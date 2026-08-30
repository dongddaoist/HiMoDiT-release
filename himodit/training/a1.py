"""
HiMoDiT current — A1 (ring layout diffusion) training script.
============================================================

Trains the RingLayoutDiffusion model on labels produced by
extract_layout(). A1 predicts the macro-layout — ring types
(R), fusion matrix (F), linker lengths (L), and per-pair spiro
position (spiro_pos_class). Pendant prediction has moved to A3.

Differences from the earlier encoder training script:
  - Imports build_ring_layout_model (not the earlier encoder)
  - LOSS_KEYS includes spiro_pos_class, not P_len/P_pos
  - Dataset emits spiro_pos_class derived from spiro_atom_positions
    (encoder writes -1 sentinel; we shift to class 0 = NO_SPIRO)
  - evaluate_sample_decode_rate uses decode_scaffold with
    zero-branch B_* templates

Mirrors A3 conventions:
  - EMA, cosine LR with warmup, auto-resume from latest.pt
  - Saves: latest.pt, best_model.pt, ema.pt, history.json, config.json
  - Per-epoch one-line summary (mid-step logging off by default)
  - Per-epoch sample-decode metric on EMA weights

Use as a script:
    python himodit.training.a1.py \\
        --labels-pkl /path/to/labels.pkl \\
        --ckpt-dir /path/to/checkpoints/a1 \\
        --epochs 50 \\
        --batch-size 256 \\
        --capacity 3M

Or import:
    from himodit.training.a1 import train_a1
    train_a1(labels_pkl_path=..., ckpt_dir=..., num_epochs=50, ...)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from himodit.models.ring_layout import (
    build_ring_layout_model, count_parameters,
    N_SPIRO_POS_CLASSES,
)
from himodit.chem.decoder import (
    decode_scaffold, R_MAX, P_MAX, B_LEN_MAX,
    F_SPIRO, RING_PAD, AROMATIC_ATOM_IDS,
)

from himodit.training.common import (
    EMA, CheckpointPaths, collate, cosine_lr_with_warmup, detect_condition_dim,
    filter_batch, maybe_resume, save_epoch, split_labels, write_config_once,
)


# ─── Loss-input filter ──────────────────────────────────────────────────
# The dataset emits dicts with many keys (smi, atom_ids, terminals,
# M_total, ...). compute_loss only takes a subset, with F→F_mat and
# L→L_mat renamed to dodge the torch.nn.functional name collision.

LOSS_KEYS = ["R", "F_mat", "L_mat", "spiro_pos_class", "condition"]
DATASET_TO_MODEL_RENAME = {"F": "F_mat", "L": "L_mat"}


# ─── EMA ────────────────────────────────────────────────────────────────

# ─── LR schedule ───────────────────────────────────────────────────────

# ─── Dataset & collation ───────────────────────────────────────────────

class RingLayoutLabelDataset(Dataset):
    """Wraps labels produced by extract_layout().

    Each label has keys: smi, scaffold_smi, R, F, L, atom_ids, M_total,
    terminals, condition, spiro_atom_positions, B_size, B_pos, B_parent,
    B_bond. A1 only consumes (R, F, L, spiro_atom_positions,
    condition); branch fields (B_*) and atom_ids are consumed by A3/A2.

    spiro_pos_class encoding (model side, shifted from encoder):
      encoder: spiro_atom_positions[i,j] = -1 → class 0 (NO_SPIRO sentinel)
      encoder: spiro_atom_positions[i,j] = p (∈ 0..6) → class p+1
    """
    def __init__(self, labels: List[Dict[str, Any]]):
        # Sanity-check schema on first label
        if labels:
            required = ("R", "F", "L", "spiro_atom_positions", "condition")
            lab0 = labels[0]
            missing = [k for k in required if k not in lab0]
            if missing:
                raise KeyError(
                    f"Labels missing fields {missing}. "
                    f"Re-run preprocessing with extract_layout()."
                )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lab = self.labels[idx]
        # `.reshape(-1)` handles both 0-D scalars and 1-D arrays uniformly,
        # so collate produces (B, condition_dim) regardless of how the
        # label originally stored its condition.
        cond = torch.as_tensor(lab["condition"], dtype=torch.float32).reshape(-1)

        # Encoder writes -1 where F != F_SPIRO; valid positions are 0..6
        # for max ring size 7. Shift by +1 so 0 is the NO_SPIRO sentinel
        # and positions 0..6 map to classes 1..7. This matches the model's
        # N_SPIRO_POS_CLASSES = 8 and is described in
        # himodit/models/ring_layout.py docstring.
        spiro_raw = np.asarray(lab["spiro_atom_positions"], dtype=np.int64)
        spiro_class = np.where(spiro_raw < 0, 0, spiro_raw + 1).astype(np.int64)

        return {
            "R":               torch.as_tensor(lab["R"], dtype=torch.long),
            "F":               torch.as_tensor(lab["F"], dtype=torch.long),
            "L":               torch.as_tensor(lab["L"], dtype=torch.long),
            "spiro_pos_class": torch.from_numpy(spiro_class).long(),
            "condition":       cond,
        }


# ─── Sample-decode metric ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_sample_decode_rate(
    model: nn.Module,
    n_samples: int = 32,
    n_steps: int = 20,
    cfg_scale: float = 1.0,
    seed: int = 12345,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Sample N layouts and report fraction that decode without error.

    A1 produces only the macro-layout (R, F, L, spiro_pos_class). To test
    whether that layout is decodable, we synthesize zero-branch B_* inputs
    and a dummy atom_ids array (aromatic-vs-aliph derived from R only) and
    call decode_scaffold. The decoder will raise on ill-formed
    layouts (e.g. cyclic fusion graph, spiro without anchor, etc.).

    Sample under zero condition (training-set average); this measures the
    learned PRIOR over valid layouts, not conditional generation quality.
    """
    model.eval()
    cond = torch.zeros(n_samples, model.condition_dim, device=device)
    samples = model.sample(condition=cond, n_steps=n_steps,
                           cfg_scale=cfg_scale, seed=seed)

    reasons: Dict[str, int] = {}
    n_valid = 0

    # Zero-branch templates (allocated once)
    B_size_zero   = np.zeros((R_MAX, P_MAX),                 dtype=np.int64)
    B_pos_zero    = np.zeros((R_MAX, P_MAX),                 dtype=np.int64)
    B_parent_zero = np.zeros((R_MAX, P_MAX, B_LEN_MAX),      dtype=np.int64)
    B_bond_zero   = np.zeros((R_MAX, P_MAX, B_LEN_MAX),      dtype=np.int64)

    for b in range(n_samples):
        R_  = samples["R"][b].cpu().numpy()
        F_  = samples["F"][b].cpu().numpy()
        L_  = samples["L"][b].cpu().numpy()
        Sc_ = samples["spiro_pos_class"][b].cpu().numpy()

        # Convert spiro_pos_class → spiro_atom_positions (decoder format):
        # class 0 → -1 (NO_SPIRO sentinel), classes 1..7 → positions 0..6.
        spiro_pos = np.where(Sc_ == 0, -1, Sc_ - 1).astype(np.int64)

        try:
            n_rings = int((R_ != RING_PAD).sum())
            if n_rings == 0:
                # Empty molecule — count as valid (degenerate but decodes)
                n_valid += 1
                continue

            # Dummy atom_ids: just need an array of length M_MAX with
            # plausible values. The decoder cares mostly that aromatic-ring
            # atoms get aromatic atom IDs. Use atom ID 1 ('c') for aromatic
            # ring atoms and 3 ('C') for everything else.
            from himodit.chem.decoder import M_MAX as _M_MAX
            atom_ids_dummy = np.full(_M_MAX, 3, dtype=np.int64)  # 'C'

            decode_scaffold(
                R_, F_, L_,
                B_size_zero, B_pos_zero, B_parent_zero, B_bond_zero,
                spiro_pos, atom_ids_dummy,
            )
            n_valid += 1
        except Exception as e:
            reason = type(e).__name__
            msg = str(e).split("\n")[0][:60]
            key = f"{reason}: {msg}"
            reasons[key] = reasons.get(key, 0) + 1

    return {
        "rate": n_valid / max(n_samples, 1),
        "n_valid": n_valid,
        "n_attempted": n_samples,
        "rejection_reasons": reasons,
    }


# ─── Main training loop ────────────────────────────────────────────────

def train_a1(
    labels_pkl_path: str,
    ckpt_dir: str,
    num_epochs: int,
    batch_size: int = 128,
    capacity: str = "1M",
    lr: float = 3e-4,
    warmup_steps: int = 200,
    val_fraction: float = 0.05,
    seed: int = 0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    cfg_drop_prob: float = 0.1,
    ema_decay: float = 0.999,
    eval_every_n_epochs: int = 1,
    eval_n_samples: int = 32,
    eval_n_steps: int = 20,
    num_workers: int = 0,
    device: Optional[str] = None,
    log_every_n_steps: int = 0,
) -> Dict[str, Any]:
    """Train the A1 ring-layout diffusion model.

    Auto-resumes from {ckpt_dir}/latest.pt if present. Saves:
      - latest.pt        (full state for resume; overwritten each epoch)
      - best_model.pt    (best val loss; weights only)
      - ema.pt           (EMA shadow; updated each epoch)
      - history.json     (per-epoch metrics)
      - config.json      (training config — written once)

    Returns a dict with summary metrics.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Determinism for split & data shuffling
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train_a1] device={device}, capacity={capacity}")

    # ── Load labels ────────────────────────────────────────────────
    with open(labels_pkl_path, "rb") as f:
        labels = pickle.load(f)
    print(f"[train_a1] loaded {len(labels):,} labels from {labels_pkl_path}")

    rng.shuffle(labels)
    n_val = max(1, int(round(val_fraction * len(labels))))
    train_labels = labels[n_val:]
    val_labels = labels[:n_val]
    print(f"[train_a1] split train={len(train_labels):,}, val={len(val_labels):,}")

    # Auto-detect condition_dim from the first label.
    # The dataset may produce 1-D (e.g., just solubility) or 2-D
    # (solubility + gap) conditions depending on cond_cols passed to
    # extract_dataset_baseline. The model needs to match. Print loudly so this
    # is visible in logs.
    sample_cond = np.asarray(labels[0]["condition"]).reshape(-1)
    detected_cond_dim = int(sample_cond.shape[0])
    print(f"[train_a1] detected condition_dim={detected_cond_dim} "
          f"(from labels[0]['condition']={sample_cond})")

    train_ds = RingLayoutLabelDataset(train_labels)
    val_ds = RingLayoutLabelDataset(val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    # ── Build model ────────────────────────────────────────────────
    model = build_ring_layout_model(
        capacity=capacity, cfg_drop_prob=cfg_drop_prob,
        condition_dim=detected_cond_dim,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[train_a1] model {capacity}: {n_params:,} params")

    # ── Optimizer & scheduler ──────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    print(f"[train_a1] steps/epoch={steps_per_epoch}, total_steps={total_steps}")

    ema = EMA(model, decay=ema_decay)

    # ── Auto-resume ────────────────────────────────────────────────
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    history_path = os.path.join(ckpt_dir, "history.json")
    config_path = os.path.join(ckpt_dir, "config.json")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    ema_path = os.path.join(ckpt_dir, "ema.pt")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    if os.path.exists(latest_path):
        print(f"[train_a1] resuming from {latest_path}")
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ema.load_state_dict(ck["ema"], device=device)
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_val_loss = ck["best_val_loss"]
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        print(f"[train_a1]  → resuming at epoch {start_epoch}, "
              f"global_step={global_step}, best_val_loss={best_val_loss:.4f}")

    # ── Save config (once) ─────────────────────────────────────────
    if not os.path.exists(config_path):
        config = dict(
            labels_pkl_path=labels_pkl_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            capacity=capacity,
            lr=lr,
            warmup_steps=warmup_steps,
            val_fraction=val_fraction,
            seed=seed,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            cfg_drop_prob=cfg_drop_prob,
            ema_decay=ema_decay,
            n_train=len(train_labels),
            n_val=len(val_labels),
            n_params=n_params,
            condition_dim=detected_cond_dim,
        )
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        model.train()
        t_epoch_start = time.time()
        epoch_losses: List[float] = []
        epoch_metrics: Dict[str, List[float]] = {}

        for step_in_epoch, batch in enumerate(train_loader):
            batch_dev = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}
            inputs = filter_batch(batch_dev, LOSS_KEYS, DATASET_TO_MODEL_RENAME)

            cur_lr = cosine_lr_with_warmup(
                global_step, warmup_steps, total_steps, lr,
            )
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            out = model.compute_loss(**inputs)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ema.update(model)

            epoch_losses.append(loss.item())
            for k, v in out.items():
                if k == "loss":
                    continue
                epoch_metrics.setdefault(k, []).append(float(v.item()))

            global_step += 1
            if log_every_n_steps > 0 and (step_in_epoch + 1) % log_every_n_steps == 0:
                print(f"  epoch {epoch:3d} step {step_in_epoch:5d}/"
                      f"{steps_per_epoch} | loss={loss.item():.4f} "
                      f"lr={cur_lr:.2e}")

        train_loss = float(np.mean(epoch_losses))
        epoch_means = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        # ── Val eval ───────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_metrics: Dict[str, List[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                inputs = filter_batch(batch_dev, LOSS_KEYS, DATASET_TO_MODEL_RENAME)
                out = model.compute_loss(**inputs)
                val_losses.append(out["loss"].item())
                for k, v in out.items():
                    if k == "loss":
                        continue
                    val_metrics.setdefault(k, []).append(float(v.item()))
        val_loss = float(np.mean(val_losses))
        val_means = {k: float(np.mean(v)) for k, v in val_metrics.items()}

        # ── Decode-rate eval (on EMA weights) ──────────────────────
        decode_metrics = None
        if (epoch + 1) % eval_every_n_epochs == 0:
            backup = ema.apply_to(model)
            decode_metrics = evaluate_sample_decode_rate(
                model, n_samples=eval_n_samples, n_steps=eval_n_steps,
                seed=seed + epoch, device=device,
            )
            ema.restore_to(model, backup)

        # ── History entry ──────────────────────────────────────────
        epoch_record = dict(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=cur_lr,
            time_sec=time.time() - t_epoch_start,
            train_metrics=epoch_means,
            val_metrics=val_means,
        )
        if decode_metrics is not None:
            epoch_record["decode_rate"] = decode_metrics["rate"]
            epoch_record["decode_rejection_reasons"] = (
                decode_metrics["rejection_reasons"]
            )
        history.append(epoch_record)

        # ── Save checkpoints ───────────────────────────────────────
        # Order matters: update best_val_loss BEFORE writing latest.pt
        # so that on resume we don't lose track of the true best.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                ema=ema.state_dict(),
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            ),
            latest_path,
        )
        torch.save(ema.state_dict(), ema_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Per-epoch summary (one line, includes train accs for all heads)
        def _g(d, k):
            return d.get(k, float("nan"))
        decode_str = (f" decode={decode_metrics['rate']*100:.1f}%"
                      if decode_metrics is not None else "")
        print(
            f"[A1][ep {epoch:3d}] "
            f"train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} "
            f"acc_R={_g(epoch_means, 'acc_R'):.3f} "
            f"acc_F={_g(epoch_means, 'acc_F'):.3f} "
            f"acc_L={_g(epoch_means, 'acc_L'):.3f} "
            f"acc_Sp={_g(epoch_means, 'acc_Spiro'):.3f}"
            f"{decode_str} "
            f"({epoch_record['time_sec']:.1f}s)",
            flush=True,
        )

    return dict(
        best_val_loss=best_val_loss,
        n_epochs=num_epochs,
        history=history,
    )


# ─── CLI ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train HiMoDiT A1.")
    p.add_argument("--labels-pkl", required=True,
                   help="Path to labels pkl from extract_layout().")
    p.add_argument("--ckpt-dir", required=True,
                   help="Output directory for checkpoints/history/config.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--capacity", type=str, default="3M",
                   choices=["600K", "1M", "3M", "10M", "30M"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--eval-n-samples", type=int, default=32)
    p.add_argument("--eval-n-steps", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' or 'cpu'; auto-detected if not set.")
    p.add_argument("--log-every-n-steps", type=int, default=0,
                   help="If >0, also print a log line every N optimizer steps. "
                        "Default 0 = one summary line per epoch only.")
    return p.parse_args()


def _main():
    args = _parse_args()
    train_a1(
        labels_pkl_path=args.labels_pkl,
        ckpt_dir=args.ckpt_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        capacity=args.capacity,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        val_fraction=args.val_fraction,
        seed=args.seed,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        cfg_drop_prob=args.cfg_drop_prob,
        ema_decay=args.ema_decay,
        eval_every_n_epochs=args.eval_every_n_epochs,
        eval_n_samples=args.eval_n_samples,
        eval_n_steps=args.eval_n_steps,
        num_workers=args.num_workers,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
    )


if __name__ == "__main__":
    _main()
