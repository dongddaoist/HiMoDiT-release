"""
HiMoDiT current — A3 (branch topology) training script.
======================================================

Trains the BranchTopologyModel on labels produced by
extract_layout() / extract_dataset(). A3 predicts per-(ring,
slot) branch topology (B_size, B_pos, B_parent, B_bond) conditioned on
the macro-layout (R, F, L, spiro_atom_positions) and property condition.

Mirrors himodit.training.a1.py conventions:
  - EMA with warmup-aware effective decay
  - Cosine LR schedule with linear warmup
  - LOSS_KEYS / _filter_batch_to_loss_inputs
  - train_a3(...) with auto-resume from latest.pt
  - Saves: latest.pt, best_model.pt, ema.pt, history.json, config.json
  - Per-epoch validity metric on EMA weights (samples a small batch,
    checks structural constraints)

Use as a script:
    python himodit.training.a3.py \
        --labels-pkl /path/to/labels.pkl \
        --ckpt-dir /path/to/checkpoints/a3 \
        --epochs 100 \
        --batch-size 256 \
        --capacity 10M

Or import:
    from himodit.training.a3 import train_a3
    train_a3(labels_pkl_path=..., ckpt_dir=..., num_epochs=100, ...)
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

from himodit.models.branch_topology import (
    BranchTopologyModel,
    build_branch_topology_model,
    count_parameters,
    remap_spiro_pos_sentinel,
    NO_SPIRO_CLS,
    N_B_SIZE_CLASSES,
    N_B_POS_CLASSES,
    N_B_PARENT_CLASSES,
    N_B_BOND_CLASSES,
)
from himodit.chem.decoder import R_MAX, P_MAX, B_LEN_MAX, RING_PAD

from himodit.training.common import (
    EMA, CheckpointPaths, collate, cosine_lr_with_warmup, detect_condition_dim,
    filter_batch, maybe_resume, save_epoch, split_labels, write_config_once,
)


# ─── Loss-input filter ────────────────────────────────────────────────

LOSS_KEYS = [
    "R", "F_mat", "L_mat", "spiro_pos",
    "B_size", "B_pos", "B_parent", "B_bond",
    "condition",
]
DATASET_TO_MODEL_RENAME = {"F": "F_mat", "L": "L_mat"}


# ─── EMA (copied verbatim from himodit.training.a1.py) ──────────────

# ─── LR schedule ──────────────────────────────────────────────────────

# ─── Dataset ──────────────────────────────────────────────────────────

class BranchTopologyDataset(Dataset):
    """Wraps the label list. Each __getitem__ loads:
      R, F, L                  (R_MAX,), (R_MAX, R_MAX), (R_MAX, R_MAX)
      spiro_pos                (R_MAX, R_MAX) — encoder's -1 remapped
      B_size, B_pos            (R_MAX, P_MAX)
      B_parent, B_bond         (R_MAX, P_MAX, B_LEN_MAX)
      condition                (condition_dim,)

    The encoder writes spiro_atom_positions[k,j] = -1 wherever
    F[k,j] != F_SPIRO. We map -1 → NO_SPIRO_CLS at dataset load time
    so the model's embedding layer can consume the values.
    """
    def __init__(self, labels: List[Dict[str, Any]]):
        # Sanity-check that labels are current (have B_size etc.)
        if labels:
            required = ("B_size", "B_pos", "B_parent", "B_bond",
                        "spiro_atom_positions")
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
        # spiro_pos sentinel remap: -1 → NO_SPIRO_CLS (=7).
        spiro_raw = np.asarray(lab["spiro_atom_positions"], dtype=np.int64)
        spiro_remapped = spiro_raw.copy()
        spiro_remapped[spiro_remapped < 0] = NO_SPIRO_CLS

        cond = torch.as_tensor(
            lab["condition"], dtype=torch.float32,
        ).reshape(-1)

        return {
            "R":         torch.as_tensor(lab["R"],         dtype=torch.long),
            "F":         torch.as_tensor(lab["F"],         dtype=torch.long),
            "L":         torch.as_tensor(lab["L"],         dtype=torch.long),
            "spiro_pos": torch.from_numpy(spiro_remapped).long(),
            "B_size":    torch.as_tensor(lab["B_size"],    dtype=torch.long),
            "B_pos":     torch.as_tensor(lab["B_pos"],     dtype=torch.long),
            "B_parent":  torch.as_tensor(lab["B_parent"],  dtype=torch.long),
            "B_bond":    torch.as_tensor(lab["B_bond"],    dtype=torch.long),
            "condition": cond,
        }


# ─── Structural-validity metric ───────────────────────────────────────

@torch.no_grad()
def evaluate_structural_validity(
    model: BranchTopologyModel,
    val_loader: DataLoader,
    n_batches: int = 4,
    cfg_scale: float = 1.0,
    seed: int = 12345,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Sample (B_size, B_pos, B_parent, B_bond) for `n_batches` of layouts
    drawn from val_loader, then check structural validity:

      - B_size in [0, B_LEN_MAX]
      - For active slots: B_pos in [0, ring_size_k)
      - For active atoms (i < B_size):
          B_parent[i] in [0, i]                     (causal tree)
          B_parent[0] == 0                          (first atom attaches to ring)
          B_bond[i] in {1, 2, 3, 4}                 (non-zero bond)

    Returns per-check pass rates and overall validity.
    """
    model.eval()
    from himodit.chem.decoder import RING_TYPE_INFO

    # Ring-size lookup table
    max_rt = max(RING_TYPE_INFO.keys())
    ring_size_lut = torch.zeros(max_rt + 1, dtype=torch.long, device=device)
    for rt, (sz, _) in RING_TYPE_INFO.items():
        ring_size_lut[rt] = sz

    tot_slots = 0
    pass_size = 0
    tot_active_slots = 0
    pass_pos = 0
    tot_active_atoms = 0
    pass_parent_causal = 0
    pass_parent_first = 0   # tracked separately
    pass_bond_nonzero = 0

    iters = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(iters)
        except StopIteration:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.sample(
            R=batch["R"], F_mat=batch["F"], L_mat=batch["L"],
            spiro_pos=batch["spiro_pos"], condition=batch["condition"],
            cfg_scale=cfg_scale, post_process=True, seed=seed,
        )
        B_size   = out["B_size"]
        B_pos    = out["B_pos"]
        B_parent = out["B_parent"]
        B_bond   = out["B_bond"]

        # B_size range check
        valid_size = (B_size >= 0) & (B_size <= B_LEN_MAX)
        tot_slots += B_size.numel()
        pass_size += valid_size.sum().item()

        # B_pos range check (only active slots)
        ring_sizes = ring_size_lut[batch["R"]]  # (B, R_MAX)
        ring_sizes_bp = ring_sizes.unsqueeze(-1).expand_as(B_pos)
        active_slot = (B_size > 0) & (ring_sizes_bp > 0)
        valid_pos = (B_pos >= 0) & (B_pos < ring_sizes_bp)
        tot_active_slots += active_slot.sum().item()
        pass_pos += (valid_pos & active_slot).sum().item()

        # B_parent / B_bond per-atom checks
        idx_range = torch.arange(B_LEN_MAX, device=device)
        atom_active = idx_range[None, None, None, :] < B_size.unsqueeze(-1)

        # Causal: parent[i] ∈ [0, i]
        upper_bound = idx_range[None, None, None, :]
        causal_ok = (B_parent >= 0) & (B_parent <= upper_bound)
        tot_active_atoms += atom_active.sum().item()
        pass_parent_causal += (causal_ok & atom_active).sum().item()

        # First-atom rule: parent[0] == 0 wherever slot is active
        first_atom_active = active_slot.unsqueeze(-1).clone()
        first_atom_active = first_atom_active & (
            (idx_range[None, None, None, :] == 0)
        )
        first_atom_ok = (B_parent == 0) & first_atom_active
        pass_parent_first += first_atom_ok.sum().item()

        # Bond non-zero (active atoms)
        bond_ok = (B_bond >= 1) & (B_bond <= 4)
        pass_bond_nonzero += (bond_ok & atom_active).sum().item()

    def _safe_rate(num: int, den: int) -> float:
        return num / den if den > 0 else 1.0

    return {
        "rate_size_range":   _safe_rate(pass_size, tot_slots),
        "rate_pos_in_ring":  _safe_rate(pass_pos, tot_active_slots),
        "rate_parent_causal": _safe_rate(pass_parent_causal, tot_active_atoms),
        "rate_parent_first":  _safe_rate(pass_parent_first, tot_active_slots),
        "rate_bond_nonzero":  _safe_rate(pass_bond_nonzero, tot_active_atoms),
        "n_slots_seen":      tot_slots,
        "n_active_slots":    tot_active_slots,
        "n_active_atoms":    tot_active_atoms,
    }


# ─── Main training loop ───────────────────────────────────────────────

def train_a3(
    labels_pkl_path: str,
    ckpt_dir: str,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    min_lr_frac: float = 0.1,
    capacity: str = "10M",
    val_fraction: float = 0.05,
    cfg_drop_prob: float = 0.3,
    grad_clip: float = 1.0,
    ema_decay: float = 0.999,
    seed: int = 42,
    val_every_n_epochs: int = 1,
    checkpoint_every_n_epochs: int = 5,
    n_val_batches: int = 4,
    num_workers: int = 2,
    init_from: Optional[str] = None,
    device: Optional[torch.device] = None,
    log_every_n_steps: int = 0,
) -> Dict[str, Any]:
    """Train A3 with auto-resume from latest.pt in ckpt_dir.

    Returns a dict with the final training history.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Accept a plain string like "cpu" or "cuda", as the other trainers and
    # the CLI do; device.type is used below.
    device = torch.device(device)

    # ── Reproducibility ──────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ── Load labels ──────────────────────────────────────────────────
    print(f"[A3] Loading labels from {labels_pkl_path}", flush=True)
    with open(labels_pkl_path, "rb") as f:
        labels = pickle.load(f)
    if not isinstance(labels, list):
        # extract_dataset_baseline writes a list; accept either {"labels": [...]}.
        labels = labels.get("labels", labels)
    print(f"[A3] Loaded {len(labels)} labels", flush=True)

    # ── Train/val split ──────────────────────────────────────────────
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    n_val = max(1, int(len(labels) * val_fraction))
    val_idx = set(indices[:n_val])
    train_labels = [labels[i] for i in indices if i not in val_idx]
    val_labels   = [labels[i] for i in indices if i in val_idx]
    print(
        f"[A3] Split: {len(train_labels)} train / {len(val_labels)} val",
        flush=True,
    )

    train_ds = BranchTopologyDataset(train_labels)
    val_ds   = BranchTopologyDataset(val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
        drop_last=False, pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────
    model = build_branch_topology_model(
        capacity=capacity,
        cfg_drop_prob=cfg_drop_prob,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[A3] Built model (capacity={capacity}, params={n_params:,})",
          flush=True)

    # Optional warm-start
    if init_from is not None and os.path.isfile(init_from):
        print(f"[A3] Warm-starting from {init_from}", flush=True)
        sd = torch.load(init_from, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:    print(f"  missing keys: {len(missing)}")
        if unexpected: print(f"  unexpected keys: {len(unexpected)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    ema = EMA(model, decay=ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    # ── Auto-resume from latest.pt ───────────────────────────────────
    history: Dict[str, List[Any]] = {
        "epoch": [], "step": [], "train_loss": [],
        "loss_size": [], "loss_pos": [], "loss_parent": [], "loss_bond": [],
        "acc_size": [], "acc_pos": [], "acc_parent": [], "acc_bond": [],
        "val_loss": [], "val_metrics": [], "wall_time": [],
    }
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest_path):
        print(f"[A3] Resuming from {latest_path}", flush=True)
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"], device=device)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", history)
        print(
            f"[A3] Resumed at epoch={start_epoch}, "
            f"global_step={global_step}, best_val_loss={best_val_loss:.4f}",
            flush=True,
        )

    # ── Save config snapshot ─────────────────────────────────────────
    config_snapshot = {
        "labels_pkl_path": labels_pkl_path,
        "ckpt_dir":        ckpt_dir,
        "num_epochs":      num_epochs,
        "batch_size":      batch_size,
        "lr":   lr,
        "weight_decay":    weight_decay,
        "warmup_frac":     warmup_frac,
        "min_lr_frac":     min_lr_frac,
        "capacity":        capacity,
        "n_params":        n_params,
        "val_fraction":        val_fraction,
        "cfg_drop_prob":   cfg_drop_prob,
        "grad_clip":       grad_clip,
        "ema_decay":       ema_decay,
        "seed":            seed,
        "steps_per_epoch": steps_per_epoch,
        "total_steps":     total_steps,
        "warmup_steps":    warmup_steps,
        "device":          str(device),
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running = {
            "loss": 0.0, "loss_size": 0.0, "loss_pos": 0.0,
            "loss_parent": 0.0, "loss_bond": 0.0,
            "acc_size": 0.0, "acc_pos": 0.0,
            "acc_parent": 0.0, "acc_bond": 0.0,
        }
        n_seen = 0

        for batch in train_loader:
            # LR schedule
            lr = cosine_lr_with_warmup(
                global_step, warmup_steps, total_steps,
                lr, min_lr_frac,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            inputs = filter_batch(batch, LOSS_KEYS, DATASET_TO_MODEL_RENAME)

            optimizer.zero_grad(set_to_none=True)
            out = model.compute_loss(**inputs)
            loss = out["loss"]
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)

            # Track
            for k in running:
                v = out.get(k, None)
                if v is not None:
                    running[k] += float(v.detach().cpu()) * batch["R"].shape[0]
            n_seen += batch["R"].shape[0]
            global_step += 1

            if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                wt = time.time() - t_start
                print(
                    f"[A3][ep {epoch:3d} step {global_step:7d}] "
                    f"loss={loss.item():.4f} "
                    f"lr={lr:.2e} "
                    f"acc_size={out['acc_size'].item():.3f} "
                    f"acc_pos={out['acc_pos'].item():.3f} "
                    f"acc_par={out['acc_parent'].item():.3f} "
                    f"acc_bnd={out['acc_bond'].item():.3f} "
                    f"wt={wt:.0f}s",
                    flush=True,
                )

        # Epoch averages
        avg = {k: v / max(n_seen, 1) for k, v in running.items()}

        # ── Always-on per-epoch summary (one line per epoch) ─────────
        epoch_wt = time.time() - t_start
        print(
            f"[A3][ep {epoch:3d}] "
            f"train_loss={avg['loss']:.4f} "
            f"lr={lr:.2e} "
            f"acc_size={avg['acc_size']:.3f} "
            f"acc_pos={avg['acc_pos']:.3f} "
            f"acc_par={avg['acc_parent']:.3f} "
            f"acc_bnd={avg['acc_bond']:.3f} "
            f"wt={epoch_wt:.0f}s",
            flush=True,
        )

        # ── Validation (on EMA weights) ──────────────────────────────
        do_val = ((epoch + 1) % val_every_n_epochs == 0) or (epoch == num_epochs - 1)
        if do_val:
            backup = ema.apply_to(model)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to(device) for k, v in vbatch.items()}
                    vinputs = filter_batch(vbatch, LOSS_KEYS, DATASET_TO_MODEL_RENAME)
                    vout = model.compute_loss(**vinputs)
                    val_losses.append(float(vout["loss"].detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            val_metrics = evaluate_structural_validity(
                model, val_loader, n_batches=n_val_batches,
                cfg_scale=1.0, seed=seed + epoch, device=device,
            )
            ema.restore_to(model, backup)
            model.train()

            print(
                f"[A3][ep {epoch:3d} EVAL] "
                f"val_loss={val_loss:.4f} "
                f"rate_pos_in_ring={val_metrics['rate_pos_in_ring']:.3f} "
                f"rate_parent_causal={val_metrics['rate_parent_causal']:.3f} "
                f"rate_bond_nonzero={val_metrics['rate_bond_nonzero']:.3f}",
                flush=True,
            )
        else:
            val_loss = float("nan")
            val_metrics = {}

        # ── Record history ───────────────────────────────────────────
        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["train_loss"].append(avg["loss"])
        history["loss_size"].append(avg["loss_size"])
        history["loss_pos"].append(avg["loss_pos"])
        history["loss_parent"].append(avg["loss_parent"])
        history["loss_bond"].append(avg["loss_bond"])
        history["acc_size"].append(avg["acc_size"])
        history["acc_pos"].append(avg["acc_pos"])
        history["acc_parent"].append(avg["acc_parent"])
        history["acc_bond"].append(avg["acc_bond"])
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        history["wall_time"].append(time.time() - t_start)

        # ── Save latest.pt (every epoch) ─────────────────────────────
        torch.save({
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "ema":           ema.state_dict(),
            "epoch":         epoch,
            "global_step":   global_step,
            "best_val_loss": best_val_loss,
            "history":       history,
            "config":        config_snapshot,
        }, latest_path)

        # ── Save best_model.pt if val_loss improved ─────────────────
        if do_val and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model":       model.state_dict(),
                "ema":         ema.state_dict(),
                "epoch":       epoch,
                "global_step": global_step,
                "val_loss":    val_loss,
                "config":      config_snapshot,
            }, os.path.join(ckpt_dir, "best_model.pt"))
            torch.save(ema.state_dict(),
                       os.path.join(ckpt_dir, "ema.pt"))
            print(f"[A3] Saved best_model.pt (val_loss={val_loss:.4f})",
                  flush=True)

        # ── Periodic snapshot ────────────────────────────────────────
        if (epoch + 1) % checkpoint_every_n_epochs == 0:
            snap = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "model":       model.state_dict(),
                "ema":         ema.state_dict(),
                "epoch":       epoch,
                "global_step": global_step,
                "config":      config_snapshot,
            }, snap)

        # ── history.json (for plotting offline) ──────────────────────
        with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
            # val_metrics may contain numpy types; coerce to plain Python.
            def _coerce(o):
                if isinstance(o, (np.integer, np.int64)): return int(o)
                if isinstance(o, (np.floating, np.float32)): return float(o)
                if isinstance(o, dict):
                    return {k: _coerce(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_coerce(x) for x in o]
                return o
            json.dump(_coerce(history), f, indent=2)

    print(f"[A3] Training complete. Best val_loss={best_val_loss:.4f}",
          flush=True)
    return {
        "best_val_loss": best_val_loss,
        "history":       history,
        "config":        config_snapshot,
    }


# ─── argparse main ────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train HiMoDiT A3 (branch topology) model.",
    )
    p.add_argument("--labels-pkl", required=True, type=str,
                   help="Path to labels.pkl (from extract_layout).")
    p.add_argument("--ckpt-dir",   required=True, type=str,
                   help="Directory for checkpoints + history.json.")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac",  type=float, default=0.05)
    p.add_argument("--min-lr-frac",  type=float, default=0.1)
    p.add_argument("--capacity",   type=str, default="10M",
                   choices=["1M", "3M", "10M", "30M"])
    p.add_argument("--val-frac",   type=float, default=0.05)
    p.add_argument("--cfg-drop-prob", type=float, default=0.3)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--ema-decay",  type=float, default=0.999)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--val-every-n-epochs",        type=int, default=1)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    p.add_argument("--n-val-batches",             type=int, default=4)
    p.add_argument("--num-workers",               type=int, default=2)
    p.add_argument("--init-from", type=str, default=None,
                   help="Optional warm-start checkpoint.")
    p.add_argument("--log-every-n-steps", type=int, default=0,
                   help="If >0, also print a log line every N optimizer steps. "
                        "Default 0 means one summary line per epoch only.")
    return p.parse_args()


def _main():
    args = _parse_args()
    train_a3(
        labels_pkl_path=args.labels_pkl,
        ckpt_dir=args.ckpt_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        min_lr_frac=args.min_lr_frac,
        capacity=args.capacity,
        val_fraction=args.val_fraction,
        cfg_drop_prob=args.cfg_drop_prob,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        seed=args.seed,
        val_every_n_epochs=args.val_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        n_val_batches=args.n_val_batches,
        num_workers=args.num_workers,
        init_from=args.init_from,
        log_every_n_steps=args.log_every_n_steps,
    )


if __name__ == "__main__":
    _main()
