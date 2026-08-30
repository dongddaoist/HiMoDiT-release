"""
HiMoDiT current — Terminal (fragment_stage2) training script.
============================================================

Trains TerminalFragmentModel on labels produced by extract_layout().
Terminal predicts a fragment id per scaffold atom, conditioned on
ground-truth scaffold (atom_ids + bond_classes + atom_mask) plus the
property condition.

Differences from the earlier encoder training script:
  - Imports decode_scaffold + aromatic_constraint_mask
  - Dataset uses fields (B_size/B_pos/B_parent/B_bond + spiro_atom_positions)
    instead of the earlier encoder's P_len/P_pos
  - NAME_TO_SMARTS_ID extended from K=9 → K=22 (full current-ZINC vocab)
  - Default num_fragments raised from 9 → 22 (was silently dropping
    Cl/Br/I/CN/NO2/OCH3/CF3/Thiol/AcylHalide/Cyanate/Thiocyanate/
    Isothiocyanate/Isonitrile into class 0)

Mirrors A1/A3 conventions:
  - EMA, cosine LR with warmup, auto-resume from latest.pt
  - Saves: latest.pt, best_model.pt, ema.pt, history.json, config.json
  - Per-epoch one-line summary (mid-step logging off by default)
  - Per-epoch sample-quality eval on EMA weights

Use as a script:
    python himodit.training.terminal.py \\
        --labels-pkl ./labels.pkl \\
        --ckpt-dir   ./checkpoints/terminal \\
        --epochs 40 --batch-size 256 --capacity 9M
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from himodit.models.terminal_fragment import (
    build_terminal_model, count_parameters, compute_class_weights,
    load_v5_3_warmstart,
    DEFAULT_NUM_FRAGMENTS,
)
from himodit.chem.decoder import (
    decode_scaffold, aromatic_constraint_mask,
    M_MAX,
)

from himodit.training.common import (
    EMA, CheckpointPaths, collate, cosine_lr_with_warmup, detect_condition_dim,
    filter_batch, maybe_resume, save_epoch, split_labels, write_config_once,
)


# terminal vocab: K=22 (was K=9 in the earlier encoder RedDB; expanded for ZINC).
# DEFAULT_NUM_FRAGMENTS in terminal_fragment_diffusion.py is 9; we override
# at every call site below. Model output dim = NUM_TERMINAL_FRAGMENTS + 1
# (class 0 = "no decoration").
NUM_TERMINAL_FRAGMENTS = 22

# Atom vocabulary: K=16, including six charged
# classes for ZINC: O-, N+, n+, N-, n-, P+). The Terminal model's atom_embed
# defaults to DEFAULT_NUM_ATOM_TYPES=10 (the RedDB vocab); must override to 16
# or atom_ids ≥ 10 trigger a CUDA index-out-of-range assert at training time.
NUM_ATOM_TYPES = 16


# ─── Loss-input filter ──────────────────────────────────────────────

LOSS_KEYS = [
    "scaffold_atom_ids", "scaffold_bond_classes", "scaffold_atom_mask",
    "site_fragment_ids", "condition",
]


# ─── EMA ────────────────────────────────────────────────────────────

# ─── LR schedule ────────────────────────────────────────────────────

# ─── Dataset ───────────────────────────────────────────────────────

# SMARTS-id → model-class: model_class = SMARTS_id + 1
# (model class 0 reserved for "no decoration")
def _smarts_id_to_model_class(smarts_id: int) -> int:
    return smarts_id + 1


class TerminalDataset(Dataset):
    """Wraps layout-label list. Builds:
      - scaffold_atom_ids (B, M_MAX) from label["atom_ids"]
      - scaffold_bond_classes (B, M_MAX, M_MAX) via build_bond_classes
      - scaffold_atom_mask (B, M_MAX) via build_bond_classes
      - site_fragment_ids (B, M_MAX) by walking label["terminals"] and
        marking each host_canonical_idx with the fragment's model class
      - condition

    Requires labels to have 'host_canonical_idx' on each terminal (added
    by the B5 patch to extract_layout_baseline). Falls back gracefully with a
    clear error if old labels are passed in.
    """
    # SMARTS id → model class (lookup; we store the inverse map for
    # named-fragment validation). current-ZINC: full K=22 vocab. The first 9
    # entries (OH..=S) are the the earlier encoder RedDB names; entries 9-21 added in
    # Indices match
    # preprocessing/terminal_smarts_v5_4_zinc.py.
    NAME_TO_SMARTS_ID = {
        "OH":   0, "COOH": 1, "NH2":  2, "SO3H": 3, "F":    4, "CH3":  5,
        "=O":   6, "=NH":  7, "=S":   8,
        "Cl":   9, "Br":  10, "I":   11, "CN":  12, "NO2": 13, "OCH3": 14, "CF3": 15,
        "Thiol": 16, "AcylHalide": 17, "Cyanate": 18,
        "Thiocyanate": 19, "Isothiocyanate": 20, "Isonitrile": 21,
    }

    def __init__(self, labels: List[Dict[str, Any]],
                 num_fragments: int = NUM_TERMINAL_FRAGMENTS):
        # Sanity-check schema: B_* fields and spiro_atom_positions
        # must be present (these replace the earlier encoder's P_len/P_pos)
        if labels:
            required_fields = ("B_size", "B_pos", "B_parent", "B_bond",
                              "spiro_atom_positions")
            lab0 = labels[0]
            missing = [k for k in required_fields if k not in lab0]
            if missing:
                raise KeyError(
                    f"Labels missing fields {missing}. "
                    f"Re-run preprocessing with extract_layout()."
                )
        # Sanity-check that labels carry host_canonical_idx
        for lab in labels:
            if not lab["terminals"]:
                continue
            t0 = lab["terminals"][0]
            if "host_canonical_idx" not in t0:
                raise KeyError(
                    "Labels are missing 'host_canonical_idx' on terminals. "
                    "Re-run extract_dataset_baseline() with the B5-patched "
                    "ring_layout_dataset.py to regenerate labels.pkl."
                )
            break
        self.labels = labels
        self.num_fragments = num_fragments

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lab = self.labels[idx]
        R  = np.asarray(lab["R"], dtype=np.int64)
        F_ = np.asarray(lab["F"], dtype=np.int64)
        L_ = np.asarray(lab["L"], dtype=np.int64)
        B_size   = np.asarray(lab["B_size"],   dtype=np.int64)
        B_pos    = np.asarray(lab["B_pos"],    dtype=np.int64)
        B_parent = np.asarray(lab["B_parent"], dtype=np.int64)
        B_bond   = np.asarray(lab["B_bond"],   dtype=np.int64)
        spiro_atom_positions = np.asarray(
            lab["spiro_atom_positions"], dtype=np.int64,
        )
        atom_ids_compact = np.asarray(lab["atom_ids"], dtype=np.int64)
        M_total = int(lab["M_total"])

        atom_ids_padded = np.zeros(M_MAX, dtype=np.int64)
        atom_ids_padded[:M_total] = atom_ids_compact

        # decoder produces bond_classes + atom_mask consistent with
        # the encoder's canonical traversal (branches included).
        bond_classes, atom_mask = decode_scaffold(
            R, F_, L_, B_size, B_pos, B_parent, B_bond,
            spiro_atom_positions, atom_ids_padded, M_MAX_out=M_MAX,
        )

        # Build site_fragment_ids: 0 everywhere, then mark hosts.
        site_fragment_ids = np.zeros(M_MAX, dtype=np.int64)
        for t in lab["terminals"]:
            host_canon = int(t["host_canonical_idx"])
            if host_canon < 0 or host_canon >= M_total:
                # Defensive: skip terminals whose host fell outside the
                # canonical scaffold. Shouldn't happen with patched B2.1.
                continue
            name = t["name"]
            sid = self.NAME_TO_SMARTS_ID.get(name, None)
            if sid is None:
                continue  # unknown terminal name — skip silently
            cls = _smarts_id_to_model_class(sid)
            if cls > self.num_fragments:
                # Terminal vocab id out of range for this model. E.g.
                # =O (cls=7) seen but model trained with K=6. Skip.
                continue
            site_fragment_ids[host_canon] = cls

        cond = torch.as_tensor(lab["condition"],
                               dtype=torch.float32).reshape(-1)
        return {
            "scaffold_atom_ids":     torch.from_numpy(atom_ids_padded).long(),
            "scaffold_bond_classes": torch.from_numpy(bond_classes).long(),
            "scaffold_atom_mask":    torch.from_numpy(atom_mask).bool(),
            "site_fragment_ids":     torch.from_numpy(
                                        site_fragment_ids).long(),
            "condition":             cond,
        }


# ─── Class-weights computation over training set ────────────────────

def compute_class_weights_from_labels(
    labels: List[Dict[str, Any]],
    num_fragments: int = NUM_TERMINAL_FRAGMENTS,
    smoothing: float = 1.0,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Walk labels, count site_fragment_id occurrences, return
    (weights tensor of shape (K+1,), counts dict)."""
    counts: Dict[int, int] = {i: 0 for i in range(num_fragments + 1)}
    name_to_sid = TerminalDataset.NAME_TO_SMARTS_ID
    for lab in labels:
        M_total = int(lab["M_total"])
        # site 0 = no decoration is the default; count atoms first
        n_total = M_total
        n_decorated = 0
        for t in lab["terminals"]:
            sid = name_to_sid.get(t["name"], None)
            if sid is None:
                continue
            cls = _smarts_id_to_model_class(sid)
            if cls > num_fragments: continue
            host_canon = int(t.get("host_canonical_idx", -1))
            if host_canon < 0 or host_canon >= M_total: continue
            counts[cls] += 1
            n_decorated += 1
        counts[0] += max(n_total - n_decorated, 0)
    w = compute_class_weights(counts, num_fragments=num_fragments,
                               smoothing=smoothing)
    return w, counts


# ─── Sample-quality eval ────────────────────────────────────────────

@torch.no_grad()
def evaluate_sample_quality(
    model: nn.Module, eval_dataset: Dataset,
    n_samples: int = 64, n_steps: int = 8, seed: int = 12345,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Sample fragment ids on ground-truth scaffolds; report acc and
    nonzero recall (how often the model picks a real fragment when one
    is expected)."""
    model.eval()
    n_eval = min(n_samples, len(eval_dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eval_dataset), size=n_eval, replace=False)
    items = [eval_dataset[int(i)] for i in indices]
    batch = {k: torch.stack([it[k] for it in items], dim=0).to(device)
             for k in items[0]}

    samples = model.sample(
        scaffold_atom_ids=batch["scaffold_atom_ids"],
        scaffold_bond_classes=batch["scaffold_bond_classes"],
        scaffold_atom_mask=batch["scaffold_atom_mask"],
        condition=batch["condition"], n_steps=n_steps, seed=seed,
    )

    target = batch["site_fragment_ids"]
    mask = batch["scaffold_atom_mask"]

    correct = (samples == target) & mask
    overall_acc = (correct.sum().float()
                   / mask.sum().clamp(min=1).float()).item()

    # Among real-fragment positions (target ≠ 0), what fraction
    # are correctly identified?
    real_frag = (target != 0) & mask
    if real_frag.any():
        nonzero_acc = ((samples[real_frag] == target[real_frag])
                        .float().mean().item())
    else:
        nonzero_acc = float("nan")

    # Among samples predicted as decorated (sample ≠ 0), what fraction
    # are at positions where target ≠ 0? (precision)
    pred_decorated = (samples != 0) & mask
    if pred_decorated.any():
        precision = (((samples == target) & pred_decorated)
                     .float().sum().item()
                     / pred_decorated.float().sum().item())
    else:
        precision = float("nan")

    # Class-0 baseline: if model always predicted 0
    baseline_acc = (((target == 0) & mask).float().sum()
                    / mask.sum().clamp(min=1).float()).item()

    return {
        "atom_acc": overall_acc,
        "nonzero_acc": nonzero_acc,
        "precision": precision,
        "baseline_acc": baseline_acc,
        "n_evaluated": n_eval,
    }


# ─── Main training loop ─────────────────────────────────────────────

def train_terminal(
    labels_pkl_path: str,
    ckpt_dir: str,
    num_epochs: int,
    batch_size: int = 256,
    capacity: str = "9M",
    num_fragments: int = NUM_TERMINAL_FRAGMENTS,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    val_fraction: float = 0.05,
    seed: int = 0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    bias_enabled: bool = True,
    use_class_weights: bool = True,
    class_weight_smoothing: float = 1.0,
    ema_decay: float = 0.999,
    eval_every_n_epochs: int = 1,
    eval_n_samples: int = 64,
    eval_n_steps: int = 8,
    init_from: Optional[str] = None,
    num_workers: int = 0,
    device: Optional[str] = None,
    log_every_n_steps: int = 0,
) -> Dict[str, Any]:
    os.makedirs(ckpt_dir, exist_ok=True)
    rng = random.Random(seed); np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train_terminal] device={device}, capacity={capacity}, "
          f"num_fragments={num_fragments}, bias_enabled={bias_enabled}")

    with open(labels_pkl_path, "rb") as f:
        labels = pickle.load(f)
    print(f"[train_terminal] loaded {len(labels):,} labels")

    rng.shuffle(labels)
    n_val = max(1, int(round(val_fraction * len(labels))))
    train_labels = labels[n_val:]
    val_labels = labels[:n_val]
    print(f"[train_terminal] split train={len(train_labels):,}, "
          f"val={len(val_labels):,}")

    # Auto-detect condition_dim
    sample_cond = np.asarray(labels[0]["condition"]).reshape(-1)
    detected_cond_dim = int(sample_cond.shape[0])
    print(f"[train_terminal] detected condition_dim={detected_cond_dim}")

    train_ds = TerminalDataset(train_labels, num_fragments=num_fragments)
    val_ds = TerminalDataset(val_labels, num_fragments=num_fragments)

    # Compute class weights from training set (once)
    class_weights = None
    counts = None
    if use_class_weights:
        class_weights, counts = compute_class_weights_from_labels(
            train_labels, num_fragments=num_fragments,
            smoothing=class_weight_smoothing,
        )
        print(f"[train_terminal] class counts (train): {counts}")
        print(f"[train_terminal] class weights: "
              f"{class_weights.detach().cpu().numpy().round(3).tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    # Model
    model = build_terminal_model(
        capacity=capacity, num_fragments=num_fragments,
        num_atom_types=NUM_ATOM_TYPES,
        condition_dim=detected_cond_dim, bias_enabled=bias_enabled,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[train_terminal] model {capacity}: {n_params:,} params "
          f"(num_atom_types={NUM_ATOM_TYPES}, num_fragments={num_fragments})")

    # Optional warm-start from the earlier version
    warmstart_status: Optional[Dict[str, str]] = None
    if init_from is not None and not os.path.exists(
            os.path.join(ckpt_dir, "latest.pt")):
        print(f"[train_terminal] warm-starting from {init_from}")
        v53_state = torch.load(init_from, map_location=device,
                                weights_only=False)
        # Accept either a raw state_dict or a full ckpt dict containing
        # "model" key.
        if isinstance(v53_state, dict) and "model" in v53_state:
            v53_state = v53_state["model"]
        warmstart_status = load_v5_3_warmstart(
            model, v53_state, strict_shape_check=False, verbose=True,
        )

    if class_weights is not None:
        class_weights = class_weights.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    print(f"[train_terminal] steps/epoch={steps_per_epoch}, "
          f"total_steps={total_steps}")

    ema = EMA(model, decay=ema_decay)

    # Auto-resume
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    history_path = os.path.join(ckpt_dir, "history.json")
    config_path = os.path.join(ckpt_dir, "config.json")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    ema_path = os.path.join(ckpt_dir, "ema.pt")

    start_epoch = 0; global_step = 0; best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    if os.path.exists(latest_path):
        print(f"[train_terminal] resuming from {latest_path}")
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ema.load_state_dict(ck["ema"], device=device)
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_val_loss = ck["best_val_loss"]
        if os.path.exists(history_path):
            with open(history_path, "r") as f: history = json.load(f)
        print(f"[train_terminal]  → epoch {start_epoch} step {global_step} "
              f"best_val={best_val_loss:.4f}")

    if not os.path.exists(config_path):
        config = dict(
            labels_pkl_path=labels_pkl_path, num_epochs=num_epochs,
            batch_size=batch_size, capacity=capacity,
            num_fragments=num_fragments,
            num_atom_types=NUM_ATOM_TYPES,
            lr=lr,
            warmup_steps=warmup_steps, val_fraction=val_fraction, seed=seed,
            weight_decay=weight_decay, grad_clip=grad_clip,
            bias_enabled=bias_enabled,
            use_class_weights=use_class_weights,
            class_weight_smoothing=class_weight_smoothing,
            ema_decay=ema_decay,
            n_train=len(train_labels), n_val=len(val_labels),
            n_params=n_params, condition_dim=detected_cond_dim,
            class_counts=counts, init_from=init_from,
        )
        with open(config_path, "w") as f: json.dump(config, f, indent=2)

    # Training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        t_start = time.time()
        epoch_losses: List[float] = []
        epoch_metrics: Dict[str, List[float]] = {}

        for step_in_epoch, batch in enumerate(train_loader):
            batch_dev = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}
            inputs = filter_batch(batch_dev, LOSS_KEYS)

            cur_lr = cosine_lr_with_warmup(global_step, warmup_steps,
                                            total_steps, lr)
            for pg in opt.param_groups: pg["lr"] = cur_lr

            out = model.compute_loss(class_weights=class_weights, **inputs)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); ema.update(model)

            epoch_losses.append(loss.item())
            for k, v in out.items():
                if k == "loss": continue
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    epoch_metrics.setdefault(k, []).append(float(v.item()))

            global_step += 1
            if (log_every_n_steps > 0
                    and (step_in_epoch + 1) % log_every_n_steps == 0):
                print(f"  epoch {epoch:3d} step {step_in_epoch:5d}/"
                      f"{steps_per_epoch} | loss={loss.item():.4f} "
                      f"acc={out['overall_acc'].item():.3f} "
                      f"lr={cur_lr:.2e}")

        train_loss = float(np.mean(epoch_losses))
        epoch_means = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        # Val eval
        model.eval()
        val_losses = []; val_metrics: Dict[str, List[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                inputs = filter_batch(batch_dev, LOSS_KEYS)
                out = model.compute_loss(class_weights=class_weights,
                                          **inputs)
                val_losses.append(out["loss"].item())
                for k, v in out.items():
                    if k == "loss": continue
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        val_metrics.setdefault(k, []).append(float(v.item()))
        val_loss = float(np.mean(val_losses))
        val_means = {k: float(np.mean(v)) for k, v in val_metrics.items()}

        # Sample-quality eval (on EMA)
        eval_metrics = None
        if (epoch + 1) % eval_every_n_epochs == 0:
            backup = ema.apply_to(model)
            eval_metrics = evaluate_sample_quality(
                model, val_ds, n_samples=eval_n_samples,
                n_steps=eval_n_steps, seed=seed + epoch, device=device,
            )
            ema.restore_to(model, backup)

        record = dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=cur_lr,
            time_sec=time.time() - t_start,
            train_metrics=epoch_means, val_metrics=val_means,
        )
        if eval_metrics is not None:
            record["sample_atom_acc"] = eval_metrics["atom_acc"]
            record["sample_nonzero_acc"] = eval_metrics["nonzero_acc"]
            record["sample_precision"] = eval_metrics["precision"]
            record["sample_baseline_acc"] = eval_metrics["baseline_acc"]
        history.append(record)

        # Save checkpoints (best updates BEFORE latest)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        torch.save(
            dict(model=model.state_dict(), optimizer=opt.state_dict(),
                 ema=ema.state_dict(), epoch=epoch, global_step=global_step,
                 best_val_loss=best_val_loss),
            latest_path,
        )
        torch.save(ema.state_dict(), ema_path)

        with open(history_path, "w") as f: json.dump(history, f, indent=2)

        # Per-epoch summary (one line, A1/A3 convention)
        def _g(d, k):
            return d.get(k, float("nan"))
        eval_str = (f" sample_acc={eval_metrics['atom_acc']*100:.1f}%"
                    f" nonzero={eval_metrics['nonzero_acc']*100:.1f}%"
                    if eval_metrics is not None else "")
        print(
            f"[Term][ep {epoch:3d}] "
            f"train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} "
            f"acc={_g(epoch_means, 'overall_acc'):.3f} "
            f"nz_acc={_g(epoch_means, 'nonzero_acc'):.3f}"
            f"{eval_str} "
            f"({record['time_sec']:.1f}s)",
            flush=True,
        )

    return dict(
        best_val_loss=best_val_loss, n_epochs=num_epochs, history=history,
        warmstart_status=warmstart_status,
    )


# ─── CLI ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train HiMoDiT Terminal stage.")
    p.add_argument("--labels-pkl", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--capacity", type=str, default="9M",
                    choices=["1M", "3M", "9M", "30M"])
    p.add_argument("--num-fragments", type=int, default=NUM_TERMINAL_FRAGMENTS,
                    help="K (default 22 for current-ZINC; 9 for the earlier encoder extended; "
                         "6 for the earlier version vocab)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bias-enabled", action="store_true", default=True)
    p.add_argument("--no-bond-bias", dest="bias_enabled", action="store_false",
                    help="Ablation: standard self-attention (no bond bias)")
    p.add_argument("--no-class-weights", dest="use_class_weights",
                    action="store_false")
    p.set_defaults(use_class_weights=True)
    p.add_argument("--class-weight-smoothing", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--eval-n-samples", type=int, default=64)
    p.add_argument("--eval-n-steps", type=int, default=8)
    p.add_argument("--init-from", type=str, default=None,
                    help="Warm-start from a the earlier version best_model.pt")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every-n-steps", type=int, default=0,
                   help="If >0, also print a log line every N optimizer steps. "
                        "Default 0 = one summary line per epoch only.")
    return p.parse_args()


def _main():
    args = _parse_args()
    train_terminal(
        labels_pkl_path=args.labels_pkl, ckpt_dir=args.ckpt_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        capacity=args.capacity, num_fragments=args.num_fragments,
        lr=args.lr, warmup_steps=args.warmup_steps,
        val_fraction=args.val_fraction, seed=args.seed,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        bias_enabled=args.bias_enabled,
        use_class_weights=args.use_class_weights,
        class_weight_smoothing=args.class_weight_smoothing,
        ema_decay=args.ema_decay,
        eval_every_n_epochs=args.eval_every_n_epochs,
        eval_n_samples=args.eval_n_samples, eval_n_steps=args.eval_n_steps,
        init_from=args.init_from,
        num_workers=args.num_workers, device=args.device,
        log_every_n_steps=args.log_every_n_steps,
    )


if __name__ == "__main__":
    _main()
