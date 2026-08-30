"""
Shared training utilities.
==========================

EMA, the LR schedule, batch collation, and checkpoint bookkeeping used
identically by all four stage trainers. These were previously duplicated
verbatim in each trainer; keeping one copy means a fix to the EMA warmup
or the resume logic applies everywhere at once.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ─── Exponential moving average ────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters.

    The effective decay at step t is min(decay, (t+1)/(t+10)), so the
    early shadow tracks the model closely instead of staying pinned near
    the random initialisation for the first few thousand steps, which is
    what a fixed decay of 0.999 would do.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.step += 1
        eff = min(self.decay, (self.step + 1) / (self.step + 10))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                self.shadow[n].mul_(eff).add_(p.detach(), alpha=1.0 - eff)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap shadow weights into the model; returns a restore backup."""
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        return backup

    def restore_to(
        self, model: nn.Module, backup: Dict[str, torch.Tensor]
    ) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup:
                    p.copy_(backup[n])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(
        self, sd: Dict[str, Any], device: torch.device
    ) -> None:
        self.step = sd["step"]
        self.decay = sd["decay"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}


# ─── Learning-rate schedule ────────────────────────────────────────────

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_frac: float = 0.1,
) -> float:
    """Linear warmup, then cosine decay to base_lr * min_lr_frac.

    step = 0             -> base_lr / warmup_steps
    step = warmup_steps  -> base_lr
    step = total_steps   -> base_lr * min_lr_frac
    beyond               -> clamped at base_lr * min_lr_frac
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


# ─── Batch helpers ─────────────────────────────────────────────────────

def collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample tensor dicts into a batch dict."""
    return {
        k: torch.stack([it[k] for it in items], dim=0)
        for k in items[0].keys()
    }


def filter_batch(
    batch: Dict[str, Any],
    loss_keys: List[str],
    rename: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Reduce a dataset batch to exactly the keys `compute_loss` expects.

    Datasets emit extra bookkeeping (smiles, M_total, terminals, ...)
    that the loss does not take. `rename` handles F -> F_mat and
    L -> L_mat, which exist to dodge the collision with the conventional
    `torch.nn.functional as F` import inside the model files.

    Raises KeyError naming the missing fields, which is almost always a
    sign that the label pickle came from the wrong encoder.
    """
    rename = rename or {}
    out = {}
    for k, v in batch.items():
        new_key = rename.get(k, k)
        if new_key in loss_keys:
            out[new_key] = v
    missing = set(loss_keys) - set(out.keys())
    if missing:
        raise KeyError(
            f"batch is missing required keys for compute_loss: "
            f"{sorted(missing)}. Check that the label pickle was produced "
            f"by himodit.chem.encoder.extract_layout."
        )
    return out


# ─── Checkpoint bookkeeping ────────────────────────────────────────────

class CheckpointPaths:
    """The five files each trainer writes into its checkpoint directory.

    latest.pt      full state for resume; overwritten every epoch
    best_model.pt  weights only, at the best validation loss so far
    ema.pt         EMA shadow weights
    history.json   per-epoch metrics
    config.json    training configuration, written once
    """

    def __init__(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.dir = ckpt_dir
        self.latest = os.path.join(ckpt_dir, "latest.pt")
        self.best = os.path.join(ckpt_dir, "best_model.pt")
        self.ema = os.path.join(ckpt_dir, "ema.pt")
        self.history = os.path.join(ckpt_dir, "history.json")
        self.config = os.path.join(ckpt_dir, "config.json")


def maybe_resume(
    paths: CheckpointPaths,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
    tag: str = "train",
) -> Dict[str, Any]:
    """Restore from latest.pt when present.

    Returns a dict with start_epoch, global_step, best_val_loss, history.
    Training scripts auto-resume by default, so an interrupted Colab
    session can simply be re-run. To restart from epoch 0, delete the
    checkpoint directory.
    """
    state = {
        "start_epoch": 0,
        "global_step": 0,
        "best_val_loss": float("inf"),
        "history": [],
    }
    if not os.path.exists(paths.latest):
        return state

    print(f"[{tag}] resuming from {paths.latest}")
    ck = torch.load(paths.latest, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    ema.load_state_dict(ck["ema"], device=device)
    state["start_epoch"] = ck["epoch"] + 1
    state["global_step"] = ck["global_step"]
    state["best_val_loss"] = ck["best_val_loss"]
    if os.path.exists(paths.history):
        with open(paths.history) as f:
            state["history"] = json.load(f)
    print(
        f"[{tag}]  -> epoch {state['start_epoch']}, "
        f"step {state['global_step']}, "
        f"best_val_loss {state['best_val_loss']:.4f}"
    )
    return state


def save_epoch(
    paths: CheckpointPaths,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    epoch: int,
    global_step: int,
    val_loss: float,
    best_val_loss: float,
    history: List[Dict[str, Any]],
) -> float:
    """Write the per-epoch checkpoint set. Returns the new best_val_loss.

    best_val_loss is updated before latest.pt is written, so a resume
    after an interrupted epoch does not lose track of the true best.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), paths.best)

    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            ema=ema.state_dict(),
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
        ),
        paths.latest,
    )
    torch.save(ema.state_dict(), paths.ema)
    with open(paths.history, "w") as f:
        json.dump(history, f, indent=2)
    return best_val_loss


def write_config_once(paths: CheckpointPaths, config: Dict[str, Any]) -> None:
    if not os.path.exists(paths.config):
        with open(paths.config, "w") as f:
            json.dump(config, f, indent=2)


def split_labels(labels: List[Dict], val_fraction: float, rng) -> tuple:
    """Shuffle in place and split off a validation set."""
    rng.shuffle(labels)
    n_val = max(1, int(round(val_fraction * len(labels))))
    return labels[n_val:], labels[:n_val]


def detect_condition_dim(labels: List[Dict]) -> int:
    """Read the conditioning width off the first label.

    Label pickles built with a single property give a 1-D condition;
    logP + SAS gives 2-D. The model has to match, and a mismatch would
    otherwise surface as an opaque shape error deep in the first forward
    pass.
    """
    import numpy as np

    cond = np.asarray(labels[0]["condition"]).reshape(-1)
    return int(cond.shape[0])
