#!/usr/bin/env python3
"""
Train one stage of the HiMoDiT cascade.

    python scripts/train.py --stage a1 \
        --labels data/labels.pkl --ckpt-dir checkpoints/a1 \
        --epochs 40 --capacity 3M --batch-size 256

Stages must be trained in this order, because each consumes the label
fields the previous one is conditioned on:

    a1        ring layout          ~40 epochs
    a3        branch topology      ~100 epochs
    a2        atom identity        ~40 epochs
    terminal  decoration           ~40 epochs

All four auto-resume from `latest.pt`, so an interrupted run is restarted
by re-issuing the same command. To restart from epoch 0, delete the
checkpoint directory.

`--all` runs the four in order with their default settings:

    python scripts/train.py --all --labels data/labels.pkl \
        --ckpt-root checkpoints/
"""
from __future__ import annotations

import argparse
import os
import sys

# cfg_drop_prob is the classifier-free-guidance condition-dropout rate.
# The Terminal stage has no CFG path, so it takes no such argument.
STAGE_DEFAULTS = {
    "a1": {"capacity": "3M", "epochs": 40, "cfg_drop_prob": 0.10},
    "a3": {"capacity": "10M", "epochs": 100, "cfg_drop_prob": 0.30},
    "a2": {"capacity": "10M", "epochs": 40, "cfg_drop_prob": 0.10},
    "terminal": {"capacity": "9M", "epochs": 40},
}

STAGE_ORDER = ["a1", "a3", "a2", "terminal"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a HiMoDiT stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=STAGE_ORDER,
                   help="Which stage to train.")
    p.add_argument("--all", action="store_true",
                   help="Train all four stages in order.")
    p.add_argument("--labels", required=True,
                   help="Label pickle from scripts/preprocess.py.")
    p.add_argument("--ckpt-dir", default=None,
                   help="Checkpoint directory for a single stage.")
    p.add_argument("--ckpt-root", default="checkpoints",
                   help="Parent directory when using --all.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Overrides the per-stage default.")
    p.add_argument("--capacity", default=None,
                   help="Overrides the per-stage default.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu'.")
    p.add_argument("--log-every-n-steps", type=int, default=0)
    return p.parse_args()


def run_stage(stage, args, ckpt_dir):
    defaults = STAGE_DEFAULTS[stage]
    capacity = args.capacity or defaults["capacity"]
    epochs = args.epochs or defaults["epochs"]

    common = dict(
        labels_pkl_path=args.labels,
        ckpt_dir=ckpt_dir,
        num_epochs=epochs,
        batch_size=args.batch_size,
        capacity=capacity,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
    )

    print(f"\n{'=' * 60}")
    print(f" stage {stage}  capacity {capacity}  epochs {epochs}")
    print(f" checkpoints -> {ckpt_dir}")
    print(f"{'=' * 60}\n")

    cfg_drop = defaults.get("cfg_drop_prob")

    if stage == "a1":
        from himodit.training.a1 import train_a1
        return train_a1(cfg_drop_prob=cfg_drop, **common)
    if stage == "a3":
        from himodit.training.a3 import train_a3
        return train_a3(cfg_drop_prob=cfg_drop, **common)
    if stage == "a2":
        from himodit.training.a2 import train_a2
        return train_a2(cfg_drop_prob=cfg_drop, **common)
    from himodit.training.terminal import train_terminal
    return train_terminal(**common)


def main():
    args = parse_args()

    if not args.all and not args.stage:
        print("error: pass either --stage or --all", file=sys.stderr)
        return 1
    if not os.path.isfile(args.labels):
        print(f"error: no label file at {args.labels}. "
              f"Run scripts/preprocess.py first.", file=sys.stderr)
        return 1

    if args.all:
        for stage in STAGE_ORDER:
            run_stage(stage, args, os.path.join(args.ckpt_root, stage))
        print("\nall four stages trained.")
        return 0

    ckpt_dir = args.ckpt_dir or os.path.join(args.ckpt_root, args.stage)
    run_stage(args.stage, args, ckpt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
