#!/usr/bin/env python3
"""
Generate molecules and evaluate them.

    python scripts/generate.py \
        --ckpt-root checkpoints/ \
        --n 1000 \
        --labels data/labels.pkl \
        --csv data/250k_rndm_zinc_drugs_clean_3.csv \
        --out generated.csv

Reports validity, uniqueness, and novelty, plus the Pearson correlation
between the requested property targets and what the generated molecules
actually achieve.

Notes on the numbers
--------------------
Novelty needs `--labels` so the training molecules can be excluded.
Controllability needs `--csv` for the property statistics used to
z-score the conditions during training; using different statistics
measures correlation against a shifted target and understates it.

Generate at least 1000 samples for a reportable figure. Smaller runs
overstate uniqueness, because collisions grow with sample count.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate and evaluate molecules with HiMoDiT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-root", required=True,
                   help="Directory holding a1/, a3/, a2/, terminal/.")
    p.add_argument("--n", type=int, default=1000,
                   help="Number of molecules to generate.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cfg-scale", type=float, default=1.5,
                   help="Classifier-free guidance. 1.0 disables it.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--a1-steps", type=int, default=20)
    p.add_argument("--a2-steps", type=int, default=20)
    p.add_argument("--term-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-ema", action="store_true",
                   help="Use raw weights instead of the EMA shadow.")
    p.add_argument("--enforce-causal-parent", action="store_true",
                   help="Clamp A3 branch parents to be causal. Off by "
                        "default; see docs/limitations.md.")
    p.add_argument("--labels", default=None,
                   help="Training labels, needed for the novelty check.")
    p.add_argument("--csv", default=None,
                   help="Training CSV, needed for controllability.")
    p.add_argument("--properties", nargs="+", default=["logP", "SAS"])
    p.add_argument("--out", default=None,
                   help="Optional CSV of generated molecules.")
    p.add_argument("--report", default=None,
                   help="Optional JSON metrics report.")
    return p.parse_args()


def main():
    args = parse_args()

    from himodit.metrics import (
        compute_controllability, compute_vun, describe_molecules,
        format_controllability, format_vun, property_stats_from_csv,
        training_set_from_labels,
    )
    from himodit.pipeline import HiMoDiT

    if not os.path.isdir(args.ckpt_root):
        print(f"error: no checkpoint root at {args.ckpt_root}",
              file=sys.stderr)
        return 1

    print(f"loading checkpoints from {args.ckpt_root}")
    model = HiMoDiT.from_checkpoints(
        args.ckpt_root, device=args.device, use_ema=not args.no_ema,
    )

    print(f"\ngenerating {args.n} molecules "
          f"(cfg_scale={args.cfg_scale}, seed={args.seed})")
    smiles, conditions = model.generate(
        n=args.n,
        batch_size=args.batch_size,
        seed=args.seed,
        return_conditions=True,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        a1_steps=args.a1_steps,
        a2_steps=args.a2_steps,
        term_steps=args.term_steps,
        enforce_causal_parent=args.enforce_causal_parent,
    )

    # ── Distribution quality ───────────────────────────────────────────
    train_canonical = None
    if args.labels:
        if not os.path.isfile(args.labels):
            print(f"warning: no label file at {args.labels}; "
                  f"novelty will be reported as nan", file=sys.stderr)
        else:
            print(f"\nreading training molecules from {args.labels}")
            with open(args.labels, "rb") as f:
                labels = pickle.load(f)
            if not isinstance(labels, list):
                labels = labels.get("labels", labels)
            train_canonical = training_set_from_labels(labels)
            print(f"  {len(train_canonical):,} distinct training molecules")

    metrics = compute_vun(smiles, train_canonical=train_canonical)
    print("\n" + format_vun(metrics))

    structure = describe_molecules(smiles)
    if structure:
        print("\nStructure of the valid samples")
        print(f"  heavy atoms     {structure['mean_heavy_atoms']:.1f}")
        print(f"  rings           {structure['mean_rings']:.2f}")
        print(f"  aromatic rings  {structure['mean_aromatic_rings']:.2f}")
        print(f"  molecular wt    {structure['mean_mol_weight']:.1f}")

    # ── Controllability ────────────────────────────────────────────────
    control = None
    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"warning: no CSV at {args.csv}; skipping controllability",
                  file=sys.stderr)
        else:
            stats = property_stats_from_csv(args.csv, args.properties)
            control = compute_controllability(
                smiles, conditions.numpy(), stats,
                property_axes=args.properties,
            )
            print("\n" + format_controllability(control))

    # ── Outputs ────────────────────────────────────────────────────────
    if args.out:
        import pandas as pd
        rows = []
        for i, smi in enumerate(smiles):
            row = {"smiles": smi}
            for j, prop in enumerate(args.properties):
                row[f"target_{prop}_z"] = float(conditions[i, j])
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if args.report:
        payload = {
            k: v for k, v in metrics.items() if k != "unique_smiles"
        }
        payload["structure"] = structure
        payload["settings"] = {
            "n": args.n, "cfg_scale": args.cfg_scale,
            "temperature": args.temperature, "seed": args.seed,
            "a1_steps": args.a1_steps, "a2_steps": args.a2_steps,
            "term_steps": args.term_steps,
            "enforce_causal_parent": args.enforce_causal_parent,
        }
        if control:
            payload["controllability"] = {
                axis: {"r": res["r"], "n": res["n"]}
                for axis, res in control.items()
            }
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
