#!/usr/bin/env python3
"""
Build the training label file from a SMILES CSV.

Reads a CSV with a SMILES column and one or more property columns,
z-scores the properties, runs the encoder over every row, and writes a
pickle of layout labels for the training stages.

    python scripts/preprocess.py \
        --csv data/250k_rndm_zinc_drugs_clean_3.csv \
        --out data/labels.pkl \
        --properties logP SAS

Rejected molecules are reported by reason, which is worth reading: a
large bucket usually means a capacity constant is set too low for your
dataset rather than that the molecules are unusable.

Pass --strict to reject molecules the encoder cannot represent without
dropping atoms. This costs roughly 6 percentage points of retention on
ZINC250K and guarantees every label round-trips exactly. See
docs/limitations.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from himodit.chem.encoder import extract_layout


def parse_args():
    p = argparse.ArgumentParser(
        description="Encode a SMILES CSV into HiMoDiT layout labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Input CSV path.")
    p.add_argument("--out", required=True, help="Output .pkl path.")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument(
        "--properties", nargs="+", default=["logP", "SAS"],
        help="Property columns used as the conditioning vector.",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Use raw property values instead of z-scoring them.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Reject molecules whose atoms are not fully accounted for.",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N rows.")
    p.add_argument("--report", default=None,
                   help="Optional JSON path for the retention report.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"{args.out} already exists; pass --overwrite to rebuild it.")
        return 0

    if not os.path.isfile(args.csv):
        print(f"error: no such CSV: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    # ZINC250K ships with a trailing newline inside every SMILES field.
    df[args.smiles_col] = df[args.smiles_col].astype(str).str.strip()
    if args.limit:
        df = df.head(args.limit)
    print(f"read {len(df):,} rows from {args.csv}")

    missing = [c for c in args.properties if c not in df.columns]
    if missing:
        print(f"error: missing property columns {missing}. "
              f"Available: {list(df.columns)}", file=sys.stderr)
        return 1

    before = len(df)
    df = df.dropna(subset=args.properties)
    if len(df) < before:
        print(f"dropped {before - len(df):,} rows with missing properties")

    # Z-score the conditioning axes. These statistics must be reused at
    # evaluation time, so they are saved alongside the labels.
    stats = {}
    cond_matrix = np.zeros((len(df), len(args.properties)), dtype=np.float32)
    for i, col in enumerate(args.properties):
        values = df[col].to_numpy(dtype=np.float64)
        mean, std = float(values.mean()), float(values.std())
        stats[col] = {"mean": mean, "std": std}
        if args.no_normalize:
            cond_matrix[:, i] = values
        else:
            cond_matrix[:, i] = (values - mean) / std
        print(f"  {col}: mean={mean:.4f} std={std:.4f}")

    smiles_list = df[args.smiles_col].tolist()
    labels, rejections = [], Counter()
    t0 = time.time()

    for idx, smiles in enumerate(smiles_list):
        try:
            label, reason = extract_layout(smiles, strict=args.strict)
        except Exception as exc:                          # noqa: BLE001
            rejections[f"exception_{type(exc).__name__}"] += 1
            continue
        if label is None:
            rejections[reason] += 1
            continue
        label["condition"] = cond_matrix[idx].copy()
        labels.append(label)

        if (idx + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(smiles_list) - idx - 1) / rate / 60
            print(f"  {idx + 1:>7,}/{len(smiles_list):,}  "
                  f"kept {len(labels):>7,}  "
                  f"retention {100 * len(labels) / (idx + 1):5.2f}%  "
                  f"eta {eta:.1f} min")

    n_total = len(smiles_list)
    n_kept = len(labels)
    retention = 100.0 * n_kept / max(n_total, 1)
    print(f"\nencoded {n_kept:,}/{n_total:,} ({retention:.2f}%) "
          f"in {(time.time() - t0) / 60:.1f} min")

    if rejections:
        print("\nrejections by reason (top 15):")
        for reason, count in rejections.most_common(15):
            print(f"  {count:>7,}  {reason}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(labels, f, protocol=4)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")

    report = {
        "csv": args.csv,
        "n_total": n_total,
        "n_kept": n_kept,
        "retention_pct": retention,
        "strict": args.strict,
        "properties": args.properties,
        "property_stats": stats,
        "rejections": dict(rejections.most_common()),
    }
    report_path = args.report or (os.path.splitext(args.out)[0]
                                  + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
