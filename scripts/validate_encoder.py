#!/usr/bin/env python3
"""
Reproduce the encoder retention and round-trip figures.

    python scripts/validate_encoder.py \
        --csv data/250k_rndm_zinc_drugs_clean_3.csv --sample 20000

Measures three things on the same sample, so they are directly
comparable:

  1. Retention of the baseline encoder (linear side chains, no spiro).
  2. Retention of the current encoder (branch trees plus spiro).
  3. Round-trip fidelity: what fraction of accepted labels rebuild the
     source molecule's scaffold exactly, in both atom and bond count.

The gap between (2) and (3) is the silent atom-dropping described in
docs/limitations.md, and is what `--strict` preprocessing removes.

Pass --full to run the whole file instead of a random sample.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

import numpy as np

from himodit.chem.decoder import M_MAX, decode_scaffold
from himodit.chem.encoder import extract_layout, extract_layout_baseline


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate the HiMoDiT encoder against a SMILES CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--sample", type=int, default=20000,
                   help="Random sample size.")
    p.add_argument("--full", action="store_true",
                   help="Use every row instead of sampling.")
    p.add_argument("--roundtrip-sample", type=int, default=4000,
                   help="Subset size for the round-trip check.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the baseline encoder comparison.")
    p.add_argument("--report", default=None, help="Optional JSON output.")
    return p.parse_args()


def measure_retention(smiles_list, encoder, label):
    kept, rejections = 0, Counter()
    t0 = time.time()
    for smiles in smiles_list:
        try:
            result, reason = encoder(smiles)
        except Exception as exc:                          # noqa: BLE001
            rejections[f"exception_{type(exc).__name__}"] += 1
            continue
        if result is None:
            rejections[reason] += 1
        else:
            kept += 1
    pct = 100.0 * kept / max(len(smiles_list), 1)
    print(f"\n{label}: {kept:,}/{len(smiles_list):,} = {pct:.2f}%  "
          f"({time.time() - t0:.0f}s)")
    for reason, count in rejections.most_common(10):
        print(f"    {count:>6,}  {reason}")
    return {"kept": kept, "n": len(smiles_list), "retention_pct": pct,
            "rejections": dict(rejections.most_common(20))}


def measure_roundtrip(smiles_list):
    """Do accepted labels rebuild the source scaffold exactly?"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    n_encoded = 0
    outcomes = Counter()
    atoms_dropped = 0

    for smiles in smiles_list:
        label, _ = extract_layout(smiles)
        if label is None:
            continue
        n_encoded += 1

        try:
            bond_classes, atom_mask = decode_scaffold(
                label["R"], label["F"], label["L"],
                label["B_size"], label["B_pos"],
                label["B_parent"], label["B_bond"],
                label["spiro_atom_positions"], label["atom_ids"],
                M_MAX_out=M_MAX,
            )
        except Exception as exc:                          # noqa: BLE001
            outcomes[f"decode_raised_{type(exc).__name__}"] += 1
            continue

        if int(atom_mask.sum()) != int(label["M_total"]):
            outcomes["atom_count_mismatch"] += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        terminal_atoms = {
            a for t in label["terminals"] for a in t["atom_indices"]
        }
        scaffold_bonds = sum(
            1 for b in mol.GetBonds()
            if b.GetBeginAtomIdx() not in terminal_atoms
            and b.GetEndAtomIdx() not in terminal_atoms
        )
        decoded_bonds = int((np.triu(bond_classes, 1) > 0).sum())

        unaccounted = mol.GetNumAtoms() - (
            int(label["M_total"]) + len(terminal_atoms)
        )
        if unaccounted:
            atoms_dropped += unaccounted

        if decoded_bonds != scaffold_bonds:
            outcomes["bond_count_mismatch"] += 1
        else:
            outcomes["exact"] += 1

    exact_pct = 100.0 * outcomes["exact"] / max(n_encoded, 1)
    print(f"\nround trip over {n_encoded:,} accepted labels:")
    for name, count in outcomes.most_common():
        print(f"    {count:>6,}  {name}")
    print(f"    exact round trip: {exact_pct:.2f}%")
    print(f"    atoms silently dropped: {atoms_dropped:,}")
    return {
        "n_encoded": n_encoded,
        "exact_pct": exact_pct,
        "atoms_dropped": atoms_dropped,
        "outcomes": dict(outcomes),
    }


def main():
    args = parse_args()
    import pandas as pd

    df = pd.read_csv(args.csv)
    df[args.smiles_col] = df[args.smiles_col].astype(str).str.strip()
    smiles_all = df[args.smiles_col].tolist()
    print(f"read {len(smiles_all):,} molecules from {args.csv}")

    if args.full:
        sample = smiles_all
    else:
        random.seed(args.seed)
        n = min(args.sample, len(smiles_all))
        sample = [smiles_all[i]
                  for i in random.sample(range(len(smiles_all)), n)]
    print(f"evaluating {len(sample):,} molecules")

    report = {"csv": args.csv, "n_evaluated": len(sample)}

    if not args.skip_baseline:
        report["baseline"] = measure_retention(
            sample, extract_layout_baseline,
            "baseline encoder (linear side chains, no spiro)",
        )
    report["current"] = measure_retention(
        sample, extract_layout, "current encoder (branch trees plus spiro)",
    )
    report["strict"] = measure_retention(
        sample, lambda s: extract_layout(s, strict=True),
        "current encoder, strict mode",
    )

    if "baseline" in report:
        delta = (report["current"]["retention_pct"]
                 - report["baseline"]["retention_pct"])
        print(f"\nimprovement over baseline: {delta:+.2f} percentage points")
        report["improvement_pp"] = delta

    rt_sample = sample[:min(args.roundtrip_sample, len(sample))]
    report["roundtrip"] = measure_roundtrip(rt_sample)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
