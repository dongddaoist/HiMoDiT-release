"""
Generation metrics.
===================

Two families of measurement, kept separate because they answer different
questions and fail independently.

Distribution quality
    validity   fraction of samples RDKit can sanitize
    uniqueness fraction of valid samples that are distinct
    novelty    fraction of distinct samples absent from the training set
    Reported individually and as their product, V.U.N.

Controllability
    Pearson correlation between the requested property target and the
    property the generated molecule actually has, computed in z-scored
    units so the two axes are comparable.

Both take plain SMILES lists, so they work on output from any source,
not only `himodit.pipeline`.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

RDLogger.DisableLog("rdApp.*")


# ─── Synthetic accessibility ───────────────────────────────────────────
# sascorer ships in RDKit's contrib tree rather than the main package, so
# it needs the path appended before it can be imported.

def _load_sascorer():
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer                                   # noqa: PLC0415
        return sascorer
    except Exception:                                     # noqa: BLE001
        return None


_SASCORER = _load_sascorer()
SAS_AVAILABLE = _SASCORER is not None


def compute_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return Crippen.MolLogP(mol)
    except Exception:                                     # noqa: BLE001
        return float("nan")


def compute_sas(smiles: str) -> float:
    if not SAS_AVAILABLE:
        return float("nan")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return _SASCORER.calculateScore(mol)
    except Exception:                                     # noqa: BLE001
        return float("nan")


# ─── Validity, uniqueness, novelty ─────────────────────────────────────

def canonical_set(smiles_iter: Iterable[str]) -> set:
    """Canonical SMILES set, skipping anything RDKit cannot parse."""
    out = set()
    for smi in smiles_iter:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol))
    return out


def training_set_from_labels(labels: Sequence[Dict]) -> set:
    """Canonical SMILES of the training molecules, for the novelty check."""
    return canonical_set(lab.get("smi") for lab in labels)


def compute_vun(
    smiles: Sequence[Optional[str]],
    train_canonical: Optional[set] = None,
) -> Dict[str, float]:
    """Validity, uniqueness, novelty over a list of generated SMILES.

    Entries that are None are counted as assembly failures; entries that
    are strings RDKit rejects are counted as parse failures. Both count
    against validity, and the split between them is reported because it
    localises the problem: assembly failures point at the layout stages,
    parse failures at atom identity or valence.

    Uniqueness is measured over valid samples, novelty over distinct
    valid samples, which is the convention used by DiGress, DeFoG, and
    Cometh, so the numbers are directly comparable.
    """
    n_total = len(smiles)
    valid, n_assembly_fail, n_parse_fail, n_empty = [], 0, 0, 0

    for smi in smiles:
        if smi is None:
            n_assembly_fail += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_parse_fail += 1
            continue
        # Chem.MolFromSmiles("") returns a valid zero-atom Mol rather than
        # None, so an empty string would otherwise be scored as a valid
        # sample. Empty output means the scaffold decode failed upstream.
        if mol.GetNumHeavyAtoms() == 0:
            n_empty += 1
            continue
        valid.append(Chem.MolToSmiles(mol))

    n_valid = len(valid)
    unique = set(valid)
    validity = n_valid / max(n_total, 1)
    uniqueness = len(unique) / max(n_valid, 1)

    if train_canonical is None:
        novelty = float("nan")
        n_novel = -1
    else:
        novel = unique - train_canonical
        n_novel = len(novel)
        novelty = n_novel / max(len(unique), 1)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unique": len(unique),
        "n_novel": n_novel,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "vun": validity * uniqueness * novelty,
        "n_assembly_failed": n_assembly_fail,
        "n_parse_failed": n_parse_fail,
        "n_empty": n_empty,
        "unique_smiles": sorted(unique),
    }


def format_vun(m: Dict[str, float]) -> str:
    """One-screen summary of a compute_vun result."""
    lines = [
        "=" * 52,
        f" V.U.N over {m['n_total']} generated samples",
        "=" * 52,
        f" Validity    {m['validity'] * 100:6.2f}%   "
        f"({m['n_valid']}/{m['n_total']})",
        f" Uniqueness  {m['uniqueness'] * 100:6.2f}%   "
        f"({m['n_unique']}/{max(m['n_valid'], 1)} valid)",
        f" Novelty     {m['novelty'] * 100:6.2f}%   "
        f"({m['n_novel']}/{max(m['n_unique'], 1)} unique)",
        f" V.U.N       {m['vun'] * 100:6.2f}%",
        "",
        " Failure breakdown",
        f"   assembly returned None   {m['n_assembly_failed']}",
        f"   RDKit could not parse    {m['n_parse_failed']}",
        f"   empty (zero-atom)        {m['n_empty']}",
    ]
    return "\n".join(lines)


# ─── Controllability ───────────────────────────────────────────────────

def compute_controllability(
    smiles: Sequence[Optional[str]],
    conditions: np.ndarray,
    property_stats: Dict[str, Dict[str, float]],
    property_axes: Sequence[str] = ("logP", "SAS"),
) -> Dict[str, Dict[str, float]]:
    """Correlate requested property targets against achieved properties.

    Parameters
    ----------
    smiles
        Generated SMILES, aligned row-for-row with `conditions`.
    conditions
        (N, n_axes) array of z-scored targets, in `property_axes` order.
    property_stats
        {"logP": {"mean": ..., "std": ...}, "SAS": {...}} from the
        training set, used to z-score the achieved values so target and
        achieved live on the same scale.
    property_axes
        Which column of `conditions` is which property.

    Each axis is accumulated independently. A molecule whose logP
    computes but whose SAS fails still contributes to the logP
    correlation, and target/achieved pairs are appended together so the
    two lists cannot drift out of alignment.
    """
    computers = {"logP": compute_logp, "SAS": compute_sas}
    results: Dict[str, Dict[str, float]] = {}
    conditions = np.asarray(conditions)

    for axis_idx, axis in enumerate(property_axes):
        if axis not in computers:
            raise ValueError(
                f"No property computer for {axis!r}; "
                f"known axes: {sorted(computers)}"
            )
        if axis == "SAS" and not SAS_AVAILABLE:
            results[axis] = {"r": float("nan"), "n": 0,
                             "note": "sascorer unavailable"}
            continue

        stats = property_stats[axis]
        mean, std = float(stats["mean"]), float(stats["std"])
        targets, achieved = [], []

        for i, smi in enumerate(smiles):
            if smi is None:
                continue
            value = computers[axis](smi)
            if not np.isfinite(value):
                continue
            z = (value - mean) / std
            if not np.isfinite(z):
                continue
            targets.append(float(conditions[i, axis_idx]))
            achieved.append(z)

        if len(targets) < 2:
            results[axis] = {"r": float("nan"), "n": len(targets)}
            continue

        r = float(np.corrcoef(np.array(targets), np.array(achieved))[0, 1])
        results[axis] = {
            "r": r,
            "n": len(targets),
            "targets": np.array(targets),
            "achieved": np.array(achieved),
        }

    return results


def format_controllability(results: Dict[str, Dict[str, float]]) -> str:
    lines = ["Controllability (Pearson r on z-scored values)"]
    for axis, res in results.items():
        note = f"   [{res['note']}]" if "note" in res else ""
        lines.append(f"  {axis:5s} r = {res['r']:.3f}   (n={res['n']}){note}")
    return "\n".join(lines)


def property_stats_from_csv(
    csv_path: str, columns: Sequence[str] = ("logP", "SAS"),
) -> Dict[str, Dict[str, float]]:
    """Read training mean/std per property from the augmented CSV.

    These must be the same statistics used to z-score the training
    conditions, or the correlation is measured against a shifted target.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    stats = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not in {csv_path}")
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }
    return stats


# ─── Structural summary ────────────────────────────────────────────────

def describe_molecules(smiles: Sequence[Optional[str]]) -> Dict[str, float]:
    """Coarse structural statistics over the valid generated molecules.

    Useful as a distribution-fidelity sanity check next to V.U.N: a model
    can score well on validity while producing molecules that are far too
    small or ring-poor compared with the training set.
    """
    heavy, rings, arom_rings, mw = [], [], [], []
    from rdkit.Chem import Descriptors

    for smi in smiles:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        heavy.append(mol.GetNumHeavyAtoms())
        ri = mol.GetRingInfo()
        rings.append(ri.NumRings())
        arom_rings.append(sum(
            1 for ring in ri.AtomRings()
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        ))
        mw.append(Descriptors.MolWt(mol))

    if not heavy:
        return {}
    return {
        "n": len(heavy),
        "mean_heavy_atoms": float(np.mean(heavy)),
        "mean_rings": float(np.mean(rings)),
        "mean_aromatic_rings": float(np.mean(arom_rings)),
        "mean_mol_weight": float(np.mean(mw)),
    }
