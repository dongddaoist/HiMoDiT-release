"""
Per-molecule terminal fragment detection.
=========================================

A SMARTS match counts as a terminal fragment only if exactly one bond
crosses out of the matched atom set. That single crossing bond defines
the fragment's anchor atom (inside the match), its host atom (outside,
on the scaffold), and the bond class joining them.

Overlapping matches are resolved by parent-child subset removal: a match
whose atom set is a strict subset of another's is dropped, so the -OH
inside a -COOH does not get counted separately. Matches with identical
atom sets keep the first by vocabulary order.

Usage
-----
    from rdkit import Chem
    from himodit.chem.terminal_smarts import CURATED_TERMINALS
    from himodit.chem.terminal_detection import (
        compile_patterns, detect_terminals_in_molecule,
    )

    compiled = compile_patterns(CURATED_TERMINALS)     # once, outside loops
    mol = Chem.MolFromSmiles(smiles)
    terminals = detect_terminals_in_molecule(mol, CURATED_TERMINALS, compiled)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem


# Bond class encoding (matches the earlier version / config.py)
BOND_NONE     = 0
BOND_SINGLE   = 1
BOND_AROMATIC = 2
BOND_DOUBLE   = 3
BOND_TRIPLE   = 4

_RDKIT_BOND_TO_CLASS = {
    Chem.BondType.SINGLE:   BOND_SINGLE,
    Chem.BondType.AROMATIC: BOND_AROMATIC,
    Chem.BondType.DOUBLE:   BOND_DOUBLE,
    Chem.BondType.TRIPLE:   BOND_TRIPLE,
}


def compile_patterns(curated_terminals: List[Dict]) -> Dict[str, Chem.Mol]:
    """Pre-compile all SMARTS once. Call this outside any per-molecule loop."""
    compiled: Dict[str, Chem.Mol] = {}
    for t in curated_terminals:
        patt = Chem.MolFromSmarts(t["detection_smarts"])
        if patt is not None:
            compiled[t["name"]] = patt
    return compiled


def _is_terminal_match(
    mol: Chem.Mol, match_atoms: Tuple[int, ...]
) -> Optional[Tuple[int, int, int]]:
    """Check if a SMARTS match is a terminal substructure.

    A match qualifies as terminal iff the matched atom set has exactly
    one bond crossing into atoms outside the match.

    Returns:
        (anchor_atom_idx, host_atom_idx, bond_class) on success.
        - anchor_atom_idx: the atom INSIDE the match that bonds to the host.
        - host_atom_idx:   the atom OUTSIDE the match (scaffold atom).
        - bond_class:      one of BOND_SINGLE, BOND_AROMATIC, BOND_DOUBLE,
                           BOND_TRIPLE.
        None if the match has 0 or 2+ crossing bonds.
    """
    match_set = set(match_atoms)
    crossings: List[Tuple[int, int, int]] = []
    for atom_idx in match_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtom(atom).GetIdx()
            if other_idx in match_set:
                continue
            bclass = _RDKIT_BOND_TO_CLASS.get(bond.GetBondType(), BOND_NONE)
            crossings.append((atom_idx, other_idx, bclass))

    if len(crossings) != 1:
        return None
    return crossings[0]


def detect_terminals_in_molecule(
    mol: Chem.Mol,
    curated_terminals: List[Dict],
    compiled_patterns: Dict[str, Chem.Mol],
    duplicate_warnings: Optional[List[str]] = None,
) -> List[Dict]:
    """Detect all terminal fragments in one molecule via parent-child subset
    removal. Returns only the maximal (parent) matches.

    Algorithm (matches the earlier detect_terminals.py):
      1. Collect ALL terminal-checked matches across all SMARTS.
      2. For each match j, drop it if its atom set is a strict subset
         of any other match i's atom set (j is a child).
      3. Equal-atom-set duplicates: keep the first by iteration order;
         optionally log to duplicate_warnings.

    Returns
    -------
    list of detection dicts; each has:
      name:                  terminal vocab name (e.g. "COOH", "=O")
      atom_indices:          tuple of RDKit atom indices in the match
      anchor_atom_idx:       fragment-side atom that bonds to host
      host_atom_idx:         scaffold-side atom (NOT in match)
      host_is_aromatic:      bool — host atom's aromaticity
      host_symbol:           atom symbol of host (e.g. 'C', 'N')
      attach_bond_class:     bond class connecting fragment to host
      flags:                 list of flag strings from the SMARTS def
    """
    candidates: List[Dict] = []
    for pattern_def in curated_terminals:
        patt = compiled_patterns.get(pattern_def["name"])
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            crossing = _is_terminal_match(mol, match)
            if crossing is None:
                continue
            anchor_idx, host_idx, bond_class = crossing
            # Verify the bond class is in the allowed list for this terminal
            allowed = pattern_def.get("allowed_attachment_bonds", [BOND_SINGLE])
            if bond_class not in allowed:
                continue
            host_atom = mol.GetAtomWithIdx(host_idx)
            candidates.append({
                "name": pattern_def["name"],
                "atom_indices": tuple(match),
                "atom_set": frozenset(match),
                "anchor_atom_idx": anchor_idx,
                "host_atom_idx": host_idx,
                "host_is_aromatic": host_atom.GetIsAromatic(),
                "host_symbol": host_atom.GetSymbol(),
                "attach_bond_class": bond_class,
                "flags": list(pattern_def.get("flags", [])),
            })

    # Strict-subset (parent-child) removal
    n = len(candidates)
    is_child = [False] * n
    for j in range(n):
        sj = candidates[j]["atom_set"]
        for i in range(n):
            if i == j:
                continue
            si = candidates[i]["atom_set"]
            if sj < si:  # strict subset
                is_child[j] = True
                break

    survivors_after_subset = [
        c for c, child in zip(candidates, is_child) if not child
    ]

    # Equal-atom-set deduplication
    seen_atom_sets: Dict[frozenset, str] = {}
    detections: List[Dict] = []
    for c in survivors_after_subset:
        s = c["atom_set"]
        if s in seen_atom_sets:
            other_name = seen_atom_sets[s]
            if duplicate_warnings is not None:
                duplicate_warnings.append(
                    f"equal-atom-set duplicate: kept '{other_name}', "
                    f"dropped '{c['name']}' on atoms {sorted(s)}"
                )
            continue
        seen_atom_sets[s] = c["name"]
        out = {k: v for k, v in c.items() if k != "atom_set"}
        detections.append(out)

    return detections
