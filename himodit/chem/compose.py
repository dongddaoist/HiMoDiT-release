"""
Scaffold + terminals -> RDKit molecule.
=======================================

Final stage of the decode path. Takes the atom identities from A2, the
bond matrix from the decoder, and the per-atom terminal assignments from
the Terminal stage, and builds a sanitized RDKit molecule.

Two mechanisms keep validity high, and both are part of the decoder
rather than post-hoc filtering: a molecule that comes out of this
function is the molecule the model specified, read as consistently as
RDKit allows.

Graft admission
---------------
Before attaching a terminal, the host atom is checked for headroom:

  * current explicit valence + the attachment bond order must not exceed
    the element's maximum valence at its formal charge;
  * an aromatic host already at degree 3 is skipped, since sp2 atoms cap
    there and adding a substituent breaks kekulization downstream.

Grafts failing either check are skipped. The host keeps an open valence
that sanitization fills with implicit hydrogen, so the molecule stays
valid and only that one decoration is lost.

Sanitization cascade
--------------------
  1. Plain `Chem.SanitizeMol`. Most molecules succeed here.
  2. On a kekulization failure, clear every aromatic flag and downgrade
     aromatic bonds to single, then retry. This treats A2's per-atom
     aromaticity as a hint and lets RDKit derive aromaticity from the
     bond topology instead. A five-membered all-carbon ring that cannot
     be kekulized as aromatic becomes a cyclopentadiene rather than
     being thrown away.
  3. On a valence failure, find each over-valent atom and remove its
     highest-order non-ring bond, then retry. Ring bonds are never
     touched, since removing one would change the scaffold topology the
     model emitted. Bounded to one removal per atom and a small number
     of passes; molecules that would need scaffold surgery are given up.

Atom construction
-----------------
RDKit's `Chem.Atom` constructor rejects element strings carrying a
charge suffix, so charged vocabulary entries (O-, N+, n+, N-, n-, P+)
are built from the bare element and given their charge with
`SetFormalCharge` afterwards.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType


# ─── Atom vocab (with charged-N/P additions) ────────────────────
# Halogens are NOT added here — they're handled as terminals only.
# Includes charged species; see docs/label_schema.md.
ATOM_VOCAB = ["<PAD>", "c", "O", "C", "N", "n", "S", "F", "s", "o",
              "O-", "N+", "n+", "N-", "n-", "P+"]

# Standard valence per atom symbol, for hydrogen-count inference at
# charged-N/P emission). Used by vocab_id_to_smiles_atom() only —
# the RDKit assembly path uses _VOCAB_TO_ELEM_CHARGE below.
_STD_VALENCE = {"N+": 4, "P+": 4, "N-": 2, "n+": 4, "n-": 2}


#  map vocab symbol -> (RDKit element, formal_charge).
# RDKit's Atom() constructor only accepts a plain element symbol, so we
# strip the +/- suffix and apply the charge afterward via SetFormalCharge.
# Aromatic flag is derived separately from the symbol's case (see
# _vocab_id_to_atom()).
_VOCAB_TO_ELEM_CHARGE = {
    "<PAD>": (None, 0),
    "c":  ("C", 0),  "C": ("C", 0),
    "n":  ("N", 0),  "N": ("N", 0),
    "o":  ("O", 0),  "O": ("O", 0),
    "s":  ("S", 0),  "S": ("S", 0),
    "F":  ("F", 0),
    "O-": ("O", -1),
    "N+": ("N", +1), "n+": ("N", +1),
    "N-": ("N", -1), "n-": ("N", -1),
    "P+": ("P", +1),
}


def vocab_id_to_smiles_atom(vid, degree=0):
    """Map vocab id -> SMILES atom token. Charged N/P need explicit nH from degree.

    Used by SMILES-string emitters that don't go through RDKit Mol
    construction. Unchanged from ; the RDKit path uses
    _vocab_id_to_atom() + _vocab_id_to_charge() instead.
    """
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>": return ""
    if sym in ("c", "C", "O", "N", "n", "S", "F", "s", "o"): return sym
    if sym == "O-": return "[O-]"
    if sym in ("N+", "P+", "N-", "n+", "n-"):
        nh = max(0, _STD_VALENCE[sym] - max(0, degree))
        elem = sym[0]
        chg = "+" if "+" in sym else "-"
        if nh == 0: return "[" + elem + chg + "]"
        if nh == 1: return "[" + elem + "H" + chg + "]"
        return "[" + elem + "H" + str(nh) + chg + "]"
    raise ValueError("Unknown vocab symbol: " + repr(sym))


def _vocab_id_to_atom(vid: int) -> Tuple[str, bool]:
    """ returns (bare_element_symbol, is_aromatic).

    Charge is queried separately via _vocab_id_to_charge() — keeping
    these two pieces of information in separate calls preserves a
    2-tuple return signature for backward compatibility (the Section 6
    diagnostic cell unpacks this as `element, is_arom = _vocab_id_to_atom(vid)`).
    The bare element returned here is always safe to pass to Chem.Atom().
    """
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>":
        raise ValueError("Cannot place PAD atom")
    elem, _ = _VOCAB_TO_ELEM_CHARGE[sym]
    # First character's case determines aromaticity:
    #   "c","n","o","s","n+","n-" → aromatic
    #   "C","N","O","S","N+","N-","O-","P+","F" → non-aromatic
    is_arom = sym[0].islower()
    return elem, is_arom


def _vocab_id_to_charge(vid: int) -> int:
    """ formal charge for vocab id (0 for neutral species, ±1 for
    charged). Used by assemble_molecule to call SetFormalCharge after
    constructing the atom with the bare element symbol returned by
    _vocab_id_to_atom().
    """
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>":
        return 0
    _, chg = _VOCAB_TO_ELEM_CHARGE[sym]
    return chg


_BOND_CLASS_TO_TYPE = {
    1: BondType.SINGLE,
    2: BondType.AROMATIC,
    3: BondType.DOUBLE,
    4: BondType.TRIPLE,
}


#  bond-order lookup for pre-graft valence accounting.
# Aromatic bond order = 1.5 (one sigma + half pi).
_BOND_ORDER = {
    BondType.SINGLE: 1.0,
    BondType.DOUBLE: 2.0,
    BondType.TRIPLE: 3.0,
    BondType.AROMATIC: 1.5,
}


#  maximum permitted explicit valence per element (neutral form).
# Charged species adjust this by ±|formal_charge| via _max_valence_for().
# This is a heuristic cap — RDKit's full valence model (which knows
# about hypervalent S, tautomers, etc.) is the final arbiter at
# Chem.SanitizeMol() time. The purpose of the heuristic is to skip
# grafts that are clearly going to fail, not to perfectly mirror RDKit.
_MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 6,
    "F": 1, "Cl": 1, "Br": 1, "I": 1,
    "P": 5,
}


def _max_valence_for(element: str, formal_charge: int) -> float:
    """Heuristic max explicit valence for `element` with given charge.

    For nitrogen: neutral N max = 3, N+ max = 4, N- max = 2. For oxygen:
    neutral O max = 2, O- max = 1. Etc. Unknown elements default to 4.
    """
    base = _MAX_VALENCE.get(element, 4)
    return float(base + formal_charge)


def _current_explicit_valence(mol: Chem.RWMol, idx: int) -> float:
    """Sum of bond orders on atom `idx` in `mol`. Aromatic bonds = 1.5.

    Used by the graft validator in assemble_molecule() before adding a
    terminal: the projected total (current + attach_order) must stay
    within _max_valence_for(element, charge) or the graft is skipped.
    """
    atom = mol.GetAtomWithIdx(idx)
    total = 0.0
    for b in atom.GetBonds():
        total += _BOND_ORDER.get(b.GetBondType(), 1.0)
    return total


#  bounded valence repair — max passes per molecule.  A pass
# tries to fix one over-valent atom by removing one non-ring bond.
# Set conservatively; in practice the molecules that need repair
# fix in 1-2 passes.  Higher cap doesn't help (atoms needing more
# than 2 fixes are usually unrecoverable without touching ring bonds).
_MAX_REPAIR_PASSES = 4


def _try_sanitize(mol: Chem.RWMol) -> bool:
    """Run Chem.SanitizeMol in-place. Returns True on success, False otherwise.

    Wraps the exception so callers can keep working with the partially-
    sanitized RWMol. (RDKit leaves the mol in a usable state even when
    SanitizeMol raises — the exception just signals which check failed.)
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def _strip_aromaticity_inplace(mol: Chem.RWMol) -> None:
    """ Q1 helper: clear all aromatic flags on atoms and convert
    every AROMATIC bond to SINGLE, so RDKit's subsequent SanitizeMol
    re-derives aromaticity from scratch using the (non-aromatic) bond
    topology.

    This is the canonical RDKit-friendly way to ask "given this bond
    skeleton, which atoms are actually aromatic?" The answer becomes
    whatever RDKit's aromaticity perceiver decides, not what A2 emitted.
    For rings that genuinely support aromaticity (e.g. benzene-like with
    proper hetero placement), the flag gets restored. For rings that
    don't (e.g. a 5-ring of all-c with no donor), the atoms stay
    non-aromatic and the bonds get assigned single/double in a way that
    makes a valid molecule (typically a diene or cyclopentadiene).
    """
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
    # Two-phase bond pass: first collect aromatic bonds, then convert.
    # Modifying bond types during iteration is supported by RDKit but
    # the safer idiom is to do it post-iteration.
    aromatic_bond_idxs = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBondType() == BondType.AROMATIC
    ]
    for idx in aromatic_bond_idxs:
        mol.GetBondWithIdx(idx).SetBondType(BondType.SINGLE)


def _find_overvalent_atoms(mol: Chem.RWMol) -> list:
    """ Q2 helper: return list of (atom_idx, current_valence, max_valence)
    for atoms that exceed their permitted explicit valence.

    Uses our heuristic max-valence model (the same one the graft validator
    uses), which is *stricter* than RDKit's full hypervalent-aware model.
    That's intentional: we'd rather repair a borderline case than fail it.
    """
    overvalent = []
    for atom in mol.GetAtoms():
        cur = sum(_BOND_ORDER.get(b.GetBondType(), 1.0)
                  for b in atom.GetBonds())
        max_v = _max_valence_for(atom.GetSymbol(), atom.GetFormalCharge())
        if cur > max_v:
            overvalent.append((atom.GetIdx(), cur, max_v))
    return overvalent


def _try_remove_one_nonring_bond(mol: Chem.RWMol, atom_idx: int) -> bool:
    """ Q2 helper: try removing the highest-order non-ring bond
    incident to atom `atom_idx`. Returns True if a bond was removed,
    False if no non-ring bond is available (i.e., all incident bonds
    are ring bonds — the molecule is unrecoverable without scaffold
    modification, which we refuse to do).

    Ring detection requires SSSR to be computed. We call GetSymmSSSR
    defensively before checking IsInRing — it works on partially-
    sanitized molecules where regular SSSR can fail.
    """
    Chem.GetSymmSSSR(mol)  # populates ring info; safe on unsanitized mol

    atom = mol.GetAtomWithIdx(atom_idx)
    # Rank candidate bonds by order, descending. Removing a higher-order
    # bond reduces valence by more — fewer iterations needed.
    candidates = []
    for b in atom.GetBonds():
        if b.IsInRing():
            continue
        order = _BOND_ORDER.get(b.GetBondType(), 1.0)
        candidates.append((order, b.GetIdx(),
                           b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    if not candidates:
        return False
    candidates.sort(reverse=True)  # highest order first
    _, _, begin, end = candidates[0]
    mol.RemoveBond(begin, end)
    return True


def _repair_and_sanitize(mol: Chem.RWMol) -> Optional[Chem.Mol]:
    """ cascade: vanilla → Q1 aromaticity re-perception → Q2
    bounded non-ring valence repair. Returns the sanitized Mol on
    success, or None if even the repair fails.

    The cascade is conservative: each attempt only fires if the
    previous one failed. Repair never modifies ring bonds.

    Post-Q2 fragment selection: if Q2's bond removal disconnects the
    molecule into multiple fragments (because the removed bond was the
    only link between two parts), keep the largest fragment by heavy
    atom count. The discarded fragment was a substituent/branch that
    couldn't be cleanly grafted; dropping it is preferable to reporting
    a multi-component "molecule" that would inflate the validity count
    in an unprincipled way.
    """
    # ── Attempt 1: vanilla sanitize ────────────────────────────────
    if _try_sanitize(mol):
        return mol

    # ── Attempt 2 (Q1): aromaticity re-perception ───────────────────
    mol_q1 = Chem.RWMol(mol)
    _strip_aromaticity_inplace(mol_q1)
    if _try_sanitize(mol_q1):
        return mol_q1

    # ── Attempt 3 (Q2): bounded valence repair on Q1 output ─────────
    used_q2 = False
    for _ in range(_MAX_REPAIR_PASSES):
        overvalent = _find_overvalent_atoms(mol_q1)
        if not overvalent:
            sanitize_ok = _try_sanitize(mol_q1)
            if not sanitize_ok:
                return None
            break
        overvalent.sort(key=lambda t: -(t[1] - t[2]))
        atom_idx, _, _ = overvalent[0]
        if not _try_remove_one_nonring_bond(mol_q1, atom_idx):
            return None  # only ring bonds available; refuse to modify scaffold
        used_q2 = True
        if _try_sanitize(mol_q1):
            break
    else:
        return None  # ran out of passes

    # If Q2 fired, check for disconnection. Most Q2 repairs preserve
    # connectivity, but removing a bridging non-ring bond can produce
    # two fragments. Keep only the largest.
    if used_q2:
        frags = Chem.GetMolFrags(mol_q1, asMols=True)
        if len(frags) > 1:
            largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            return largest

    return mol_q1


# ─── Terminal fragment graft specifications (K=22) ───────────────────
# Each entry:
#   atoms:  list of (element, is_aromatic, num_explicit_h)
#   bonds:  list of (a_idx, b_idx, BondType) — bonds INTERNAL to fragment
#   attach: BondType for the bond from scaffold host to fragment atom 0
#
# Atom 0 is the anchor — it bonds to the scaffold host. Internal bonds
# always reference previously-added atoms within the fragment.
#
# Model class index (1..22) → fragment spec.
_TERMINAL_SPECS = {
    # ───── the earlier encoder K=9 originals (unchanged) ─────
    1: dict(atoms=[("O", False, 1)], bonds=[], attach=BondType.SINGLE),
    2: dict(atoms=[("C", False, 0), ("O", False, 0), ("O", False, 1)],
            bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.SINGLE)],
            attach=BondType.SINGLE),
    3: dict(atoms=[("N", False, 2)], bonds=[], attach=BondType.SINGLE),
    4: dict(atoms=[("S", False, 0), ("O", False, 0), ("O", False, 0),
                   ("O", False, 1)],
            bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.DOUBLE),
                   (0, 3, BondType.SINGLE)],
            attach=BondType.SINGLE),
    5: dict(atoms=[("F", False, 0)], bonds=[], attach=BondType.SINGLE),
    6: dict(atoms=[("C", False, 3)], bonds=[], attach=BondType.SINGLE),
    7: dict(atoms=[("O", False, 0)], bonds=[], attach=BondType.DOUBLE),
    8: dict(atoms=[("N", False, 1)], bonds=[], attach=BondType.DOUBLE),
    9: dict(atoms=[("S", False, 0)], bonds=[], attach=BondType.DOUBLE),

    # ───── the earlier encoder additions (model classes 10-16) ─────
    10: dict(atoms=[("Cl", False, 0)], bonds=[], attach=BondType.SINGLE),
    11: dict(atoms=[("Br", False, 0)], bonds=[], attach=BondType.SINGLE),
    12: dict(atoms=[("I", False, 0)], bonds=[], attach=BondType.SINGLE),
    13: dict(atoms=[("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    14: dict(atoms=[("N", False, 0), ("O", False, 0), ("O", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.DOUBLE)],
             attach=BondType.SINGLE),
    15: dict(atoms=[("O", False, 0), ("C", False, 3)],
             bonds=[(0, 1, BondType.SINGLE)],
             attach=BondType.SINGLE),
    16: dict(atoms=[("C", False, 0), ("F", False, 0),
                    ("F", False, 0), ("F", False, 0)],
             bonds=[(0, 1, BondType.SINGLE),
                    (0, 2, BondType.SINGLE),
                    (0, 3, BondType.SINGLE)],
             attach=BondType.SINGLE),

    # ───── additions (model classes 17-22) ─────
    17: dict(atoms=[("S", False, 1)], bonds=[], attach=BondType.SINGLE),
    18: dict(atoms=[("C", False, 0), ("O", False, 0), ("Cl", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.SINGLE)],
             attach=BondType.SINGLE),
    19: dict(atoms=[("O", False, 0), ("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.SINGLE), (1, 2, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    20: dict(atoms=[("S", False, 0), ("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.SINGLE), (1, 2, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    21: dict(atoms=[("N", False, 0), ("C", False, 0), ("S", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (1, 2, BondType.DOUBLE)],
             attach=BondType.SINGLE),
    22: dict(atoms=[("N", False, 0), ("C", False, 0)],
             bonds=[(0, 1, BondType.TRIPLE)],
             attach=BondType.SINGLE),
}


def assemble_molecule(
    atom_ids: np.ndarray,
    bond_classes: np.ndarray,
    atom_mask: np.ndarray,
    fragment_ids: np.ndarray,
    sanitize: bool = True,
) -> Optional[Chem.Mol]:
    """Assemble (scaffold + terminals) into an RDKit Mol.

    Notes:
      • Charged scaffold atoms (vocab ids 10-15: O-, N+, n+, N-, n-, P+)
        are constructed with the bare element symbol and SetFormalCharge,
        avoiding RDKit's "Element 'N+' not found" crash.
      • Terminals that would over-saturate the host atom — either
        exceeding the element's max valence, or pushing an aromatic atom
        beyond degree 3 (sp2 limit) — are silently skipped instead of
        being added. The host's open valence is filled with implicit H
        by SanitizeMol.

    Returns the sanitized Mol on success, None on any failure.
    """
    try:
        mol = Chem.RWMol()
        slot_to_rdkit = {}

        # 1. Scaffold atoms —  handle formal charges
        for i in range(len(atom_ids)):
            if not atom_mask[i]: continue
            vid = int(atom_ids[i])
            if vid == 0: return None
            element, is_arom = _vocab_id_to_atom(vid)
            charge = _vocab_id_to_charge(vid)
            atom = Chem.Atom(element)
            atom.SetIsAromatic(is_arom)
            if charge != 0:
                atom.SetFormalCharge(charge)
            slot_to_rdkit[i] = mol.AddAtom(atom)

        # An all-padding atom_mask places no atoms. RDKit will happily
        # sanitize the empty RWMol and MolToSmiles it to "", which
        # MolFromSmiles then parses back into a valid zero-atom molecule -
        # so an empty result would silently count as a valid sample
        # downstream. Fail explicitly instead.
        if not slot_to_rdkit:
            return None

        # 2. Scaffold bonds (upper triangle)
        for i in range(len(atom_ids)):
            if i not in slot_to_rdkit: continue
            for j in range(i + 1, len(atom_ids)):
                if j not in slot_to_rdkit: continue
                bc = int(bond_classes[i, j])
                if bc == 0: continue
                if bc not in _BOND_CLASS_TO_TYPE: return None
                mol.AddBond(slot_to_rdkit[i], slot_to_rdkit[j],
                            _BOND_CLASS_TO_TYPE[bc])

        # 3. Graft terminals —  valence-aware, skip infeasible grafts
        for i in range(len(fragment_ids)):
            if not atom_mask[i]: continue
            fid = int(fragment_ids[i])
            if fid == 0: continue
            spec = _TERMINAL_SPECS.get(fid)
            if spec is None: return None

            host_idx = slot_to_rdkit[i]
            host = mol.GetAtomWithIdx(host_idx)
            host_elem = host.GetSymbol()
            host_charge = host.GetFormalCharge()
            host_is_arom = host.GetIsAromatic()

            attach_order = _BOND_ORDER.get(spec["attach"], 1.0)
            current_val = _current_explicit_valence(mol, host_idx)
            max_val = _max_valence_for(host_elem, host_charge)

            # (a) Skip if graft would push host past its valence cap
            #     (e.g., a saturated sp3 C with 4 bonds, or pyridine N
            #     with 2 aromatic bonds = valence 3 already maxed out).
            if current_val + attach_order > max_val:
                continue
            # (b) Skip if host is aromatic and already at the sp2
            #     degree-3 cap (fusion/spiro aromatic atom). Adding any
            #     substituent breaks the ring geometry and kekulization
            #     fails downstream.
            if host_is_arom and host.GetDegree() >= 3:
                continue

            # Graft accepted: add fragment atoms and bonds
            frag_idx_to_rdkit = {}
            for k, (element, is_arom, num_h) in enumerate(spec["atoms"]):
                a = Chem.Atom(element)
                a.SetIsAromatic(is_arom)
                if num_h > 0: a.SetNumExplicitHs(num_h)
                frag_idx_to_rdkit[k] = mol.AddAtom(a)
            mol.AddBond(host_idx, frag_idx_to_rdkit[0], spec["attach"])
            for (a, b, bt) in spec["bonds"]:
                mol.AddBond(frag_idx_to_rdkit[a], frag_idx_to_rdkit[b], bt)

        # 4. Sanitize —  three-stage cascade
        #    vanilla -> Q1 aromaticity re-perception -> Q2 bounded valence repair
        #    See _repair_and_sanitize() docstring for details. Ring bonds
        #    are never touched; molecules that need scaffold modification
        #    to be valid are given up (return None).
        if sanitize:
            repaired = _repair_and_sanitize(mol)
            if repaired is None:
                return None
            return repaired
        return mol
    except Exception:
        return None


def assemble_to_smiles(
    atom_ids: np.ndarray,
    bond_classes: np.ndarray,
    atom_mask: np.ndarray,
    fragment_ids: np.ndarray,
) -> Optional[str]:
    mol = assemble_molecule(atom_ids, bond_classes, atom_mask, fragment_ids)
    if mol is None: return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def assemble_batch_to_smiles(
    atom_ids_b, bond_classes_b, atom_mask_b, fragment_ids_b,
):
    import torch
    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    a = _to_np(atom_ids_b); b = _to_np(bond_classes_b)
    m = _to_np(atom_mask_b); f = _to_np(fragment_ids_b)
    return [assemble_to_smiles(a[i], b[i], m[i], f[i]) for i in range(a.shape[0])]
