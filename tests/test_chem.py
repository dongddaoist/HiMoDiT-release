"""
Chemistry-layer tests: encoder, decoder, round trip, and composition.

Run with:  pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger

from himodit.chem.compose import (
    ATOM_VOCAB, _TERMINAL_SPECS, assemble_to_smiles,
)
from himodit.chem.decoder import (
    B_LEN_MAX, F_FUSED, F_LINKED, F_NONE, F_SPIRO, M_MAX, P_MAX_BRANCH,
    R_MAX, RING_5_AROM, RING_6_AROM, RING_6_ALIPH, RING_PAD, RING_TYPE_INFO,
    aromatic_constraint_mask, decode_scaffold, ring_is_aromatic, ring_size,
)
from himodit.chem.encoder import (
    ATOM_VOCAB as ENCODER_ATOM_VOCAB, extract_layout, extract_layout_baseline,
)
from himodit.chem.terminal_smarts import CURATED_TERMINALS

RDLogger.DisableLog("rdApp.*")


# A spread of drug-like structures covering every topology the encoder
# supports: single rings, fusion, linkers, spiro, branches, charges,
# heteroaromatics, and halogens.
REPRESENTATIVE_SMILES = [
    "CC(=O)Nc1ccc(O)cc1",                          # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",                       # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                  # caffeine, fused
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                  # ibuprofen, branched
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",              # atenolol, long linker
    "OC(=O)c1ccccc1Nc1ccccc1Cl",                   # diclofenac, linked rings
    "c1ccc2ccccc2c1",                              # naphthalene
    "c1ccc(-c2ccccc2)cc1",                         # biphenyl, L=0 link
    "C1CCC2(CC1)CCCCC2",                           # spiro[5.5]undecane
    "CC[NH+](CC)Cc1ccccc1",                        # charged nitrogen
    "c1ccsc1",                                     # thiophene
    "Cc1ccc(S(=O)(=O)O)cc1",                       # tosylate
]


def _encode_ok(smiles):
    label, reason = extract_layout(smiles)
    assert label is not None, f"{smiles} rejected: {reason}"
    return label


# ─── Vocabulary consistency ────────────────────────────────────────────

def test_terminal_vocab_size_is_22():
    assert len(CURATED_TERMINALS) == 22


def test_every_terminal_has_a_construction_spec():
    """Model class = SMARTS id + 1, and compose must know every class."""
    for idx, term in enumerate(CURATED_TERMINALS):
        model_class = idx + 1
        assert model_class in _TERMINAL_SPECS, (
            f"terminal {term['name']!r} (id {idx}, model class "
            f"{model_class}) has no spec in compose._TERMINAL_SPECS"
        )
    assert len(_TERMINAL_SPECS) == len(CURATED_TERMINALS)


def test_all_terminal_smarts_compile():
    for term in CURATED_TERMINALS:
        patt = Chem.MolFromSmarts(term["detection_smarts"])
        assert patt is not None, f"{term['name']} SMARTS failed to compile"


def test_atom_vocab_matches_between_encoder_and_compose():
    assert ENCODER_ATOM_VOCAB == ATOM_VOCAB


def test_ring_type_table_is_self_consistent():
    for rt, (size, arom) in RING_TYPE_INFO.items():
        assert ring_size(rt) == size
        assert ring_is_aromatic(rt) == arom
    assert ring_size(RING_PAD) == 0


# ─── Encoder ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("smiles", REPRESENTATIVE_SMILES)
def test_encoder_accepts_representative_molecules(smiles):
    label = _encode_ok(smiles)
    assert label["M_total"] > 0
    assert label["atom_ids"].shape == (label["M_total"],)
    assert label["R"].shape == (R_MAX,)
    assert label["F"].shape == (R_MAX, R_MAX)
    assert label["B_size"].shape == (R_MAX, P_MAX_BRANCH)
    assert label["B_parent"].shape == (R_MAX, P_MAX_BRANCH, B_LEN_MAX)


def test_encoder_emits_all_required_label_fields():
    label = _encode_ok("CC(=O)Nc1ccc(O)cc1")
    required = {
        "smi", "R", "F", "L", "spiro_atom_positions",
        "B_size", "B_pos", "B_parent", "B_bond",
        "atom_ids", "M_total", "terminals",
    }
    assert required <= set(label)


def test_encoder_f_matrix_is_symmetric():
    for smiles in REPRESENTATIVE_SMILES:
        label, _ = extract_layout(smiles)
        if label is None:
            continue
        assert np.array_equal(label["F"], label["F"].T), smiles
        assert np.array_equal(label["L"], label["L"].T), smiles


def test_encoder_rejects_unparseable_smiles():
    label, reason = extract_layout("this is not a molecule")
    assert label is None
    assert reason == "smiles_parse_failed"


def test_encoder_rejects_ring_free_molecules():
    label, reason = extract_layout("CCCCCC")
    assert label is None
    assert "no_rings" in reason


def test_encoder_accepts_spiro():
    """Spiro at an sp3-quaternary carbon is supported (it was not before)."""
    label = _encode_ok("C1CCC2(CC1)CCCCC2")
    assert (label["F"] == F_SPIRO).any()
    assert label["n_spiro_junctions"] == 1


def test_encoder_accepts_branched_side_chains():
    """A branching side chain becomes a tree, not a rejection."""
    label = _encode_ok("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    assert label["n_branches"] > 0


def test_encoder_accepts_seven_membered_rings():
    """Ring sizes 3-7 are in the vocabulary."""
    label = _encode_ok("C1CCCCCC1")
    assert label["R"][0] != RING_PAD


def test_baseline_encoder_is_stricter_than_current():
    """The baseline rejects spiro and branched pendants; the current one
    accepts both. This is the headline improvement, asserted directly."""
    spiro = "C1CCC2(CC1)CCCCC2"
    base_label, base_reason = extract_layout_baseline(spiro)
    assert base_label is None and "spiro" in base_reason
    assert extract_layout(spiro)[0] is not None


# ─── Strict mode ───────────────────────────────────────────────────────

def test_strict_mode_rejects_molecules_with_unaccounted_atoms():
    """A substituent on a linker atom is not collected by the ring-rooted
    branch walk. Default mode silently drops it; strict mode rejects."""
    # The ethyl group sits on a linker atom between the two rings.
    smiles = "CC[C@@H](OC(=O)Cc1c(C)nc(C)[nH]c1=O)c1cccc([N+](=O)[O-])c1"

    loose, _ = extract_layout(smiles, strict=False)
    assert loose is not None
    mol = Chem.MolFromSmiles(smiles)
    terminal_atoms = {a for t in loose["terminals"] for a in t["atom_indices"]}
    accounted = loose["M_total"] + len(terminal_atoms)
    assert accounted < mol.GetNumAtoms(), (
        "expected this molecule to lose atoms in non-strict mode"
    )

    strict, reason = extract_layout(smiles, strict=True)
    assert strict is None
    assert reason.startswith("atoms_unaccounted_")


def test_strict_mode_accepts_clean_molecules():
    for smiles in ["CC(=O)Nc1ccc(O)cc1", "c1ccc2ccccc2c1", "c1ccsc1"]:
        label, reason = extract_layout(smiles, strict=True)
        assert label is not None, f"{smiles} rejected in strict mode: {reason}"
        assert label["n_atoms_unaccounted"] == 0


# ─── Decoder ───────────────────────────────────────────────────────────

def _empty_layout():
    return dict(
        R=np.zeros(R_MAX, dtype=np.int64),
        F=np.zeros((R_MAX, R_MAX), dtype=np.int64),
        L=np.zeros((R_MAX, R_MAX), dtype=np.int64),
        B_size=np.zeros((R_MAX, P_MAX_BRANCH), dtype=np.int64),
        B_pos=np.zeros((R_MAX, P_MAX_BRANCH), dtype=np.int64),
        B_parent=np.zeros((R_MAX, P_MAX_BRANCH, B_LEN_MAX), dtype=np.int64),
        B_bond=np.zeros((R_MAX, P_MAX_BRANCH, B_LEN_MAX), dtype=np.int64),
        spiro_atom_positions=np.full((R_MAX, R_MAX), -1, dtype=np.int64),
        atom_ids=np.zeros(M_MAX, dtype=np.int64),
    )


def _decode(layout):
    return decode_scaffold(
        layout["R"], layout["F"], layout["L"],
        layout["B_size"], layout["B_pos"],
        layout["B_parent"], layout["B_bond"],
        layout["spiro_atom_positions"], layout["atom_ids"],
        M_MAX_out=M_MAX,
    )


def test_decode_single_benzene():
    layout = _empty_layout()
    layout["R"][0] = RING_6_AROM
    bond_classes, atom_mask = _decode(layout)
    assert atom_mask.sum() == 6
    assert (np.triu(bond_classes, 1) > 0).sum() == 6      # a 6-cycle


def test_decode_fused_naphthalene():
    layout = _empty_layout()
    layout["R"][0] = RING_6_AROM
    layout["R"][1] = RING_6_AROM
    layout["F"][0, 1] = layout["F"][1, 0] = F_FUSED
    bond_classes, atom_mask = _decode(layout)
    assert atom_mask.sum() == 10                          # 6 + 6 - 2 shared
    assert (np.triu(bond_classes, 1) > 0).sum() == 11


def test_decode_linked_rings_with_linker():
    layout = _empty_layout()
    layout["R"][0] = RING_6_AROM
    layout["R"][1] = RING_6_AROM
    layout["F"][0, 1] = layout["F"][1, 0] = F_LINKED
    layout["L"][0, 1] = layout["L"][1, 0] = 2
    _, atom_mask = _decode(layout)
    assert atom_mask.sum() == 14                          # 6 + 2 + 6


def test_decode_spiro_shares_exactly_one_atom():
    layout = _empty_layout()
    layout["R"][0] = RING_6_ALIPH
    layout["R"][1] = RING_6_ALIPH
    layout["F"][0, 1] = layout["F"][1, 0] = F_SPIRO
    layout["spiro_atom_positions"][0, 1] = 0
    layout["spiro_atom_positions"][1, 0] = 0
    _, atom_mask = _decode(layout)
    assert atom_mask.sum() == 11                          # 6 + 6 - 1 shared


def test_decode_branch_attaches_to_ring():
    layout = _empty_layout()
    layout["R"][0] = RING_6_AROM
    layout["B_size"][0, 0] = 2
    layout["B_pos"][0, 0] = 0
    layout["B_parent"][0, 0, 0] = 0      # first branch atom -> ring atom
    layout["B_parent"][0, 0, 1] = 1      # second -> first branch atom
    layout["B_bond"][0, 0, 0] = 1
    layout["B_bond"][0, 0, 1] = 1
    bond_classes, atom_mask = _decode(layout)
    assert atom_mask.sum() == 8                           # 6 ring + 2 branch
    assert (np.triu(bond_classes, 1) > 0).sum() == 8      # 6 ring + 2 chain


def test_decode_raises_a_clear_error_on_overflow():
    """A layout larger than M_MAX must say so, not throw IndexError."""
    layout = _empty_layout()
    for k in range(R_MAX):
        layout["R"][k] = RING_6_ALIPH
        if k > 0:
            layout["F"][k - 1, k] = layout["F"][k, k - 1] = F_LINKED
            layout["L"][k - 1, k] = layout["L"][k, k - 1] = 10
    for k in range(R_MAX):
        for slot in range(P_MAX_BRANCH):
            layout["B_size"][k, slot] = B_LEN_MAX
    with pytest.raises(ValueError, match="M_MAX_out"):
        _decode(layout)


def test_decode_empty_layout_returns_empty_scaffold():
    bond_classes, atom_mask = _decode(_empty_layout())
    assert atom_mask.sum() == 0
    assert bond_classes.sum() == 0


def test_aromatic_mask_marks_only_aromatic_ring_atoms():
    layout = _empty_layout()
    layout["R"][0] = RING_6_AROM        # aromatic
    layout["R"][1] = RING_6_ALIPH       # aliphatic
    layout["F"][0, 1] = layout["F"][1, 0] = F_LINKED
    bond_classes, atom_mask = _decode(layout)
    arom = aromatic_constraint_mask(bond_classes, atom_mask)
    assert arom[:6].all(), "benzene atoms must be aromatic-constrained"
    assert not arom[6:12].any(), "cyclohexane atoms must not be"


# ─── Round trip ────────────────────────────────────────────────────────

@pytest.mark.parametrize("smiles", REPRESENTATIVE_SMILES)
def test_encode_decode_atom_count_round_trip(smiles):
    """The decoder must rebuild exactly the atoms the encoder counted."""
    label = _encode_ok(smiles)
    _, atom_mask = decode_scaffold(
        label["R"], label["F"], label["L"],
        label["B_size"], label["B_pos"], label["B_parent"], label["B_bond"],
        label["spiro_atom_positions"], label["atom_ids"], M_MAX_out=M_MAX,
    )
    assert int(atom_mask.sum()) == int(label["M_total"])


@pytest.mark.parametrize("smiles", REPRESENTATIVE_SMILES)
def test_encode_decode_bond_count_round_trip(smiles):
    """Decoded bonds must match the scaffold bonds of the input molecule.

    Only molecules that pass strict encoding are checked: in non-strict
    mode the encoder may drop substituents on linker atoms, and those
    labels legitimately carry fewer bonds than the source.
    """
    label, _ = extract_layout(smiles, strict=True)
    if label is None:
        pytest.skip("not round-trip-clean in strict mode")

    bond_classes, _ = decode_scaffold(
        label["R"], label["F"], label["L"],
        label["B_size"], label["B_pos"], label["B_parent"], label["B_bond"],
        label["spiro_atom_positions"], label["atom_ids"], M_MAX_out=M_MAX,
    )
    mol = Chem.MolFromSmiles(smiles)
    terminal_atoms = {a for t in label["terminals"] for a in t["atom_indices"]}
    scaffold_bonds = sum(
        1 for b in mol.GetBonds()
        if b.GetBeginAtomIdx() not in terminal_atoms
        and b.GetEndAtomIdx() not in terminal_atoms
    )
    decoded_bonds = int((np.triu(bond_classes, 1) > 0).sum())
    assert decoded_bonds == scaffold_bonds


# ─── Composition ───────────────────────────────────────────────────────

def _benzene_scaffold():
    atom_ids = np.zeros(M_MAX, dtype=np.int64)
    atom_ids[:6] = 1                                      # aromatic carbon
    atom_mask = np.zeros(M_MAX, dtype=bool)
    atom_mask[:6] = True
    bond_classes = np.zeros((M_MAX, M_MAX), dtype=np.int64)
    for i in range(6):
        j = (i + 1) % 6
        bond_classes[i, j] = bond_classes[j, i] = 2       # aromatic
    return atom_ids, bond_classes, atom_mask


def test_compose_bare_benzene():
    atom_ids, bond_classes, atom_mask = _benzene_scaffold()
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    smiles = assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    )
    assert smiles == "c1ccccc1"


@pytest.mark.parametrize("model_class", sorted(_TERMINAL_SPECS))
def test_compose_every_terminal_grafts_onto_benzene(model_class):
    """Each terminal class must produce a sanitizable molecule."""
    atom_ids, bond_classes, atom_mask = _benzene_scaffold()
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    fragment_ids[0] = model_class
    smiles = assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    )
    assert smiles is not None, f"class {model_class} failed to assemble"
    assert Chem.MolFromSmiles(smiles) is not None


def test_compose_multiple_terminals():
    atom_ids, bond_classes, atom_mask = _benzene_scaffold()
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    fragment_ids[0] = 14      # NO2
    fragment_ids[2] = 15      # OCH3
    fragment_ids[4] = 16      # CF3
    smiles = assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    )
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == 6 + 3 + 2 + 4


def test_compose_returns_none_for_empty_scaffold():
    """An all-padding mask must fail, not produce an empty molecule.

    RDKit parses "" into a valid zero-atom Mol, so returning "" here
    would let decode failures score as valid samples downstream.
    """
    atom_ids = np.zeros(M_MAX, dtype=np.int64)
    atom_mask = np.zeros(M_MAX, dtype=bool)
    bond_classes = np.zeros((M_MAX, M_MAX), dtype=np.int64)
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    assert assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    ) is None


def test_compose_skips_grafts_that_would_overload_a_host():
    """Grafting onto an already-saturated aromatic atom is skipped, and
    the molecule still sanitizes rather than being thrown away."""
    atom_ids, bond_classes, atom_mask = _benzene_scaffold()
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    fragment_ids[:6] = 6                                  # CH3 everywhere
    smiles = assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    )
    assert smiles is not None
    assert Chem.MolFromSmiles(smiles) is not None


def test_compose_handles_charged_atoms():
    """Charged vocabulary entries must not hit RDKit's element lookup."""
    atom_ids = np.zeros(M_MAX, dtype=np.int64)
    atom_ids[0] = ATOM_VOCAB.index("N+")
    atom_ids[1:4] = ATOM_VOCAB.index("C")
    atom_mask = np.zeros(M_MAX, dtype=bool)
    atom_mask[:4] = True
    bond_classes = np.zeros((M_MAX, M_MAX), dtype=np.int64)
    for i in range(3):
        bond_classes[i, i + 1] = bond_classes[i + 1, i] = 1
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    smiles = assemble_to_smiles(
        atom_ids, bond_classes, atom_mask, fragment_ids
    )
    assert smiles is not None
    assert "+" in smiles
