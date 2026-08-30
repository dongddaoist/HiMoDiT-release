"""
Model and pipeline tests.

These run on CPU with the smallest capacity presets and untrained
weights. They check shapes, vocabulary ranges, gradient flow, and that
the four stages compose into a working cascade - not chemistry quality,
which requires trained checkpoints.

Run with:  pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from himodit.chem.decoder import (
    B_LEN_MAX, F_FUSED, M_MAX, P_MAX_BRANCH, R_MAX, RING_6_AROM,
    aromatic_constraint_mask, decode_scaffold,
)
from himodit.chem.encoder import extract_layout
from himodit.metrics import compute_vun, describe_molecules
from himodit.models.branch_topology import (
    NO_SPIRO_CLS, N_B_BOND_CLASSES, N_B_PARENT_CLASSES, N_B_POS_CLASSES,
    N_B_SIZE_CLASSES, P_MAX, build_branch_topology_model,
    remap_spiro_pos_sentinel,
)
from himodit.models.ring_atom import (
    AROMATIC_ATOM_IDS, N_ATOM_CLASSES, build_ring_atom_model,
)
from himodit.models.ring_layout import (
    N_F_CLASSES, N_L_CLASSES, N_R_CLASSES, N_SPIRO_POS_CLASSES, N_TOKENS,
    build_ring_layout_model, postprocess_layout,
)
from himodit.models.terminal_fragment import build_terminal_model
from himodit.pipeline import (
    HiMoDiT, to_a3_spiro, to_decoder_spiro,
)

DEVICE = torch.device("cpu")
COND_DIM = 2
NUM_BOND_CLASSES = 5
NUM_FRAGMENTS = 22


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def a1():
    torch.manual_seed(0)
    return build_ring_layout_model("600K", condition_dim=COND_DIM).to(DEVICE)


@pytest.fixture(scope="module")
def a3():
    torch.manual_seed(0)
    return build_branch_topology_model("3M").to(DEVICE)


@pytest.fixture(scope="module")
def a2():
    torch.manual_seed(0)
    return build_ring_atom_model(
        "1M", condition_dim=COND_DIM, n_bond_classes=NUM_BOND_CLASSES
    ).to(DEVICE)


@pytest.fixture(scope="module")
def terminal():
    torch.manual_seed(0)
    return build_terminal_model(
        "3M", num_fragments=NUM_FRAGMENTS, num_atom_types=N_ATOM_CLASSES
    ).to(DEVICE)


@pytest.fixture(scope="module")
def real_scaffold():
    """A decoded scaffold from a real molecule, for A2 and Terminal."""
    label, _ = extract_layout("CC(=O)Nc1ccc(O)cc1")
    bond_classes, atom_mask = decode_scaffold(
        label["R"], label["F"], label["L"],
        label["B_size"], label["B_pos"], label["B_parent"], label["B_bond"],
        label["spiro_atom_positions"], label["atom_ids"], M_MAX_out=M_MAX,
    )
    arom = aromatic_constraint_mask(bond_classes, atom_mask)
    return {
        "bond_classes": torch.from_numpy(bond_classes).long()[None],
        "atom_mask": torch.from_numpy(atom_mask).bool()[None],
        "arom_mask": torch.from_numpy(arom).bool()[None],
        "atom_ids": torch.from_numpy(
            np.pad(label["atom_ids"], (0, M_MAX - len(label["atom_ids"])))
        ).long()[None],
    }


def _layout_batch(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        R=torch.randint(0, N_R_CLASSES, (batch, R_MAX), generator=g),
        F_mat=torch.randint(0, N_F_CLASSES, (batch, R_MAX, R_MAX), generator=g),
        L_mat=torch.randint(0, N_L_CLASSES, (batch, R_MAX, R_MAX), generator=g),
        condition=torch.randn(batch, COND_DIM, generator=g),
    )


# ─── A1: ring layout ───────────────────────────────────────────────────

def test_a1_token_count_matches_ring_and_pair_slots():
    assert N_TOKENS == R_MAX + R_MAX * (R_MAX - 1) // 2


def test_a1_forward_shapes(a1):
    b = _layout_batch()
    out = a1(
        R_t=b["R"], F_full_t=b["F_mat"], L_full_t=b["L_mat"],
        Spiro_full_t=torch.zeros_like(b["F_mat"]),
        alpha=torch.rand(3), condition=b["condition"],
    )
    n_pairs = R_MAX * (R_MAX - 1) // 2
    assert out["R_logits"].shape == (3, R_MAX, N_R_CLASSES)
    assert out["F_logits"].shape == (3, n_pairs, N_F_CLASSES)
    assert out["L_logits"].shape == (3, n_pairs, N_L_CLASSES)
    assert out["Spiro_logits"].shape == (3, n_pairs, N_SPIRO_POS_CLASSES)


def test_a1_loss_is_finite_and_has_gradient(a1):
    b = _layout_batch()
    out = a1.compute_loss(
        R=b["R"], F_mat=b["F_mat"], L_mat=b["L_mat"],
        spiro_pos_class=torch.zeros_like(b["F_mat"]),
        condition=b["condition"],
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [
        p.grad for p in a1.parameters() if p.grad is not None and p.grad.any()
    ]
    assert grads, "no parameter received a gradient"
    a1.zero_grad()


def test_a1_sample_stays_in_vocabulary(a1):
    out = a1.sample(condition=torch.randn(2, COND_DIM), n_steps=4, seed=0)
    assert out["R"].max() < N_R_CLASSES
    assert out["F"].max() < N_F_CLASSES
    assert out["L"].max() < N_L_CLASSES
    assert out["spiro_pos_class"].max() < N_SPIRO_POS_CLASSES


def test_a1_postprocess_enforces_decoder_invariants():
    torch.manual_seed(0)
    layout = {
        "R": torch.randint(1, N_R_CLASSES, (4, R_MAX)),
        "F": torch.randint(0, N_F_CLASSES, (4, R_MAX, R_MAX)),
        "L": torch.randint(0, N_L_CLASSES, (4, R_MAX, R_MAX)),
        "spiro_pos_class": torch.randint(
            0, N_SPIRO_POS_CLASSES, (4, R_MAX, R_MAX)
        ),
    }
    out = postprocess_layout(layout)
    F_, L_, S_ = out["F"], out["L"], out["spiro_pos_class"]

    assert torch.equal(F_, F_.transpose(1, 2)), "F must be symmetric"
    assert (F_.diagonal(dim1=1, dim2=2) == 0).all(), "F diagonal must be zero"
    assert (L_[F_ != 2] == 0).all(), "L must be zero off linked pairs"
    assert (S_[F_ != 3] == 0).all(), "spiro must be zero off spiro pairs"
    # A spiro pair with the NO_SPIRO sentinel is downgraded, since the
    # decoder cannot place a junction without a position.
    assert not ((F_ == 3) & (S_ == 0)).any()


def test_a1_postprocess_left_packs_rings():
    layout = {
        "R": torch.tensor([[0, 1, 0, 2, 0, 0]]),
        "F": torch.zeros(1, R_MAX, R_MAX, dtype=torch.long),
        "L": torch.zeros(1, R_MAX, R_MAX, dtype=torch.long),
        "spiro_pos_class": torch.zeros(1, R_MAX, R_MAX, dtype=torch.long),
    }
    R = postprocess_layout(layout)["R"][0]
    assert R.tolist() == [1, 2, 0, 0, 0, 0]


# ─── A3: branch topology ───────────────────────────────────────────────

def test_a3_slot_count_matches_encoder_branch_capacity():
    assert P_MAX == P_MAX_BRANCH


def test_a3_forward_shapes(a3):
    b = _layout_batch()
    out = a3(
        R=b["R"], F_mat=b["F_mat"], L_mat=b["L_mat"],
        spiro_pos=torch.full_like(b["F_mat"], NO_SPIRO_CLS),
        condition=b["condition"],
    )
    assert out["size_logits"].shape == (3, R_MAX, P_MAX, N_B_SIZE_CLASSES)
    assert out["pos_logits"].shape == (3, R_MAX, P_MAX, N_B_POS_CLASSES)
    assert out["parent_logits"].shape == (
        3, R_MAX, P_MAX, B_LEN_MAX, N_B_PARENT_CLASSES
    )
    assert out["bond_logits"].shape == (
        3, R_MAX, P_MAX, B_LEN_MAX, N_B_BOND_CLASSES
    )


def test_a3_loss_is_finite_and_has_gradient(a3):
    b = _layout_batch()
    g = torch.Generator().manual_seed(1)
    out = a3.compute_loss(
        R=b["R"], F_mat=b["F_mat"], L_mat=b["L_mat"],
        spiro_pos=torch.full_like(b["F_mat"], NO_SPIRO_CLS),
        B_size=torch.randint(0, N_B_SIZE_CLASSES, (3, R_MAX, P_MAX), generator=g),
        B_pos=torch.randint(0, N_B_POS_CLASSES, (3, R_MAX, P_MAX), generator=g),
        B_parent=torch.randint(
            0, N_B_PARENT_CLASSES, (3, R_MAX, P_MAX, B_LEN_MAX), generator=g
        ),
        B_bond=torch.randint(
            0, N_B_BOND_CLASSES, (3, R_MAX, P_MAX, B_LEN_MAX), generator=g
        ),
        condition=b["condition"],
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(
        p.grad is not None and p.grad.any() for p in a3.parameters()
    )
    a3.zero_grad()


def test_a3_postprocess_zeroes_inactive_slots(a3):
    b = _layout_batch()
    out = a3.sample(
        R=b["R"], F_mat=b["F_mat"], L_mat=b["L_mat"],
        spiro_pos=torch.full_like(b["F_mat"], NO_SPIRO_CLS),
        condition=b["condition"], post_process=True, seed=0,
    )
    inactive = out["B_size"] == 0
    assert (out["B_pos"][inactive] == 0).all()
    idx = torch.arange(B_LEN_MAX)
    beyond = idx.view(1, 1, 1, -1) >= out["B_size"].unsqueeze(-1)
    assert (out["B_parent"][beyond] == 0).all()
    assert (out["B_bond"][beyond] == 0).all()


def test_a3_spiro_sentinel_remap_removes_negatives():
    raw = torch.tensor([[-1, 3], [0, -1]])
    out = remap_spiro_pos_sentinel(raw)
    assert (out >= 0).all()
    assert out[0, 0] == NO_SPIRO_CLS
    assert out[0, 1] == 3
    assert raw[0, 0] == -1, "input must not be modified in place"


# ─── A2: atom identity ─────────────────────────────────────────────────

def test_a2_sample_respects_padding_and_aromaticity(a2, real_scaffold):
    atom_ids = a2.sample(
        bond_classes=real_scaffold["bond_classes"],
        atom_mask=real_scaffold["atom_mask"],
        arom_mask=real_scaffold["arom_mask"],
        condition=torch.randn(1, COND_DIM), n_steps=4, seed=0,
    )
    assert atom_ids.shape == (1, M_MAX)
    assert atom_ids.max() < N_ATOM_CLASSES

    pad = ~real_scaffold["atom_mask"]
    assert (atom_ids[pad] == 0).all(), "padding must stay PAD"

    arom = real_scaffold["arom_mask"]
    aromatic_ids = torch.tensor(list(AROMATIC_ATOM_IDS))
    assert torch.isin(atom_ids[arom], aromatic_ids).all(), (
        "atoms in aromatic rings must take aromatic identities"
    )


def test_a2_loss_is_finite(a2, real_scaffold):
    out = a2.compute_loss(
        atom_ids=real_scaffold["atom_ids"],
        bond_classes=real_scaffold["bond_classes"],
        atom_mask=real_scaffold["atom_mask"],
        arom_mask=real_scaffold["arom_mask"],
        condition=torch.randn(1, COND_DIM),
    )
    assert torch.isfinite(out["loss"])


def test_a2_needs_five_bond_classes_for_branch_bonds():
    """Branch bonds can be double or triple, so a 3-class model overflows.

    This is the failure that motivates NUM_BOND_CLASSES_SCAFFOLD = 5.
    """
    small = build_ring_atom_model("1M", condition_dim=COND_DIM,
                                  n_bond_classes=3).to(DEVICE)
    bond_classes = torch.zeros(1, M_MAX, M_MAX, dtype=torch.long)
    bond_classes[0, 0, 1] = bond_classes[0, 1, 0] = 3      # a double bond
    atom_mask = torch.zeros(1, M_MAX, dtype=torch.bool)
    atom_mask[0, :2] = True
    with pytest.raises(Exception):
        small.sample(
            bond_classes=bond_classes, atom_mask=atom_mask,
            arom_mask=torch.zeros(1, M_MAX, dtype=torch.bool),
            condition=torch.randn(1, COND_DIM), n_steps=2, seed=0,
        )


# ─── Terminal ──────────────────────────────────────────────────────────

def test_terminal_sample_shapes_and_vocabulary(terminal, real_scaffold):
    fragment_ids = terminal.sample(
        scaffold_atom_ids=real_scaffold["atom_ids"],
        scaffold_bond_classes=real_scaffold["bond_classes"],
        scaffold_atom_mask=real_scaffold["atom_mask"],
        condition=torch.randn(1, COND_DIM), n_steps=3, seed=0,
    )
    assert fragment_ids.shape == (1, M_MAX)
    assert fragment_ids.min() >= 0
    assert fragment_ids.max() <= NUM_FRAGMENTS
    pad = ~real_scaffold["atom_mask"]
    assert (fragment_ids[pad] == 0).all(), "padding must not be decorated"


# ─── Spiro convention conversions ──────────────────────────────────────

def test_spiro_conversions_round_trip():
    """A1 class 0 is NO_SPIRO; classes 1..7 are positions 0..6."""
    a1_classes = torch.tensor([[0, 1, 4, 7]])

    a3_form = to_a3_spiro(a1_classes)
    assert a3_form[0, 0] == NO_SPIRO_CLS
    assert a3_form[0, 1] == 0
    assert a3_form[0, 2] == 3
    assert a3_form[0, 3] == 6

    dec_form = to_decoder_spiro(a1_classes)
    assert dec_form[0, 0] == -1
    assert dec_form[0, 1] == 0
    assert dec_form[0, 2] == 3
    assert dec_form[0, 3] == 6


def test_spiro_conversions_never_collide_with_real_positions():
    """The sentinel must be distinguishable from position 0 in every
    encoding - the bug this conversion layer exists to prevent."""
    no_spiro = torch.tensor([[0]])
    position_zero = torch.tensor([[1]])
    assert to_a3_spiro(no_spiro) != to_a3_spiro(position_zero)
    assert to_decoder_spiro(no_spiro) != to_decoder_spiro(position_zero)


# ─── Pipeline ──────────────────────────────────────────────────────────

def test_pipeline_runs_end_to_end(a1, a3, a2, terminal):
    """The four stages compose without shape or dtype errors.

    With untrained weights the chemistry is noise and most samples fail
    to assemble; what matters here is that the cascade runs and returns
    one entry per requested molecule.
    """
    model = HiMoDiT(a1, a3, a2, terminal, DEVICE, condition_dim=COND_DIM)
    smiles = model.generate_batch(
        torch.randn(4, COND_DIM), a1_steps=3, a2_steps=3, term_steps=2, seed=0,
    )
    assert len(smiles) == 4
    assert all(s is None or isinstance(s, str) for s in smiles)


def test_pipeline_generate_returns_conditions(a1, a3, a2, terminal):
    model = HiMoDiT(a1, a3, a2, terminal, DEVICE, condition_dim=COND_DIM)
    smiles, conds = model.generate(
        n=4, batch_size=2, seed=0, return_conditions=True, progress=False,
        a1_steps=2, a2_steps=2, term_steps=2,
    )
    assert len(smiles) == 4
    assert conds.shape == (4, COND_DIM)


def test_pipeline_causal_parent_clamp_is_within_range(a1, a3, a2, terminal):
    from himodit.pipeline import _clamp_causal_parent

    B_size = torch.tensor([[[3]]])
    B_parent = torch.tensor([[[[9, 9, 9, 0, 0]]]])
    out = _clamp_causal_parent(B_parent, B_size)
    idx = torch.arange(B_parent.shape[-1])
    assert (out[0, 0, 0] <= idx).all(), "parent must not point forwards"
    assert (out <= B_size.unsqueeze(-1)).all(), "parent must stay in branch"


# ─── Metrics ───────────────────────────────────────────────────────────

def test_vun_counts_failures_separately():
    m = compute_vun(
        [None, "c1ccccc1", "c1ccccc1", "not-a-smiles", ""],
        train_canonical=set(),
    )
    assert m["n_assembly_failed"] == 1
    assert m["n_parse_failed"] == 1
    assert m["n_empty"] == 1
    assert m["n_valid"] == 2
    assert m["n_unique"] == 1


def test_vun_empty_smiles_is_not_valid():
    """Guards the RDKit quirk where "" parses to a zero-atom molecule."""
    m = compute_vun(["", "", ""], train_canonical=set())
    assert m["n_valid"] == 0
    assert m["validity"] == 0.0


def test_vun_novelty_excludes_training_molecules():
    train = {"c1ccccc1"}
    m = compute_vun(["c1ccccc1", "CCO"], train_canonical=train)
    assert m["n_valid"] == 2
    assert m["n_novel"] == 1
    assert m["novelty"] == 0.5


def test_describe_molecules_summarises_structure():
    stats = describe_molecules(["c1ccccc1", "c1ccc2ccccc2c1", None])
    assert stats["n"] == 2
    assert stats["mean_rings"] == 1.5
