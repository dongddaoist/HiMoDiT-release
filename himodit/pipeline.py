"""
End-to-end generation pipeline.
===============================

Loads the four trained stages and runs them in cascade to produce SMILES:

    condition
       |
       v
    A1  -> (R, F, L, spiro_pos_class)
       |
       v
    A3  -> (B_size, B_pos, B_parent, B_bond)
       |
       v
    decode_scaffold  -> bond_classes, atom_mask, aromatic mask
       |
       v
    A2  -> atom_ids
       |
       v
    Terminal -> fragment_ids
       |
       v
    compose -> SMILES

Usage
-----
    from himodit.pipeline import HiMoDiT

    model = HiMoDiT.from_checkpoints("checkpoints/")
    smiles = model.generate(n=1000, cfg_scale=1.5)

    # or steer toward a property target, in z-scored units
    import torch
    cond = torch.tensor([[1.5, -0.5]])       # high logP, low SAS
    smiles = model.generate(condition=cond)

The three spiro conventions
---------------------------
The spiro position field is encoded differently by each component, which
is the single most error-prone seam in the cascade. All three
conversions happen here and nowhere else:

    A1 output    0 = NO_SPIRO,        1..7 = position 0..6
    A3 input     0..6 = position,     7    = NO_SPIRO  (NO_SPIRO_CLS)
    decoder      0..6 = position,     -1   = NO_SPIRO

Any new consumer of a spiro field should go through `to_a3_spiro` or
`to_decoder_spiro` rather than open-coding the arithmetic.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from himodit.chem.compose import assemble_batch_to_smiles
from himodit.chem.decoder import (
    M_MAX, aromatic_constraint_mask, decode_scaffold,
)
from himodit.models.branch_topology import (
    NO_SPIRO_CLS, build_branch_topology_model,
)
from himodit.models.ring_atom import build_ring_atom_model
from himodit.models.ring_layout import build_ring_layout_model
from himodit.models.terminal_fragment import build_terminal_model

# Bond vocabulary the scaffold decoder can emit: none, single, aromatic,
# double, triple. A2's default is 3; branch bonds add double and triple,
# so the model must be built with 5 or one-hot encoding overflows.
NUM_BOND_CLASSES_SCAFFOLD = 5
NUM_TERMINAL_FRAGMENTS = 22
NUM_ATOM_TYPES = 16

DEFAULT_SUBDIRS = {
    "a1": "a1",
    "a3": "a3",
    "a2": "a2",
    "terminal": "terminal",
}


# ─── Spiro convention conversions ──────────────────────────────────────

def to_a3_spiro(a1_spiro_pos_class: torch.Tensor) -> torch.Tensor:
    """A1 encoding (0 = NO_SPIRO) -> A3 encoding (NO_SPIRO_CLS = 7)."""
    return torch.where(
        a1_spiro_pos_class == 0,
        torch.full_like(a1_spiro_pos_class, NO_SPIRO_CLS),
        a1_spiro_pos_class - 1,
    )


def to_decoder_spiro(a1_spiro_pos_class: torch.Tensor) -> torch.Tensor:
    """A1 encoding (0 = NO_SPIRO) -> decoder encoding (-1 = NO_SPIRO)."""
    return torch.where(
        a1_spiro_pos_class == 0,
        torch.full_like(a1_spiro_pos_class, -1),
        a1_spiro_pos_class - 1,
    )


# ─── Checkpoint loading ────────────────────────────────────────────────

def _load_state(path: str, device: torch.device):
    """Read a checkpoint, handling both raw state dicts and resume bundles.

    The trainers write `best_model.pt` as a bare `state_dict` and keep the
    EMA shadow in a sibling `ema.pt`, except A3 which writes a bundle
    carrying both. Look in both places so `use_ema` means the same thing
    for every stage - otherwise it silently does nothing for three of the
    four, and generation quietly runs on raw weights.
    """
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        state, ema = sd["model"], sd.get("ema")
        config = sd.get("config")
    else:
        state, ema, config = sd, None, None

    if ema is None:
        sibling = os.path.join(os.path.dirname(path), "ema.pt")
        if os.path.isfile(sibling):
            candidate = torch.load(
                sibling, map_location=device, weights_only=False
            )
            if isinstance(candidate, dict) and "shadow" in candidate:
                ema = candidate

    return state, ema, config


def _apply_ema(model: torch.nn.Module, ema_dict, device) -> bool:
    """Copy EMA shadow weights into the model. Returns whether it applied."""
    if not ema_dict or "shadow" not in ema_dict:
        return False
    shadow = ema_dict["shadow"]
    for n, p in model.named_parameters():
        if n in shadow:
            p.data.copy_(shadow[n].to(device))
    return True


class HiMoDiT:
    """The four trained stages, wired together for generation."""

    def __init__(self, a1, a3, a2, terminal, device, condition_dim: int = 2):
        self.a1 = a1
        self.a3 = a3
        self.a2 = a2
        self.terminal = terminal
        self.device = device
        self.condition_dim = condition_dim
        for m in (a1, a3, a2, terminal):
            m.eval()

    # ── Construction ───────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        ckpt_root: str,
        subdirs: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        use_ema: bool = True,
        weights: str = "best_model.pt",
        verbose: bool = True,
    ) -> "HiMoDiT":
        """Load all four stages from a checkpoint root directory.

        Expects `ckpt_root/{a1,a3,a2,terminal}/` each containing
        `config.json` and `best_model.pt`, which is what the trainers
        write. Pass `subdirs` to override the layout.
        """
        subdirs = subdirs or DEFAULT_SUBDIRS
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        def _dir(stage: str) -> str:
            path = os.path.join(ckpt_root, subdirs[stage])
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"No checkpoint directory for stage {stage!r} at {path}. "
                    f"Train it first with: python scripts/train.py "
                    f"--stage {stage} ..."
                )
            return path

        def _cfg(path: str) -> dict:
            cfg_path = os.path.join(path, "config.json")
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(f"Missing config.json in {path}")
            with open(cfg_path) as f:
                return json.load(f)

        loaded = {}
        cond_dim = 2

        # A1
        d = _dir("a1")
        cfg = _cfg(d)
        cond_dim = cfg.get("condition_dim", 2)
        state, ema, _ = _load_state(os.path.join(d, weights), device)
        a1 = build_ring_layout_model(
            capacity=cfg["capacity"], condition_dim=cond_dim,
        ).to(device)
        a1.load_state_dict(state)
        loaded["a1"] = use_ema and _apply_ema(a1, ema, device)

        # A3
        d = _dir("a3")
        cfg = _cfg(d)
        state, ema, _ = _load_state(os.path.join(d, weights), device)
        a3 = build_branch_topology_model(capacity=cfg["capacity"]).to(device)
        a3.load_state_dict(state)
        loaded["a3"] = use_ema and _apply_ema(a3, ema, device)

        # A2
        d = _dir("a2")
        cfg = _cfg(d)
        state, ema, _ = _load_state(os.path.join(d, weights), device)
        a2 = build_ring_atom_model(
            capacity=cfg["capacity"],
            condition_dim=cfg.get("condition_dim", cond_dim),
            n_bond_classes=cfg.get(
                "n_bond_classes", NUM_BOND_CLASSES_SCAFFOLD
            ),
        ).to(device)
        a2.load_state_dict(state)
        loaded["a2"] = use_ema and _apply_ema(a2, ema, device)

        # Terminal
        d = _dir("terminal")
        cfg = _cfg(d)
        state, ema, _ = _load_state(os.path.join(d, weights), device)
        term = build_terminal_model(
            capacity=cfg["capacity"],
            num_fragments=cfg.get("num_fragments", NUM_TERMINAL_FRAGMENTS),
            num_atom_types=cfg.get("num_atom_types", NUM_ATOM_TYPES),
        ).to(device)
        term.load_state_dict(state)
        loaded["terminal"] = use_ema and _apply_ema(term, ema, device)

        if verbose:
            for stage, with_ema in loaded.items():
                suffix = "EMA weights" if with_ema else "raw weights"
                print(f"  loaded {stage:9s} ({suffix})")

        return cls(a1, a3, a2, term, device, condition_dim=cond_dim)

    # ── Generation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_batch(
        self,
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        temperature: float = 1.0,
        a1_steps: int = 20,
        a2_steps: int = 20,
        term_steps: int = 8,
        seed: int = 0,
        enforce_causal_parent: bool = False,
        debug: bool = False,
    ) -> List[Optional[str]]:
        """Generate one batch. Returns SMILES, with None where assembly failed.

        `condition` is (B, condition_dim), already z-scored the same way
        the training labels were.

        `enforce_causal_parent` clamps A3's B_parent so that atom i can
        only attach to the ring root or an earlier atom of the same
        branch. Off by default, which reproduces the published numbers;
        see docs/limitations.md.
        """
        B = condition.shape[0]
        device = self.device

        # ── A1: ring layout ────────────────────────────────────────────
        a1_out = self.a1.sample(
            condition=condition, n_steps=a1_steps,
            temperature=temperature, cfg_scale=cfg_scale, seed=seed,
        )
        R = a1_out["R"]
        F = a1_out["F"]
        L = a1_out["L"]
        spiro_cls = a1_out["spiro_pos_class"]
        if debug:
            print(
                f"  A1: R in [{int(R.min())}, {int(R.max())}], "
                f"F values {torch.unique(F).cpu().tolist()}"
            )

        # ── A3: branch topology ────────────────────────────────────────
        a3_out = self.a3.sample(
            R=R, F_mat=F, L_mat=L,
            spiro_pos=to_a3_spiro(spiro_cls),
            condition=condition, cfg_scale=cfg_scale,
            temperature=temperature, post_process=True, seed=seed + 1,
        )
        B_size = a3_out["B_size"]
        B_pos = a3_out["B_pos"]
        B_parent = a3_out["B_parent"]
        B_bond = a3_out["B_bond"]

        if enforce_causal_parent:
            B_parent = _clamp_causal_parent(B_parent, B_size)

        if debug:
            active = (B_size > 0).float().sum(dim=(1, 2)).mean()
            print(
                f"  A3: B_size in [{int(B_size.min())}, {int(B_size.max())}], "
                f"mean active slots {active:.2f}"
            )

        # ── Decode the scaffold (numpy, per molecule) ──────────────────
        spiro_dec = to_decoder_spiro(spiro_cls)
        arrays = {
            "R": R.cpu().numpy(), "F": F.cpu().numpy(), "L": L.cpu().numpy(),
            "B_size": B_size.cpu().numpy(), "B_pos": B_pos.cpu().numpy(),
            "B_parent": B_parent.cpu().numpy(),
            "B_bond": B_bond.cpu().numpy(),
            "spiro": spiro_dec.cpu().numpy(),
        }
        placeholder_ids = np.zeros(M_MAX, dtype=np.int64)

        bond_list, mask_list, arom_list = [], [], []
        decode_failures = []
        for i in range(B):
            try:
                bc, am = decode_scaffold(
                    arrays["R"][i], arrays["F"][i], arrays["L"][i],
                    arrays["B_size"][i], arrays["B_pos"][i],
                    arrays["B_parent"][i], arrays["B_bond"][i],
                    arrays["spiro"][i], placeholder_ids, M_MAX_out=M_MAX,
                )
                arom = aromatic_constraint_mask(bc, am)
            except Exception as exc:                       # noqa: BLE001
                decode_failures.append((i, type(exc).__name__, str(exc)[:80]))
                bc = np.zeros((M_MAX, M_MAX), dtype=np.int64)
                am = np.zeros(M_MAX, dtype=bool)
                arom = np.zeros(M_MAX, dtype=bool)
            bond_list.append(bc)
            mask_list.append(am)
            arom_list.append(arom)

        bond_classes = torch.from_numpy(np.stack(bond_list)).long().to(device)
        atom_mask = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        arom_mask = torch.from_numpy(np.stack(arom_list)).bool().to(device)

        if debug and decode_failures:
            print(f"  decode failures: {len(decode_failures)}/{B}")
            for idx, kind, msg in decode_failures[:3]:
                print(f"    [{idx}] {kind}: {msg}")

        # ── A2: atom identities ────────────────────────────────────────
        atom_ids = self.a2.sample(
            bond_classes=bond_classes, atom_mask=atom_mask,
            arom_mask=arom_mask, condition=condition, n_steps=a2_steps,
            temperature=temperature, cfg_scale=cfg_scale, seed=seed + 2,
        )
        for i, _, _ in decode_failures:
            atom_ids[i] = 0        # empty scaffold, will assemble to None

        # ── Terminal: decoration ───────────────────────────────────────
        fragment_ids = self.terminal.sample(
            scaffold_atom_ids=atom_ids,
            scaffold_bond_classes=bond_classes,
            scaffold_atom_mask=atom_mask,
            condition=condition, n_steps=term_steps,
            temperature=temperature, seed=seed + 3,
        )
        if debug:
            decorated = (fragment_ids > 0).sum(dim=1).float().mean()
            print(f"  Terminal: mean decorations per molecule {decorated:.2f}")

        return assemble_batch_to_smiles(
            atom_ids, bond_classes, atom_mask, fragment_ids,
        )

    @torch.no_grad()
    def generate(
        self,
        n: int = 1000,
        batch_size: int = 64,
        condition: Optional[torch.Tensor] = None,
        seed: int = 0,
        return_conditions: bool = False,
        progress: bool = True,
        **kwargs,
    ):
        """Generate `n` molecules.

        With no `condition`, samples conditions from N(0, 1) per axis,
        which approximates the z-scored training distribution. Pass an
        explicit (n, condition_dim) tensor to steer toward specific
        property targets.

        Returns a list of SMILES (None where assembly failed), or a
        (smiles, conditions) pair when `return_conditions` is set - which
        is what the controllability metric needs.
        """
        torch.manual_seed(seed)
        if condition is not None and condition.shape[0] != n:
            raise ValueError(
                f"condition has {condition.shape[0]} rows but n={n}"
            )

        smiles: List[Optional[str]] = []
        conds: List[torch.Tensor] = []
        n_batches = (n + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="generating")
            except ImportError:
                pass

        for bi in iterator:
            lo = bi * batch_size
            bs = min(batch_size, n - lo)
            if bs <= 0:
                break
            if condition is None:
                cond = torch.randn(bs, self.condition_dim, device=self.device)
            else:
                cond = condition[lo:lo + bs].to(self.device)
            smiles.extend(
                self.generate_batch(cond, seed=seed + bi * 7, **kwargs)
            )
            conds.append(cond.cpu())

        smiles = smiles[:n]
        if return_conditions:
            return smiles, torch.cat(conds, dim=0)[:n]
        return smiles


def _clamp_causal_parent(
    B_parent: torch.Tensor, B_size: torch.Tensor
) -> torch.Tensor:
    """Force B_parent[..., i] <= i so every branch tree is causal.

    A3 samples each parent index independently, so nothing stops it from
    naming a parent that comes later in the branch than the atom itself,
    or one past the end of the branch entirely. The latter raises an
    IndexError inside the decoder and costs the whole molecule.

    Clamping is the conservative repair: an out-of-range parent falls
    back to the nearest legal one rather than the atom being dropped.
    """
    idx = torch.arange(B_parent.shape[-1], device=B_parent.device)
    upper = idx.view(1, 1, 1, -1).expand_as(B_parent)
    clamped = torch.minimum(B_parent, upper)
    # An atom may not exceed its own branch length either.
    size_bound = B_size.unsqueeze(-1).expand_as(B_parent)
    return torch.minimum(clamped, size_bound)
