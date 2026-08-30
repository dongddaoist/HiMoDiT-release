"""
Stage A3 - branch topology.
===========================

Predicts the side-chain trees hanging off each ring, conditioned on the
macro layout that A1 produced and on the property condition.

    (R, F, L, spiro_pos) + condition  ->  (B_size, B_pos, B_parent, B_bond)

Per branch slot (ring k, slot p):

    B_size    number of atoms in this branch tree; 0 = empty slot
    B_pos     ring position the branch is rooted at
    B_parent  per branch atom i: parent index (0 = ring root, k>=1 = atom
              k-1 of this branch)
    B_bond    per branch atom i: bond class joining it to its parent

Design decisions
----------------
A3 predicts topology only, never atom identity. A2 owns every atom id in
the A1 -> A3 -> A2 -> Terminal order, and sees the fully decoded bond
matrix when it does so. A3 likewise sees only the macro layout and the
condition, never atom ids, which keeps training and inference inputs
identical.

Unlike A1 and A2 this is a single forward pass rather than an iterative
denoiser. Classifier-free guidance is available through condition
dropout and a paired unconditional pass at sample time.

Token sequence
--------------
  ring tokens   embed R[k]
  pair tokens   embed F[i,j], L[i,j], spiro_pos[i,j], spiro_pos[j,i]
  slot tokens   positional only; gather context through self-attention

R_MAX = 6 and P_MAX_BRANCH = 8 give 6 + 15 + 48 = 69 tokens. Slot tokens
are read out through four heads (size, pos, parent, bond).

Loss is masked cross-entropy: size is supervised everywhere since 0 is a
real class, position only on active slots, and parent and bond only at
atom indices below that slot's B_size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse DiT primitives from A1's model file (they're vocab-agnostic
# neural building blocks; safe to import despite A1's the earlier encoder constants).
from himodit.models.layers import AdaLN, DiTBlock

# Authoritative constants from the decoder
from himodit.chem.decoder import (
    R_MAX, L_MAX, B_LEN_MAX, P_MAX_BRANCH,
    F_NONE, F_FUSED, F_LINKED, F_SPIRO,
    RING_PAD, RING_TYPE_INFO,
)

# encoder uses P_MAX_BRANCH=8 (himodit/chem/encoder.py line 1098),
# while the decoder's module-level P_MAX is still 6 (its the earlier encoder value). Override
# the encoder, since the label tensors are shaped (R_MAX, P_MAX_BRANCH).
# A3's token layout, heads, and post-processing all derive from this constant
# parametrically, so the override is sufficient.
P_MAX = 8


# ─── A3 vocabulary constants ──────────────────────────────────────────

# Ring vocabulary: PAD plus 10 ring types.
N_R_CLASSES = 11

# F vocab (current): NONE, FUSED, LINKED, SPIRO.
N_F_CLASSES = 4

# L vocab: 0..L_MAX inclusive (linker length).
N_L_CLASSES = L_MAX + 1   # = 11

# Maximum ring size in the vocabulary.
MAX_RING_SIZE = 7

# Spiro position: positions 0..MAX_RING_SIZE-1 plus NO_SPIRO sentinel.
N_SPIRO_POS_CLASSES = MAX_RING_SIZE + 1   # = 8
NO_SPIRO_CLS = N_SPIRO_POS_CLASSES - 1    # = 7

# Branch-topology output vocabularies.
N_B_SIZE_CLASSES   = B_LEN_MAX + 1   # = 16 (sizes 0..15)
N_B_POS_CLASSES    = MAX_RING_SIZE   # = 7  (positions 0..6)
N_B_PARENT_CLASSES = B_LEN_MAX + 1   # = 16 (parent index 0..15)
N_B_BOND_CLASSES   = 5               # NONE, SINGLE, AROMATIC, DOUBLE, TRIPLE


# ─── Token-sequence layout ────────────────────────────────────────────

N_RING_TOKENS = R_MAX                                  # 6
N_PAIR_TOKENS = R_MAX * (R_MAX - 1) // 2               # 15
N_SLOT_TOKENS = R_MAX * P_MAX                          # 48 (P_MAX=8 from encoder)
N_TOKENS = N_RING_TOKENS + N_PAIR_TOKENS + N_SLOT_TOKENS   # 69

RING_TOKEN_START = 0
RING_TOKEN_END   = N_RING_TOKENS                       # 6
PAIR_TOKEN_START = RING_TOKEN_END                      # 6
PAIR_TOKEN_END   = PAIR_TOKEN_START + N_PAIR_TOKENS    # 21
SLOT_TOKEN_START = PAIR_TOKEN_END                      # 21
SLOT_TOKEN_END   = SLOT_TOKEN_START + N_SLOT_TOKENS    # 57


def _upper_tri_pairs(r_max: int = R_MAX) -> List[Tuple[int, int]]:
    """Canonical upper-triangular (i, j) pairs, i < j."""
    return [(i, j) for i in range(r_max) for j in range(i + 1, r_max)]


PAIR_INDICES = _upper_tri_pairs(R_MAX)
assert len(PAIR_INDICES) == N_PAIR_TOKENS


# ─── Helpers ──────────────────────────────────────────────────────────

def remap_spiro_pos_sentinel(
    spiro_pos: torch.Tensor,
    no_spiro_cls: int = NO_SPIRO_CLS,
) -> torch.Tensor:
    """Convert encoder's -1 sentinel to the in-vocab NO_SPIRO class.

    Encoder writes spiro_atom_positions[k,j] = -1 wherever F[k,j] != F_SPIRO.
    Embedding layers can't accept -1, so we map it to a valid class.

    Returns a new tensor (does NOT modify the input)."""
    out = spiro_pos.clone()
    out[out < 0] = no_spiro_cls
    return out


# ─── Main model ───────────────────────────────────────────────────────

class BranchTopologyModel(nn.Module):
    """One-pass branch-topology predictor (A3 in pipeline).

    Parameters
    ----------
    d_model : transformer hidden dim
    n_layers : number of DiT blocks
    n_heads : attention heads
    d_ff : FFN hidden dim (defaults to 4 * d_model)
    d_cond : conditioning broadcast dim (defaults to d_model)
    condition_dim : external property-vector size (default 2 for
                    (logP_norm, SAS_norm))
    cfg_drop_prob : training-time prob of dropping cond → zeros
                    (matches A1's tuned value of 0.3)
    dropout : applied after attention and FFN
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        d_ff: Optional[int] = None,
        d_cond: Optional[int] = None,
        condition_dim: int = 2,
        cfg_drop_prob: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if d_cond is None:
            d_cond = d_model

        self.d_model = d_model
        self.d_cond = d_cond
        self.condition_dim = condition_dim
        self.cfg_drop_prob = cfg_drop_prob
        self.n_tokens = N_TOKENS

        # ── Per-token role+slot positional embedding ───────────────────
        self.token_pos_embed = nn.Embedding(N_TOKENS, d_model)

        # ── Value embeddings for context categoricals ──────────────────
        # Ring tokens embed R[k] (no MASK class — A3 is one-pass, not diffusion)
        self.r_value_embed     = nn.Embedding(N_R_CLASSES, d_model)
        # Pair tokens embed four categoricals, summed
        self.f_value_embed     = nn.Embedding(N_F_CLASSES, d_model)
        self.l_value_embed     = nn.Embedding(N_L_CLASSES, d_model)
        self.spiro_value_embed = nn.Embedding(N_SPIRO_POS_CLASSES, d_model)
        # (Slot tokens are positional only — they aggregate context via SA.)

        # ── Condition projection ───────────────────────────────────────
        # No time embed: one-pass model, no diffusion timestep.
        self.cond_in_proj = nn.Sequential(
            nn.Linear(condition_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # ── DiT stack ──────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_cond,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Output heads (operate on slot tokens only) ─────────────────
        self.size_head   = nn.Linear(d_model, N_B_SIZE_CLASSES)
        self.pos_head    = nn.Linear(d_model, N_B_POS_CLASSES)
        self.parent_head = nn.Linear(
            d_model, B_LEN_MAX * N_B_PARENT_CLASSES,
        )
        self.bond_head   = nn.Linear(
            d_model, B_LEN_MAX * N_B_BOND_CLASSES,
        )
        # Zero-init biases for soft prior at start (matches A1 convention).
        for head in (self.size_head, self.pos_head,
                     self.parent_head, self.bond_head):
            nn.init.zeros_(head.bias)

        # ── Cached pair indices for upper-tri gather ───────────────────
        pi = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long)
        pj = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long)
        self.register_buffer("pair_i", pi, persistent=False)
        self.register_buffer("pair_j", pj, persistent=False)

    # ──────────────────────────────────────────────────────────────────
    #  Token assembly
    # ──────────────────────────────────────────────────────────────────

    def _gather_pair_upper_sym(
        self, mat_BNN: torch.Tensor
    ) -> torch.Tensor:
        """(B, R_MAX, R_MAX) → (B, N_PAIR_TOKENS) using pair_i, pair_j."""
        return mat_BNN[:, self.pair_i, self.pair_j]

    def _gather_pair_lower_sym(
        self, mat_BNN: torch.Tensor
    ) -> torch.Tensor:
        """(B, R_MAX, R_MAX) → (B, N_PAIR_TOKENS) using pair_j, pair_i.
        Used to read spiro_pos[j,i] (the asymmetric spiro_atom_positions
        field has different meaning at (i,j) vs (j,i))."""
        return mat_BNN[:, self.pair_j, self.pair_i]

    def _assemble_tokens(
        self,
        R: torch.Tensor,         # (B, R_MAX)
        F_mat: torch.Tensor,     # (B, R_MAX, R_MAX)
        L_mat: torch.Tensor,     # (B, R_MAX, R_MAX)
        spiro_pos: torch.Tensor, # (B, R_MAX, R_MAX) — already remapped (no -1)
    ) -> torch.Tensor:
        """Build (B, N_TOKENS, d_model) token input."""
        B = R.shape[0]
        device = R.device

        # Position embeddings — same for every batch element
        pos_ids = torch.arange(N_TOKENS, device=device)
        pos_emb = self.token_pos_embed(pos_ids)              # (N_TOKENS, d)
        tokens = pos_emb.unsqueeze(0).expand(B, -1, -1).clone()

        # Ring tokens: add r_value_embed(R[k])
        r_emb = self.r_value_embed(R)                        # (B, R_MAX, d)
        tokens[:, RING_TOKEN_START:RING_TOKEN_END] = \
            tokens[:, RING_TOKEN_START:RING_TOKEN_END] + r_emb

        # Pair tokens: add f + l + spiro_i + spiro_j
        F_pair  = self._gather_pair_upper_sym(F_mat)         # (B, N_PAIR)
        L_pair  = self._gather_pair_upper_sym(L_mat)         # (B, N_PAIR)
        SPi_pair = self._gather_pair_upper_sym(spiro_pos)    # (B, N_PAIR) — pos in ring i
        SPj_pair = self._gather_pair_lower_sym(spiro_pos)    # (B, N_PAIR) — pos in ring j
        pair_emb = (
            self.f_value_embed(F_pair)
            + self.l_value_embed(L_pair)
            + self.spiro_value_embed(SPi_pair)
            + self.spiro_value_embed(SPj_pair)
        )                                                    # (B, N_PAIR, d)
        tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] = \
            tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] + pair_emb

        # Slot tokens: positional only (already added). No value embed.
        return tokens

    # ──────────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────────

    def _project_cond(
        self,
        condition: torch.Tensor,    # (B, condition_dim)
        cfg_drop: bool = False,
    ) -> torch.Tensor:
        """Map external cond → d_cond via small MLP.

        If cfg_drop=True and training, drop the condition (replace with
        zeros) per-batch-element with prob cfg_drop_prob, enabling
        classifier-free guidance at inference.
        """
        if cfg_drop and self.training and self.cfg_drop_prob > 0:
            keep_mask = (
                torch.rand(condition.shape[0], device=condition.device)
                > self.cfg_drop_prob
            ).float().unsqueeze(-1)
            condition = condition * keep_mask
        return self.cond_in_proj(condition)

    def forward(
        self,
        R: torch.Tensor,           # (B, R_MAX)               int64
        F_mat: torch.Tensor,       # (B, R_MAX, R_MAX)        int64
        L_mat: torch.Tensor,       # (B, R_MAX, R_MAX)        int64
        spiro_pos: torch.Tensor,   # (B, R_MAX, R_MAX)        int64 — pre-remapped
        condition: torch.Tensor,   # (B, condition_dim)       float32
        cfg_drop: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Returns a dict of logits tensors:
          size_logits:    (B, R_MAX, P_MAX, N_B_SIZE_CLASSES)
          pos_logits:     (B, R_MAX, P_MAX, N_B_POS_CLASSES)
          parent_logits:  (B, R_MAX, P_MAX, B_LEN_MAX, N_B_PARENT_CLASSES)
          bond_logits:    (B, R_MAX, P_MAX, B_LEN_MAX, N_B_BOND_CLASSES)
        """
        B = R.shape[0]
        cond = self._project_cond(condition, cfg_drop=cfg_drop)   # (B, d_cond)

        tokens = self._assemble_tokens(R, F_mat, L_mat, spiro_pos)
        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.final_norm(tokens)

        # Read slot tokens and reshape to (B, R_MAX, P_MAX, d)
        slot_tokens = tokens[:, SLOT_TOKEN_START:SLOT_TOKEN_END]
        slot_tokens = slot_tokens.reshape(B, R_MAX, P_MAX, self.d_model)

        size_logits   = self.size_head(slot_tokens)
        pos_logits    = self.pos_head(slot_tokens)
        parent_flat   = self.parent_head(slot_tokens)
        bond_flat     = self.bond_head(slot_tokens)
        parent_logits = parent_flat.reshape(
            B, R_MAX, P_MAX, B_LEN_MAX, N_B_PARENT_CLASSES,
        )
        bond_logits   = bond_flat.reshape(
            B, R_MAX, P_MAX, B_LEN_MAX, N_B_BOND_CLASSES,
        )

        return {
            "size_logits":   size_logits,
            "pos_logits":    pos_logits,
            "parent_logits": parent_logits,
            "bond_logits":   bond_logits,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Loss (masked cross-entropy)
    # ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        R: torch.Tensor,           # (B, R_MAX)               int64
        F_mat: torch.Tensor,       # (B, R_MAX, R_MAX)        int64
        L_mat: torch.Tensor,       # (B, R_MAX, R_MAX)        int64
        spiro_pos: torch.Tensor,   # (B, R_MAX, R_MAX)        int64 (remapped)
        B_size: torch.Tensor,      # (B, R_MAX, P_MAX)        int64 targets
        B_pos: torch.Tensor,       # (B, R_MAX, P_MAX)        int64 targets
        B_parent: torch.Tensor,    # (B, R_MAX, P_MAX, B_LEN_MAX) int64
        B_bond: torch.Tensor,      # (B, R_MAX, P_MAX, B_LEN_MAX) int64
        condition: torch.Tensor,   # (B, condition_dim)       float32
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with 'loss' (scalar) and per-head detached
        losses + accuracies for logging."""
        out = self.forward(
            R=R, F_mat=F_mat, L_mat=L_mat, spiro_pos=spiro_pos,
            condition=condition, cfg_drop=True,
        )

        size_logits   = out["size_logits"]    # (B, R_MAX, P_MAX, S)
        pos_logits    = out["pos_logits"]     # (B, R_MAX, P_MAX, P)
        parent_logits = out["parent_logits"]  # (B, R_MAX, P_MAX, L, par)
        bond_logits   = out["bond_logits"]    # (B, R_MAX, P_MAX, L, bnd)

        # ── L_size: full supervision ──────────────────────────────────
        loss_size = F.cross_entropy(
            size_logits.reshape(-1, N_B_SIZE_CLASSES),
            B_size.reshape(-1),
        )

        # ── L_pos: mask slots with B_size == 0 ────────────────────────
        slot_active = (B_size > 0)   # (B, R_MAX, P_MAX)
        n_active = slot_active.sum().clamp(min=1)
        pos_ce = F.cross_entropy(
            pos_logits.reshape(-1, N_B_POS_CLASSES),
            B_pos.reshape(-1),
            reduction="none",
        ).reshape_as(B_size)
        loss_pos = (pos_ce * slot_active).sum() / n_active

        # ── L_parent and L_bond: mask per-atom-index ──────────────────
        # atom_active[b, k, p, i] = (i < B_size[b, k, p])
        idx_range = torch.arange(B_LEN_MAX, device=B_size.device)
        atom_active = idx_range[None, None, None, :] < B_size.unsqueeze(-1)
        n_atom_active = atom_active.sum().clamp(min=1)

        parent_ce = F.cross_entropy(
            parent_logits.reshape(-1, N_B_PARENT_CLASSES),
            B_parent.reshape(-1),
            reduction="none",
        ).reshape_as(B_parent)
        loss_parent = (parent_ce * atom_active).sum() / n_atom_active

        bond_ce = F.cross_entropy(
            bond_logits.reshape(-1, N_B_BOND_CLASSES),
            B_bond.reshape(-1),
            reduction="none",
        ).reshape_as(B_bond)
        loss_bond = (bond_ce * atom_active).sum() / n_atom_active

        # ── Accuracies (no_grad) ──────────────────────────────────────
        with torch.no_grad():
            pred_size = size_logits.argmax(-1)
            acc_size = (pred_size == B_size).float().mean()

            pred_pos = pos_logits.argmax(-1)
            pos_correct = ((pred_pos == B_pos) & slot_active).float().sum()
            acc_pos = pos_correct / n_active

            pred_par = parent_logits.argmax(-1)
            par_correct = ((pred_par == B_parent) & atom_active).float().sum()
            acc_parent = par_correct / n_atom_active

            pred_bnd = bond_logits.argmax(-1)
            bnd_correct = ((pred_bnd == B_bond) & atom_active).float().sum()
            acc_bond = bnd_correct / n_atom_active

        # ── Combine ───────────────────────────────────────────────────
        if loss_weights is None:
            loss_weights = {"size": 1.0, "pos": 1.0, "parent": 1.0, "bond": 1.0}
        total = (
            loss_weights["size"]   * loss_size
            + loss_weights["pos"]    * loss_pos
            + loss_weights["parent"] * loss_parent
            + loss_weights["bond"]   * loss_bond
        )

        return {
            "loss":        total,
            "loss_size":   loss_size.detach(),
            "loss_pos":    loss_pos.detach(),
            "loss_parent": loss_parent.detach(),
            "loss_bond":   loss_bond.detach(),
            "acc_size":    acc_size,
            "acc_pos":     acc_pos,
            "acc_parent":  acc_parent,
            "acc_bond":    acc_bond,
            "n_active_slots":      n_active.detach(),
            "n_active_atoms":      n_atom_active.detach(),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Sampling (one-pass with optional CFG)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        R: torch.Tensor,              # (B, R_MAX)
        F_mat: torch.Tensor,          # (B, R_MAX, R_MAX)
        L_mat: torch.Tensor,          # (B, R_MAX, R_MAX)
        spiro_pos: torch.Tensor,      # (B, R_MAX, R_MAX) — remapped
        condition: torch.Tensor,      # (B, condition_dim)
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
        post_process: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample (B_size, B_pos, B_parent, B_bond) from the model.

        cfg_scale: classifier-free guidance. 1.0 = conditional only.
                   >1.0 = sharpen toward conditional; <1.0 = blend toward
                   unconditional. Applied per-head, post-softmax-mixing
                   in log-prob space.
        temperature: applied to combined logits before sampling.
                     <1 = sharper, >1 = more diverse.
        post_process: zero out positions inconsistent with B_size
                      (B_pos, B_parent, B_bond at padding indices set to 0).
        """
        was_training = self.training
        self.eval()

        if seed is not None:
            gen = torch.Generator(device=R.device).manual_seed(seed)
        else:
            gen = None

        # Conditional pass
        out_c = self.forward(
            R=R, F_mat=F_mat, L_mat=L_mat, spiro_pos=spiro_pos,
            condition=condition, cfg_drop=False,
        )
        # Unconditional pass (if CFG)
        if cfg_scale != 1.0:
            uncond = torch.zeros_like(condition)
            out_u = self.forward(
                R=R, F_mat=F_mat, L_mat=L_mat, spiro_pos=spiro_pos,
                condition=uncond, cfg_drop=False,
            )
            logits = {
                k: out_u[k] + cfg_scale * (out_c[k] - out_u[k])
                for k in out_c
            }
        else:
            logits = out_c

        def _sample_categorical(lg: torch.Tensor) -> torch.Tensor:
            shape = lg.shape[:-1]
            n_cls = lg.shape[-1]
            probs = F.softmax(lg / max(temperature, 1e-6), dim=-1)
            flat = probs.reshape(-1, n_cls)
            idx = torch.multinomial(flat, num_samples=1, generator=gen)
            return idx.reshape(shape)

        B_size_s   = _sample_categorical(logits["size_logits"])
        B_pos_s    = _sample_categorical(logits["pos_logits"])
        B_parent_s = _sample_categorical(logits["parent_logits"])
        B_bond_s   = _sample_categorical(logits["bond_logits"])

        if post_process:
            # Where B_size == 0, force B_pos = 0
            slot_inactive = (B_size_s == 0)
            B_pos_s = B_pos_s.masked_fill(slot_inactive, 0)
            # Where i >= B_size, force B_parent = B_bond = 0
            idx_range = torch.arange(B_LEN_MAX, device=R.device)
            atom_inactive = (
                idx_range[None, None, None, :] >= B_size_s.unsqueeze(-1)
            )
            B_parent_s = B_parent_s.masked_fill(atom_inactive, 0)
            B_bond_s   = B_bond_s.masked_fill(atom_inactive, 0)
            # Force B_parent[..., 0] = 0 (first branch atom always attaches
            # to the ring root, by encoder convention). Done only for
            # active slots.
            slot_active_4d = (
                ~slot_inactive.unsqueeze(-1)
            ).expand_as(B_parent_s)
            first_atom_mask = torch.zeros_like(B_parent_s, dtype=torch.bool)
            first_atom_mask[..., 0] = True
            B_parent_s = B_parent_s.masked_fill(
                first_atom_mask & slot_active_4d, 0,
            )
            # Optional: also constrain B_pos to be < ring_size_k.
            # Requires looking up ring size from R[k] via RING_TYPE_INFO.
            ring_sizes = torch.zeros_like(R)
            for rt, (sz, _) in RING_TYPE_INFO.items():
                ring_sizes = torch.where(R == rt,
                                          torch.full_like(R, sz),
                                          ring_sizes)
            # Where slot is active AND B_pos >= ring_size, wrap modulo
            # ring_size (clamping is also fine; wrap is fairer to the
            # learned distribution).
            ring_sz_bp = ring_sizes.unsqueeze(-1).expand_as(B_pos_s)
            invalid_pos = (B_pos_s >= ring_sz_bp) & (~slot_inactive)
            # Wrap with modulo to keep determinism; ring_sz>0 because R!=PAD
            # for active slots. Use clamp_min(1) for PAD rings to avoid /0.
            ring_sz_safe = ring_sz_bp.clamp_min(1)
            B_pos_s = torch.where(invalid_pos,
                                   B_pos_s % ring_sz_safe,
                                   B_pos_s)
            # Also: force B_size=0 for any PAD ring (no branches on a
            # ring that doesn't exist).
            R_pad_per_slot = (R == RING_PAD).unsqueeze(-1).expand_as(B_size_s)
            B_size_s = B_size_s.masked_fill(R_pad_per_slot, 0)
            # Re-apply downstream masks after the above edits.
            slot_inactive = (B_size_s == 0)
            B_pos_s = B_pos_s.masked_fill(slot_inactive, 0)
            atom_inactive = (
                idx_range[None, None, None, :] >= B_size_s.unsqueeze(-1)
            )
            B_parent_s = B_parent_s.masked_fill(atom_inactive, 0)
            B_bond_s   = B_bond_s.masked_fill(atom_inactive, 0)

        if was_training:
            self.train()

        return {
            "B_size":   B_size_s,
            "B_pos":    B_pos_s,
            "B_parent": B_parent_s,
            "B_bond":   B_bond_s,
        }


# ─── Factory + utilities (mirror ring_layout_diffusion.py conventions) ─

@dataclass
class BranchTopologyConfig:
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: Optional[int] = None
    d_cond: Optional[int] = None
    condition_dim: int = 2
    cfg_drop_prob: float = 0.3
    dropout: float = 0.1


CAPACITY_PRESETS: Dict[str, BranchTopologyConfig] = {
    "1M":  BranchTopologyConfig(d_model=128, n_layers=4, n_heads=4),
    "3M":  BranchTopologyConfig(d_model=192, n_layers=6, n_heads=4),
    "10M": BranchTopologyConfig(d_model=384, n_layers=8, n_heads=6),
    "30M": BranchTopologyConfig(d_model=512, n_layers=12, n_heads=8),
}


def build_branch_topology_model(
    capacity: str = "10M",
    **overrides,
) -> BranchTopologyModel:
    """Construct an A3 model at a named capacity. Pass keyword overrides
    to tweak individual fields (e.g. cfg_drop_prob=0.2)."""
    if capacity not in CAPACITY_PRESETS:
        raise ValueError(
            f"Unknown capacity '{capacity}'. "
            f"Choose from {list(CAPACITY_PRESETS.keys())}"
        )
    cfg = CAPACITY_PRESETS[capacity]
    kwargs = dict(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        d_cond=cfg.d_cond,
        condition_dim=cfg.condition_dim,
        cfg_drop_prob=cfg.cfg_drop_prob,
        dropout=cfg.dropout,
    )
    kwargs.update(overrides)
    return BranchTopologyModel(**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
