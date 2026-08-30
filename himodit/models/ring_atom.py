"""
Stage A2 - atom identity diffusion.
===================================

Masked absorbing-state discrete diffusion over scaffold atom identities,
conditioned on the bond skeleton the decoder built from A1 and A3, plus
the property condition.

    decoded scaffold + condition  ->  atom_ids

The bond graph is a known, clean input here, not something being
denoised alongside the atoms. That separation is the point: bonds come
from a deterministic decoder, so A2 only has to answer "which element
sits at each position", and the two cannot contradict each other.

Architecture
------------
One token per atom slot (T = M_MAX). Each token sums a learnable
positional embedding and a learnable atom-value embedding whose input
vocabulary carries one extra MASK class. DiT blocks use edge-biased
attention: the score between tokens i and j is biased by a learned
per-head function of the bond class at edge (i, j), so heads can route
information preferentially along, say, aromatic bonds. Padding atoms are
masked out through `key_padding_mask`. AdaLN modulation carries time and
condition.

Loss is cross-entropy at valid positions only; padded positions
contribute nothing, since scaffold sizes vary per molecule.

Aromaticity constraint
----------------------
Atoms sitting in an aromatic ring must take an aromatic identity. During
training this is left to the data, which contains only valid
combinations. At sampling it is enforced as a hard logit mask, so the
constraint holds even for out-of-distribution layouts.

Sampling is iterative confidence-based unmasking with per-row top-k
selection and Gumbel perturbation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from himodit.models.layers import (
    AdaLN, DiTBlock, EdgeBiasedMultiheadAttention, SinusoidalTimeEmbed,
)


# ─── Vocab constants (mirror ring_layout_decoder.py) ─────────────────

# Atom vocab: {0:<PAD>, 1:c, 2:O, 3:C, 4:N, 5:n, 6:S, 7:F, 8:s, 9:o}
ATOM_PAD = 0
# Includes charged species; see docs/label_schema.md.
N_ATOM_CLASSES = 16                                # 0..15
MASK_ATOM = N_ATOM_CLASSES                         # 16 (input vocab only)
AROMATIC_ATOM_IDS = (1, 5, 8, 9, 12, 14)           # c, n, s, o, n+, n-
ALIPHATIC_ATOM_IDS = (2, 3, 4, 6, 7, 10, 11, 13, 15)  # O, C, N, S, F, O-, N+, N-, P+

# Scaffold bond vocab: {0:none, 1:single, 2:aromatic}
N_BOND_CLASSES = 3

# Atom-sequence cap (matches decoder's M_MAX)
# M_MAX hardening: single source of truth in ring_layout_decoder.
from himodit.chem.decoder import M_MAX  # noqa: E402,F401


# ─── Diffusion noise schedule ────────────────────────────────────────

def alpha_bar(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """Mask probability α(t) ∈ [0,1]. Cosine: α(t)=1-cos²(πt/2)."""
    if schedule == "cosine":
        return 1.0 - torch.cos(math.pi * t / 2.0) ** 2
    elif schedule == "linear":
        return t.clone()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def corrupt_atoms(
    atom_ids: torch.Tensor,        # (B, T) int
    atom_mask: torch.Tensor,       # (B, T) bool — True at valid atoms
    alpha: torch.Tensor,           # (B,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace valid atom positions with MASK_ATOM independently with
    prob α. Padding positions stay at ATOM_PAD (never corrupted).

    Returns (atom_ids_t, is_masked).
    """
    B, T = atom_ids.shape
    alpha_b = alpha.view(B, 1)
    rand = torch.rand_like(atom_ids, dtype=torch.float32)
    is_masked = (rand < alpha_b) & atom_mask
    atom_ids_t = torch.where(
        is_masked, torch.full_like(atom_ids, MASK_ATOM), atom_ids
    )
    # Force PAD positions to ATOM_PAD (defensive)
    atom_ids_t = torch.where(atom_mask, atom_ids_t,
                              torch.full_like(atom_ids_t, ATOM_PAD))
    return atom_ids_t, is_masked


# ─── Model components ────────────────────────────────────────────────




class GraphDiTBlock(nn.Module):
    """DiT block with edge-biased self-attention.

    Mirrors the earlier `dit_matrix.py` GraphDiT block but without cross-
    attention (A2 has no terminal-fragment memory) and with the bond
    class count fixed to 3 (scaffold-only — no double/triple bonds at
    this stage; those come in via Stage 2 terminals).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        dropout: float = 0.1,
        edge_attn_enabled: bool = True,
        bias_temperature: float = 1.0,
        n_bond_classes: int = N_BOND_CLASSES,
    ):
        super().__init__()
        self.adaln1 = AdaLN(d_model, d_cond)
        self.self_attn = EdgeBiasedMultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            bias_enabled=edge_attn_enabled,
            bias_temperature=bias_temperature,
            n_bond_classes=n_bond_classes,
        )
        self.adaln2 = AdaLN(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, d_model)
        cond: torch.Tensor,                # (B, d_cond)
        edge_probs: torch.Tensor,          # (B, T, T, n_bond_classes)
        key_padding_mask: torch.Tensor,    # (B, T) — True at IGNORED keys
    ) -> torch.Tensor:
        h = self.adaln1(x, cond)
        a = self.self_attn(
            h, h, h,
            edge_probs=edge_probs,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(a)
        h = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ─── Main model ──────────────────────────────────────────────────────

class RingAtomModel(nn.Module):
    """A2: discrete absorbing-state diffusion over scaffold atom IDs.

    Inputs at training (clean):
        atom_ids:     (B, T) long — target, in [0, N_ATOM_CLASSES)
        bond_classes: (B, T, T) long — clean from decoder, in [0, N_BOND_CLASSES)
        atom_mask:    (B, T) bool — True at valid (non-padding) atoms
        arom_mask:    (B, T) bool — True at aromatic-required positions
        condition:    (B, condition_dim) float

    Output: (B, T, N_ATOM_CLASSES) logits over clean atom classes.

    The aromatic constraint is NOT enforced in the loss — the dataset
    only contains valid atom/ring combinations, so the model learns
    it from data. The constraint is enforced as a hard logit mask
    during sampling, which matters for OOD layouts coming from A1.
    """
    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 8,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        d_cond: Optional[int] = None,
        condition_dim: int = 2,
        cfg_drop_prob: float = 0.1,
        dropout: float = 0.1,
        schedule: str = "cosine",
        time_embed_dim: int = 64,
        edge_attn_enabled: bool = True,
        bias_temperature: float = 1.0,
        m_max: int = M_MAX,
        n_bond_classes: int = N_BOND_CLASSES,
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
        self.schedule = schedule
        self.m_max = m_max
        self.n_bond_classes = n_bond_classes
        self.edge_attn_enabled = edge_attn_enabled

        # Per-position embedding. The decoder's atom ordering is canonical
        # (ring 0's atoms, ring 1's atoms, ..., linker atoms, pendant atoms),
        # so positions carry weak structural information. Edge-biased
        # attention dominates structural learning, but a small per-slot
        # embedding lets the model exploit canonical ordering.
        self.token_pos_embed = nn.Embedding(m_max, d_model)

        # Atom-value embedding with +1 MASK class (input side only).
        self.atom_value_embed = nn.Embedding(N_ATOM_CLASSES + 1, d_model)

        # Time + condition → (B, d_cond)
        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_in_proj = nn.Sequential(
            nn.Linear(condition_dim + time_embed_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Graph DiT stack
        self.blocks = nn.ModuleList([
            GraphDiTBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, d_cond=d_cond,
                dropout=dropout, edge_attn_enabled=edge_attn_enabled,
                bias_temperature=bias_temperature,
                n_bond_classes=n_bond_classes,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head over no-MASK atom vocab
        self.atom_head = nn.Linear(d_model, N_ATOM_CLASSES)
        # Zero-init bias only — keep weight at default Kaiming so gradient
        # flows through the body on step 0 (lesson from B3's gradient-flow
        # bug; see ring_layout_diffusion.py head comment).
        nn.init.zeros_(self.atom_head.bias)

    # ──────────────────────────────────────────────────────────────────
    #  Edge-prob construction
    # ──────────────────────────────────────────────────────────────────

    def _build_edge_probs(
        self, bond_classes: torch.Tensor
    ) -> torch.Tensor:
        """One-hot (B, T, T) → (B, T, T, n_bond_classes) float.

        Bond classes come from the decoder as CLEAN integers; we expose
        them to attention as one-hot probabilities. (The edge-biased
        attention applies softmax internally, which is a no-op on one-
        hot but correctly handles any future noisy-bond scenario.)
        """
        return F.one_hot(bond_classes, num_classes=self.n_bond_classes
                          ).to(self._param_dtype())

    def _param_dtype(self) -> torch.dtype:
        # Get the dtype of the first parameter (works under autocast).
        return next(self.parameters()).dtype

    # ──────────────────────────────────────────────────────────────────
    #  Conditioning
    # ──────────────────────────────────────────────────────────────────

    def _build_cond(
        self, condition: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_embed(alpha)
        cat = torch.cat([condition, t_emb], dim=-1)
        return self.cond_in_proj(cat)

    # ──────────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        atom_ids_t: torch.Tensor,      # (B, T) noisy in [0..N_ATOM_CLASSES] (incl. MASK_ATOM)
        bond_classes: torch.Tensor,    # (B, T, T) clean
        atom_mask: torch.Tensor,       # (B, T) bool, True = valid atom
        alpha: torch.Tensor,           # (B,) ∈ [0,1]
        condition: torch.Tensor,       # (B, condition_dim)
    ) -> torch.Tensor:
        """Returns (B, T, N_ATOM_CLASSES) logits."""
        B, T = atom_ids_t.shape
        device = atom_ids_t.device

        # CFG: at training, drop condition with prob cfg_drop_prob
        if self.training and self.cfg_drop_prob > 0:
            drop = (torch.rand(B, device=device) < self.cfg_drop_prob).unsqueeze(-1)
            condition = torch.where(drop, torch.zeros_like(condition), condition)

        # Token input = pos_embed + atom_value_embed
        positions = torch.arange(T, device=device)
        pos = self.token_pos_embed(positions)            # (T, d)
        val = self.atom_value_embed(atom_ids_t)          # (B, T, d)
        tokens = pos.unsqueeze(0) + val                  # (B, T, d)

        # Edge probs (one-hot over clean bonds)
        edge_probs = self._build_edge_probs(bond_classes)  # (B, T, T, n_bond)

        # key_padding_mask: True at positions to IGNORE (i.e., NOT atom_mask)
        kpm = ~atom_mask                                  # (B, T)

        # Conditioning
        cond_emb = self._build_cond(condition, alpha)

        # DiT stack
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond_emb, edge_probs=edge_probs, key_padding_mask=kpm)
        h = self.final_norm(h)
        logits = self.atom_head(h)                        # (B, T, N_ATOM)
        return logits

    # ──────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        atom_ids: torch.Tensor,        # (B, T) clean target
        bond_classes: torch.Tensor,    # (B, T, T) clean
        atom_mask: torch.Tensor,       # (B, T) bool
        condition: torch.Tensor,       # (B, condition_dim)
        arom_mask: Optional[torch.Tensor] = None,   # (B, T) bool — used only for diagnostics
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """One training step. Loss is CE on atom logits, masked to valid
        atoms (per atom_mask). Padding positions contribute 0 to loss.
        """
        device = atom_ids.device
        B, T = atom_ids.shape

        t = torch.empty(B, device=device).uniform_(alpha_min, alpha_max)
        alpha = alpha_bar(t, self.schedule)

        atom_ids_t, _ = corrupt_atoms(atom_ids, atom_mask, alpha)

        logits = self.forward(
            atom_ids_t=atom_ids_t,
            bond_classes=bond_classes,
            atom_mask=atom_mask,
            alpha=alpha,
            condition=condition,
        )                                            # (B, T, N_ATOM)

        # Masked CE: average over valid positions only.
        loss_per_pos = F.cross_entropy(
            logits.reshape(-1, N_ATOM_CLASSES),
            atom_ids.reshape(-1),
            reduction="none",
        ).reshape(B, T)                              # (B, T)
        # Zero out padding positions; normalize by valid count.
        flat_mask = atom_mask.float()
        denom = flat_mask.sum().clamp(min=1.0)
        loss = (loss_per_pos * flat_mask).sum() / denom

        # Diagnostic: per-position accuracy on valid atoms.
        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = (pred == atom_ids) & atom_mask
            acc = correct.sum().float() / denom

            # Aromatic-violation rate on argmax predictions
            if arom_mask is not None:
                arom_set = torch.tensor(AROMATIC_ATOM_IDS, device=device)
                pred_aromatic = (pred.unsqueeze(-1) == arom_set).any(dim=-1)
                # A violation = at an aromatic-required valid position,
                # the prediction is NOT in AROMATIC_ATOM_IDS.
                violations = arom_mask & atom_mask & (~pred_aromatic)
                arom_denom = (arom_mask & atom_mask).sum().float().clamp(min=1.0)
                arom_violation_rate = violations.sum().float() / arom_denom
            else:
                arom_violation_rate = torch.tensor(float("nan"), device=device)

        return {
            "loss": loss,
            "acc": acc,
            "arom_violation_rate": arom_violation_rate,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Sampling
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        bond_classes: torch.Tensor,    # (B, T, T) clean — from A1+decoder
        atom_mask: torch.Tensor,       # (B, T) bool
        arom_mask: torch.Tensor,       # (B, T) bool — aromatic constraint
        condition: torch.Tensor,       # (B, condition_dim)
        n_steps: int = 20,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Iterative confidence-based unmasking sampler with aromatic
        constraint enforced by hard logit masking.

        Returns (B, T) clean atom IDs in [0, N_ATOM_CLASSES).
        """
        device = condition.device
        B, T = atom_mask.shape
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Build constraint mask over (B, T, N_ATOM_CLASSES) — True where
        # a class is FORBIDDEN at that position.
        forbid = self._build_class_forbidden_mask(atom_mask, arom_mask)

        # Initialize: all valid positions to MASK_ATOM, padding to ATOM_PAD.
        atom_ids_t = torch.full((B, T), MASK_ATOM, dtype=torch.long, device=device)
        atom_ids_t = torch.where(atom_mask, atom_ids_t,
                                  torch.full_like(atom_ids_t, ATOM_PAD))
        is_masked = atom_mask.clone()  # currently masked positions

        total_masked_start = int(is_masked.sum().item())
        if total_masked_start == 0:
            return atom_ids_t  # no atoms to predict (all-padding batch)

        for step in range(n_steps):
            # Schedule: target #masked AT END of this step
            t_next = (n_steps - step - 1) / n_steps
            alpha_next = float(alpha_bar(torch.tensor([t_next]), self.schedule).item())
            t_now = torch.full(
                (B,), (n_steps - step) / n_steps, dtype=torch.float32, device=device,
            )

            logits = self.forward(
                atom_ids_t=atom_ids_t, bond_classes=bond_classes,
                atom_mask=atom_mask, alpha=t_now, condition=condition,
            )

            if cfg_scale != 1.0:
                uncond_logits = self.forward(
                    atom_ids_t=atom_ids_t, bond_classes=bond_classes,
                    atom_mask=atom_mask, alpha=t_now,
                    condition=torch.zeros_like(condition),
                )
                logits = uncond_logits + cfg_scale * (logits - uncond_logits)

            logits = logits / max(temperature, 1e-6)
            # Apply hard constraints: forbidden classes get -inf
            logits = logits.masked_fill(forbid, float("-inf"))

            probs = F.softmax(logits, dim=-1)            # (B, T, N_ATOM)
            conf, pred = probs.max(dim=-1)               # (B, T)

            # Determine how many to unmask
            n_currently = int(is_masked.sum().item())
            n_target = int(round(total_masked_start * alpha_next))
            n_to_unmask = max(n_currently - n_target, 0)
            if step == n_steps - 1:
                n_to_unmask = n_currently

            if n_to_unmask <= 0:
                continue

            # Per-row: pick top-K confident MASKED positions
            n_per_row = max(1, int(round(n_to_unmask / B)))
            if step == n_steps - 1:
                n_per_row = T  # plenty

            # Score: confidence at masked positions, -inf elsewhere
            u = torch.rand_like(conf).clamp_(1e-9, 1 - 1e-9)
            gumbel = -torch.log(-torch.log(u))
            scores = torch.where(
                is_masked, conf + 0.0 * gumbel,  # set Gumbel weight low for stability
                torch.full_like(conf, float("-inf")),
            )
            n_sel = min(n_per_row, T)
            _, topk_idx = scores.topk(n_sel, dim=1)        # (B, n_sel)

            # Vectorized scatter using gather: build a flat row index
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_sel)
            # Only unmask positions where score is finite (i.e., were masked)
            score_at_topk = scores.gather(1, topk_idx)
            valid = torch.isfinite(score_at_topk)

            # Pick the predicted class at those positions
            pred_at_topk = pred.gather(1, topk_idx)
            new_atom_ids_t = atom_ids_t.clone()
            new_is_masked = is_masked.clone()
            for k in range(n_sel):
                ridx = row_idx[:, k]
                cidx = topk_idx[:, k]
                v = valid[:, k]
                new_atom_ids_t[ridx, cidx] = torch.where(
                    v, pred_at_topk[:, k], new_atom_ids_t[ridx, cidx],
                )
                new_is_masked[ridx, cidx] = torch.where(
                    v, torch.zeros_like(new_is_masked[ridx, cidx]),
                    new_is_masked[ridx, cidx],
                )
            atom_ids_t = new_atom_ids_t
            is_masked = new_is_masked

        # Final cleanup: any remaining MASK → greedy fill with constraint-respecting argmax
        if is_masked.any():
            t_zero = torch.zeros(B, device=device)
            logits = self.forward(
                atom_ids_t=atom_ids_t, bond_classes=bond_classes,
                atom_mask=atom_mask, alpha=t_zero, condition=condition,
            )
            logits = logits.masked_fill(forbid, float("-inf"))
            pred = logits.argmax(-1)
            atom_ids_t = torch.where(is_masked, pred, atom_ids_t)

        # Defensive: force pad positions to ATOM_PAD
        atom_ids_t = torch.where(atom_mask, atom_ids_t,
                                  torch.full_like(atom_ids_t, ATOM_PAD))
        return atom_ids_t

    def _build_class_forbidden_mask(
        self,
        atom_mask: torch.Tensor,    # (B, T) bool
        arom_mask: torch.Tensor,    # (B, T) bool
    ) -> torch.Tensor:
        """Build (B, T, N_ATOM_CLASSES) bool: True = class is FORBIDDEN
        at this position.

        Rules:
          - At valid (atom_mask=True) aromatic-required positions: only
            AROMATIC_ATOM_IDS allowed; ATOM_PAD and ALIPHATIC_ATOM_IDS
            are forbidden.
          - At valid aliphatic-required positions: ATOM_PAD is forbidden;
            all 9 non-PAD IDs are allowed (model picks aliphatic OR
            aromatic; aromatic OK in linker/pendant chains theoretically,
            though rare — we don't forbid them).
          - At padding (atom_mask=False) positions: only ATOM_PAD allowed.
        """
        device = atom_mask.device
        B, T = atom_mask.shape
        forbid = torch.zeros(B, T, N_ATOM_CLASSES, dtype=torch.bool, device=device)

        # Padding: forbid everything except ATOM_PAD
        forbid[~atom_mask] = True
        forbid[~atom_mask, ATOM_PAD] = False

        # Valid + aromatic-required: forbid ATOM_PAD and ALIPHATIC_ATOM_IDS
        valid_arom = atom_mask & arom_mask
        forbid[valid_arom, ATOM_PAD] = True
        for aliph in ALIPHATIC_ATOM_IDS:
            forbid[valid_arom, aliph] = True

        # Valid + aliphatic-required: forbid only ATOM_PAD
        valid_aliph = atom_mask & (~arom_mask)
        forbid[valid_aliph, ATOM_PAD] = True

        return forbid


# ─── Capacity presets ────────────────────────────────────────────────

CAPACITY_PRESETS: Dict[str, Dict] = {
    # (d_model, n_layers, n_heads, d_ff) — actual param counts on next line
    "1M":   dict(d_model=128, n_layers=4,  n_heads=4, d_ff=512),    # ~1.09M
    "3M":   dict(d_model=192, n_layers=6,  n_heads=4, d_ff=768),    # ~3.61M
    "10M":  dict(d_model=256, n_layers=8,  n_heads=8, d_ff=1024),   # ~8.51M
    "30M":  dict(d_model=384, n_layers=10, n_heads=8, d_ff=1536),   # ~23.83M
}


def build_ring_atom_model(
    capacity: str = "10M",
    condition_dim: int = 2,
    cfg_drop_prob: float = 0.1,
    dropout: float = 0.1,
    schedule: str = "cosine",
    edge_attn_enabled: bool = True,
    n_bond_classes: int = N_BOND_CLASSES,
    **overrides,
) -> RingAtomModel:
    """Build A2 at a named capacity.

    capacity ∈ {'1M', '3M', '10M', '30M'}. Default '10M' for RedDB.

    A2 has more semantic capacity to use than A1 (larger atom vocab × T=24
    sequence length × structural information from bonds). For ZINC250K,
    use '30M'.

    Set edge_attn_enabled=False for the no-edge-bias ablation (mirrors
    the earlier version spec §6.3).

    n_bond_classes: number of bond classes the model expects. Default
    N_BOND_CLASSES=3 (the earlier version/the earlier encoder RedDB vocab: none/single/aromatic).
    Set to 5 for current (adds double, triple — produced by the current
    decoder from branch bonds).
    """
    if capacity not in CAPACITY_PRESETS:
        raise ValueError(
            f"Unknown capacity '{capacity}'. Options: {list(CAPACITY_PRESETS.keys())}"
        )
    cfg = dict(CAPACITY_PRESETS[capacity])
    cfg.update(overrides)
    return RingAtomModel(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        condition_dim=condition_dim, cfg_drop_prob=cfg_drop_prob,
        dropout=dropout, schedule=schedule,
        edge_attn_enabled=edge_attn_enabled,
        n_bond_classes=n_bond_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
