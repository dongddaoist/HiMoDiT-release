"""
Stage 2 - terminal decoration.
==============================

Per-atom categorical over K+1 classes: 0 means "leave this atom bare",
1..K select a functional group from the terminal vocabulary.

    scaffold (atom ids + bonds) + condition  ->  per-atom fragment id

This is the functionalization stage. A1, A3, and A2 between them produce
a decorated-nothing scaffold; this stage hangs -OH, -CH3, =O, halogens,
and the rest of the K = 22 vocabulary onto it, and
`himodit.chem.compose` grafts them.

Architecture: bond-biased self-attention over atom tokens, single
forward pass, no cross-attention.

Vocabulary mapping
------------------
Model class 0 is "no decoration" and has no SMARTS entry. Every real
fragment satisfies

    model_class = terminal_smarts_id + 1

so class 1 is OH (id 0), class 7 is =O (id 6), class 22 is isonitrile
(id 21). `himodit.chem.compose._TERMINAL_SPECS` is keyed by model class
and must stay aligned with `CURATED_TERMINALS`; the test suite asserts
this.

NUM_BOND_CLASSES is 5 (none, single, aromatic, double, triple) even
though scaffold bond matrices only ever contain the first three. The
extra two slots cost a few hundred unused parameters and keep the
`bond_bias` tensor shape stable across checkpoints.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants ──────────────────────────────────────────────────────────
# Kept at 5 for the earlier version weight-transfer compatibility. the earlier encoder scaffolds only
# use {0=none, 1=single, 2=aromatic}; slots 3, 4 are unused at inference
# but retained so saved the earlier version `bond_bias` tensors load with no surgery.
NUM_BOND_CLASSES = 5

# Default atom vocab size (matches ring_atom_diffusion.N_ATOM_CLASSES = 10).
DEFAULT_NUM_ATOM_TYPES = 10

# the earlier encoder extended terminal vocab (K=9). the earlier version used K=6.
DEFAULT_NUM_FRAGMENTS = 9

# the earlier vocab for warm-start checks.
V5_3_NUM_FRAGMENTS = 6


def _mask_class_idx(num_fragments: int) -> int:
    """The input-side MASK token index (NOT a prediction class)."""
    return num_fragments + 1   # 0..K used; K+1 = MASK


# ════════════════════════════════════════════════════════════════════════
#  CORRUPTION
# ════════════════════════════════════════════════════════════════════════

def corrupt_fragment_ids(
    fragment_ids_0: torch.Tensor,        # (B, N) long
    alpha: torch.Tensor,                  # (B,) in [0, 1]
    atom_mask: torch.Tensor,              # (B, N) bool
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask a Bernoulli-alpha fraction of real-atom targets to MASK.

    Returns (fragment_ids_t, is_masked).
    """
    B, N = fragment_ids_0.shape
    rand = torch.rand(B, N, device=fragment_ids_0.device)
    is_masked = (rand < alpha.view(B, 1)) & atom_mask
    mask_idx = _mask_class_idx(num_fragments)
    fragment_ids_t = torch.where(
        is_masked,
        torch.full_like(fragment_ids_0, mask_idx),
        fragment_ids_0,
    )
    return fragment_ids_t, is_masked


# ════════════════════════════════════════════════════════════════════════
#  CLASS WEIGHTING (inverse-frequency)
# ════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    fragment_id_counts: Dict[int, int],
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency weights over (K+1) classes, normalized so mean=1.

    `fragment_id_counts` maps {0: n_no_decoration, 1: n_OH, ...}.
    `smoothing` adds to each count before inversion (default 1.0).
    """
    n_classes = num_fragments + 1
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for fid, n in fragment_id_counts.items():
        if 0 <= fid < n_classes:
            counts[fid] = float(n)
    inv_freq = 1.0 / (counts + smoothing)
    weights = inv_freq / inv_freq.mean()
    return weights.float()


# ════════════════════════════════════════════════════════════════════════
#  DiT BACKBONE
# ════════════════════════════════════════════════════════════════════════

# NOTE: this stage defines its own SinusoidalTimeEmbed, AdaLN, and DiTBlock
# rather than importing the shared ones from himodit.models.layers. They are
# deliberately different: this AdaLN projects with a bare Linear (parameter
# `proj.weight`) where the shared one wraps SiLU + Linear in a Sequential
# (`proj.1.weight`), and this time embedding scales alpha by 1000 before the
# sinusoid. Both differences are load-bearing for checkpoint compatibility, so
# replacing them with the shared classes would silently break weight loading.


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=alpha.device)
            / max(half - 1, 1)
        )
        args = alpha.view(-1, 1) * 1000.0 * freqs.view(1, -1)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLN(nn.Module):
    """Zero-init AdaLN: γ, β predicted from cond. Initially identity.

    Note: same shape as the earlier AdaLN — single Linear(d_cond, 2*d_model)
    with zero-init weight + bias. Loads the earlier version checkpoints directly.
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BondBiasedAttention(nn.Module):
    """Multi-head self-attention with a per-pair learned bond-class bias.

    Tensor names match the earlier version verbatim (qkv, out_proj, bond_bias) so the earlier version
    state_dicts load directly when shapes match.

    bias_enabled=False reproduces standard multi-head self-attention,
    used for the no-edge-bias ablation.
    """

    def __init__(
        self, d_model: int, n_heads: int,
        num_bond_classes: int = NUM_BOND_CLASSES,
        dropout: float = 0.1,
        bias_enabled: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bias_enabled = bias_enabled
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        if bias_enabled:
            self.bond_bias = nn.Parameter(
                torch.zeros(n_heads, num_bond_classes)
            )
        else:
            self.register_parameter("bond_bias", None)

    def forward(
        self,
        x: torch.Tensor,                  # (B, N, d_model)
        bond_classes: torch.Tensor,       # (B, N, N) long
        atom_mask: torch.Tensor,          # (B, N) bool
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)   # (B, H, N, d_h)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        attn_logits = torch.einsum(
            "bhnd,bhmd->bhnm", q, k
        ) / math.sqrt(self.d_head)

        if self.bias_enabled:
            bias = self.bond_bias[:, bond_classes]   # (H, B, N, N)
            bias = bias.permute(1, 0, 2, 3)           # (B, H, N, N)
            attn_logits = attn_logits + bias

        key_mask = atom_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, N)
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))

        attn = attn_logits.softmax(dim=-1)
        # NaN guard: if a query has no valid keys (whole row -inf), softmax
        # yields NaN. Replace with 0; output for that position is then 0.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)
        return out


class DiTBlock(nn.Module):
    """One transformer block. Shape-compatible with the earlier version."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, d_cond: int,
        num_bond_classes: int = NUM_BOND_CLASSES,
        dropout: float = 0.1,
        bias_enabled: bool = True,
    ):
        super().__init__()
        self.norm_attn = AdaLN(d_model, d_cond)
        self.attn = BondBiasedAttention(
            d_model, n_heads, num_bond_classes, dropout=dropout,
            bias_enabled=bias_enabled,
        )
        self.norm_ff = AdaLN(d_model, d_cond)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
        bond_classes: torch.Tensor, atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm_attn(x, cond)
        h = self.attn(h, bond_classes, atom_mask)
        x = x + self.dropout(h)

        h = self.norm_ff(x, cond)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x


# ════════════════════════════════════════════════════════════════════════
#  THE STAGE 2 MODEL
# ════════════════════════════════════════════════════════════════════════

class TerminalFragmentModel(nn.Module):
    """Per-atom fragment classifier. Tensor naming matches the earlier version so state
    dicts transfer directly when shapes align."""

    def __init__(
        self,
        num_atom_types: int = DEFAULT_NUM_ATOM_TYPES,
        num_fragments: int = DEFAULT_NUM_FRAGMENTS,
        num_bond_classes: int = NUM_BOND_CLASSES,
        d_model: int = 192,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        n_layers: int = 8,
        d_cond: int = 64,
        condition_dim: int = 2,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
        bias_enabled: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.num_atom_types = num_atom_types
        self.num_fragments = num_fragments
        self.num_bond_classes = num_bond_classes
        self.d_model = d_model
        self.d_cond = d_cond
        self.condition_dim = condition_dim
        self.bias_enabled = bias_enabled

        # ── Atom embedding (matches the earlier version) ───────────────────────────
        self.atom_embed = nn.Embedding(num_atom_types, d_model)

        # ── Fragment-target embedding for diffusion input ──
        # Input: 0..K + MASK = K+2 classes.
        n_input_frag_classes = num_fragments + 2
        self.frag_input_embed = nn.Embedding(n_input_frag_classes, d_model)

        # ── Time + condition ──
        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_in_proj = nn.Linear(condition_dim + time_embed_dim, d_cond)

        # ── DiT stack ──
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                d_cond=d_cond, num_bond_classes=num_bond_classes,
                dropout=dropout, bias_enabled=bias_enabled,
            )
            for _ in range(n_layers)
        ])
        self.norm_out = AdaLN(d_model, d_cond)

        # ── Output head: (K+1) classes (no MASK) ──
        # Bias init zero (lesson from B3/B4 — head zero-init is fine for
        # bias but zero-init on weights blocks gradient flow on step 0).
        self.frag_head = nn.Linear(d_model, num_fragments + 1)
        nn.init.zeros_(self.frag_head.bias)

    @property
    def num_input_classes(self) -> int:
        return self.num_fragments + 2  # incl. MASK

    @property
    def num_output_classes(self) -> int:
        return self.num_fragments + 1

    def _build_cond(
        self, cond: torch.Tensor, alpha: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(alpha)
        cat = torch.cat([cond, t_emb], dim=-1)
        return self.cond_in_proj(cat)

    def forward(
        self,
        scaffold_atom_ids: torch.Tensor,
        scaffold_bond_classes: torch.Tensor,
        scaffold_atom_mask: torch.Tensor,
        fragment_ids_t: torch.Tensor,
        alpha: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Returns fragment_logits: (B, N, num_fragments+1)."""
        atom_h = self.atom_embed(scaffold_atom_ids)         # (B, N, d)
        frag_h = self.frag_input_embed(fragment_ids_t)
        h = atom_h + frag_h

        cond_emb = self._build_cond(condition, alpha)

        for block in self.blocks:
            h = block(h, cond_emb, scaffold_bond_classes,
                      scaffold_atom_mask)

        h = self.norm_out(h, cond_emb)
        logits = self.frag_head(h)
        return logits

    # ──────────────────────────────────────────────────────────────────
    #  Training step
    # ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        scaffold_atom_ids: torch.Tensor,
        scaffold_bond_classes: torch.Tensor,
        scaffold_atom_mask: torch.Tensor,
        site_fragment_ids: torch.Tensor,
        condition: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """One training step: sample alpha, corrupt targets, predict,
        CE loss on real-atom positions only.
        """
        B, N = site_fragment_ids.shape
        device = site_fragment_ids.device

        alpha = torch.empty(B, device=device).uniform_(alpha_min, alpha_max)
        fragment_ids_t, is_masked = corrupt_fragment_ids(
            site_fragment_ids, alpha, scaffold_atom_mask,
            num_fragments=self.num_fragments,
        )

        logits = self.forward(
            scaffold_atom_ids=scaffold_atom_ids,
            scaffold_bond_classes=scaffold_bond_classes,
            scaffold_atom_mask=scaffold_atom_mask,
            fragment_ids_t=fragment_ids_t,
            alpha=alpha,
            condition=condition,
        )

        K = self.num_output_classes
        flat_logits = logits.reshape(B * N, K)
        flat_targets = site_fragment_ids.reshape(B * N)
        flat_mask = scaffold_atom_mask.reshape(B * N)

        if class_weights is not None:
            ce = F.cross_entropy(
                flat_logits[flat_mask],
                flat_targets[flat_mask],
                weight=class_weights.to(flat_logits.device),
                reduction="mean",
            )
        else:
            ce = F.cross_entropy(
                flat_logits[flat_mask],
                flat_targets[flat_mask],
                reduction="mean",
            )

        with torch.no_grad():
            preds = flat_logits[flat_mask].argmax(dim=-1)
            targets_real = flat_targets[flat_mask]
            n_real = targets_real.numel()
            overall_acc = (preds == targets_real).float().mean()
            # Class 0 (no decoration) baseline — what fraction of targets
            # are class 0? If the model just predicts 0 always, it gets
            # this acc.
            baseline_acc = (targets_real == 0).float().mean()
            # Non-zero-class accuracy: among real fragments (target ≠ 0),
            # what fraction are correctly identified?
            nonzero_mask = targets_real != 0
            if nonzero_mask.sum() > 0:
                nonzero_acc = (
                    (preds[nonzero_mask] == targets_real[nonzero_mask])
                    .float().mean()
                )
            else:
                nonzero_acc = torch.tensor(float("nan"), device=device)

        return {
            "loss": ce,
            "overall_acc": overall_acc,
            "baseline_acc": baseline_acc,
            "nonzero_acc": nonzero_acc,
            "n_real_atoms": torch.tensor(float(n_real), device=device),
            "n_masked": is_masked.sum().float(),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        scaffold_atom_ids: torch.Tensor,
        scaffold_bond_classes: torch.Tensor,
        scaffold_atom_mask: torch.Tensor,
        condition: torch.Tensor,
        n_steps: int = 8,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Iterative confidence-naive unmasking: at each step reveal a
        random subset of currently-masked positions using the model's
        sampled prediction.

        Returns sampled fragment_ids: (B, N) long, in {0..K}.
        Padding stays at 0 (no decoration on padding).
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, N = scaffold_atom_ids.shape
        device = scaffold_atom_ids.device
        mask_idx = _mask_class_idx(self.num_fragments)

        frag_ids = torch.where(
            scaffold_atom_mask,
            torch.full_like(scaffold_atom_ids, mask_idx),
            torch.zeros_like(scaffold_atom_ids),
        )

        for step in range(n_steps):
            alpha_val = 1.0 - (step + 1) / n_steps
            alpha = torch.full((B,), alpha_val, device=device)

            logits = self.forward(
                scaffold_atom_ids=scaffold_atom_ids,
                scaffold_bond_classes=scaffold_bond_classes,
                scaffold_atom_mask=scaffold_atom_mask,
                fragment_ids_t=frag_ids,
                alpha=alpha,
                condition=condition,
            )

            still_masked = (frag_ids == mask_idx) & scaffold_atom_mask
            if temperature == 0.0:
                pred = logits.argmax(dim=-1)
            else:
                probs = (logits / temperature).softmax(dim=-1)
                pred = torch.distributions.Categorical(probs).sample()

            n_to_reveal_per_b = (
                still_masked.sum(dim=1).float()
                / max(n_steps - step, 1)
            ).ceil().long()
            for b in range(B):
                masked_idx = torch.where(still_masked[b])[0]
                if masked_idx.numel() == 0:
                    continue
                k = min(int(n_to_reveal_per_b[b].item()),
                        masked_idx.numel())
                perm = torch.randperm(masked_idx.numel(), device=device)
                reveal = masked_idx[perm[:k]]
                frag_ids[b, reveal] = pred[b, reveal]

        # Final cleanup: anything still MASK → argmax fill.
        if (frag_ids == mask_idx).any():
            alpha = torch.zeros(B, device=device)
            logits = self.forward(
                scaffold_atom_ids=scaffold_atom_ids,
                scaffold_bond_classes=scaffold_bond_classes,
                scaffold_atom_mask=scaffold_atom_mask,
                fragment_ids_t=frag_ids,
                alpha=alpha,
                condition=condition,
            )
            pred = logits.argmax(dim=-1)
            still_m = (frag_ids == mask_idx) & scaffold_atom_mask
            frag_ids = torch.where(still_m, pred, frag_ids)

        # Force padding back to 0.
        frag_ids = torch.where(
            scaffold_atom_mask, frag_ids, torch.zeros_like(frag_ids),
        )
        return frag_ids


# ════════════════════════════════════════════════════════════════════════
#  the earlier version → the earlier encoder WARM-START
# ════════════════════════════════════════════════════════════════════════

def load_v5_3_warmstart(
    model: TerminalFragmentModel,
    v5_3_state_dict: Dict[str, torch.Tensor],
    strict_shape_check: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """Load the earlier version weights into a B5 model, expanding fragment-aware tensors.

    What gets transferred verbatim:
      - atom_embed.weight                     (assuming num_atom_types matches)
      - cond_in_proj.weight, .bias            (assuming condition_dim matches)
      - time_embed (no params)
      - blocks.*.norm_attn.proj.{weight,bias}  (AdaLN)
      - blocks.*.attn.qkv.{weight,bias}
      - blocks.*.attn.out_proj.{weight,bias}
      - blocks.*.attn.bond_bias               (NUM_BOND_CLASSES kept at 5)
      - blocks.*.norm_ff.proj.{weight,bias}
      - blocks.*.mlp.0.{weight,bias}, .3.{weight,bias}
      - norm_out.proj.{weight,bias}

    What gets EXPANDED (zero-padded for new classes):
      - frag_input_embed.weight: (8, d) → (11, d)
          the earlier version row 0 (no decoration)  → B5 row 0
          the earlier version rows 1..6 (OH..CH3)    → B5 rows 1..6
          B5 rows 7, 8, 9 (=O,=NH,=S) ← zeros (new)
          the earlier version row 7 (MASK)           → B5 row 10 (MASK)
      - frag_head.weight: (7, d) → (10, d)
          the earlier version rows 0..6              → B5 rows 0..6
          B5 rows 7, 8, 9             ← zeros (new heads)
      - frag_head.bias: same expansion.

    Returns a dict mapping state-dict key → status (e.g. 'transferred',
    'expanded', 'shape_mismatch', 'missing_in_v5_3', 'extra_in_v5_3').
    Use this for diagnostics; warm-start will not raise on shape gaps if
    strict_shape_check=False.
    """
    status: Dict[str, str] = {}
    own = dict(model.state_dict())
    src = v5_3_state_dict

    own_keys = set(own.keys())
    src_keys = set(src.keys())

    for k in own_keys:
        if k.startswith("frag_input_embed"):
            v53 = src.get(k)
            if v53 is None:
                status[k] = "missing_in_v5_3"
                continue
            v54 = own[k]
            # v53: (8, d), v54: (11, d). v53 row 7 = MASK → v54 row 10.
            if v53.shape[0] != V5_3_NUM_FRAGMENTS + 2:
                status[k] = (
                    f"shape_mismatch: v5_3 has {v53.shape[0]} rows, "
                    f"expected {V5_3_NUM_FRAGMENTS + 2}"
                )
                if strict_shape_check:
                    raise ValueError(status[k])
                continue
            new = torch.zeros_like(v54)
            # 0..6 transfer
            new[:V5_3_NUM_FRAGMENTS + 1] = v53[:V5_3_NUM_FRAGMENTS + 1]
            # MASK row: v53 row 7 → v54 row K+1 (= num_fragments + 1)
            new[model.num_fragments + 1] = v53[V5_3_NUM_FRAGMENTS + 1]
            own[k] = new
            status[k] = "expanded"
        elif k.startswith("frag_head"):
            v53 = src.get(k)
            if v53 is None:
                status[k] = "missing_in_v5_3"
                continue
            v54 = own[k]
            if v53.shape[0] != V5_3_NUM_FRAGMENTS + 1:
                status[k] = (
                    f"shape_mismatch: v5_3 has {v53.shape[0]} rows, "
                    f"expected {V5_3_NUM_FRAGMENTS + 1}"
                )
                if strict_shape_check:
                    raise ValueError(status[k])
                continue
            new = torch.zeros_like(v54)
            # Rows 0..6 transfer; rows 7..9 stay zero.
            new[:V5_3_NUM_FRAGMENTS + 1] = v53[:V5_3_NUM_FRAGMENTS + 1]
            own[k] = new
            status[k] = "expanded"
        elif k in src:
            if own[k].shape == src[k].shape:
                own[k] = src[k]
                status[k] = "transferred"
            else:
                status[k] = (
                    f"shape_mismatch: v5_3 {tuple(src[k].shape)} vs "
                    f"v5_4 {tuple(own[k].shape)}"
                )
                if strict_shape_check:
                    raise ValueError(f"{k}: {status[k]}")
        else:
            status[k] = "missing_in_v5_3"

    for k in src_keys - own_keys:
        status[k] = "extra_in_v5_3"

    model.load_state_dict(own)

    if verbose:
        n_transfer = sum(1 for v in status.values() if v == "transferred")
        n_expand = sum(1 for v in status.values() if v == "expanded")
        n_miss = sum(1 for v in status.values() if v.startswith("missing"))
        n_extra = sum(1 for v in status.values() if v == "extra_in_v5_3")
        n_mismatch = sum(1 for v in status.values()
                         if v.startswith("shape_mismatch"))
        print(f"[load_v5_3_warmstart] transferred={n_transfer} "
              f"expanded={n_expand} missing={n_miss} "
              f"extra={n_extra} mismatch={n_mismatch}")
    return status


# ════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FACTORY
# ════════════════════════════════════════════════════════════════════════

CAPACITY_PRESETS: Dict[str, Dict] = {
    "1M":  dict(d_model=128, n_heads=4,  n_layers=4),
    "3M":  dict(d_model=160, n_heads=8,  n_layers=6),
    "9M":  dict(d_model=288, n_heads=8,  n_layers=8),    # ~8.6M
    "30M": dict(d_model=512, n_heads=16, n_layers=10),   # ~30M
}


def build_terminal_model(
    capacity: str = "9M",
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
    num_atom_types: int = DEFAULT_NUM_ATOM_TYPES,
    condition_dim: int = 2,
    bias_enabled: bool = True,
    dropout: float = 0.1,
    **overrides,
) -> TerminalFragmentModel:
    """Build Stage 2 at a named capacity.

    capacity ∈ {'1M', '3M', '9M', '30M'}. Default '9M' for RedDB (matches
    the earlier version baseline). For ZINC250K, use '30M'.

    Set bias_enabled=False for the no-bond-bias ablation.
    """
    if capacity not in CAPACITY_PRESETS:
        raise ValueError(
            f"Unknown capacity '{capacity}'. "
            f"Options: {list(CAPACITY_PRESETS.keys())}"
        )
    cfg = dict(CAPACITY_PRESETS[capacity])
    cfg.update(overrides)
    return TerminalFragmentModel(
        num_atom_types=num_atom_types,
        num_fragments=num_fragments,
        condition_dim=condition_dim,
        bias_enabled=bias_enabled,
        dropout=dropout,
        **cfg,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
