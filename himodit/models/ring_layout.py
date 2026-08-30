"""
Stage A1 - ring layout diffusion.
=================================

Masked absorbing-state discrete diffusion over the macro layout:
ring types, ring relations, linker lengths, and spiro positions. This is
the first and most abstract stage of the cascade - it decides what the
molecule's ring skeleton looks like before any atom identity exists.

    condition  ->  (R, F, L, spiro_pos_class)

Token sequence
--------------
  [0 .. R_MAX-1]        ring tokens, one per ring slot, predict R[k]
  [R_MAX .. N_TOKENS-1] pair tokens, one per unordered ring pair (i<j),
                        predict F[i,j], L[i,j], spiro_pos_class[i,j]

R_MAX = 6 gives 6 ring tokens and 15 pair tokens, 21 in total. Attention
is plain self-attention: the pairwise structure here is the prediction
target, not a known input graph, so there is nothing to bias on.

Conditioning enters through AdaLN-Zero modulation of a DiT stack, with
classifier-free guidance via condition dropout at training time.

spiro_pos_class encoding
------------------------
The model works in a shifted encoding so that embeddings never see a
negative index:

    class 0     NO_SPIRO sentinel (pair is not a spiro junction)
    class 1..7  position 0..6 in the anchor ring's canonical traversal

The encoder writes -1 for "no spiro" and 0..6 for positions, so the
dataset shifts by +1 on load. The decoder expects the encoder's
convention, so `himodit.pipeline` shifts back by -1 before decoding.
A third convention exists in A3, which uses an in-vocabulary sentinel at
class 7. All three conversions are done explicitly in one place; see
`himodit/pipeline.py`.

Post-processing
---------------
`postprocess_layout` enforces the structural invariants the decoder
requires: F symmetric with zero diagonal, L zeroed off F_LINKED pairs,
spiro positions zeroed off F_SPIRO pairs, and F_SPIRO downgraded to
F_NONE where the sampled spiro position is the NO_SPIRO sentinel (the
decoder cannot place a spiro junction without a position). Ring slots
are left-packed so that R has no interior padding.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single source of truth for all capacity / vocab constants.
from himodit.models.layers import AdaLN, DiTBlock, SinusoidalTimeEmbed
from himodit.chem.decoder import (
    M_MAX, R_MAX, P_MAX, L_MAX, B_LEN_MAX,
    F_NONE, F_FUSED, F_LINKED, F_SPIRO,
    RING_TYPE_INFO, RING_PAD,
)

# Derived class counts
N_R_CLASSES = len(RING_TYPE_INFO)               # 11 (PAD + 10 ring types)
N_F_CLASSES = 4                                  # NONE, FUSED, LINKED, SPIRO
N_L_CLASSES = L_MAX + 1                          # 11 (0..10)

# spiro_pos_class encoding:
#   0           = NO_SPIRO (pair is not a spiro junction)
#   1..7        = position 0..6 in the i-side ring's canonical traversal
# Maximum ring size is 7, so a spiro position lies
# in [0, 6]; shift by +1 to make 0 free for the NO_SPIRO sentinel.
MAX_RING_SIZE = max(sz for sz, _ in RING_TYPE_INFO.values() if sz > 0)  # 7
N_SPIRO_POS_CLASSES = MAX_RING_SIZE + 1          # 8

# MASK class IDs (one beyond the vocab on the input side only)
MASK_R = N_R_CLASSES                              # 11
MASK_F = N_F_CLASSES                              # 4
MASK_L = N_L_CLASSES                              # 11
MASK_SPIRO = N_SPIRO_POS_CLASSES                  # 8

# Token-sequence layout (NO pendant tokens in current)
N_RING_TOKENS = R_MAX                              # 6
N_PAIR_TOKENS = R_MAX * (R_MAX - 1) // 2           # 15
N_TOKENS = N_RING_TOKENS + N_PAIR_TOKENS           # 21

RING_TOKEN_START = 0
RING_TOKEN_END = N_RING_TOKENS                     # 6
PAIR_TOKEN_START = RING_TOKEN_END                  # 6
PAIR_TOKEN_END = PAIR_TOKEN_START + N_PAIR_TOKENS  # 21


def _upper_tri_pairs(r_max: int = R_MAX) -> List[Tuple[int, int]]:
    """Canonical list of upper-triangular (i, j) pairs with i < j, ordered
    by row-then-column. For R_MAX = 6: 15 pairs."""
    return [(i, j) for i in range(r_max) for j in range(i + 1, r_max)]


PAIR_INDICES: List[Tuple[int, int]] = _upper_tri_pairs()
assert len(PAIR_INDICES) == N_PAIR_TOKENS, (
    f"Expected {N_PAIR_TOKENS} pairs, got {len(PAIR_INDICES)}"
)


def _pair_index(i: int, j: int, r_max: int = R_MAX) -> int:
    if i >= j:
        i, j = j, i
    return i * r_max - (i * (i + 1)) // 2 + (j - i - 1)


# Sanity
for _idx, (_i, _j) in enumerate(PAIR_INDICES):
    assert _pair_index(_i, _j) == _idx, "pair index mismatch"


# ─── Diffusion noise schedule ──────────────────────────────────────────

def alpha_bar(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "cosine":
        return 1.0 - torch.cos(math.pi * t / 2.0) ** 2
    elif schedule == "linear":
        return t.clone()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def corrupt_categorical(
    x_0: torch.Tensor,
    alpha: torch.Tensor,
    mask_class_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace each element of x_0 with mask_class_id with prob alpha."""
    B = x_0.shape[0]
    extra_dims = (1,) * (x_0.dim() - 1)
    alpha_b = alpha.view(B, *extra_dims)
    rand = torch.rand_like(x_0, dtype=torch.float32)
    is_masked = rand < alpha_b
    x_t = torch.where(is_masked, torch.full_like(x_0, mask_class_id), x_0)
    return x_t, is_masked


# ─── Building blocks (unchanged from the earlier encoder) ─────────────────────────────





# ─── A1 model ──────────────────────────────────────────────────────

class RingLayoutDiffusion(nn.Module):
    """A1 for current: discrete diffusion over (R, F, L, spiro_pos_class).

    Token sequence:
      [0..5]    ring tokens — predict R[k] (11 classes)
      [6..20]   pair tokens — predict F[i,j] (4 cls), L[i,j] (11 cls),
                              spiro_pos[i,j] (8 cls), upper triangle only
    Total 21 tokens.

    Loss is plain cross-entropy on every position. PAD is just one of R's
    classes; non-relations sit at F=F_NONE / L=0 / spiro=0 and the model
    naturally learns those defaults.
    """
    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        d_cond: Optional[int] = None,
        condition_dim: int = 2,
        cfg_drop_prob: float = 0.1,
        dropout: float = 0.1,
        schedule: str = "cosine",
        time_embed_dim: int = 64,
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
        self.r_max = R_MAX
        self.n_tokens = N_TOKENS

        # Per-token role+slot positional embedding (21 unique vectors)
        self.token_pos_embed = nn.Embedding(N_TOKENS, d_model)

        # Value embeddings (input side has +1 MASK class)
        self.r_value_embed     = nn.Embedding(N_R_CLASSES + 1,         d_model)
        self.f_value_embed     = nn.Embedding(N_F_CLASSES + 1,         d_model)
        self.l_value_embed     = nn.Embedding(N_L_CLASSES + 1,         d_model)
        self.spiro_value_embed = nn.Embedding(N_SPIRO_POS_CLASSES + 1, d_model)

        # Time + condition
        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_in_proj = nn.Sequential(
            nn.Linear(condition_dim + time_embed_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # DiT stack
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

        # Heads — ring (1), pair (3: F, L, spiro_pos)
        self.r_head     = nn.Linear(d_model, N_R_CLASSES)
        self.f_head     = nn.Linear(d_model, N_F_CLASSES)
        self.l_head     = nn.Linear(d_model, N_L_CLASSES)
        self.spiro_head = nn.Linear(d_model, N_SPIRO_POS_CLASSES)

        # Soft prior at init: zero-init biases only (weights default
        # Kaiming-uniform). Pairs the AdaLN zero-init for stable start.
        for head in (self.r_head, self.f_head, self.l_head, self.spiro_head):
            nn.init.zeros_(head.bias)

        # Pair indices as buffers for gather/scatter
        pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long)
        pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

    # ── Helpers ────────────────────────────────────────────────────────

    def gather_pair_upper(self, mat_BNN: torch.Tensor) -> torch.Tensor:
        """Extract upper-tri entries (B, R_MAX, R_MAX) → (B, N_PAIR_TOKENS)."""
        return mat_BNN[:, self.pair_i, self.pair_j]

    def scatter_pair_upper_symmetric(
        self, vec_BP: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric matrix from upper-tri vector, zero diagonal."""
        B = vec_BP.shape[0]
        out = torch.zeros(
            B, self.r_max, self.r_max,
            dtype=vec_BP.dtype, device=vec_BP.device,
        )
        out[:, self.pair_i, self.pair_j] = vec_BP
        out[:, self.pair_j, self.pair_i] = vec_BP
        return out

    def _mirror_upper_to_full(self, mat: torch.Tensor) -> torch.Tensor:
        """Used during sampling when the lower triangle should track upper."""
        out = mat.clone()
        for ii in range(self.r_max):
            for jj in range(ii):
                out[:, ii, jj] = out[:, jj, ii]
        return out

    # ── Token assembly ─────────────────────────────────────────────────

    def _assemble_input_tokens(
        self,
        R_t: torch.Tensor,          # (B, R_MAX)
        F_upper_t: torch.Tensor,    # (B, N_PAIR_TOKENS)
        L_upper_t: torch.Tensor,    # (B, N_PAIR_TOKENS)
        Spiro_upper_t: torch.Tensor,# (B, N_PAIR_TOKENS)
    ) -> torch.Tensor:
        B = R_t.shape[0]
        device = R_t.device
        positions = torch.arange(N_TOKENS, device=device)
        pos_emb = self.token_pos_embed(positions)
        tokens = pos_emb.unsqueeze(0).expand(B, -1, -1).clone()

        # Ring slots [0..5]
        r_emb = self.r_value_embed(R_t)
        tokens[:, RING_TOKEN_START:RING_TOKEN_END] = (
            tokens[:, RING_TOKEN_START:RING_TOKEN_END] + r_emb
        )

        # Pair slots [6..20]: F + L + spiro values stacked into one token
        f_emb = self.f_value_embed(F_upper_t)
        l_emb = self.l_value_embed(L_upper_t)
        s_emb = self.spiro_value_embed(Spiro_upper_t)
        tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] = (
            tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] + f_emb + l_emb + s_emb
        )
        return tokens

    def _build_cond(
        self,
        condition: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(alpha)
        cat = torch.cat([condition, t_emb], dim=-1)
        return self.cond_in_proj(cat)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        R_t: torch.Tensor,                  # (B, R_MAX)
        F_full_t: torch.Tensor,             # (B, R_MAX, R_MAX)
        L_full_t: torch.Tensor,             # (B, R_MAX, R_MAX)
        Spiro_full_t: torch.Tensor,         # (B, R_MAX, R_MAX)
        alpha: torch.Tensor,                # (B,)
        condition: torch.Tensor,            # (B, condition_dim)
    ) -> Dict[str, torch.Tensor]:
        B = R_t.shape[0]

        F_upper     = self.gather_pair_upper(F_full_t)
        L_upper     = self.gather_pair_upper(L_full_t)
        Spiro_upper = self.gather_pair_upper(Spiro_full_t)

        # CFG drop
        if self.training and self.cfg_drop_prob > 0:
            drop = (torch.rand(B, device=condition.device)
                    < self.cfg_drop_prob).unsqueeze(-1)
            condition = torch.where(drop, torch.zeros_like(condition),
                                    condition)

        tokens = self._assemble_input_tokens(
            R_t, F_upper, L_upper, Spiro_upper,
        )
        cond_emb = self._build_cond(condition, alpha)

        h = tokens
        for blk in self.blocks:
            h = blk(h, cond_emb)
        h = self.final_norm(h)

        h_ring = h[:, RING_TOKEN_START:RING_TOKEN_END]
        h_pair = h[:, PAIR_TOKEN_START:PAIR_TOKEN_END]

        return {
            "R_logits":     self.r_head(h_ring),       # (B, R_MAX, N_R)
            "F_logits":     self.f_head(h_pair),       # (B, N_PAIR, N_F)
            "L_logits":     self.l_head(h_pair),       # (B, N_PAIR, N_L)
            "Spiro_logits": self.spiro_head(h_pair),   # (B, N_PAIR, N_SPIRO)
        }

    # ── Loss ───────────────────────────────────────────────────────────

    def compute_loss(
        self,
        R: torch.Tensor,                    # (B, R_MAX)
        F_mat: torch.Tensor,                # (B, R_MAX, R_MAX)
        L_mat: torch.Tensor,                # (B, R_MAX, R_MAX)
        spiro_pos_class: torch.Tensor,      # (B, R_MAX, R_MAX) — 0=NO_SPIRO
        condition: torch.Tensor,            # (B, condition_dim)
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = R.device
        B = R.shape[0]

        # Sample timestep + corrupt
        t = torch.empty(B, device=device).uniform_(alpha_min, alpha_max)
        alpha = alpha_bar(t, self.schedule)

        R_t,     _ = corrupt_categorical(R,                alpha, MASK_R)
        F_t,     _ = corrupt_categorical(F_mat,            alpha, MASK_F)
        L_t,     _ = corrupt_categorical(L_mat,            alpha, MASK_L)
        Spiro_t, _ = corrupt_categorical(spiro_pos_class,  alpha, MASK_SPIRO)

        out = self.forward(
            R_t=R_t, F_full_t=F_t, L_full_t=L_t,
            Spiro_full_t=Spiro_t,
            alpha=alpha, condition=condition,
        )

        F_target     = self.gather_pair_upper(F_mat)
        L_target     = self.gather_pair_upper(L_mat)
        Spiro_target = self.gather_pair_upper(spiro_pos_class)

        loss_R = F.cross_entropy(
            out["R_logits"].reshape(-1, N_R_CLASSES),
            R.reshape(-1),
        )
        loss_F = F.cross_entropy(
            out["F_logits"].reshape(-1, N_F_CLASSES),
            F_target.reshape(-1),
        )
        loss_L = F.cross_entropy(
            out["L_logits"].reshape(-1, N_L_CLASSES),
            L_target.reshape(-1),
        )
        loss_Spiro = F.cross_entropy(
            out["Spiro_logits"].reshape(-1, N_SPIRO_POS_CLASSES),
            Spiro_target.reshape(-1),
        )

        with torch.no_grad():
            acc_R     = (out["R_logits"].argmax(-1) == R).float().mean()
            acc_F     = (out["F_logits"].argmax(-1) == F_target).float().mean()
            acc_L     = (out["L_logits"].argmax(-1) == L_target).float().mean()
            acc_Spiro = (out["Spiro_logits"].argmax(-1) == Spiro_target).float().mean()

        if loss_weights is None:
            # Spiro signal is rare (~2% of pairs nontrivial); weight up.
            loss_weights = {"R": 1.0, "F": 1.0, "L": 1.0, "Spiro": 3.0}

        loss = (
            loss_weights["R"]     * loss_R
            + loss_weights["F"]   * loss_F
            + loss_weights["L"]   * loss_L
            + loss_weights["Spiro"] * loss_Spiro
        )

        return {
            "loss": loss,
            "loss_R": loss_R.detach(),
            "loss_F": loss_F.detach(),
            "loss_L": loss_L.detach(),
            "loss_Spiro": loss_Spiro.detach(),
            "acc_R": acc_R, "acc_F": acc_F,
            "acc_L": acc_L, "acc_Spiro": acc_Spiro,
        }

    # ── Sampling ───────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,            # (B, condition_dim)
        n_steps: int = 20,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        post_process: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Iterative confidence-based unmasking. Same algorithm as the earlier encoder
        but operating on the current (R, F, L, spiro_pos_class) streams.

        Returns a dict with keys "R", "F", "L", "spiro_pos_class".
        """
        device = condition.device
        B = condition.shape[0]
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        diag_idx = torch.arange(self.r_max, device=device)

        # Initialize all-MASK
        R_t     = torch.full((B, self.r_max), MASK_R,
                             dtype=torch.long, device=device)
        F_t     = torch.full((B, self.r_max, self.r_max), MASK_F,
                             dtype=torch.long, device=device)
        L_t     = torch.full((B, self.r_max, self.r_max), MASK_L,
                             dtype=torch.long, device=device)
        Spiro_t = torch.full((B, self.r_max, self.r_max), MASK_SPIRO,
                             dtype=torch.long, device=device)
        F_t[:, diag_idx, diag_idx]     = 0
        L_t[:, diag_idx, diag_idx]     = 0
        Spiro_t[:, diag_idx, diag_idx] = 0

        # Maskedness: upper-tri only for pair tensors
        R_masked     = torch.ones_like(R_t,     dtype=torch.bool)
        F_masked     = torch.ones_like(F_t,     dtype=torch.bool)
        L_masked     = torch.ones_like(L_t,     dtype=torch.bool)
        Spiro_masked = torch.ones_like(Spiro_t, dtype=torch.bool)
        F_masked[:,     diag_idx, diag_idx] = False
        L_masked[:,     diag_idx, diag_idx] = False
        Spiro_masked[:, diag_idx, diag_idx] = False
        for i in range(self.r_max):
            for j in range(i):
                F_masked[:,     i, j] = False
                L_masked[:,     i, j] = False
                Spiro_masked[:, i, j] = False

        total_masked_start = (
            R_masked.sum().item()
            + F_masked.sum().item()
            + L_masked.sum().item()
            + Spiro_masked.sum().item()
        )

        for step in range(n_steps):
            t_next = torch.tensor(
                [(n_steps - step - 1) / n_steps],
                device=device, dtype=torch.float32,
            )
            alpha_next = alpha_bar(t_next, self.schedule).item()
            t_now = torch.tensor(
                [(n_steps - step) / n_steps],
                device=device, dtype=torch.float32,
            ).expand(B)

            F_full     = self._mirror_upper_to_full(F_t)
            L_full     = self._mirror_upper_to_full(L_t)
            Spiro_full = self._mirror_upper_to_full(Spiro_t)

            out = self.forward(
                R_t=R_t, F_full_t=F_full, L_full_t=L_full,
                Spiro_full_t=Spiro_full,
                alpha=t_now, condition=condition,
            )

            if cfg_scale != 1.0:
                uncond = torch.zeros_like(condition)
                out_uncond = self.forward(
                    R_t=R_t, F_full_t=F_full, L_full_t=L_full,
                    Spiro_full_t=Spiro_full,
                    alpha=t_now, condition=uncond,
                )
                for k in out:
                    out[k] = out_uncond[k] + cfg_scale * (out[k] - out_uncond[k])

            for k in out:
                out[k] = out[k] / max(temperature, 1e-6)

            R_probs = F.softmax(out["R_logits"], dim=-1)
            R_conf, R_pred = R_probs.max(dim=-1)

            F_probs = F.softmax(out["F_logits"], dim=-1)
            F_conf_u, F_pred_u = F_probs.max(dim=-1)
            L_probs = F.softmax(out["L_logits"], dim=-1)
            L_conf_u, L_pred_u = L_probs.max(dim=-1)
            Spiro_probs = F.softmax(out["Spiro_logits"], dim=-1)
            Spiro_conf_u, Spiro_pred_u = Spiro_probs.max(dim=-1)

            # Scatter pair preds to (B, R_MAX, R_MAX)
            F_conf = torch.zeros_like(F_t, dtype=torch.float32)
            F_conf[:, self.pair_i, self.pair_j] = F_conf_u
            F_pred = F_t.clone()
            F_pred[:, self.pair_i, self.pair_j] = F_pred_u

            L_conf = torch.zeros_like(L_t, dtype=torch.float32)
            L_conf[:, self.pair_i, self.pair_j] = L_conf_u
            L_pred = L_t.clone()
            L_pred[:, self.pair_i, self.pair_j] = L_pred_u

            Spiro_conf = torch.zeros_like(Spiro_t, dtype=torch.float32)
            Spiro_conf[:, self.pair_i, self.pair_j] = Spiro_conf_u
            Spiro_pred = Spiro_t.clone()
            Spiro_pred[:, self.pair_i, self.pair_j] = Spiro_pred_u

            # Count remaining masks
            n_R     = R_masked.sum().item()
            n_F     = F_masked.sum().item()
            n_L     = L_masked.sum().item()
            n_Spiro = Spiro_masked.sum().item()
            n_currently = n_R + n_F + n_L + n_Spiro

            n_target = int(round(total_masked_start * alpha_next))
            n_to_unmask = max(n_currently - n_target, 0)
            if step == n_steps - 1:
                n_to_unmask = n_currently
            if n_to_unmask <= 0:
                continue

            def gumbel_perturb(conf):
                u = torch.rand_like(conf).clamp(min=1e-9, max=1.0 - 1e-9)
                return conf - torch.log(-torch.log(u))

            scores_R = torch.where(
                R_masked, gumbel_perturb(R_conf),
                torch.full_like(R_conf, float("-inf")),
            )
            scores_F = torch.where(
                F_masked, gumbel_perturb(F_conf),
                torch.full_like(F_conf, float("-inf")),
            )
            scores_L = torch.where(
                L_masked, gumbel_perturb(L_conf),
                torch.full_like(L_conf, float("-inf")),
            )
            scores_Spiro = torch.where(
                Spiro_masked, gumbel_perturb(Spiro_conf),
                torch.full_like(Spiro_conf, float("-inf")),
            )

            n_per_row = max(1, int(round(n_to_unmask / B)))
            if step == n_steps - 1:
                n_per_row = total_masked_start // B + 1

            # Per-row flat-concat top-K (same trick as the earlier encoder)
            sR = scores_R.reshape(B, -1)
            sF = scores_F.reshape(B, -1)
            sL = scores_L.reshape(B, -1)
            sS = scores_Spiro.reshape(B, -1)
            sizes = [sR.shape[1], sF.shape[1], sL.shape[1], sS.shape[1]]
            cuts = [0]
            for s in sizes:
                cuts.append(cuts[-1] + s)
            big = torch.cat([sR, sF, sL, sS], dim=1)
            n_sel = min(n_per_row, big.shape[1])
            if n_sel <= 0:
                continue

            _, topk_idx = big.topk(n_sel, dim=1)

            for r in range(B):
                for k in range(n_sel):
                    flat_idx = topk_idx[r, k].item()
                    score_val = big[r, flat_idx].item()
                    if not math.isfinite(score_val):
                        continue
                    if cuts[0] <= flat_idx < cuts[1]:
                        local = flat_idx - cuts[0]
                        R_t[r, local] = R_pred[r, local]
                        R_masked[r, local] = False
                    elif cuts[1] <= flat_idx < cuts[2]:
                        local = flat_idx - cuts[1]
                        a = local // self.r_max
                        b = local % self.r_max
                        F_t[r, a, b] = F_pred[r, a, b]
                        F_t[r, b, a] = F_pred[r, a, b]
                        F_masked[r, a, b] = False
                    elif cuts[2] <= flat_idx < cuts[3]:
                        local = flat_idx - cuts[2]
                        a = local // self.r_max
                        b = local % self.r_max
                        L_t[r, a, b] = L_pred[r, a, b]
                        L_t[r, b, a] = L_pred[r, a, b]
                        L_masked[r, a, b] = False
                    else:
                        local = flat_idx - cuts[3]
                        a = local // self.r_max
                        b = local % self.r_max
                        Spiro_t[r, a, b] = Spiro_pred[r, a, b]
                        Spiro_t[r, b, a] = Spiro_pred[r, a, b]
                        Spiro_masked[r, a, b] = False

        # Defensive final fill if anything stayed masked
        any_left = (R_masked.any() or F_masked.any()
                    or L_masked.any() or Spiro_masked.any())
        if any_left:
            F_full = self._mirror_upper_to_full(F_t)
            L_full = self._mirror_upper_to_full(L_t)
            Spiro_full = self._mirror_upper_to_full(Spiro_t)
            out = self.forward(
                R_t=R_t.where(~R_masked,
                              torch.full_like(R_t, MASK_R)),
                F_full_t=F_full, L_full_t=L_full,
                Spiro_full_t=Spiro_full,
                alpha=torch.zeros(B, device=device),
                condition=condition,
            )
            R_pred = out["R_logits"].argmax(-1)
            F_pred_u = out["F_logits"].argmax(-1)
            L_pred_u = out["L_logits"].argmax(-1)
            S_pred_u = out["Spiro_logits"].argmax(-1)
            R_t = torch.where(R_masked, R_pred, R_t)

            F_pred = torch.zeros_like(F_t)
            F_pred[:, self.pair_i, self.pair_j] = F_pred_u
            F_pred[:, self.pair_j, self.pair_i] = F_pred_u
            F_t = torch.where(F_masked, F_pred, F_t)

            L_pred = torch.zeros_like(L_t)
            L_pred[:, self.pair_i, self.pair_j] = L_pred_u
            L_pred[:, self.pair_j, self.pair_i] = L_pred_u
            L_t = torch.where(L_masked, L_pred, L_t)

            S_pred = torch.zeros_like(Spiro_t)
            S_pred[:, self.pair_i, self.pair_j] = S_pred_u
            S_pred[:, self.pair_j, self.pair_i] = S_pred_u
            Spiro_t = torch.where(Spiro_masked, S_pred, Spiro_t)

        out_layout = {
            "R": R_t,
            "F": F_t,
            "L": L_t,
            "spiro_pos_class": Spiro_t,
        }
        if post_process:
            out_layout = postprocess_layout(out_layout)
        return out_layout


# ─── Post-processing for current ──────────────────────────────────────────

def postprocess_layout(
    layout: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Minimal cleanup so the layout is more likely to satisfy the
    decoder's structural constraints.

    Steps:
      (1) Mirror F upper→lower; zero diagonal.
      (2) Mirror L; zero where F != F_LINKED; zero diagonal.
      (3) Mirror spiro_pos_class; zero where F != F_SPIRO; zero diagonal.
      (4) For pairs where F == F_SPIRO but spiro_pos_class == 0
          (i.e., NO_SPIRO sentinel), downgrade F to F_NONE — the
          decoder requires a valid spiro position to handle F_SPIRO.
      (5) Left-pack R; permute F/L/spiro accordingly.

    Does NOT enforce tree-shaped ring graph, anchor uniqueness, etc.;
    those violations cause decode_scaffold to raise.
    """
    R = layout["R"].clone()
    F_ = layout["F"].clone()
    L = layout["L"].clone()
    S = layout["spiro_pos_class"].clone()
    B = R.shape[0]
    r_max = R.shape[1]
    device = R.device

    diag = torch.arange(r_max, device=device)

    # (1) Symmetrize F via upper triangle
    F_[:, diag, diag] = 0
    upper = torch.triu(
        torch.ones(r_max, r_max, dtype=torch.bool, device=device),
        diagonal=1,
    )
    F_upper = F_ * upper.unsqueeze(0)
    F_ = F_upper + F_upper.transpose(-1, -2)

    # (2) Symmetrize L; zero where F != F_LINKED
    L[:, diag, diag] = 0
    L_upper = L * upper.unsqueeze(0)
    L = L_upper + L_upper.transpose(-1, -2)
    L = torch.where(F_ == F_LINKED, L, torch.zeros_like(L))

    # (3) Symmetrize spiro_pos_class; zero where F != F_SPIRO
    S[:, diag, diag] = 0
    S_upper = S * upper.unsqueeze(0)
    S = S_upper + S_upper.transpose(-1, -2)
    S = torch.where(F_ == F_SPIRO, S, torch.zeros_like(S))

    # (4) Inconsistency: F == F_SPIRO but spiro_pos_class == 0
    #     Downgrade F to F_NONE (otherwise decoder will raise)
    inconsistent = (F_ == F_SPIRO) & (S == 0)
    F_ = torch.where(inconsistent, torch.full_like(F_, F_NONE), F_)
    # Re-zero L and S after F was modified (defensive)
    L = torch.where(F_ == F_LINKED, L, torch.zeros_like(L))
    S = torch.where(F_ == F_SPIRO, S, torch.zeros_like(S))

    # (5) Left-pack R per row, permute pairwise tensors
    for b in range(B):
        order = []
        for k in range(r_max):
            if int(R[b, k]) != 0:
                order.append(k)
        for k in range(r_max):
            if int(R[b, k]) == 0:
                order.append(k)
        perm = torch.tensor(order, dtype=torch.long, device=device)
        R[b]  = R[b][perm]
        F_[b] = F_[b][perm][:, perm]
        L[b]  = L[b][perm][:, perm]
        S[b]  = S[b][perm][:, perm]

    return {"R": R, "F": F_, "L": L, "spiro_pos_class": S}


# ─── Capacity presets ──────────────────────────────────────────────────
# Slightly larger defaults than the earlier encoder — A1 in has a harder vocab
# (K=11 ring types vs 5; K=4 F vs 3; rare F_SPIRO; spiro_pos head).

CAPACITY_PRESETS: Dict[str, Dict] = {
    "600K": dict(d_model=96,  n_layers=4, n_heads=4, d_ff=384),
    "1M":   dict(d_model=128, n_layers=4, n_heads=4, d_ff=512),
    "3M":   dict(d_model=192, n_layers=6, n_heads=4, d_ff=768),
    "10M":  dict(d_model=288, n_layers=8, n_heads=8, d_ff=1152),
    "30M":  dict(d_model=384, n_layers=12, n_heads=8, d_ff=1536),
}


def build_ring_layout_model(
    capacity: str = "3M",
    condition_dim: int = 2,
    cfg_drop_prob: float = 0.1,
    dropout: float = 0.1,
    schedule: str = "cosine",
    **overrides,
) -> RingLayoutDiffusion:
    """Factory for the A1 model at a named capacity.

    Default '3M' (was '1M' in the earlier encoder) — A1 has more outputs in and
    benefits from extra capacity to fit ZINC's larger scaffold diversity.
    """
    if capacity not in CAPACITY_PRESETS:
        raise ValueError(
            f"Unknown capacity '{capacity}'. Options: "
            f"{list(CAPACITY_PRESETS.keys())}"
        )
    cfg = dict(CAPACITY_PRESETS[capacity])
    cfg.update(overrides)
    return RingLayoutDiffusion(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        condition_dim=condition_dim,
        cfg_drop_prob=cfg_drop_prob,
        dropout=dropout,
        schedule=schedule,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
