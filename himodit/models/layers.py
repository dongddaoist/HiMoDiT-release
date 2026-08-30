"""
Shared neural building blocks for all HiMoDiT stages.
=====================================================

These are vocabulary-agnostic primitives. Every stage (A1, A2, A3,
Terminal) builds its tower out of them, so they live in one place
rather than being redefined per stage.

  SinusoidalTimeEmbed          diffusion-timestep embedding
  AdaLN                        DiT-style adaptive LayerNorm (zero-init)
  DiTBlock                     AdaLN -> MHA -> AdaLN -> FFN, both residual
  EdgeBiasedMultiheadAttention attention with additive per-head bond bias

Parameter names are unchanged from the original per-stage definitions,
so `state_dict` keys of models built before this refactor still load.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Timestep embedding ────────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ─── Adaptive LayerNorm ────────────────────────────────────────────────

class AdaLN(nn.Module):
    """Adaptive LayerNorm: gamma, beta predicted from a conditioning vector.

    Output = LN(x) * (1 + gamma) + beta, where (gamma, beta) =
    Linear(SiLU(cond)). Following DiT, gamma and beta are zero-initialised
    so each block starts as the identity.
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model), cond: (B, d_cond)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ─── DiT block (pure self-attention) ───────────────────────────────────

class DiTBlock(nn.Module):
    """DiT block: AdaLN -> MHA -> AdaLN -> FFN, both with residuals.

    No edge bias on attention. A1 and A3 attend over an abstract token
    sequence whose pairwise structure is itself the prediction target,
    not a fixed input graph to bias on, so plain self-attention is the
    right primitive there. Stages that DO have a known input graph
    (A2, Terminal) use EdgeBiasedMultiheadAttention instead.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adaln1 = AdaLN(d_model, d_cond)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.adaln2 = AdaLN(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        h = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ─── Edge-biased attention ─────────────────────────────────────────────

class EdgeBiasedMultiheadAttention(nn.Module):
    """Multihead attention with an additive, learned bond-class bias.

    The score between tokens i and j is

        attn_score(i, j) = (Q_i . K_j) / sqrt(d_head) + bias_h(bond_ij)
        bias_h(bond_ij)  = sum_c softmax(bond_ij / tau)[c] * w_h[c]

    where w_h is a learnable per-head vector over bond classes. This lets
    a head route information preferentially along, say, aromatic bonds,
    which per-edge independent logits cannot do.

    Implemented with explicit einsums rather than fused attention: the
    token counts here are small (<= M_MAX = 40), so readability wins over
    kernel efficiency.

    Forward signature matches nn.MultiheadAttention except for:
      edge_probs        (B, T_q, T_k, n_bond_classes) soft bond distribution
                        between query and key positions. Required when
                        bias_enabled=True, ignored otherwise.
      key_padding_mask  (B, T_k) bool, True at positions to IGNORE
                        (PyTorch convention).

    Works for self-attention (T_q == T_k) and cross-attention (T_q != T_k).

    Parameters
    ----------
    embed_dim        total d_model
    num_heads        number of attention heads
    dropout          dropout probability on attention weights
    bias_enabled     if False, reproduces standard multihead attention
                     exactly (used for the edge-attention ablation)
    n_bond_classes   5 here: none, single, aromatic, double, triple
    bias_temperature tau for the softmax over bond probabilities before
                     the bias projection. Lower = sharper commitment to
                     the current bond state. Default 1.0 (no sharpening).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0,
                 bias_enabled=True, n_bond_classes=5,
                 bias_temperature=1.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by "
                f"num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.dropout = dropout
        self.bias_enabled = bias_enabled
        self.n_bond_classes = n_bond_classes
        self.bias_temperature = bias_temperature

        # Q/K/V kept as separate projections (rather than one fused matrix)
        # so cross-attention with different query and key sources stays
        # readable.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if bias_enabled:
            # Zero-init so the block starts as standard attention and each
            # head has to actively learn whether edge biasing helps it.
            self.bond_bias = nn.Parameter(
                torch.zeros(num_heads, n_bond_classes)
            )
        else:
            self.register_parameter("bond_bias", None)

    def forward(self, query, key, value, edge_probs=None,
                key_padding_mask=None):
        """
        Parameters
        ----------
        query, key, value : (B, T_q / T_k / T_k, embed_dim)
        edge_probs        : (B, T_q, T_k, n_bond_classes), or None when
                            bias_enabled=False
        key_padding_mask  : (B, T_k) bool, True at positions to mask out

        Returns
        -------
        (B, T_q, embed_dim)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H = self.num_heads
        D = self.d_head

        Q = self.q_proj(query).reshape(B, T_q, H, D).transpose(1, 2)
        K = self.k_proj(key).reshape(B, T_k, H, D).transpose(1, 2)
        V = self.v_proj(value).reshape(B, T_k, H, D).transpose(1, 2)

        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale

        if self.bias_enabled:
            if edge_probs is None:
                raise ValueError(
                    "bias_enabled=True but edge_probs is None. Pass bond "
                    "probabilities between query/key positions."
                )
            # edge_probs may be a raw one-hot-with-noise tensor, so softmax
            # with temperature to get well-defined probabilities.
            bond_probs = F.softmax(
                edge_probs / self.bias_temperature, dim=-1
            )                                        # (B, T_q, T_k, C)
            bias = torch.einsum(
                "bqkc,hc->bhqk", bond_probs, self.bond_bias
            )                                        # (B, H, T_q, T_k)
            scores = scores + bias

        if key_padding_mask is not None:
            # True = ignore, so drive those logits to -inf pre-softmax.
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                float("-inf"),
            )

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        # A query row that has no valid keys softmaxes to all-NaN. Those are
        # padding positions that get masked at the loss, so zero them out
        # rather than letting NaN propagate into the whole batch.
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.einsum("bhqk,bhkd->bhqd", attn, V)
        out = out.transpose(1, 2).contiguous().reshape(B, T_q, self.embed_dim)
        return self.out_proj(out)
