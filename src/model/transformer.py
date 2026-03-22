"""
Core Transformer building blocks shared by the Chord Predictor and Texture Generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.positional import RotaryPositionalEmbedding, apply_rotary_emb


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and KV-cache support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = None
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: Query input (batch, seq_len, embed_dim)
            kv: Key/Value source for cross-attention. If None, uses self-attention.
            kv_cache: (cached_keys, cached_values) from previous steps.
            position_offset: Position offset for RoPE (for KV-cache continuation).

        Returns:
            (output, new_kv_cache) — cache is None if not using causal attention.
        """
        B, T, D = x.shape
        kv_source = kv if kv is not None else x
        B_kv, S, _ = kv_source.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_source).view(B_kv, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(B_kv, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (only for self-attention)
        if self.rope is not None and kv is None:
            cos_q, sin_q = self.rope(q, offset=position_offset)
            cos_k, sin_k = self.rope(k, offset=0 if kv_cache is None else position_offset)
            q = apply_rotary_emb(q, cos_q, sin_q)
            k = apply_rotary_emb(k, cos_k, sin_k)

        # Append to KV cache
        new_cache = None
        if self.is_causal and kv is None:
            if kv_cache is not None:
                cached_k, cached_v = kv_cache
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.is_causal and kv_cache is None and kv is None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, new_cache


class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention, optional cross-attention, and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        has_cross_attention: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        activation: str = "gelu",
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, is_causal, use_rope, max_seq_len
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention (optional)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = MultiHeadAttention(
                embed_dim, num_heads, dropout, is_causal=False, use_rope=False
            )
            self.norm_cross = nn.LayerNorm(embed_dim)

        # FFN
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        cross_kv: torch.Tensor | None = None,
        self_attn_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            cross_kv: Cross-attention source (batch, src_len, embed_dim)
            self_attn_cache: KV-cache from previous step
            position_offset: For RoPE with KV-cache

        Returns:
            (output, new_self_attn_cache)
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        attn_out, new_cache = self.self_attn(
            x, kv_cache=self_attn_cache, position_offset=position_offset
        )
        x = residual + attn_out

        # Cross-attention
        if self.has_cross_attention and cross_kv is not None:
            residual = x
            x = self.norm_cross(x)
            cross_out, _ = self.cross_attn(x, kv=cross_kv)
            x = residual + cross_out

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x, new_cache
