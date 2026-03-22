"""
Positional encoding modules for the Transformer models.

Uses Rotary Position Embedding (RoPE) for relative position awareness,
which is critical for streaming inference where absolute position is meaningless.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al.

    Encodes relative positions by rotating query/key vectors in pairs.
    This gives the attention mechanism a sense of "how far apart" two tokens are
    without relying on absolute position indices.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # Duplicate for both sin and cos parts
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for positions [offset, offset + seq_len).

        Args:
            x: Input tensor, shape (..., seq_len, dim). Only used for seq_len/device.
            offset: Starting position (for KV-cache continuation).

        Returns:
            (cos, sin) each of shape (seq_len, dim)
        """
        seq_len = x.shape[-2]
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: (batch, heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        Tensor with rotary position encoding applied.
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into pairs and rotate
    x_rot = _rotate_half(x)
    return (x * cos) + (x_rot * sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (fallback).

    Used when absolute positions are acceptable (e.g., fixed-length inputs).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, dim)
            offset: Starting position for streaming.

        Returns:
            x + positional encoding
        """
        seq_len = x.shape[1]
        return self.dropout(x + self.pe[:, offset : offset + seq_len])
