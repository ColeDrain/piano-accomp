"""
Stage 1: Chord Predictor (Structured Decomposition)

Predicts chords by decomposing into four sub-problems:
  1. Root: 12 classes (C, Db, D, ..., B)
  2. Triad: 5 classes (maj, min, dim, aug, sus)
  3. Seventh: 4 classes (none, dom7, maj7, min7)
  4. Bass: 12 classes (for inversions/slash chords, optional)

Follows ChordFormer (2025) and Jiang et al. (ISMIR 2019) structured prediction.
Each head is easy (80%+ accuracy) vs a single 360-class head (18% accuracy).

Includes learnable bigram transition bias for temporal smoothing
(per ERLD-HC, MDPI 2025).

Architecture: ~5M parameters
"""

import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock
from src.tokenizer.vocab import NUM_POSITIONS

NUM_ROOTS = 12
NUM_TRIADS = 5
NUM_SEVENTHS = 4
NUM_BASS = 12


class ChordPredictor(nn.Module):
    """Predicts chord root, triad, seventh, and bass from melody context."""

    def __init__(
        self,
        vocab_size: int = 580,
        num_chord_classes: int = 360,  # Unused, kept for config compat
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_melody_tokens: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
        bass_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_melody_tokens = max_melody_tokens
        self.num_roots = NUM_ROOTS
        self.num_triads = NUM_TRIADS
        self.num_sevenths = NUM_SEVENTHS
        self.num_bass = NUM_BASS
        self.bass_loss_weight = bass_loss_weight

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                is_causal=True,
                has_cross_attention=False,
                use_rope=True,
                max_seq_len=max_melody_tokens,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Four classification heads
        self.root_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_ROOTS),
        )

        self.triad_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_TRIADS),
        )

        self.seventh_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_SEVENTHS),
        )

        self.bass_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_BASS),
        )

        # Bigram transition bias for root (learned, initialized to uniform)
        # transition_bias[prev_root] gives logit bias for current root prediction
        self.root_transition = nn.Parameter(torch.zeros(NUM_ROOTS, NUM_ROOTS))

    def forward(
        self,
        melody_tokens: torch.Tensor,
        prev_root: torch.Tensor | None = None,
        return_embedding: bool = False,
    ) -> dict | tuple[dict, torch.Tensor]:
        """
        Args:
            melody_tokens: (batch, seq_len) token IDs
            prev_root: (batch,) previous root index for transition bias. None = no bias.
            return_embedding: If True, also return melody context embedding

        Returns:
            dict with root_logits (B,12), triad_logits (B,5), seventh_logits (B,4), bass_logits (B,12)
        """
        x = self.token_embed(melody_tokens)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.final_norm(x)

        last_hidden = x[:, -1, :]

        root_logits = self.root_head(last_hidden)

        # Apply transition bias if previous root is known
        if prev_root is not None:
            bias = self.root_transition[prev_root]  # (batch, NUM_ROOTS)
            root_logits = root_logits + bias

        result = {
            "root_logits": root_logits,               # (batch, 12)
            "triad_logits": self.triad_head(last_hidden),    # (batch, 5)
            "seventh_logits": self.seventh_head(last_hidden), # (batch, 4)
            "bass_logits": self.bass_head(last_hidden),       # (batch, 12)
        }

        if return_embedding:
            return result, x

        return result

    def predict_chord(self, melody_tokens: torch.Tensor, prev_root: torch.Tensor | None = None) -> dict:
        """Convenience: predict all component indices.

        Returns:
            dict with root (B,), triad (B,), seventh (B,), bass (B,)
        """
        result = self.forward(melody_tokens, prev_root)
        return {
            "root": result["root_logits"].argmax(dim=-1),
            "triad": result["triad_logits"].argmax(dim=-1),
            "seventh": result["seventh_logits"].argmax(dim=-1),
            "bass": result["bass_logits"].argmax(dim=-1),
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
