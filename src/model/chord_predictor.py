"""
Stage 1: Chord Predictor (Decomposed)

Predicts chords by decomposing into two easier sub-problems:
  1. Root prediction: 12 classes (C, Db, D, ..., B) — easy
  2. Quality prediction: 7 classes (maj, min, 7, maj7, min7, dim, sus) — easy

This follows the ChordFormer (2025) approach of structured prediction.
Combined, this gives 12 × 7 = 84 possible chords — enough for accompaniment,
and each head achieves much higher accuracy than a single 360-class head.

Architecture: ~4M parameters
"""

import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock

# Reduced quality vocabulary — group similar chords
QUALITY_GROUPS = {
    # Group 0: Major family
    "maj": 0, "maj9": 0, "add9": 0, "6": 0, "6/9": 0,
    # Group 1: Minor family
    "min": 1, "min9": 1, "min6": 1,
    # Group 2: Dominant 7 family
    "7": 2, "9": 2, "13": 2, "7#9": 2, "7b9": 2, "7#11": 2, "aug7": 2,
    # Group 3: Major 7 family
    "maj7": 3, "maj9": 3, "maj7#11": 3, "add11": 3,
    # Group 4: Minor 7 family
    "min7": 4, "min7b5": 4, "min11": 4, "min7add11": 4,
    # Group 5: Diminished family
    "dim": 5, "dim7": 5,
    # Group 6: Suspended / other
    "sus2": 6, "sus4": 6, "9sus4": 6, "13sus4": 6, "aug": 6,
}

NUM_ROOTS = 12
NUM_QUALITIES = 7
QUALITY_NAMES = ["maj", "min", "dom7", "maj7", "min7", "dim", "sus"]


class ChordPredictor(nn.Module):
    """Predicts chord root and quality separately from melody context."""

    def __init__(
        self,
        vocab_size: int = 564,
        num_chord_classes: int = 360,  # Kept for backward compat, not used
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_melody_tokens: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_melody_tokens = max_melody_tokens
        self.num_roots = NUM_ROOTS
        self.num_qualities = NUM_QUALITIES

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

        # Two separate heads — much easier than one 360-class head
        self.root_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_ROOTS),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_QUALITIES),
        )

    def forward(
        self,
        melody_tokens: torch.Tensor,
        return_embedding: bool = False,
    ) -> dict | tuple[dict, torch.Tensor]:
        """
        Args:
            melody_tokens: (batch, seq_len) token IDs
            return_embedding: If True, also return melody context embedding

        Returns:
            dict with "root_logits" (batch, 12) and "quality_logits" (batch, 7)
            If return_embedding: (dict, melody_embedding)
        """
        x = self.token_embed(melody_tokens)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.final_norm(x)

        last_hidden = x[:, -1, :]  # (batch, embed_dim)

        result = {
            "root_logits": self.root_head(last_hidden),      # (batch, 12)
            "quality_logits": self.quality_head(last_hidden), # (batch, 7)
        }

        if return_embedding:
            return result, x

        return result

    def predict_chord(self, melody_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: predict root and quality indices.

        Returns:
            (root_ids, quality_ids) each (batch,)
        """
        result = self.forward(melody_tokens)
        root_ids = result["root_logits"].argmax(dim=-1)
        quality_ids = result["quality_logits"].argmax(dim=-1)
        return root_ids, quality_ids

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
