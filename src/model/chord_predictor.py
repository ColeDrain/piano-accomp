"""
Stage 1: Chord Predictor

Encoder-only Transformer that predicts the current chord symbol from a window
of recent melody tokens. Runs once per beat/chord change (~200-500ms intervals).

Input: Last N melody tokens (N=32, ~2-4 bars)
Output: Chord class (softmax over ~200 gospel-extended chord types)

Architecture: ~4M parameters
"""

import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock


class ChordPredictor(nn.Module):
    """Predicts chord symbols from melody context."""

    def __init__(
        self,
        vocab_size: int = 500,
        num_chord_classes: int = 200,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_melody_tokens: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_melody_tokens = max_melody_tokens

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers (causal-masked for streaming compatibility)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                is_causal=True,  # Causal so it works in streaming mode
                has_cross_attention=False,
                use_rope=True,
                max_seq_len=max_melody_tokens,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification head: last token's representation -> chord class
        self.chord_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_chord_classes),
        )

    def forward(
        self,
        melody_tokens: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            melody_tokens: (batch, seq_len) token IDs, seq_len <= max_melody_tokens
            return_embedding: If True, also return the melody context embedding
                (used as cross-attention input for the texture generator)

        Returns:
            chord_logits: (batch, num_chord_classes)
            melody_embedding: (batch, seq_len, embed_dim) — only if return_embedding=True
        """
        x = self.token_embed(melody_tokens)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.final_norm(x)

        # Use the last token's representation for chord classification
        last_hidden = x[:, -1, :]  # (batch, embed_dim)
        chord_logits = self.chord_head(last_hidden)

        if return_embedding:
            return chord_logits, x

        return chord_logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
