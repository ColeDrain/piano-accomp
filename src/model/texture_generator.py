"""
Stage 2: Texture Generator (Structured Chord Conditioning)

Decoder-only Transformer with cross-attention to melody context.
Generates piano accompaniment tokens autoregressively.

Chord conditioning uses structured embeddings (per TSHE 2025):
  - Separate embeddings for root, triad, seventh, bass
  - Concatenated and projected (not summed — preserves discriminability
    between gospel chord qualities like 7 vs 7#9)

Architecture: ~34M parameters, KV-cache for streaming inference.
"""

import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock
from src.model.chord_predictor import NUM_ROOTS, NUM_TRIADS, NUM_SEVENTHS, NUM_BASS


class TextureGenerator(nn.Module):
    """Generates piano accompaniment conditioned on structured chord + melody."""

    def __init__(
        self,
        vocab_size: int = 580,
        num_chord_classes: int = 84,  # Unused, kept for config compat
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 256,
        melody_context_dim: int = 256,
        chord_embed_dim: int = 64,
        style_embed_dim: int = 64,
        num_styles: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.chord_embed_dim = chord_embed_dim

        # Token embedding for accompaniment sequence
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Structured chord embeddings — one per component
        # Each component gets chord_embed_dim // 4 dimensions
        comp_dim = chord_embed_dim // 4
        self.root_embed = nn.Embedding(NUM_ROOTS, comp_dim)
        self.triad_embed = nn.Embedding(NUM_TRIADS, comp_dim)
        self.seventh_embed = nn.Embedding(NUM_SEVENTHS, comp_dim)
        self.bass_embed = nn.Embedding(NUM_BASS, comp_dim)

        # Project concatenated components to chord_embed_dim
        self.chord_proj = nn.Linear(comp_dim * 4, chord_embed_dim)

        # Style conditioning
        self.style_embed = nn.Embedding(num_styles, style_embed_dim)

        # Project melody context + chord + style to cross-attention dimension
        cross_input_dim = melody_context_dim + chord_embed_dim + style_embed_dim
        self.cross_proj = nn.Linear(cross_input_dim, embed_dim)

        # Decoder layers with cross-attention
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                is_causal=True,
                has_cross_attention=True,
                use_rope=True,
                max_seq_len=max_seq_len,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # Tie weights between token embedding and output projection
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        accomp_tokens: torch.Tensor,
        melody_context: torch.Tensor,
        chord_components: dict[str, torch.Tensor],
        style_ids: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            accomp_tokens: (batch, seq_len) previous accompaniment token IDs
            melody_context: (batch, melody_len, melody_dim) from chord predictor
            chord_components: dict with "root" (B,), "triad" (B,), "seventh" (B,), "bass" (B,)
            style_ids: (batch,) optional style label IDs
            kv_caches: List of (cached_k, cached_v) per layer
            position_offset: Current position for RoPE

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: Updated caches
        """
        x = self.token_embed(accomp_tokens)
        x = self.embed_dropout(x)

        # Build cross-attention source with structured chord embedding
        cross_kv = self._build_cross_input(melody_context, chord_components, style_ids)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x,
                cross_kv=cross_kv,
                self_attn_cache=layer_cache,
                position_offset=position_offset,
            )
            new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_caches

    def _build_cross_input(
        self,
        melody_context: torch.Tensor,
        chord_components: dict[str, torch.Tensor],
        style_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build cross-attention source with structured chord embedding.

        Instead of one flat chord embedding, we embed each component
        separately and concatenate + project. This gives the model
        compositional understanding of chord structure.
        """
        B, T_mel, _ = melody_context.shape

        # Structured chord embedding: concat 4 component embeddings
        root_emb = self.root_embed(chord_components["root"])       # (B, comp_dim)
        triad_emb = self.triad_embed(chord_components["triad"])    # (B, comp_dim)
        seventh_emb = self.seventh_embed(chord_components["seventh"])  # (B, comp_dim)
        bass_emb = self.bass_embed(chord_components["bass"])       # (B, comp_dim)

        chord_emb = self.chord_proj(
            torch.cat([root_emb, triad_emb, seventh_emb, bass_emb], dim=-1)
        )  # (B, chord_embed_dim)

        # Broadcast across melody positions
        chord_emb = chord_emb.unsqueeze(1).expand(-1, T_mel, -1)

        # Style embedding
        if style_ids is not None:
            style_emb = self.style_embed(style_ids).unsqueeze(1).expand(-1, T_mel, -1)
        else:
            style_emb = torch.zeros(
                B, T_mel, self.style_embed.embedding_dim,
                device=melody_context.device
            )

        cross_input = torch.cat([melody_context, chord_emb, style_emb], dim=-1)
        return self.cross_proj(cross_input)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        melody_context: torch.Tensor,
        chord_components: dict[str, torch.Tensor],
        style_ids: torch.Tensor | None = None,
        max_tokens: int = 30,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        bos_id: int = 1,
        eos_id: int = 2,
        allowed_token_mask: torch.Tensor | None = None,
    ) -> list[int]:
        """Generate accompaniment tokens autoregressively."""
        device = melody_context.device
        generated = [bos_id]
        kv_caches = None
        position = 0

        for _ in range(max_tokens):
            input_ids = torch.tensor([[generated[-1]]], device=device)

            logits, kv_caches = self.forward(
                accomp_tokens=input_ids,
                melody_context=melody_context,
                chord_components=chord_components,
                style_ids=style_ids,
                kv_caches=kv_caches,
                position_offset=position,
            )

            next_logits = logits[0, -1, :] / temperature

            if allowed_token_mask is not None:
                next_logits[~allowed_token_mask] = float("-inf")

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break

            generated.append(next_token)
            position += 1

        return generated[1:]
