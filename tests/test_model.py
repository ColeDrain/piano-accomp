"""Tests for model instantiation and forward pass shapes."""

import pytest
import torch

from src.tokenizer.vocab import Vocabulary
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator


@pytest.fixture
def vocab():
    return Vocabulary()


@pytest.fixture
def device():
    return torch.device("cpu")


class TestChordPredictor:
    def test_instantiation(self, vocab):
        model = ChordPredictor(vocab_size=vocab.size)
        assert model.get_num_params() > 0

    def test_forward_shape(self, vocab, device):
        model = ChordPredictor(
            vocab_size=vocab.size,
            num_chord_classes=200,
            embed_dim=64,  # Small for testing
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        logits = model(batch)
        assert logits.shape == (4, 200)

    def test_forward_with_embedding(self, vocab, device):
        model = ChordPredictor(
            vocab_size=vocab.size,
            num_chord_classes=200,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        logits, embedding = model(batch, return_embedding=True)
        assert logits.shape == (4, 200)
        assert embedding.shape == (4, 32, 64)


class TestTextureGenerator:
    def test_instantiation(self, vocab):
        model = TextureGenerator(vocab_size=vocab.size)
        assert model.get_num_params() > 0

    def test_forward_shape(self, vocab, device):
        model = TextureGenerator(
            vocab_size=vocab.size,
            num_chord_classes=200,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            melody_context_dim=64,
            chord_embed_dim=32,
            style_embed_dim=32,
        ).to(device)

        accomp = torch.randint(0, vocab.size, (4, 16), device=device)
        melody_ctx = torch.randn(4, 32, 64, device=device)
        chord_ids = torch.randint(0, 200, (4,), device=device)

        logits, caches = model(accomp, melody_ctx, chord_ids)
        assert logits.shape == (4, 16, vocab.size)
        assert len(caches) == 2  # num_layers

    def test_generate(self, vocab, device):
        model = TextureGenerator(
            vocab_size=vocab.size,
            num_chord_classes=200,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            melody_context_dim=64,
            chord_embed_dim=32,
            style_embed_dim=32,
        ).to(device)
        model.eval()

        melody_ctx = torch.randn(1, 8, 64, device=device)
        chord_ids = torch.tensor([5], device=device)

        tokens = model.generate(
            melody_context=melody_ctx,
            chord_ids=chord_ids,
            max_tokens=10,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
        )
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
