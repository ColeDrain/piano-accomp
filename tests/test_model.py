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
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        result = model(batch)

        assert result["root_logits"].shape == (4, 12)
        assert result["triad_logits"].shape == (4, 5)
        assert result["seventh_logits"].shape == (4, 4)
        assert result["bass_logits"].shape == (4, 12)

    def test_forward_with_embedding(self, vocab, device):
        model = ChordPredictor(
            vocab_size=vocab.size,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        result, embedding = model(batch, return_embedding=True)

        assert result["root_logits"].shape == (4, 12)
        assert embedding.shape == (4, 32, 64)

    def test_transition_bias(self, vocab, device):
        model = ChordPredictor(
            vocab_size=vocab.size,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        prev_root = torch.randint(0, 12, (4,), device=device)

        result_no_bias = model(batch)
        result_bias = model(batch, prev_root=prev_root)

        # With transition bias, root logits should differ
        assert not torch.equal(result_no_bias["root_logits"], result_bias["root_logits"])

    def test_predict_chord(self, vocab, device):
        model = ChordPredictor(
            vocab_size=vocab.size,
            embed_dim=64, num_layers=2, num_heads=4, ffn_dim=128,
        ).to(device)

        batch = torch.randint(0, vocab.size, (4, 32), device=device)
        pred = model.predict_chord(batch)

        assert pred["root"].shape == (4,)
        assert pred["triad"].shape == (4,)
        assert pred["seventh"].shape == (4,)
        assert pred["bass"].shape == (4,)


class TestTextureGenerator:
    def test_instantiation(self, vocab):
        model = TextureGenerator(vocab_size=vocab.size)
        assert model.get_num_params() > 0

    def test_forward_shape(self, vocab, device):
        model = TextureGenerator(
            vocab_size=vocab.size,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            melody_context_dim=64,
            chord_embed_dim=64,
            style_embed_dim=32,
        ).to(device)

        accomp = torch.randint(0, vocab.size, (4, 16), device=device)
        melody_ctx = torch.randn(4, 32, 64, device=device)
        chord_components = {
            "root": torch.randint(0, 12, (4,), device=device),
            "triad": torch.randint(0, 5, (4,), device=device),
            "seventh": torch.randint(0, 4, (4,), device=device),
            "bass": torch.randint(0, 12, (4,), device=device),
        }

        logits, caches = model(accomp, melody_ctx, chord_components)
        assert logits.shape == (4, 16, vocab.size)
        assert len(caches) == 2  # num_layers

    def test_generate(self, vocab, device):
        model = TextureGenerator(
            vocab_size=vocab.size,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            melody_context_dim=64,
            chord_embed_dim=64,
            style_embed_dim=32,
        ).to(device)
        model.eval()

        melody_ctx = torch.randn(1, 8, 64, device=device)
        chord_components = {
            "root": torch.tensor([0], device=device),
            "triad": torch.tensor([0], device=device),
            "seventh": torch.tensor([1], device=device),
            "bass": torch.tensor([0], device=device),
        }

        tokens = model.generate(
            melody_context=melody_ctx,
            chord_components=chord_components,
            max_tokens=10,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
        )
        assert isinstance(tokens, list)
        assert len(tokens) <= 10
