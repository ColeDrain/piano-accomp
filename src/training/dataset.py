"""
PyTorch Datasets for training the Chord Predictor and Texture Generator.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer.vocab import (
    Vocabulary, PITCH_NAMES,
    quality_to_triad, quality_to_seventh,
)


def _decompose_chord(chord_vocab_id: int, vocab: Vocabulary) -> dict[str, int]:
    """Decompose a chord vocab token ID into root, triad, seventh, bass indices."""
    if not vocab.is_chord(chord_vocab_id):
        return {"root": 0, "triad": 0, "seventh": 0, "bass": 0}

    root_str, quality_str = vocab.decode_chord(chord_vocab_id)
    root_idx = PITCH_NAMES.index(root_str) if root_str in PITCH_NAMES else 0
    triad_idx = quality_to_triad(quality_str)
    seventh_idx = quality_to_seventh(quality_str)
    bass_idx = root_idx  # Default bass = root (root position)

    return {
        "root": root_idx,
        "triad": triad_idx,
        "seventh": seventh_idx,
        "bass": bass_idx,
    }


class ChordDataset(Dataset):
    """Dataset for training the decomposed Chord Predictor.

    Each sample: (melody_tokens, root, triad, seventh, bass)
    """

    def __init__(
        self,
        data_path: str | Path,
        max_melody_len: int = 32,
        chord_offset: int = 0,  # Kept for API compat
    ):
        self.data = torch.load(data_path, weights_only=False)
        self.max_melody_len = max_melody_len
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        melody = item["melody_tokens"][:self.max_melody_len]

        components = _decompose_chord(item["chord_label"], self.vocab)

        return {
            "melody_tokens": melody,
            "root_label": torch.tensor(components["root"], dtype=torch.long),
            "triad_label": torch.tensor(components["triad"], dtype=torch.long),
            "seventh_label": torch.tensor(components["seventh"], dtype=torch.long),
            "bass_label": torch.tensor(components["bass"], dtype=torch.long),
        }


class TextureDataset(Dataset):
    """Dataset for training the Texture Generator."""

    def __init__(
        self,
        data_path: str | Path,
        max_melody_len: int = 32,
        max_accomp_len: int = 256,
        chord_offset: int = 0,
    ):
        self.data = torch.load(data_path, weights_only=False)
        self.max_melody_len = max_melody_len
        self.max_accomp_len = max_accomp_len
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        melody = item["melody_tokens"][:self.max_melody_len]
        accomp = item["accomp_tokens"][:self.max_accomp_len]

        accomp_input = accomp[:-1]
        accomp_target = accomp[1:]

        components = _decompose_chord(item["chord_label"], self.vocab)

        return {
            "melody_tokens": melody,
            "root_label": torch.tensor(components["root"], dtype=torch.long),
            "triad_label": torch.tensor(components["triad"], dtype=torch.long),
            "seventh_label": torch.tensor(components["seventh"], dtype=torch.long),
            "bass_label": torch.tensor(components["bass"], dtype=torch.long),
            "accomp_input": accomp_input,
            "accomp_target": accomp_target,
        }


def collate_chord(batch: list[dict], pad_id: int = 0) -> dict:
    melody = pad_sequence(
        [b["melody_tokens"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    return {
        "melody_tokens": melody,
        "root_label": torch.stack([b["root_label"] for b in batch]),
        "triad_label": torch.stack([b["triad_label"] for b in batch]),
        "seventh_label": torch.stack([b["seventh_label"] for b in batch]),
        "bass_label": torch.stack([b["bass_label"] for b in batch]),
    }


def collate_texture(batch: list[dict], pad_id: int = 0) -> dict:
    melody = pad_sequence(
        [b["melody_tokens"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    accomp_input = pad_sequence(
        [b["accomp_input"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    accomp_target = pad_sequence(
        [b["accomp_target"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )

    return {
        "melody_tokens": melody,
        "root_label": torch.stack([b["root_label"] for b in batch]),
        "triad_label": torch.stack([b["triad_label"] for b in batch]),
        "seventh_label": torch.stack([b["seventh_label"] for b in batch]),
        "bass_label": torch.stack([b["bass_label"] for b in batch]),
        "accomp_input": accomp_input,
        "accomp_target": accomp_target,
    }
