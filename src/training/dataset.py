"""
PyTorch Datasets for training the Chord Predictor and Texture Generator.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer.vocab import Vocabulary, PITCH_NAMES, CHORD_QUALITIES
from src.model.chord_predictor import QUALITY_GROUPS


def _chord_id_to_root_quality(chord_vocab_id: int, vocab: Vocabulary) -> tuple[int, int]:
    """Convert a chord vocab token ID to (root_index, quality_group_index).

    Returns:
        (root: 0-11, quality_group: 0-6)
    """
    if not vocab.is_chord(chord_vocab_id):
        return 0, 0  # Default to C major

    root_str, quality_str = vocab.decode_chord(chord_vocab_id)
    root_idx = PITCH_NAMES.index(root_str) if root_str in PITCH_NAMES else 0
    quality_idx = QUALITY_GROUPS.get(quality_str, 0)  # Default to maj group
    return root_idx, quality_idx


class ChordDataset(Dataset):
    """Dataset for training the decomposed Chord Predictor.

    Each sample: (melody_tokens, root_label, quality_label)
    """

    def __init__(
        self,
        data_path: str | Path,
        max_melody_len: int = 32,
        chord_offset: int = 0,  # Kept for API compat but we use vocab directly now
    ):
        self.data = torch.load(data_path, weights_only=False)
        self.max_melody_len = max_melody_len
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        melody = item["melody_tokens"][:self.max_melody_len]

        # Decompose chord vocab ID into root (0-11) and quality group (0-6)
        root_idx, quality_idx = _chord_id_to_root_quality(
            item["chord_label"], self.vocab
        )

        return {
            "melody_tokens": melody,
            "root_label": torch.tensor(root_idx, dtype=torch.long),
            "quality_label": torch.tensor(quality_idx, dtype=torch.long),
            # Keep combined label for texture generator compatibility
            "chord_label": torch.tensor(item["chord_label"], dtype=torch.long),
        }


class TextureDataset(Dataset):
    """Dataset for training the Texture Generator.

    Each sample: (melody_tokens, chord_label, accomp_input, accomp_target)
    where accomp_target is accomp_input shifted right by 1 (teacher forcing).
    """

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
        self.chord_offset = chord_offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        melody = item["melody_tokens"][:self.max_melody_len]
        accomp = item["accomp_tokens"][:self.max_accomp_len]

        # Teacher forcing: input is all tokens except last, target is all except first
        accomp_input = accomp[:-1]
        accomp_target = accomp[1:]

        chord_label = item["chord_label"] - self.chord_offset

        return {
            "melody_tokens": melody,
            "chord_label": torch.tensor(chord_label, dtype=torch.long),
            "accomp_input": accomp_input,
            "accomp_target": accomp_target,
        }


def collate_chord(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate function for ChordDataset — pads melody to max length in batch."""
    melody = pad_sequence(
        [b["melody_tokens"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    root = torch.stack([b["root_label"] for b in batch])
    quality = torch.stack([b["quality_label"] for b in batch])
    chord = torch.stack([b["chord_label"] for b in batch])

    return {
        "melody_tokens": melody,
        "root_label": root,
        "quality_label": quality,
        "chord_label": chord,
    }


def collate_texture(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate function for TextureDataset — pads both melody and accompaniment."""
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
    chord = torch.stack([b["chord_label"] for b in batch])

    return {
        "melody_tokens": melody,
        "chord_label": chord,
        "accomp_input": accomp_input,
        "accomp_target": accomp_target,
    }
