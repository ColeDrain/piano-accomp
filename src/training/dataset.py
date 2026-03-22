"""
PyTorch Datasets for training the Chord Predictor and Texture Generator.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ChordDataset(Dataset):
    """Dataset for training the Chord Predictor.

    Each sample: (melody_tokens, chord_label)
    """

    def __init__(
        self,
        data_path: str | Path,
        max_melody_len: int = 32,
        chord_offset: int = 0,  # Offset to convert vocab chord ID to class index
    ):
        self.data = torch.load(data_path, weights_only=False)
        self.max_melody_len = max_melody_len
        self.chord_offset = chord_offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        melody = item["melody_tokens"][:self.max_melody_len]

        # Convert chord vocab ID to class index (0-based)
        chord_label = item["chord_label"] - self.chord_offset

        return {
            "melody_tokens": melody,
            "chord_label": torch.tensor(chord_label, dtype=torch.long),
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
    chord = torch.stack([b["chord_label"] for b in batch])
    return {"melody_tokens": melody, "chord_label": chord}


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
