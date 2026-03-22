"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordPredictionLoss(nn.Module):
    """Cross-entropy loss with label smoothing for chord classification."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,) class indices
        """
        return self.loss_fn(logits, targets)


class TextureGenerationLoss(nn.Module):
    """Cross-entropy loss for next-token prediction, ignoring padding."""

    def __init__(self, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len) token IDs
        """
        # Reshape for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        B, T, V = logits.shape
        return self.loss_fn(logits.reshape(B * T, V), targets.reshape(B * T))
