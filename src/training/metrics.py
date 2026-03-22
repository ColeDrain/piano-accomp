"""
Evaluation metrics for chord prediction and texture generation.
"""

import torch


def chord_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """Compute top-k accuracy for chord prediction.

    Args:
        logits: (batch, num_classes)
        targets: (batch,)
        top_k: Number of top predictions to consider

    Returns:
        Accuracy as a float [0, 1]
    """
    _, top_indices = logits.topk(top_k, dim=-1)  # (batch, top_k)
    correct = (top_indices == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def token_perplexity(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> float:
    """Compute perplexity for next-token prediction, ignoring padding.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        pad_id: Padding token to ignore

    Returns:
        Perplexity (lower is better)
    """
    B, T, V = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather the log-prob of the target token at each position
    target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Mask out padding positions
    mask = (targets != pad_id).float()
    num_tokens = mask.sum()

    if num_tokens == 0:
        return float("inf")

    avg_neg_log_prob = -(target_log_probs * mask).sum() / num_tokens
    return torch.exp(avg_neg_log_prob).item()


def harmonic_compatibility(
    generated_pitches: list[int],
    chord_root: int,
    chord_quality: str,
) -> float:
    """Check what fraction of generated pitches belong to the current chord.

    Args:
        generated_pitches: List of MIDI pitch values
        chord_root: MIDI pitch of chord root (mod 12)
        chord_quality: Chord quality string

    Returns:
        Fraction of pitches that are chord tones [0, 1]
    """
    if not generated_pitches:
        return 1.0

    # Define chord tones as semitone intervals from root
    quality_intervals = {
        "maj": {0, 4, 7},
        "min": {0, 3, 7},
        "dim": {0, 3, 6},
        "aug": {0, 4, 8},
        "7": {0, 4, 7, 10},
        "maj7": {0, 4, 7, 11},
        "min7": {0, 3, 7, 10},
        "dim7": {0, 3, 6, 9},
        "min7b5": {0, 3, 6, 10},
        "9": {0, 2, 4, 7, 10},
        "maj9": {0, 2, 4, 7, 11},
        "min9": {0, 2, 3, 7, 10},
        "sus4": {0, 5, 7},
        "sus2": {0, 2, 7},
        "6": {0, 4, 7, 9},
        "min6": {0, 3, 7, 9},
        "13": {0, 2, 4, 7, 9, 10},
        "13sus4": {0, 2, 5, 7, 9, 10},
    }

    intervals = quality_intervals.get(chord_quality, {0, 4, 7})  # Default to major
    chord_pcs = {(chord_root + i) % 12 for i in intervals}

    num_compatible = sum(1 for p in generated_pitches if p % 12 in chord_pcs)
    return num_compatible / len(generated_pitches)
