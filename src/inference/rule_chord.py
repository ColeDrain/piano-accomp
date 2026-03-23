"""
Rule-based chord detection from melody using music theory.

No ML — just template matching against known chord profiles.
This gives reliable ~70%+ accuracy on diatonic music, which is
enough to prove the rest of the pipeline works.

Approach:
1. Collect pitch classes in a window, weighted by duration
2. Score against all chord templates (major, minor, dom7, etc.)
3. Pick the best match

This replaces the ML chord predictor for now.
"""

from __future__ import annotations

import numpy as np
from src.tokenizer.midi_tokenizer import NoteEvent

# Chord templates: pitch class intervals from root
CHORD_TEMPLATES = {
    "maj":  [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
    "min":  [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
    "7":    [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.6, 0.0],
    "maj7": [1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.6],
    "min7": [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.6, 0.0],
    "dim":  [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sus4": [1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
}

# Map quality names to triad group indices (matching chord_predictor.py)
QUALITY_TO_TRIAD = {
    "maj": 0, "7": 0, "maj7": 0,
    "min": 1, "min7": 1,
    "dim": 2,
    "sus4": 4,
}

PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def detect_chord_from_notes(
    notes: list[NoteEvent],
    key_root: int = 0,
    prefer_diatonic: bool = True,
) -> tuple[int, int, str]:
    """Detect the most likely chord from a list of melody notes.

    Args:
        notes: List of NoteEvents
        key_root: The key of the piece (0=C, for diatonic preference)
        prefer_diatonic: Boost diatonic chords (chords in the key)

    Returns:
        (root_pc, triad_group, quality_name)
    """
    if not notes:
        return 0, 0, "maj"

    # Build pitch class histogram weighted by duration
    chroma = np.zeros(12)
    for note in notes:
        pc = note.pitch % 12
        chroma[pc] += note.duration_beats

    if chroma.sum() == 0:
        return 0, 0, "maj"

    chroma = chroma / chroma.sum()

    # Score every possible chord (12 roots × 7 qualities)
    best_score = -1.0
    best_root = 0
    best_quality = "maj"

    # Diatonic chord roots for the key (major scale degrees)
    diatonic_roots = set()
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    for degree in major_scale:
        diatonic_roots.add((key_root + degree) % 12)

    for root in range(12):
        for quality, template in CHORD_TEMPLATES.items():
            # Rotate template to this root
            rotated = np.roll(template, root)

            # Correlation with the observed chroma
            score = np.dot(chroma, rotated)

            # Boost diatonic chords
            if prefer_diatonic and root in diatonic_roots:
                score *= 1.3

            # Slight boost for major/minor over complex chords
            if quality in ("maj", "min"):
                score *= 1.1

            if score > best_score:
                best_score = score
                best_root = root
                best_quality = quality

    triad_group = QUALITY_TO_TRIAD.get(best_quality, 0)
    return best_root, triad_group, best_quality


def detect_key_from_notes(notes: list[NoteEvent]) -> int:
    """Simple key detection — returns pitch class of likely key root."""
    if not notes:
        return 0

    chroma = np.zeros(12)
    for note in notes:
        chroma[note.pitch % 12] += note.duration_beats

    if chroma.sum() == 0:
        return 0

    chroma = chroma / chroma.sum()

    # Krumhansl major key profile
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major_profile = major_profile / major_profile.sum()

    best_corr = -2.0
    best_key = 0

    for shift in range(12):
        shifted = np.roll(chroma, -shift)
        corr = np.corrcoef(shifted, major_profile)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_key = shift

    return best_key
