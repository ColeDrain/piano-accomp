"""
Gospel-style data augmentation.

Takes chord progressions (from POP909 or chord charts) and voices them
in gospel style to create synthetic training data.

This is how we bootstrap gospel data without thousands of real recordings:
1. Take a pop chord progression (e.g., C - Am - F - G)
2. Optionally reharmonize with gospel substitutions (C maj9 - Am9 - Fmaj7 - G13)
3. Voice it using gospel voicing rules (stride, walking bass, block)
4. Pair with the original melody to create a training example

Augmentation pipeline:
    POP909 chord progression
    → gospel reharmonization (optional)
    → gospel voicing (multiple styles)
    → key transposition (all 12 keys)
    → tempo variation
    → training data
"""

from __future__ import annotations

import random
from pathlib import Path

import pretty_midi

from src.tokenizer.vocab import PITCH_NAMES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.gospel.chord_vocabulary import GOSPEL_PROGRESSIONS
from src.gospel.voicing_rules import voice_progression


# --- Gospel reharmonization rules ---
# Replace simple pop chords with richer gospel equivalents

REHARMONIZE_MAP: dict[str, list[str]] = {
    "maj":   ["maj9", "maj7", "6/9", "add9"],
    "min":   ["min9", "min7", "min11"],
    "7":     ["9", "13", "7#9", "13sus4"],
    "maj7":  ["maj9", "maj7#11", "6/9"],
    "min7":  ["min9", "min7add11", "min11"],
    "dim":   ["dim7", "min7b5"],
    "sus4":  ["9sus4", "13sus4"],
}


def reharmonize_chord(quality: str) -> str:
    """Replace a simple chord quality with a gospel-extended one."""
    options = REHARMONIZE_MAP.get(quality)
    if options:
        return random.choice(options)
    return quality


def gospelize_progression(
    chords: list[tuple[int, str]],
    reharmonize: bool = True,
) -> list[tuple[int, str]]:
    """Apply gospel reharmonization to a chord progression.

    Args:
        chords: List of (root_pitch_class, quality) tuples
        reharmonize: Whether to substitute chord qualities

    Returns:
        Gospel-reharmonized progression
    """
    result = []
    for root, quality in chords:
        if reharmonize and random.random() < 0.7:
            quality = reharmonize_chord(quality)
        result.append((root, quality))

    # Add passing diminished chords (gospel hallmark)
    gospelized = []
    for i, (root, quality) in enumerate(result):
        gospelized.append((root, quality))

        # Insert passing dim7 between chords a whole step apart
        if i + 1 < len(result):
            next_root = result[i + 1][0]
            interval = (next_root - root) % 12
            if interval == 2 and random.random() < 0.4:
                # Chromatic passing diminished
                passing_root = (root + 1) % 12
                gospelized.append((passing_root, "dim7"))

    return gospelized


def generate_synthetic_gospel_data(
    num_examples: int = 1000,
    beats_per_chord: float = 4.0,
) -> list[dict]:
    """Generate synthetic gospel accompaniment training data.

    Uses gospel progressions and voicing rules to create
    (chord_progression, voiced_accompaniment) pairs.

    Returns:
        List of dicts with keys:
            - "chords": list of (root_pc, quality) tuples
            - "events": list of NoteEvent (the voiced accompaniment)
            - "style": str (voicing style used)
            - "key": int (root key, 0-11)
    """
    styles = ["block", "stride", "walking"]
    examples = []

    for _ in range(num_examples):
        # Pick a base progression
        prog = random.choice(GOSPEL_PROGRESSIONS)
        base_chords = prog["chords"]

        # Random key transposition
        key = random.randint(0, 11)

        # Transpose
        chords = [((root + key) % 12, quality) for root, quality in base_chords]

        # Optionally gospelize further
        if random.random() < 0.5:
            chords = gospelize_progression(chords)

        # Voice in a random style
        style = random.choice(styles)
        events = voice_progression(chords, beats_per_chord=beats_per_chord, style=style)

        examples.append({
            "chords": chords,
            "events": events,
            "style": style,
            "key": key,
        })

    return examples


def generate_gospel_midi(
    output_dir: str | Path,
    num_files: int = 100,
    bpm: float = 80.0,
):
    """Generate synthetic gospel MIDI files for training.

    Creates MIDI files with both a simple melody and gospel accompaniment.
    """
    from src.tokenizer.vocab import Vocabulary

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_synthetic_gospel_data(num_files)

    for i, example in enumerate(examples):
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        beat_dur = 60.0 / bpm

        # Accompaniment track
        piano = pretty_midi.Instrument(program=0, name="Piano")
        for event in example["events"]:
            note = pretty_midi.Note(
                velocity=event.velocity,
                pitch=event.pitch,
                start=event.start_beat * beat_dur,
                end=(event.start_beat + event.duration_beats) * beat_dur,
            )
            piano.notes.append(note)
        midi.instruments.append(piano)

        # Simple melody track (root notes of chords, for pairing)
        melody = pretty_midi.Instrument(program=0, name="Melody")
        beat = 0.0
        for root_pc, quality in example["chords"]:
            melody_pitch = 72 + root_pc  # C5 range
            note = pretty_midi.Note(
                velocity=80,
                pitch=melody_pitch,
                start=beat * beat_dur,
                end=(beat + 2.0) * beat_dur,  # Half-note melody
            )
            melody.notes.append(note)
            beat += 4.0  # One chord per bar
        midi.instruments.append(melody)

        key_name = PITCH_NAMES[example["key"]]
        filename = f"gospel_{i:04d}_{key_name}_{example['style']}.mid"
        midi.write(str(output_dir / filename))

    print(f"Generated {num_files} gospel MIDI files in {output_dir}")
