"""
Preprocess POP909 MIDI files into tokenized training data.

For each song in POP909:
1. Load the MIDI file
2. Identify melody and accompaniment tracks
3. Load chord annotations
4. Tokenize into (melody_tokens, chord_label, accompaniment_tokens) windows
5. Apply data augmentation (transposition, tempo jitter)
6. Save as .pt files for training

Output format per window:
{
    "melody_tokens": Tensor[int],       # Token IDs for melody window
    "chord_label": int,                  # Chord class ID for this window
    "accomp_tokens": Tensor[int],        # Token IDs for accompaniment
    "song_id": int,                      # For train/val/test splitting
}
"""

import json
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.tokenizer.vocab import Vocabulary, PITCH_NAMES, CHORD_QUALITIES


DATA_DIR = Path(__file__).parent.parent
RAW_DIR = DATA_DIR / "raw" / "pop909"
OUT_DIR = DATA_DIR / "processed"


def parse_chord_annotation(chord_file: Path) -> list[tuple[float, str, str]]:
    """Parse POP909 chord annotation file.

    Returns list of (beat_position, root, quality) tuples.
    """
    chords = []
    with open(chord_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                beat = float(parts[0])
                chord_str = parts[2] if len(parts) > 2 else parts[1]
                root, quality = _parse_chord_symbol(chord_str)
                if root is not None:
                    chords.append((beat, root, quality))
    return chords


def _parse_chord_symbol(symbol: str) -> tuple[str | None, str]:
    """Parse a chord symbol like 'C:maj7' or 'Db:min' into (root, quality)."""
    if symbol in ("N", "N.C.", "NC", "X"):
        return None, "N"

    if ":" in symbol:
        root, quality = symbol.split(":", 1)
    elif len(symbol) >= 2 and symbol[1] in ("b", "#"):
        root = symbol[:2]
        quality = symbol[2:] or "maj"
    else:
        root = symbol[0]
        quality = symbol[1:] or "maj"

    # Normalize root name
    root = root.replace("#", "").replace("b", "b")  # Keep flats
    if root not in PITCH_NAMES:
        # Try enharmonic
        enharmonic = {
            "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
            "Cb": "B", "Fb": "E",
        }
        root = enharmonic.get(root, root)

    if root not in PITCH_NAMES:
        return None, "N"

    # Normalize quality to our vocabulary
    quality_map = {
        "": "maj", "M": "maj", "major": "maj",
        "m": "min", "minor": "min",
        "dom": "7", "dom7": "7",
        "M7": "maj7", "m7": "min7",
        "hdim7": "min7b5", "hdim": "min7b5",
    }
    quality = quality_map.get(quality, quality)

    if quality not in CHORD_QUALITIES:
        # Fall back to basic triad
        if "min" in quality:
            quality = "min"
        elif "dim" in quality:
            quality = "dim"
        elif "aug" in quality:
            quality = "aug"
        else:
            quality = "maj"

    return root, quality


def transpose_events(
    events: list[NoteEvent], semitones: int
) -> list[NoteEvent]:
    """Transpose all notes by given number of semitones."""
    transposed = []
    for e in events:
        new_pitch = e.pitch + semitones
        if 0 <= new_pitch <= 127:
            transposed.append(NoteEvent(
                start_beat=e.start_beat,
                pitch=new_pitch,
                duration_beats=e.duration_beats,
                velocity=e.velocity,
            ))
    return transposed


def transpose_chord(root: str, quality: str, semitones: int) -> tuple[str, str]:
    """Transpose a chord root by semitones."""
    root_idx = PITCH_NAMES.index(root)
    new_idx = (root_idx + semitones) % 12
    return PITCH_NAMES[new_idx], quality


def process_song(
    song_dir: Path,
    tokenizer: MidiTokenizer,
    vocab: Vocabulary,
    song_id: int,
    augment: bool = True,
) -> list[dict]:
    """Process a single POP909 song into training windows."""
    midi_files = list(song_dir.rglob("*.mid"))
    if not midi_files:
        return []

    midi_file = midi_files[0]
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
    except Exception as e:
        print(f"  Error loading {midi_file}: {e}")
        return []

    if len(midi.instruments) < 2:
        return []

    # POP909 convention: track 0 = melody, track 1 = bridge, track 2 = piano
    melody_track_idx = 0
    accomp_track_idx = min(2, len(midi.instruments) - 1)

    # Extract note events
    bpm = midi.get_tempo_changes()[1][0] if len(midi.get_tempo_changes()[1]) > 0 else 120.0
    beat_dur = 60.0 / bpm

    def notes_to_events(instrument):
        return [
            NoteEvent(
                start_beat=n.start / beat_dur,
                pitch=n.pitch,
                duration_beats=(n.end - n.start) / beat_dur,
                velocity=n.velocity,
            )
            for n in instrument.notes
        ]

    melody_events = notes_to_events(midi.instruments[melody_track_idx])
    accomp_events = notes_to_events(midi.instruments[accomp_track_idx])

    # Load chord annotations if available
    chord_file = song_dir / "chord_midi.txt"
    if not chord_file.exists():
        chord_file = next(song_dir.glob("*chord*"), None)

    chords = parse_chord_annotation(chord_file) if chord_file and chord_file.exists() else []

    # Generate windows
    windows = _create_windows(
        melody_events, accomp_events, chords, tokenizer, vocab, song_id
    )

    # Augmentation: transpose to all 12 keys
    if augment:
        all_windows = []
        for semitones in range(12):
            if semitones == 0:
                all_windows.extend(windows)
                continue

            t_melody = transpose_events(melody_events, semitones)
            t_accomp = transpose_events(accomp_events, semitones)
            t_chords = [
                (beat, *transpose_chord(root, qual, semitones))
                for beat, root, qual in chords
            ]
            all_windows.extend(_create_windows(
                t_melody, t_accomp, t_chords, tokenizer, vocab, song_id
            ))
        return all_windows

    return windows


def _create_windows(
    melody_events: list[NoteEvent],
    accomp_events: list[NoteEvent],
    chords: list[tuple[float, str, str]],
    tokenizer: MidiTokenizer,
    vocab: Vocabulary,
    song_id: int,
    window_beats: float = 8.0,     # 2 bars of 4/4
    hop_beats: float = 4.0,        # 1 bar hop
    max_melody_tokens: int = 32,
    max_accomp_tokens: int = 128,
) -> list[dict]:
    """Segment events into fixed-size windows for training."""
    if not melody_events:
        return []

    max_beat = max(e.start_beat for e in melody_events)
    windows = []

    beat = 0.0
    while beat < max_beat:
        end_beat = beat + window_beats

        # Get melody events in this window
        mel_win = [
            NoteEvent(e.start_beat - beat, e.pitch, e.duration_beats, e.velocity)
            for e in melody_events
            if beat <= e.start_beat < end_beat
        ]
        # Get accompaniment events in this window
        acc_win = [
            NoteEvent(e.start_beat - beat, e.pitch, e.duration_beats, e.velocity)
            for e in accomp_events
            if beat <= e.start_beat < end_beat
        ]

        if not mel_win or not acc_win:
            beat += hop_beats
            continue

        # Find chord for this window (use chord at window start or most common)
        chord_label = _get_window_chord(chords, beat, end_beat, vocab)

        # Tokenize
        mel_tokens = tokenizer.encode_note_events(mel_win)
        acc_tokens = tokenizer.encode_note_events(acc_win)

        # Truncate / pad
        mel_tokens = mel_tokens[:max_melody_tokens]
        acc_tokens = acc_tokens[:max_accomp_tokens]

        windows.append({
            "melody_tokens": torch.tensor(mel_tokens, dtype=torch.long),
            "chord_label": chord_label,
            "accomp_tokens": torch.tensor(acc_tokens, dtype=torch.long),
            "song_id": song_id,
        })

        beat += hop_beats

    return windows


def _get_window_chord(
    chords: list[tuple[float, str, str]],
    start_beat: float,
    end_beat: float,
    vocab: Vocabulary,
) -> int:
    """Get the chord label for a time window. Returns vocab token ID."""
    if not chords:
        return vocab.encode_chord("C", "maj")  # Default

    # Find the chord active at the window start
    active_chord = None
    for beat, root, quality in chords:
        if beat <= start_beat:
            active_chord = (root, quality)
        elif beat > start_beat:
            break

    if active_chord is None:
        active_chord = (chords[0][1], chords[0][2])

    root, quality = active_chord
    try:
        return vocab.encode_chord(root, quality)
    except KeyError:
        return vocab.encode_chord("C", "maj")


def preprocess_all(augment: bool = True):
    """Process all POP909 songs and save tokenized data."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    song_dirs = sorted([
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Processing {len(song_dirs)} songs...")
    all_windows = []

    for i, song_dir in enumerate(song_dirs):
        windows = process_song(song_dir, tokenizer, vocab, song_id=i, augment=augment)
        all_windows.extend(windows)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(song_dirs)} songs ({len(all_windows)} windows)")

    print(f"\nTotal: {len(all_windows)} training windows")

    # Split by song_id
    song_ids = list(set(w["song_id"] for w in all_windows))
    random.shuffle(song_ids)
    n = len(song_ids)
    train_ids = set(song_ids[: int(0.8 * n)])
    val_ids = set(song_ids[int(0.8 * n) : int(0.9 * n)])
    test_ids = set(song_ids[int(0.9 * n) :])

    splits = {"train": [], "val": [], "test": []}
    for w in all_windows:
        if w["song_id"] in train_ids:
            splits["train"].append(w)
        elif w["song_id"] in val_ids:
            splits["val"].append(w)
        else:
            splits["test"].append(w)

    for split_name, windows in splits.items():
        out_path = OUT_DIR / f"{split_name}.pt"
        torch.save(windows, out_path)
        print(f"Saved {split_name}: {len(windows)} windows -> {out_path}")

    # Save metadata
    meta = {
        "vocab_size": vocab.size,
        "num_songs": len(song_dirs),
        "num_windows": {k: len(v) for k, v in splits.items()},
        "augmented": augment,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    preprocess_all()
