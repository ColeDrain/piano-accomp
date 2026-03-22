"""
Preprocess POP909 MIDI files into tokenized training data.

Key preprocessing steps (following published best practices):
1. Detect key and normalize all songs to C major / A minor
2. Align windows to chord boundaries (not fixed 2-bar cuts)
3. One chord per window (window = one chord span)
4. No 12-key transposition (normalization makes it unnecessary)
5. Augment with tempo jitter and onset jitter instead

References:
- HarmonyTok (MDPI 2025): key normalization + chord spelling
- Structure-Aware Piano (arXiv 2026): key normalization + functional chords
- "Translating Melody to Chord" (IEEE 2022): Hooktheory preprocessing
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
POP909_SONGS_DIR = RAW_DIR / "POP909"
OUT_DIR = DATA_DIR / "processed"


# --- Key detection (simplified Krumhansl-Schmuckler) ---

# Major and minor key profiles (Krumhansl)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(notes: list[NoteEvent]) -> tuple[int, str]:
    """Detect the key of a piece using Krumhansl-Schmuckler algorithm.

    Returns:
        (root_semitone, mode) where root_semitone is 0-11 and mode is "major" or "minor"
    """
    if not notes:
        return 0, "major"

    # Build pitch class histogram weighted by duration
    histogram = np.zeros(12)
    for note in notes:
        pc = note.pitch % 12
        histogram[pc] += note.duration_beats

    if histogram.sum() == 0:
        return 0, "major"

    histogram = histogram / histogram.sum()

    # Correlate with all major and minor key profiles
    best_corr = -2.0
    best_key = 0
    best_mode = "major"

    for shift in range(12):
        shifted = np.roll(histogram, -shift)
        corr_major = np.corrcoef(shifted, MAJOR_PROFILE)[0, 1]
        corr_minor = np.corrcoef(shifted, MINOR_PROFILE)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = shift
            best_mode = "major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = shift
            best_mode = "minor"

    return best_key, best_mode


def normalize_to_c(events: list[NoteEvent], key_root: int, mode: str) -> list[NoteEvent]:
    """Transpose events so that key_root becomes C (major) or A (minor)."""
    if mode == "minor":
        # Normalize to A minor (A = 9 semitones)
        shift = (9 - key_root) % 12
    else:
        # Normalize to C major (C = 0)
        shift = (0 - key_root) % 12

    if shift == 0:
        return events

    return [
        NoteEvent(
            start_beat=e.start_beat,
            pitch=max(0, min(127, e.pitch + shift)),
            duration_beats=e.duration_beats,
            velocity=e.velocity,
        )
        for e in events
    ]


def normalize_chord(root: str, quality: str, key_root: int, mode: str) -> tuple[str, str]:
    """Transpose a chord to the normalized key."""
    if mode == "minor":
        shift = (9 - key_root) % 12
    else:
        shift = (0 - key_root) % 12

    root_idx = PITCH_NAMES.index(root) if root in PITCH_NAMES else 0
    new_idx = (root_idx + shift) % 12
    return PITCH_NAMES[new_idx], quality


# --- Chord annotation parsing ---

def parse_chord_annotation(chord_file: Path) -> list[tuple[float, float, str, str]]:
    """Parse POP909 chord annotation file.

    Returns list of (start_beat, end_beat, root, quality) tuples.
    """
    chords = []
    with open(chord_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1]) if len(parts) > 2 else start + 1.0
                chord_str = parts[2] if len(parts) > 2 else parts[1]
                root, quality = _parse_chord_symbol(chord_str)
                if root is not None:
                    chords.append((start, end, root, quality))
    return chords


def _parse_chord_symbol(symbol: str) -> tuple[str | None, str]:
    """Parse a chord symbol like 'C:maj7' into (root, quality)."""
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

    # Normalize root
    enharmonic = {
        "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
        "Cb": "B", "Fb": "E",
    }
    root = enharmonic.get(root, root)

    if root not in PITCH_NAMES:
        return None, "N"

    # Normalize quality
    quality_map = {
        "": "maj", "M": "maj", "major": "maj",
        "m": "min", "minor": "min",
        "dom": "7", "dom7": "7",
        "M7": "maj7", "m7": "min7",
        "hdim7": "min7b5", "hdim": "min7b5",
    }
    quality = quality_map.get(quality, quality)

    if quality not in CHORD_QUALITIES:
        if "min" in quality:
            quality = "min"
        elif "dim" in quality:
            quality = "dim"
        elif "aug" in quality:
            quality = "aug"
        else:
            quality = "maj"

    return root, quality


# --- Window creation aligned to chord boundaries ---

def process_song(
    song_dir: Path,
    tokenizer: MidiTokenizer,
    vocab: Vocabulary,
    song_id: int,
) -> list[dict]:
    """Process a single POP909 song into chord-aligned training windows."""
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

    # POP909: track 0 = melody, track 1 = bridge, track 2 = piano
    melody_track_idx = 0
    accomp_track_idx = min(2, len(midi.instruments) - 1)

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

    if not melody_events:
        return []

    # Step 1: Detect key and normalize to C major / A minor
    key_root, mode = detect_key(melody_events)
    melody_events = normalize_to_c(melody_events, key_root, mode)
    accomp_events = normalize_to_c(accomp_events, key_root, mode)

    # Step 2: Load and normalize chord annotations
    chord_file = song_dir / "chord_midi.txt"
    if not chord_file.exists():
        chord_file = next(song_dir.glob("*chord*"), None)

    if not chord_file or not chord_file.exists():
        return []

    raw_chords = parse_chord_annotation(chord_file)
    chords = [
        (start, end, *normalize_chord(root, qual, key_root, mode))
        for start, end, root, qual in raw_chords
    ]

    if not chords:
        return []

    # Step 3: Create chord-aligned windows
    # Each window = one chord span (or merged short chords up to ~2 bars)
    return _create_chord_aligned_windows(
        melody_events, accomp_events, chords, tokenizer, vocab, song_id
    )


def _create_chord_aligned_windows(
    melody_events: list[NoteEvent],
    accomp_events: list[NoteEvent],
    chords: list[tuple[float, float, str, str]],
    tokenizer: MidiTokenizer,
    vocab: Vocabulary,
    song_id: int,
    min_window_beats: float = 2.0,
    max_window_beats: float = 8.0,
    context_beats: float = 4.0,  # Include preceding melody as context
    max_melody_tokens: int = 48,
    max_accomp_tokens: int = 128,
) -> list[dict]:
    """Create windows aligned to chord boundaries.

    Each window covers one chord span. Short chords are merged.
    Preceding melody is included as context for the chord predictor.
    """
    windows = []

    i = 0
    while i < len(chords):
        start_beat, end_beat, root, quality = chords[i]
        window_dur = end_beat - start_beat

        # Merge short chords (< min_window_beats) with next chord
        while window_dur < min_window_beats and i + 1 < len(chords):
            i += 1
            _, end_beat, _, _ = chords[i]
            window_dur = end_beat - start_beat
            if window_dur >= max_window_beats:
                break

        # Cap window duration
        end_beat = min(end_beat, start_beat + max_window_beats)

        # Include context: melody from before this chord
        context_start = max(0.0, start_beat - context_beats)

        # Get melody events (with context)
        mel_win = [
            NoteEvent(e.start_beat - context_start, e.pitch, e.duration_beats, e.velocity)
            for e in melody_events
            if context_start <= e.start_beat < end_beat
        ]

        # Get accompaniment events (only for this chord span)
        acc_win = [
            NoteEvent(e.start_beat - start_beat, e.pitch, e.duration_beats, e.velocity)
            for e in accomp_events
            if start_beat <= e.start_beat < end_beat
        ]

        if mel_win and acc_win:
            try:
                chord_label = vocab.encode_chord(root, quality)
            except KeyError:
                chord_label = vocab.encode_chord("C", "maj")

            mel_tokens = tokenizer.encode_note_events(mel_win)[:max_melody_tokens]
            acc_tokens = tokenizer.encode_note_events(acc_win)[:max_accomp_tokens]

            windows.append({
                "melody_tokens": torch.tensor(mel_tokens, dtype=torch.long),
                "chord_label": chord_label,
                "accomp_tokens": torch.tensor(acc_tokens, dtype=torch.long),
                "song_id": song_id,
            })

        i += 1

    return windows


# --- Augmentation (tempo jitter + onset jitter, NOT key transposition) ---

def augment_window(window: dict, tokenizer: MidiTokenizer) -> list[dict]:
    """Create augmented copies with tempo and onset jitter."""
    augmented = [window]  # Original

    # Tempo jitter: speed up or slow down by 5-10%
    for factor in [0.9, 0.95, 1.05, 1.1]:
        mel_events = tokenizer.decode_to_events(window["melody_tokens"].tolist())
        acc_events = tokenizer.decode_to_events(window["accomp_tokens"].tolist())

        mel_scaled = [
            NoteEvent(e.start_beat * factor, e.pitch, e.duration_beats * factor, e.velocity)
            for e in mel_events
        ]
        acc_scaled = [
            NoteEvent(e.start_beat * factor, e.pitch, e.duration_beats * factor, e.velocity)
            for e in acc_events
        ]

        augmented.append({
            "melody_tokens": torch.tensor(tokenizer.encode_note_events(mel_scaled), dtype=torch.long),
            "chord_label": window["chord_label"],
            "accomp_tokens": torch.tensor(tokenizer.encode_note_events(acc_scaled), dtype=torch.long),
            "song_id": window["song_id"],
        })

    return augmented


# --- Main preprocessing ---

def preprocess_all():
    """Process all POP909 songs and save tokenized data."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    search_dir = POP909_SONGS_DIR if POP909_SONGS_DIR.exists() else RAW_DIR
    song_dirs = sorted([
        d for d in search_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Processing {len(song_dirs)} songs (key-normalized, chord-aligned)...")
    all_windows = []

    for i, song_dir in enumerate(song_dirs):
        windows = process_song(song_dir, tokenizer, vocab, song_id=i)

        # Augment with tempo jitter (5x per window)
        for w in windows:
            all_windows.extend(augment_window(w, tokenizer))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(song_dirs)} songs ({len(all_windows)} windows)")

    print(f"\nTotal: {len(all_windows)} training windows")

    # Split by song_id (80/10/10)
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

    meta = {
        "vocab_size": vocab.size,
        "num_songs": len(song_dirs),
        "num_windows": {k: len(v) for k, v in splits.items()},
        "key_normalized": True,
        "chord_aligned": True,
        "augmentation": "tempo_jitter_5x",
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    preprocess_all()
