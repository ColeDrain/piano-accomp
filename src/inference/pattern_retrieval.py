"""
Pattern retrieval engine for piano accompaniment.

Instead of generating notes from scratch (which requires massive training data),
this retrieves actual piano patterns from POP909 and transposes them to fit
the predicted chord. Based on Structure-Aware Piano Accompaniment (Feb 2026).

The pattern library contains ~60K real piano measures from professional musicians.
Each pattern is indexed by chord root, chord quality, register, and density.
At inference time, the system finds the best matching pattern and adapts it.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pretty_midi

from src.tokenizer.vocab import PITCH_NAMES, quality_to_triad, TRIAD_NAMES


@dataclass
class PianoPattern:
    """A single measure of piano accompaniment from POP909."""
    notes: list[tuple[float, float, int, int]]  # (start_beat, duration_beat, pitch, velocity)
    chord_root: int       # 0-11
    chord_quality: str    # "maj", "min", etc.
    triad_group: int      # 0-4
    num_notes: int
    pitch_range: int      # max - min pitch
    avg_pitch: float
    has_bass: bool        # Has notes below C3 (48)
    density: float        # notes per beat
    song_id: int


@dataclass
class PatternLibrary:
    """Indexed collection of piano patterns for fast retrieval."""
    patterns: list[PianoPattern] = field(default_factory=list)

    # Indices for fast lookup
    _by_triad: dict[int, list[int]] = field(default_factory=dict)
    _by_root: dict[int, list[int]] = field(default_factory=dict)

    def build_index(self):
        """Build lookup indices after all patterns are added."""
        self._by_triad = {}
        self._by_root = {}
        for i, p in enumerate(self.patterns):
            self._by_triad.setdefault(p.triad_group, []).append(i)
            self._by_root.setdefault(p.chord_root, []).append(i)

    def retrieve(
        self,
        target_root: int,
        target_triad: int,
        prev_pattern: PianoPattern | None = None,
        prefer_bass: bool = True,
        min_notes: int = 3,
        max_notes: int = 30,
        top_k: int = 10,
    ) -> PianoPattern | None:
        """Find the best matching pattern for a target chord.

        Scoring considers:
        1. Chord quality match (triad group)
        2. Note density (prefer 5-15 notes)
        3. Has bass notes
        4. Voice-leading continuity with previous pattern
        """
        # Get candidates matching the triad group
        candidates = self._by_triad.get(target_triad, [])

        if not candidates:
            # Fallback: any major/minor pattern
            candidates = self._by_triad.get(0, []) + self._by_triad.get(1, [])

        if not candidates:
            return None

        # Score each candidate
        scored = []
        for idx in candidates:
            p = self.patterns[idx]

            if p.num_notes < min_notes or p.num_notes > max_notes:
                continue

            score = 0.0

            # Exact triad match
            if p.triad_group == target_triad:
                score += 5.0

            # Prefer patterns with bass
            if prefer_bass and p.has_bass:
                score += 2.0

            # Prefer moderate density (5-15 notes per 4 beats)
            if 5 <= p.num_notes <= 15:
                score += 2.0
            elif p.num_notes < 5:
                score += 0.5

            # Prefer wider pitch range (sounds more like piano)
            if p.pitch_range > 24:  # 2+ octaves
                score += 2.0
            elif p.pitch_range > 12:  # 1+ octave
                score += 1.0

            # Voice-leading: prefer patterns whose register is close to previous
            if prev_pattern is not None:
                pitch_diff = abs(p.avg_pitch - prev_pattern.avg_pitch)
                if pitch_diff < 6:
                    score += 2.0  # Smooth connection
                elif pitch_diff < 12:
                    score += 1.0

            # Small random factor for variety
            score += random.random() * 1.0

            scored.append((score, idx))

        if not scored:
            return None

        # Pick from top-k by score
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]
        _, chosen_idx = random.choice(top)

        return self._transpose_pattern(self.patterns[chosen_idx], target_root)

    def _transpose_pattern(self, pattern: PianoPattern, target_root: int) -> PianoPattern:
        """Transpose a pattern's notes so its root becomes target_root."""
        shift = (target_root - pattern.chord_root) % 12

        # Handle shift direction — prefer smaller shift
        if shift > 6:
            shift -= 12

        transposed_notes = []
        for start, dur, pitch, vel in pattern.notes:
            new_pitch = pitch + shift
            if 21 <= new_pitch <= 108:  # Keep in piano range
                transposed_notes.append((start, dur, new_pitch, vel))

        return PianoPattern(
            notes=transposed_notes,
            chord_root=target_root,
            chord_quality=pattern.chord_quality,
            triad_group=pattern.triad_group,
            num_notes=len(transposed_notes),
            pitch_range=max(n[2] for n in transposed_notes) - min(n[2] for n in transposed_notes) if transposed_notes else 0,
            avg_pitch=np.mean([n[2] for n in transposed_notes]) if transposed_notes else 60,
            has_bass=any(n[2] < 48 for n in transposed_notes),
            density=pattern.density,
            song_id=pattern.song_id,
        )


def build_library_from_pop909(pop909_dir: str | Path) -> PatternLibrary:
    """Extract all piano patterns from POP909 into a searchable library."""
    pop909_dir = Path(pop909_dir)
    songs_dir = pop909_dir / "POP909"

    if not songs_dir.exists():
        raise FileNotFoundError(f"POP909 songs not found at {songs_dir}")

    library = PatternLibrary()

    song_dirs = sorted([d for d in songs_dir.iterdir() if d.is_dir()])
    print(f"Building pattern library from {len(song_dirs)} songs...")

    for song_idx, song_dir in enumerate(song_dirs):
        midi_files = list(song_dir.glob("*.mid"))
        if not midi_files:
            continue

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
        except Exception:
            continue

        if len(midi.instruments) < 2:
            continue

        # POP909: track 0 = melody, track 2 = piano (or last track)
        piano_idx = min(2, len(midi.instruments) - 1)
        piano = midi.instruments[piano_idx]

        if not piano.notes:
            continue

        # Get tempo
        bpm = 120.0
        tempos = midi.get_tempo_changes()
        if len(tempos[1]) > 0:
            bpm = tempos[1][0]
        beat_dur = 60.0 / bpm

        # Load chord annotations
        chord_file = song_dir / "chord_midi.txt"
        chords = _parse_chords(chord_file, beat_dur) if chord_file.exists() else []

        # Extract patterns per measure (4 beats)
        max_beat = max(n.end / beat_dur for n in piano.notes)
        measure = 0

        while measure * 4 < max_beat:
            start_beat = measure * 4
            end_beat = start_beat + 4

            # Get notes in this measure
            measure_notes = []
            for n in piano.notes:
                note_start = n.start / beat_dur
                if start_beat <= note_start < end_beat:
                    measure_notes.append((
                        note_start - start_beat,  # Relative to measure start
                        (n.end - n.start) / beat_dur,
                        n.pitch,
                        n.velocity,
                    ))

            if len(measure_notes) >= 2:
                # Find chord for this measure
                chord_root, chord_quality = _get_chord_at_beat(chords, start_beat)

                pitches = [n[2] for n in measure_notes]
                library.patterns.append(PianoPattern(
                    notes=measure_notes,
                    chord_root=chord_root,
                    chord_quality=chord_quality,
                    triad_group=quality_to_triad(chord_quality),
                    num_notes=len(measure_notes),
                    pitch_range=max(pitches) - min(pitches),
                    avg_pitch=np.mean(pitches),
                    has_bass=any(p < 48 for p in pitches),
                    density=len(measure_notes) / 4.0,
                    song_id=song_idx,
                ))

            measure += 1

        if (song_idx + 1) % 100 == 0:
            print(f"  Processed {song_idx + 1}/{len(song_dirs)} songs ({len(library.patterns)} patterns)")

    print(f"Built library: {len(library.patterns)} patterns from {len(song_dirs)} songs")
    library.build_index()
    return library


def _parse_chords(chord_file: Path, beat_dur: float) -> list[tuple[float, int, str]]:
    """Parse chord file into (beat, root_pc, quality) list."""
    chords = []
    enharmonic = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

    with open(chord_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                beat = float(parts[0])
            except ValueError:
                continue

            symbol = parts[2] if len(parts) > 2 else parts[1]
            if symbol in ("N", "N.C.", "X"):
                continue

            if ":" in symbol:
                root_str, quality = symbol.split(":", 1)
            else:
                root_str = symbol[0]
                if len(symbol) > 1 and symbol[1] in "b#":
                    root_str = symbol[:2]
                    quality = symbol[2:] or "maj"
                else:
                    quality = symbol[1:] or "maj"

            root_str = enharmonic.get(root_str, root_str)
            if root_str in PITCH_NAMES:
                root_pc = PITCH_NAMES.index(root_str)
                # Simplify quality
                if "min" in quality:
                    quality = "min"
                elif "dim" in quality:
                    quality = "dim"
                elif "aug" in quality:
                    quality = "aug"
                elif "sus" in quality:
                    quality = "sus4"
                elif "7" in quality or "9" in quality or "11" in quality or "13" in quality:
                    quality = "7"
                else:
                    quality = "maj"
                chords.append((beat, root_pc, quality))

    return chords


def _get_chord_at_beat(chords: list[tuple[float, int, str]], beat: float) -> tuple[int, str]:
    """Get the chord active at a given beat position."""
    if not chords:
        return 0, "maj"

    active = (0, "maj")
    for b, root, quality in chords:
        if b <= beat:
            active = (root, quality)
        else:
            break
    return active


def pattern_to_midi_notes(
    pattern: PianoPattern,
    offset_beat: float,
    bpm: float,
) -> list[pretty_midi.Note]:
    """Convert a pattern to MIDI notes at a specific beat offset."""
    beat_dur = 60.0 / bpm
    notes = []
    for start, dur, pitch, vel in pattern.notes:
        notes.append(pretty_midi.Note(
            velocity=vel,
            pitch=pitch,
            start=(offset_beat + start) * beat_dur,
            end=(offset_beat + start + dur) * beat_dur,
        ))
    return notes


def save_library(library: PatternLibrary, path: str | Path):
    with open(path, "wb") as f:
        pickle.dump(library, f)
    print(f"Saved library to {path}")


def load_library(path: str | Path) -> PatternLibrary:
    with open(path, "rb") as f:
        library = pickle.load(f)
    print(f"Loaded library: {len(library.patterns)} patterns")
    return library
