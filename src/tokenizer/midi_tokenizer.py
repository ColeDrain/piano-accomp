"""
REMI-style MIDI tokenizer for melody and accompaniment.

Converts between pretty_midi.PrettyMIDI objects and token ID sequences.
Handles both melody (single voice) and accompaniment (polyphonic piano).

Encoding flow:
    PrettyMIDI -> list of NoteEvents -> sort by time -> quantize -> token IDs

Decoding flow:
    token IDs -> NoteEvents -> PrettyMIDI
"""

from __future__ import annotations

from dataclasses import dataclass

import pretty_midi
import numpy as np

from src.tokenizer.vocab import Vocabulary, DURATION_BINS, TIME_SHIFT_BINS


@dataclass
class NoteEvent:
    """A single note with timing, pitch, velocity, and duration."""
    start_beat: float
    pitch: int
    duration_beats: float
    velocity: int


class MidiTokenizer:
    """Tokenizes MIDI data using REMI-style representation."""

    def __init__(self, vocab: Vocabulary | None = None):
        self.vocab = vocab or Vocabulary()

    # ------------------------------------------------------------------
    # Encoding: MIDI -> tokens
    # ------------------------------------------------------------------

    def encode_midi(
        self,
        midi: pretty_midi.PrettyMIDI,
        track_idx: int = 0,
        max_bars: int | None = None,
    ) -> list[int]:
        """Encode a single MIDI track to a token sequence.

        Args:
            midi: PrettyMIDI object
            track_idx: Which instrument track to encode
            max_bars: Optionally limit to first N bars

        Returns:
            List of token IDs
        """
        if not midi.instruments or track_idx >= len(midi.instruments):
            return [self.vocab.bos_id, self.vocab.eos_id]

        instrument = midi.instruments[track_idx]
        tempo_changes = midi.get_tempo_changes()

        # Get tempo (use first tempo, or default 120 BPM)
        if len(tempo_changes[1]) > 0:
            bpm = tempo_changes[1][0]
        else:
            bpm = 120.0

        beat_duration = 60.0 / bpm  # seconds per beat

        # Convert notes to NoteEvents in beat-space
        events = []
        for note in instrument.notes:
            start_beat = note.start / beat_duration
            dur_beat = (note.end - note.start) / beat_duration
            events.append(NoteEvent(
                start_beat=start_beat,
                pitch=note.pitch,
                duration_beats=dur_beat,
                velocity=note.velocity,
            ))

        # Sort by start time, then by pitch (ascending)
        events.sort(key=lambda e: (e.start_beat, e.pitch))

        if max_bars is not None:
            max_beat = max_bars * 4  # Assume 4/4 time
            events = [e for e in events if e.start_beat < max_beat]

        return self._events_to_tokens(events)

    def encode_note_events(self, events: list[NoteEvent]) -> list[int]:
        """Encode a list of NoteEvents directly (useful for real-time pipeline)."""
        events = sorted(events, key=lambda e: (e.start_beat, e.pitch))
        return self._events_to_tokens(events)

    def _events_to_tokens(self, events: list[NoteEvent]) -> list[int]:
        """Convert sorted NoteEvents to token IDs."""
        tokens = [self.vocab.bos_id]
        current_beat = 0.0

        for event in events:
            # Time shift from current position
            dt = event.start_beat - current_beat
            if dt > 0.01:  # Skip negligible shifts
                # Break large shifts into multiple tokens
                while dt > TIME_SHIFT_BINS[-1]:
                    tokens.append(self.vocab.encode_time_shift(TIME_SHIFT_BINS[-1]))
                    dt -= TIME_SHIFT_BINS[-1]
                if dt > 0.01:
                    tokens.append(self.vocab.encode_time_shift(dt))

            # Note: Pitch -> Duration -> Velocity
            tokens.append(self.vocab.encode_pitch(event.pitch))
            tokens.append(self.vocab.encode_duration(event.duration_beats))
            tokens.append(self.vocab.encode_velocity(event.velocity))

            current_beat = event.start_beat

        tokens.append(self.vocab.eos_id)
        return tokens

    # ------------------------------------------------------------------
    # Encoding: melody + accompaniment pair
    # ------------------------------------------------------------------

    def encode_pair(
        self,
        midi: pretty_midi.PrettyMIDI,
        melody_track: int = 0,
        accomp_track: int = 1,
        max_bars: int | None = None,
    ) -> list[int]:
        """Encode a melody-accompaniment pair with SEP token between them.

        Returns:
            [BOS, ...melody tokens..., SEP, ...accompaniment tokens..., EOS]
        """
        melody_tokens = self.encode_midi(midi, melody_track, max_bars)
        accomp_tokens = self.encode_midi(midi, accomp_track, max_bars)

        # Strip BOS/EOS from individual encodings, combine with SEP
        melody_body = melody_tokens[1:-1]  # Remove BOS and EOS
        accomp_body = accomp_tokens[1:-1]

        return [self.vocab.bos_id] + melody_body + [self.vocab.sep_id] + accomp_body + [self.vocab.eos_id]

    # ------------------------------------------------------------------
    # Decoding: tokens -> MIDI
    # ------------------------------------------------------------------

    def decode_to_midi(
        self,
        token_ids: list[int],
        bpm: float = 120.0,
        program: int = 0,
    ) -> pretty_midi.PrettyMIDI:
        """Decode token IDs back to a PrettyMIDI object.

        Args:
            token_ids: List of token IDs
            bpm: Tempo for timing conversion
            program: MIDI program number (0 = Acoustic Grand Piano)

        Returns:
            PrettyMIDI object
        """
        events = self.decode_to_events(token_ids)
        return self._events_to_midi(events, bpm, program)

    def decode_to_events(self, token_ids: list[int]) -> list[NoteEvent]:
        """Decode token IDs to NoteEvents."""
        events = []
        current_beat = 0.0
        i = 0

        while i < len(token_ids):
            tid = token_ids[i]

            # Skip special tokens
            if self.vocab.is_special(tid):
                # SEP resets position (start of accompaniment)
                if tid == self.vocab.sep_id:
                    current_beat = 0.0
                i += 1
                continue

            # Time shift
            if self.vocab.is_time_shift(tid):
                current_beat += self.vocab.decode_time_shift(tid)
                i += 1
                continue

            # Chord tokens (skip for now, used by model not for playback)
            if self.vocab.is_chord(tid):
                i += 1
                continue

            # Note: expect Pitch, Duration, Velocity in sequence
            if self.vocab.is_pitch(tid):
                pitch = self.vocab.decode_pitch(tid)
                duration = DURATION_BINS[0]  # default
                velocity = 80  # default

                if i + 1 < len(token_ids) and self.vocab.is_duration(token_ids[i + 1]):
                    duration = self.vocab.decode_duration(token_ids[i + 1])
                    i += 1

                if i + 1 < len(token_ids) and self.vocab.is_velocity(token_ids[i + 1]):
                    velocity = self.vocab.decode_velocity(token_ids[i + 1])
                    i += 1

                events.append(NoteEvent(
                    start_beat=current_beat,
                    pitch=pitch,
                    duration_beats=duration,
                    velocity=velocity,
                ))

            i += 1

        return events

    def _events_to_midi(
        self,
        events: list[NoteEvent],
        bpm: float,
        program: int,
    ) -> pretty_midi.PrettyMIDI:
        """Convert NoteEvents to a PrettyMIDI object."""
        beat_duration = 60.0 / bpm

        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(program=program)

        for event in events:
            start_sec = event.start_beat * beat_duration
            end_sec = (event.start_beat + event.duration_beats) * beat_duration
            note = pretty_midi.Note(
                velocity=event.velocity,
                pitch=event.pitch,
                start=start_sec,
                end=end_sec,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)
        return midi

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def split_pair(self, token_ids: list[int]) -> tuple[list[int], list[int]]:
        """Split a paired sequence at the SEP token.

        Returns:
            (melody_tokens, accompaniment_tokens) — each with BOS/EOS
        """
        if self.vocab.sep_id not in token_ids:
            return token_ids, []

        sep_idx = token_ids.index(self.vocab.sep_id)
        melody = [self.vocab.bos_id] + token_ids[1:sep_idx] + [self.vocab.eos_id]
        accomp = [self.vocab.bos_id] + token_ids[sep_idx + 1:-1] + [self.vocab.eos_id]
        return melody, accomp

    def tokens_to_str(self, token_ids: list[int]) -> str:
        """Convert token IDs to a human-readable string for debugging."""
        return " ".join(self.vocab.id_to_token.get(t, f"<UNK:{t}>") for t in token_ids)
