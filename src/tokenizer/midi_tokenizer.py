"""
True REMI MIDI tokenizer for melody and accompaniment.

Converts between pretty_midi.PrettyMIDI objects and token ID sequences.
Uses Bar + Position tokens for explicit metrical structure (not TimeShift).

REMI encoding per bar:
    <BAR> Position_0 Pitch_60 Duration_9 Velocity_20 Position_4 Pitch_64 ...

This gives the model explicit knowledge of where each note falls in the bar,
which is critical for chord prediction (notes on beat 1 vs beat 3 have
very different harmonic implications).
"""

from __future__ import annotations

from dataclasses import dataclass

import pretty_midi
import numpy as np

from src.tokenizer.vocab import (
    Vocabulary, DURATION_BINS, TIME_SHIFT_BINS,
    BEATS_PER_BAR, SUBDIVISIONS_PER_BEAT, NUM_POSITIONS,
)


@dataclass
class NoteEvent:
    """A single note with timing, pitch, velocity, and duration."""
    start_beat: float
    pitch: int
    duration_beats: float
    velocity: int


class MidiTokenizer:
    """Tokenizes MIDI data using true REMI representation with Bar + Position."""

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
        """Encode a single MIDI track to a token sequence."""
        if not midi.instruments or track_idx >= len(midi.instruments):
            return [self.vocab.bos_id, self.vocab.eos_id]

        instrument = midi.instruments[track_idx]
        tempo_changes = midi.get_tempo_changes()

        if len(tempo_changes[1]) > 0:
            bpm = tempo_changes[1][0]
        else:
            bpm = 120.0

        beat_duration = 60.0 / bpm

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

        events.sort(key=lambda e: (e.start_beat, e.pitch))

        if max_bars is not None:
            max_beat = max_bars * BEATS_PER_BAR
            events = [e for e in events if e.start_beat < max_beat]

        return self._events_to_tokens(events)

    def encode_note_events(self, events: list[NoteEvent]) -> list[int]:
        """Encode a list of NoteEvents directly (useful for real-time pipeline)."""
        events = sorted(events, key=lambda e: (e.start_beat, e.pitch))
        return self._events_to_tokens(events)

    def _events_to_tokens(self, events: list[NoteEvent]) -> list[int]:
        """Convert sorted NoteEvents to token IDs using Bar + Position.

        Encoding format per note:
            [<BAR>] Position_X Pitch_P Duration_D Velocity_V

        <BAR> is emitted at the start of each new bar.
        Position_X gives the 16th-note position within the current bar.
        """
        tokens = [self.vocab.bos_id]

        if not events:
            tokens.append(self.vocab.eos_id)
            return tokens

        current_bar = -1  # Track which bar we're in

        for event in events:
            # Determine bar number and position within bar
            bar_num = int(event.start_beat // BEATS_PER_BAR)
            beat_in_bar = event.start_beat % BEATS_PER_BAR

            # Emit <BAR> token at the start of each new bar
            if bar_num > current_bar:
                # Emit BAR for each bar (including skipped empty bars)
                for _ in range(bar_num - current_bar):
                    tokens.append(self.vocab.bar_id)
                current_bar = bar_num

            # Position within bar (quantized to 16th note)
            tokens.append(self.vocab.encode_position(beat_in_bar))

            # Note: Pitch -> Duration -> Velocity
            tokens.append(self.vocab.encode_pitch(event.pitch))
            tokens.append(self.vocab.encode_duration(event.duration_beats))
            tokens.append(self.vocab.encode_velocity(event.velocity))

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
        """Encode a melody-accompaniment pair with SEP token between them."""
        melody_tokens = self.encode_midi(midi, melody_track, max_bars)
        accomp_tokens = self.encode_midi(midi, accomp_track, max_bars)

        melody_body = melody_tokens[1:-1]
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
        events = self.decode_to_events(token_ids)
        return self._events_to_midi(events, bpm, program)

    def decode_to_events(self, token_ids: list[int]) -> list[NoteEvent]:
        """Decode token IDs to NoteEvents. Handles both REMI (Bar+Position) and legacy (TimeShift)."""
        events = []
        current_bar = 0
        current_beat = 0.0  # Absolute beat position
        i = 0

        while i < len(token_ids):
            tid = token_ids[i]

            # Special tokens
            if self.vocab.is_special(tid):
                if tid == self.vocab.bar_id:
                    current_bar += 1
                    current_beat = current_bar * BEATS_PER_BAR
                elif tid == self.vocab.sep_id:
                    current_bar = 0
                    current_beat = 0.0
                i += 1
                continue

            # Position token -> set absolute position within current bar
            if self.vocab.is_position(tid):
                beat_in_bar = self.vocab.decode_position(tid)
                current_beat = (current_bar * BEATS_PER_BAR) + beat_in_bar
                i += 1
                continue

            # Legacy TimeShift support
            if self.vocab.is_time_shift(tid):
                current_beat += self.vocab.decode_time_shift(tid)
                current_bar = int(current_beat // BEATS_PER_BAR)
                i += 1
                continue

            # Chord tokens (skip during playback)
            if self.vocab.is_chord(tid):
                i += 1
                continue

            # Note: expect Pitch, Duration, Velocity in sequence
            if self.vocab.is_pitch(tid):
                pitch = self.vocab.decode_pitch(tid)
                duration = DURATION_BINS[0]
                velocity = 80

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
        if self.vocab.sep_id not in token_ids:
            return token_ids, []

        sep_idx = token_ids.index(self.vocab.sep_id)
        melody = [self.vocab.bos_id] + token_ids[1:sep_idx] + [self.vocab.eos_id]
        accomp = [self.vocab.bos_id] + token_ids[sep_idx + 1:-1] + [self.vocab.eos_id]
        return melody, accomp

    def tokens_to_str(self, token_ids: list[int]) -> str:
        return " ".join(self.vocab.id_to_token.get(t, f"<UNK:{t}>") for t in token_ids)
