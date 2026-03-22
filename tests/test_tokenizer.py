"""Tests for the true REMI MIDI tokenizer and vocabulary."""

import pytest
from src.tokenizer.vocab import Vocabulary, DURATION_BINS, PITCH_NAMES, CHORD_QUALITIES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent


@pytest.fixture
def vocab():
    return Vocabulary()


@pytest.fixture
def tokenizer(vocab):
    return MidiTokenizer(vocab)


class TestVocabulary:
    def test_vocab_size(self, vocab):
        assert vocab.size > 0
        assert vocab.size == len(vocab.id_to_token)
        assert vocab.size == len(vocab.token_to_id)

    def test_special_tokens(self, vocab):
        assert vocab.pad_id == 0
        assert vocab.bos_id == 1
        assert vocab.eos_id == 2
        assert vocab.sep_id == 3
        assert vocab.bar_id == 4

    def test_position_encoding(self, vocab):
        # Beat 0 (downbeat) = Position_0
        tid = vocab.encode_position(0.0)
        assert vocab.is_position(tid)
        assert vocab.decode_position(tid) == 0.0

        # Beat 1 = Position_4
        tid = vocab.encode_position(1.0)
        assert vocab.decode_position(tid) == 1.0

        # Beat 2.5 = Position_10
        tid = vocab.encode_position(2.5)
        assert abs(vocab.decode_position(tid) - 2.5) < 0.01

    def test_pitch_encoding_roundtrip(self, vocab):
        for pitch in [0, 60, 127]:
            token_id = vocab.encode_pitch(pitch)
            assert vocab.decode_pitch(token_id) == pitch
            assert vocab.is_pitch(token_id)

    def test_duration_encoding_roundtrip(self, vocab):
        for dur in DURATION_BINS:
            token_id = vocab.encode_duration(dur)
            decoded = vocab.decode_duration(token_id)
            assert abs(decoded - dur) < 0.01
            assert vocab.is_duration(token_id)

    def test_velocity_encoding_roundtrip(self, vocab):
        token_id = vocab.encode_velocity(80)
        decoded = vocab.decode_velocity(token_id)
        assert abs(decoded - 80) <= 4
        assert vocab.is_velocity(token_id)

    def test_chord_encoding_roundtrip(self, vocab):
        token_id = vocab.encode_chord("C", "maj7")
        root, quality = vocab.decode_chord(token_id)
        assert root == "C"
        assert quality == "maj7"
        assert vocab.is_chord(token_id)

    def test_all_chord_types_encoded(self, vocab):
        for root in PITCH_NAMES:
            for quality in CHORD_QUALITIES:
                token_id = vocab.encode_chord(root, quality)
                assert vocab.is_chord(token_id)

    def test_no_overlap_between_ranges(self, vocab):
        for tid in range(vocab.size):
            categories = [
                vocab.is_special(tid),
                vocab.is_position(tid),
                vocab.is_pitch(tid),
                vocab.is_duration(tid),
                vocab.is_velocity(tid),
                vocab.is_time_shift(tid),
                vocab.is_chord(tid),
            ]
            assert sum(categories) == 1, f"Token {tid} belongs to {sum(categories)} categories"


class TestMidiTokenizer:
    def test_encode_emits_bar_and_position(self, tokenizer, vocab):
        """True REMI should emit BAR + Position tokens, not TimeShift."""
        events = [
            NoteEvent(start_beat=0.0, pitch=60, duration_beats=1.0, velocity=80),
            NoteEvent(start_beat=1.0, pitch=64, duration_beats=0.5, velocity=90),
        ]
        tokens = tokenizer.encode_note_events(events)

        # Should contain BAR token
        assert vocab.bar_id in tokens
        # Should contain Position tokens
        has_position = any(vocab.is_position(t) for t in tokens)
        assert has_position
        # Should NOT contain TimeShift tokens (true REMI uses Position instead)
        has_timeshift = any(vocab.is_time_shift(t) for t in tokens)
        assert not has_timeshift

    def test_roundtrip_single_note(self, tokenizer):
        events = [NoteEvent(start_beat=0.0, pitch=60, duration_beats=1.0, velocity=80)]
        tokens = tokenizer.encode_note_events(events)
        decoded = tokenizer.decode_to_events(tokens)

        assert len(decoded) == 1
        assert decoded[0].pitch == 60
        assert abs(decoded[0].duration_beats - 1.0) < 0.1

    def test_roundtrip_multiple_notes(self, tokenizer):
        events = [
            NoteEvent(start_beat=0.0, pitch=60, duration_beats=1.0, velocity=80),
            NoteEvent(start_beat=1.0, pitch=64, duration_beats=0.5, velocity=90),
            NoteEvent(start_beat=2.0, pitch=67, duration_beats=1.0, velocity=70),
        ]
        tokens = tokenizer.encode_note_events(events)
        decoded = tokenizer.decode_to_events(tokens)

        assert len(decoded) == 3
        for orig, dec in zip(events, decoded):
            assert orig.pitch == dec.pitch

    def test_bar_boundaries(self, tokenizer, vocab):
        """Notes in different bars should have separate BAR tokens."""
        events = [
            NoteEvent(start_beat=0.0, pitch=60, duration_beats=1.0, velocity=80),   # Bar 0
            NoteEvent(start_beat=4.0, pitch=64, duration_beats=1.0, velocity=80),   # Bar 1
            NoteEvent(start_beat=8.0, pitch=67, duration_beats=1.0, velocity=80),   # Bar 2
        ]
        tokens = tokenizer.encode_note_events(events)
        bar_count = sum(1 for t in tokens if t == vocab.bar_id)
        assert bar_count == 3  # Bar 0, Bar 1, Bar 2

    def test_pair_encoding_split(self, tokenizer, vocab):
        events = [NoteEvent(0.0, 60, 1.0, 80)]
        mel_tokens = tokenizer.encode_note_events(events)
        acc_tokens = tokenizer.encode_note_events(events)

        pair = [vocab.bos_id] + mel_tokens[1:-1] + [vocab.sep_id] + acc_tokens[1:-1] + [vocab.eos_id]
        mel, acc = tokenizer.split_pair(pair)

        assert mel[0] == vocab.bos_id
        assert mel[-1] == vocab.eos_id
        assert acc[0] == vocab.bos_id
        assert acc[-1] == vocab.eos_id

    def test_empty_events(self, tokenizer, vocab):
        tokens = tokenizer.encode_note_events([])
        assert tokens == [vocab.bos_id, vocab.eos_id]
