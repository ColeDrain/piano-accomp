"""
Accompaniment using rule-based chords + pattern retrieval.

No ML at all — pure music theory + real piano patterns from POP909.
This proves whether the pipeline works when given correct chords.

Usage:
    uv run python scripts/run_retrieval_rulechord.py \
        --melody test_long_melody.mid \
        --output test_rulechord.mid
"""

import argparse
from pathlib import Path

import pretty_midi

from src.tokenizer.vocab import PITCH_NAMES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.inference.rule_chord import detect_chord_from_notes, detect_key_from_notes
from src.inference.pattern_retrieval import (
    PatternLibrary, build_library_from_pop909,
    save_library, load_library, pattern_to_midi_notes,
)


LIBRARY_CACHE = "data/processed/pattern_library.pkl"
TRIAD_NAMES = ["maj", "min", "dim", "aug", "sus"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--melody", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_rulechord.mid")
    parser.add_argument("--pop909-dir", type=str, default="data/raw/pop909")
    args = parser.parse_args()

    # Load or build pattern library
    if Path(LIBRARY_CACHE).exists():
        library = load_library(LIBRARY_CACHE)
    else:
        print("Building pattern library...")
        library = build_library_from_pop909(args.pop909_dir)
        Path(LIBRARY_CACHE).parent.mkdir(parents=True, exist_ok=True)
        save_library(library, LIBRARY_CACHE)

    # Load melody
    melody_midi = pretty_midi.PrettyMIDI(args.melody)
    bpm = 80.0
    tempos = melody_midi.get_tempo_changes()
    if len(tempos[1]) > 0:
        bpm = tempos[1][0]
    beat_dur = 60.0 / bpm

    melody_notes = melody_midi.instruments[0].notes
    max_beat = max(n.end / beat_dur for n in melody_notes)

    # Convert to NoteEvents for analysis
    all_events = [
        NoteEvent(n.start / beat_dur, n.pitch, (n.end - n.start) / beat_dur, n.velocity)
        for n in melody_notes
    ]

    # Detect key from entire melody
    key_root = detect_key_from_notes(all_events)
    print(f"Detected key: {PITCH_NAMES[key_root]} major")
    print(f"Melody: {len(melody_notes)} notes, {max_beat:.1f} beats at {bpm} BPM")

    # Build output
    output = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Melody track — use flute so it's distinguishable
    mel_inst = pretty_midi.Instrument(program=73, name="Vocal (Flute)")
    mel_inst.notes = list(melody_notes)
    output.instruments.append(mel_inst)

    # Accompaniment track
    acc_inst = pretty_midi.Instrument(program=0, name="Piano")

    prev_pattern = None
    measure = 0

    while measure * 4 < max_beat:
        start_beat = measure * 4
        end_beat = start_beat + 4

        # Get melody notes in this measure
        measure_events = [
            NoteEvent(
                (n.start / beat_dur) - start_beat,
                n.pitch,
                (n.end - n.start) / beat_dur,
                n.velocity,
            )
            for n in melody_notes
            if start_beat * beat_dur <= n.start < end_beat * beat_dur
        ]

        if measure_events:
            # Rule-based chord detection
            root, triad, quality = detect_chord_from_notes(
                measure_events, key_root=key_root, prefer_diatonic=True
            )

            print(f"  Measure {measure+1}: {PITCH_NAMES[root]} {quality}")

            # Retrieve matching pattern
            pattern = library.retrieve(
                target_root=root,
                target_triad=triad,
                prev_pattern=prev_pattern,
                prefer_bass=True,
            )

            if pattern:
                notes = pattern_to_midi_notes(pattern, start_beat, bpm)
                acc_inst.notes.extend(notes)
                prev_pattern = pattern
        else:
            print(f"  Measure {measure+1}: (rest)")

        measure += 1

    output.instruments.append(acc_inst)
    output.write(args.output)
    print(f"\nSaved: {args.output} ({len(acc_inst.notes)} accompaniment notes)")


if __name__ == "__main__":
    main()
