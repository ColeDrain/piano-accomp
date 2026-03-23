"""
Generate accompaniment using pattern retrieval instead of neural generation.

Melody → Chord Predictor → Pattern Retrieval → MIDI output

Usage:
    uv run python scripts/run_retrieval_demo.py \
        --melody test_melody.mid \
        --output test_retrieval.mid \
        --chord-checkpoint checkpoints/chord_best.pt
"""

import argparse
from pathlib import Path

import pretty_midi
import torch
import yaml

from src.tokenizer.vocab import Vocabulary, PITCH_NAMES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.model.chord_predictor import ChordPredictor, NUM_ROOTS, NUM_TRIADS
from src.inference.pattern_retrieval import (
    PatternLibrary, build_library_from_pop909,
    save_library, load_library, pattern_to_midi_notes,
)


LIBRARY_CACHE = "data/processed/pattern_library.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--melody", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_retrieval.mid")
    parser.add_argument("--chord-checkpoint", type=str, required=True)
    parser.add_argument("--pop909-dir", type=str, default="data/raw/pop909")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    # Load or build pattern library
    if Path(LIBRARY_CACHE).exists():
        library = load_library(LIBRARY_CACHE)
    else:
        print("Building pattern library from POP909 (first time only)...")
        library = build_library_from_pop909(args.pop909_dir)
        Path(LIBRARY_CACHE).parent.mkdir(parents=True, exist_ok=True)
        save_library(library, LIBRARY_CACHE)

    # Load chord predictor
    with open("configs/model.yaml") as f:
        chord_cfg = yaml.safe_load(f)["chord_predictor"]

    chord_model = ChordPredictor(
        vocab_size=vocab.size,
        embed_dim=chord_cfg["embed_dim"],
        num_layers=chord_cfg["num_layers"],
        num_heads=chord_cfg["num_heads"],
        ffn_dim=chord_cfg["ffn_dim"],
        max_melody_tokens=chord_cfg["max_melody_tokens"],
        dropout=0.0,
    ).to(device)
    chord_model.load_state_dict(torch.load(args.chord_checkpoint, map_location=device))
    chord_model.eval()
    print(f"Loaded chord predictor on {device}")

    # Load melody
    melody_midi = pretty_midi.PrettyMIDI(args.melody)
    bpm = 80.0
    tempos = melody_midi.get_tempo_changes()
    if len(tempos[1]) > 0:
        bpm = tempos[1][0]
    beat_dur = 60.0 / bpm

    melody_notes = melody_midi.instruments[0].notes
    max_beat = max(n.end / beat_dur for n in melody_notes)

    print(f"Melody: {len(melody_notes)} notes, {max_beat:.1f} beats at {bpm} BPM")

    # Generate accompaniment measure by measure
    output = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Melody track
    mel_inst = pretty_midi.Instrument(program=0, name="Melody")
    mel_inst.notes = list(melody_notes)
    output.instruments.append(mel_inst)

    # Accompaniment track
    acc_inst = pretty_midi.Instrument(program=0, name="Accompaniment")

    prev_pattern = None
    measure = 0
    triad_names = ["maj", "min", "dim", "aug", "sus"]

    while measure * 4 < max_beat:
        start_beat = measure * 4
        end_beat = start_beat + 4

        # Get melody notes in this measure
        window_notes = [
            n for n in melody_notes
            if start_beat * beat_dur <= n.start < end_beat * beat_dur
        ]

        if window_notes:
            # Tokenize melody
            mel_events = [
                NoteEvent(
                    start_beat=(n.start / beat_dur) - start_beat,
                    pitch=n.pitch,
                    duration_beats=(n.end - n.start) / beat_dur,
                    velocity=n.velocity,
                )
                for n in window_notes
            ]
            mel_tokens = tokenizer.encode_note_events(mel_events)[:32]
            mel_tensor = torch.tensor([mel_tokens], device=device)

            # Predict chord
            with torch.no_grad():
                result = chord_model(mel_tensor)
                root = result["root_logits"].argmax(dim=-1).item()
                triad = result["triad_logits"].argmax(dim=-1).item()

            print(f"  Measure {measure+1}: {PITCH_NAMES[root]} {triad_names[triad]}")

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
            print(f"  Measure {measure+1}: (no melody)")

        measure += 1

    output.instruments.append(acc_inst)
    output.write(args.output)
    print(f"\nSaved: {args.output} ({len(acc_inst.notes)} accompaniment notes)")


if __name__ == "__main__":
    main()
