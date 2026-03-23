"""
Evaluate model quality with objective metrics and multiple test cases.

Tests:
1. Harmonic compatibility — are generated notes in the right key/chord?
2. Register analysis — are notes in piano range? Left/right hand separation?
3. Rhythmic density — reasonable number of notes per beat?
4. Multiple melodies — test across different keys and styles

Usage:
    uv run python scripts/evaluate.py \
        --chord-checkpoint checkpoints/chord_best.pt \
        --texture-checkpoint checkpoints/texture_best.pt
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pretty_midi
import torch
import yaml

from src.tokenizer.vocab import Vocabulary, PITCH_NAMES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator


# --- Test melodies ---

def make_c_major_scale():
    """C major scale — should produce C/Am type chords."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    inst = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        inst.notes.append(pretty_midi.Note(80, pitch, i * 0.75, i * 0.75 + 0.7))
    midi.instruments.append(inst)
    return midi, "C major scale"


def make_a_minor_melody():
    """A minor melody — should produce Am/Dm/Em type chords."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    inst = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate([69, 72, 71, 69, 67, 69, 71, 72]):
        inst.notes.append(pretty_midi.Note(80, pitch, i * 0.75, i * 0.75 + 0.7))
    midi.instruments.append(inst)
    return midi, "A minor melody"


def make_gospel_progression_melody():
    """Melody implying I-IV-V-I in C — classic gospel."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    inst = pretty_midi.Instrument(program=0)
    # I chord notes (C E G), IV chord notes (F A C), V chord notes (G B D), I
    notes = [
        (64, 0), (67, 0.75), (72, 1.5),   # I: E G C
        (65, 2.25), (69, 3.0), (72, 3.75), # IV: F A C
        (67, 4.5), (71, 5.25), (74, 6.0),  # V: G B D
        (72, 6.75), (67, 7.5), (64, 8.25), # I: C G E
    ]
    for pitch, start in notes:
        inst.notes.append(pretty_midi.Note(80, pitch, start, start + 0.7))
    midi.instruments.append(inst)
    return midi, "Gospel I-IV-V-I"


def make_repeated_note():
    """Single repeated note — stress test, should still produce varied accompaniment."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    inst = pretty_midi.Instrument(program=0)
    for i in range(8):
        inst.notes.append(pretty_midi.Note(80, 60, i * 0.75, i * 0.75 + 0.5))
    midi.instruments.append(inst)
    return midi, "Repeated C4"


def make_chromatic():
    """Chromatic run — should be challenging for the model."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    inst = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate(range(60, 73)):
        inst.notes.append(pretty_midi.Note(80, pitch, i * 0.5, i * 0.5 + 0.45))
    midi.instruments.append(inst)
    return midi, "Chromatic run"


# --- Metrics ---

def analyze_output(melody_midi, accomp_midi, name):
    """Analyze generated accompaniment quality."""
    melody_notes = melody_midi.instruments[0].notes
    accomp_notes = accomp_midi.instruments[1].notes if len(accomp_midi.instruments) > 1 else []

    if not accomp_notes:
        print(f"\n{'='*50}")
        print(f"  {name}: NO ACCOMPANIMENT GENERATED")
        print(f"{'='*50}")
        return

    mel_pitches = [n.pitch for n in melody_notes]
    acc_pitches = [n.pitch for n in accomp_notes]
    acc_durations = [n.end - n.start for n in accomp_notes]

    # Key analysis: what pitch classes are used?
    mel_pcs = Counter([p % 12 for p in mel_pitches])
    acc_pcs = Counter([p % 12 for p in acc_pitches])

    # C major / A minor diatonic pitch classes
    c_major_pcs = {0, 2, 4, 5, 7, 9, 11}  # C D E F G A B
    acc_diatonic = sum(1 for p in acc_pitches if p % 12 in c_major_pcs)
    diatonic_ratio = acc_diatonic / len(acc_pitches) if acc_pitches else 0

    # Pitch overlap with melody (should be some but not too much)
    overlap = len(set(acc_pitches) & set(mel_pitches))

    # Register analysis
    bass_notes = [p for p in acc_pitches if p < 55]  # Below G3
    mid_notes = [p for p in acc_pitches if 55 <= p < 72]  # G3 to C5
    treble_notes = [p for p in acc_pitches if p >= 72]  # C5 and above

    # Density: notes per beat
    if accomp_notes:
        total_time = max(n.end for n in accomp_notes) - min(n.start for n in accomp_notes)
        notes_per_sec = len(accomp_notes) / max(total_time, 0.1)
    else:
        notes_per_sec = 0

    # Print results
    pc_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    top_pcs = acc_pcs.most_common(5)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accompaniment: {len(accomp_notes)} notes")
    print(f"  Pitch range: {min(acc_pitches)} ({pretty_midi.note_number_to_name(min(acc_pitches))}) "
          f"- {max(acc_pitches)} ({pretty_midi.note_number_to_name(max(acc_pitches))})")
    print(f"  Register: {len(bass_notes)} bass / {len(mid_notes)} mid / {len(treble_notes)} treble")
    print(f"  Diatonic ratio: {diatonic_ratio:.1%} (notes in C major/A minor)")
    print(f"  Top pitch classes: {', '.join(f'{pc_names[pc]}({c})' for pc, c in top_pcs)}")
    print(f"  Avg duration: {np.mean(acc_durations):.2f}s")
    print(f"  Notes/sec: {notes_per_sec:.1f}")
    print(f"  Melody overlap: {overlap} shared pitches")

    # Verdict
    issues = []
    if diatonic_ratio < 0.7:
        issues.append(f"LOW DIATONIC ({diatonic_ratio:.0%}) — notes outside key")
    if len(accomp_notes) < 3:
        issues.append("TOO SPARSE — fewer than 3 notes")
    if len(accomp_notes) > 50:
        issues.append("TOO DENSE — more than 50 notes")
    if max(acc_pitches) - min(acc_pitches) < 12:
        issues.append("NARROW RANGE — less than 1 octave")
    if not bass_notes:
        issues.append("NO BASS — missing left hand")
    if notes_per_sec > 15:
        issues.append("MACHINE GUN — too many notes per second")

    if issues:
        print(f"  ⚠ Issues: {'; '.join(issues)}")
    else:
        print(f"  ✓ All checks passed")

    return {
        "name": name,
        "num_notes": len(accomp_notes),
        "diatonic_ratio": diatonic_ratio,
        "range": max(acc_pitches) - min(acc_pitches),
        "notes_per_sec": notes_per_sec,
        "has_bass": len(bass_notes) > 0,
        "issues": issues,
    }


def generate_and_evaluate(melody_midi, chord_model, texture_model, tokenizer, vocab, device, name, temperature=0.8):
    """Generate accompaniment and evaluate it."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_offline_demo import generate_accompaniment

    output = generate_accompaniment(
        melody_midi, chord_model, texture_model, tokenizer, vocab, device,
        temperature=temperature,
    )

    return analyze_output(melody_midi, output, name), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chord-checkpoint", type=str, required=True)
    parser.add_argument("--texture-checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--render", action="store_true", help="Render WAVs with FluidSynth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_offline_demo import load_models
    chord_model, texture_model = load_models(args, vocab, device)

    test_cases = [
        make_c_major_scale(),
        make_a_minor_melody(),
        make_gospel_progression_melody(),
        make_repeated_note(),
        make_chromatic(),
    ]

    print(f"\nEvaluating with temperature={args.temperature} on {device}")
    print(f"Texture checkpoint: {args.texture_checkpoint}")

    results = []
    for melody_midi, name in test_cases:
        result, output = generate_and_evaluate(
            melody_midi, chord_model, texture_model, tokenizer, vocab, device, name,
            temperature=args.temperature,
        )
        results.append(result)

        if args.render:
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            midi_path = f"eval_{safe_name}.mid"
            wav_path = f"eval_{safe_name}.wav"
            output.write(midi_path)

            import subprocess
            subprocess.run(
                ["fluidsynth", "-ni", "-F", wav_path, "soundfonts/piano.sf2", midi_path],
                capture_output=True,
            )
            print(f"  Rendered: {wav_path}")

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    valid = [r for r in results if r]
    avg_diatonic = np.mean([r["diatonic_ratio"] for r in valid])
    avg_notes = np.mean([r["num_notes"] for r in valid])
    total_issues = sum(len(r["issues"]) for r in valid)
    has_bass_pct = sum(1 for r in valid if r["has_bass"]) / len(valid)

    print(f"  Avg diatonic ratio: {avg_diatonic:.1%}")
    print(f"  Avg notes generated: {avg_notes:.0f}")
    print(f"  Bass notes present: {has_bass_pct:.0%} of tests")
    print(f"  Total issues: {total_issues}")

    if avg_diatonic > 0.8 and total_issues <= 2:
        print(f"\n  ✓ Model quality: GOOD")
    elif avg_diatonic > 0.6:
        print(f"\n  ~ Model quality: FAIR — some out-of-key notes")
    else:
        print(f"\n  ✗ Model quality: POOR — significant harmonic issues")


if __name__ == "__main__":
    main()
