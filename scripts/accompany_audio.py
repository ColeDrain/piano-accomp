"""
Generate piano accompaniment for an audio file (acapella/vocal recording).

Audio file → torchcrepe pitch detection → melody notes
→ key detection → chord detection → pattern retrieval → MIDI + WAV output

Usage:
    uv run python scripts/accompany_audio.py --input vocals.wav --output accomp.wav
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

import sys, types
if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = types.ModuleType("pkg_resources")
import torchcrepe

import pretty_midi

from src.inference.pitch_detector import freq_to_midi
from src.inference.rule_chord import detect_chord_from_notes, detect_key_from_notes, PITCH_NAMES
from src.inference.pattern_retrieval import load_library, pattern_to_midi_notes
from src.tokenizer.midi_tokenizer import NoteEvent

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LIBRARY_CACHE = "data/processed/pattern_library.pkl"


def extract_melody(audio_path: str, device: str = "cpu") -> list[NoteEvent]:
    """Extract melody notes from an audio file using torchcrepe."""
    print(f"Loading {audio_path}...")
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz for torchcrepe
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    duration = waveform.shape[1] / sr
    print(f"Audio: {duration:.1f}s, {sr}Hz")

    # Run torchcrepe on the full audio
    print("Detecting pitch with torchcrepe (this may take a moment)...")
    hop_length = 160  # 10ms steps at 16kHz

    pitch_hz, confidence = torchcrepe.predict(
        waveform,
        sample_rate=sr,
        model="tiny",
        hop_length=hop_length,
        batch_size=2048,
        device=device,
        return_periodicity=True,
        pad=True,
    )

    pitch_hz = pitch_hz.squeeze().numpy()
    confidence = confidence.squeeze().numpy()

    print(f"Got {len(pitch_hz)} pitch frames")

    # Convert to note events
    # Group consecutive frames with same MIDI note into single notes
    notes = []
    current_note = None
    current_start = 0
    frame_duration = hop_length / sr  # seconds per frame

    for i in range(len(pitch_hz)):
        if confidence[i] > 0.5 and 70 < pitch_hz[i] < 1000:
            midi = freq_to_midi(pitch_hz[i])
            t = i * frame_duration

            if current_note is None:
                current_note = midi
                current_start = t
            elif abs(midi - current_note) >= 2:
                # Note changed
                dur = t - current_start
                if dur > 0.05:  # Min 50ms
                    notes.append(NoteEvent(
                        start_beat=current_start,
                        pitch=current_note,
                        duration_beats=dur,
                        velocity=80,
                    ))
                current_note = midi
                current_start = t
        else:
            # Silence/unvoiced
            if current_note is not None:
                dur = i * frame_duration - current_start
                if dur > 0.05:
                    notes.append(NoteEvent(
                        start_beat=current_start,
                        pitch=current_note,
                        duration_beats=dur,
                        velocity=80,
                    ))
                current_note = None

    # Final note
    if current_note is not None:
        dur = len(pitch_hz) * frame_duration - current_start
        if dur > 0.05:
            notes.append(NoteEvent(current_start, current_note, dur, 80))

    print(f"Extracted {len(notes)} melody notes")
    return notes, duration


def generate_accompaniment(
    notes: list[NoteEvent],
    duration: float,
    library,
    bpm: float = 80.0,
) -> pretty_midi.PrettyMIDI:
    """Generate accompaniment for extracted melody notes."""

    # Detect key from all notes
    key_root = detect_key_from_notes(notes)
    print(f"Detected key: {PITCH_NAMES[key_root]} major")

    major_scale = [(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
    print(f"Diatonic roots: {[PITCH_NAMES[r] for r in major_scale]}")

    NOTE_NAMES_LOCAL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Estimate BPM from note density
    if len(notes) > 4:
        avg_gap = np.mean([notes[i+1].start_beat - notes[i].start_beat
                          for i in range(min(20, len(notes)-1))])
        if avg_gap > 0:
            estimated_bpm = 60.0 / avg_gap * 2  # Rough estimate
            bpm = max(60, min(140, estimated_bpm))
    print(f"Using BPM: {bpm:.0f}")

    beat_dur = 60.0 / bpm

    # Build a clean melody line: one dominant note per 2-second window
    # This eliminates pitch detection noise
    window_sec = 2.0
    clean_melody = []
    t = 0.0
    while t < duration:
        window_notes = [n for n in notes if t <= n.start_beat < t + window_sec]
        if window_notes:
            # Pick the most common pitch class (weighted by duration)
            from collections import Counter
            pc_weights = Counter()
            for n in window_notes:
                pc_weights[n.pitch % 12] += n.duration_beats
            dominant_pc = pc_weights.most_common(1)[0][0]

            # Pick the actual note closest to median pitch with that PC
            matching = [n for n in window_notes if n.pitch % 12 == dominant_pc]
            if matching:
                median_pitch = int(np.median([n.pitch for n in matching]))
                clean_melody.append(NoteEvent(t, median_pitch, window_sec, 80))
        t += window_sec

    print(f"Clean melody: {len(clean_melody)} notes")
    for n in clean_melody[:20]:
        name = f"{NOTE_NAMES_LOCAL[n.pitch % 12]}{n.pitch // 12 - 1}"
        print(f"  {n.start_beat:.0f}s: {name}")

    # Now generate chords from the clean melody
    # Use a sliding window of 3-4 notes for chord detection
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    acc_inst = pretty_midi.Instrument(program=0, name="Piano")

    prev_pattern = None
    prev_root = -1

    for i, note in enumerate(clean_melody):
        # Use this note + surrounding context for chord detection
        context_start = max(0, i - 2)
        context_end = min(len(clean_melody), i + 2)
        context = clean_melody[context_start:context_end]

        root, triad, quality = detect_chord_from_notes(
            context, key_root=key_root, prefer_diatonic=True
        )

        # Force diatonic
        if root not in major_scale:
            dists = [((root - r) % 12 if (root - r) % 12 <= 6 else 12 - (root - r) % 12, r)
                     for r in major_scale]
            root = min(dists)[1]
            triad = 0  # Default to major when snapping

        chord_str = f"{PITCH_NAMES[root]} {quality}"
        if root != prev_root:
            print(f"  {note.start_beat:.0f}s: {chord_str}")
            prev_root = root

        pattern = library.retrieve(
            target_root=root,
            target_triad=triad,
            prev_pattern=prev_pattern,
            prefer_bass=True,
            prefer_gospel=True,
            min_notes=4,
            max_notes=15,
        )

        if pattern:
            prev_pattern = pattern
            offset_beat = note.start_beat / beat_dur
            pattern_notes = pattern_to_midi_notes(pattern, offset_beat, bpm)
            acc_inst.notes.extend(pattern_notes)

    midi.instruments.append(acc_inst)
    print(f"Generated {len(acc_inst.notes)} accompaniment notes")
    return midi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Vocal audio file (WAV, MP3, etc.)")
    parser.add_argument("--output", type=str, default="accompaniment.wav")
    parser.add_argument("--bpm", type=float, default=80.0)
    parser.add_argument("--soundfont", type=str, default="soundfonts/piano.sf2")
    args = parser.parse_args()

    library = load_library(LIBRARY_CACHE)

    # Extract melody
    notes, duration = extract_melody(args.input)

    if not notes:
        print("No melody detected in audio!")
        return

    # Generate accompaniment MIDI
    midi = generate_accompaniment(notes, duration, library, bpm=args.bpm)

    # Save MIDI
    midi_path = args.output.replace(".wav", ".mid")
    midi.write(midi_path)
    print(f"Saved MIDI: {midi_path}")

    # Render to WAV with FluidSynth
    import subprocess
    subprocess.run(
        ["fluidsynth", "-ni", "-F", "accomp_piano.wav", args.soundfont, midi_path],
        capture_output=True,
    )

    # Mix vocal + piano
    print("Mixing vocal + piano...")
    vocal_wav, vocal_sr = torchaudio.load(args.input)
    if vocal_wav.shape[0] > 1:
        vocal_wav = vocal_wav.mean(dim=0, keepdim=True)

    piano_wav, piano_sr = torchaudio.load("accomp_piano.wav")
    if piano_wav.shape[0] > 1:
        piano_wav = piano_wav.mean(dim=0, keepdim=True)

    # Resample piano to match vocal sample rate
    if piano_sr != vocal_sr:
        piano_wav = torchaudio.functional.resample(piano_wav, piano_sr, vocal_sr)

    # Match lengths
    max_len = max(vocal_wav.shape[1], piano_wav.shape[1])
    if vocal_wav.shape[1] < max_len:
        vocal_wav = torch.nn.functional.pad(vocal_wav, (0, max_len - vocal_wav.shape[1]))
    if piano_wav.shape[1] < max_len:
        piano_wav = torch.nn.functional.pad(piano_wav, (0, max_len - piano_wav.shape[1]))

    # Mix: vocal at full volume, piano at 150% (louder accompaniment)
    mixed = vocal_wav + piano_wav * 1.5

    # Normalize to prevent clipping
    peak = mixed.abs().max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    torchaudio.save(args.output, mixed, vocal_sr)
    print(f"\n✓ Saved: {args.output} (vocal + piano mixed)")
    print(f"  Play with: afplay {args.output}")


if __name__ == "__main__":
    main()
