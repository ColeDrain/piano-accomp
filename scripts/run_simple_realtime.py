"""
Real-time accompaniment with torchcrepe + pattern retrieval.

Mic → torchcrepe pitch detection → chord detection → retrieve POP909 pattern → FluidSynth.
"""

import time
import numpy as np
import sounddevice as sd
import torch

import sys, types
if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = types.ModuleType("pkg_resources")
import torchcrepe

from src.inference.pitch_detector import freq_to_midi
from src.inference.synthesizer import Synthesizer
from src.inference.rule_chord import detect_chord_from_notes, detect_key_from_notes, PITCH_NAMES
from src.inference.pattern_retrieval import load_library, pattern_to_midi_notes
from src.tokenizer.midi_tokenizer import NoteEvent
from pathlib import Path

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LIBRARY_CACHE = "data/processed/pattern_library.pkl"


def main():
    print("Real-Time Piano Accompaniment")
    print("=" * 50)

    # Load pattern library
    library = load_library(LIBRARY_CACHE)

    synth = Synthesizer(soundfont_path="soundfonts/piano.sf2")
    synth.start()
    print("FluidSynth started")

    sample_rate = 16000
    chunk_seconds = 0.5
    chunk_size = int(sample_rate * chunk_seconds)

    recent_pitches: list[NoteEvent] = []
    current_chord = ""
    current_root = -1
    current_triad = -1
    key_root = 0
    last_chord_time = 0.0
    prev_pattern = None
    playing_notes: list[int] = []
    chord_hold_count = 0  # How many times in a row same chord was detected

    print("\n🎤 Sing a few bars so I can detect your key...")
    print("   (I'll listen for 8 seconds, then start accompanying)\n")

    # Phase 1: Listen for 8 seconds to detect key
    key_notes = []
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                       blocksize=chunk_size) as stream:
        start = time.time()
        while time.time() - start < 8.0:
            audio, _ = stream.read(chunk_size)
            audio = audio[:, 0]
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.02:
                continue

            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            pitch_hz, confidence = torchcrepe.predict(
                audio_tensor, sample_rate=sample_rate, model="tiny",
                hop_length=160, batch_size=1, device="cpu",
                return_periodicity=True, pad=False,
            )
            pitch_hz = pitch_hz.squeeze()
            confidence = confidence.squeeze()
            if pitch_hz.dim() == 0:
                pitch_hz = pitch_hz.unsqueeze(0)
                confidence = confidence.unsqueeze(0)

            mask = confidence > 0.6
            if mask.sum() == 0:
                continue

            median_freq = pitch_hz[mask].median().item()
            if 70 < median_freq < 1000:
                midi_note = freq_to_midi(median_freq)
                note_name = f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"
                print(f"  ♪ {note_name}", end="", flush=True)
                key_notes.append(NoteEvent(0, midi_note, 0.5, 80))

    if key_notes:
        key_root = detect_key_from_notes(key_notes)
        print(f"\n\n✓ Detected key: {PITCH_NAMES[key_root]} major")
    else:
        key_root = 0
        print(f"\n\n⚠ No singing detected, defaulting to C major")

    # Build diatonic chords for this key
    major_scale = [(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
    print(f"  Diatonic roots: {[PITCH_NAMES[r] for r in major_scale]}")
    print(f"\n🎹 Now accompanying! Sing and I'll follow. Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                           blocksize=chunk_size) as stream:
            while True:
                audio, overflowed = stream.read(chunk_size)
                audio = audio[:, 0]

                rms = np.sqrt(np.mean(audio ** 2))
                if rms < 0.02:
                    time.sleep(0.01)
                    continue

                # torchcrepe pitch detection
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                pitch_hz, confidence = torchcrepe.predict(
                    audio_tensor,
                    sample_rate=sample_rate,
                    model="tiny",
                    hop_length=160,
                    batch_size=1,
                    device="cpu",
                    return_periodicity=True,
                    pad=False,
                )

                pitch_hz = pitch_hz.squeeze()
                confidence = confidence.squeeze()
                if pitch_hz.dim() == 0:
                    pitch_hz = pitch_hz.unsqueeze(0)
                    confidence = confidence.unsqueeze(0)

                mask = confidence > 0.6
                if mask.sum() == 0:
                    continue

                median_freq = pitch_hz[mask].median().item()
                if median_freq < 70 or median_freq > 1000:
                    continue

                midi_note = freq_to_midi(median_freq)
                note_name = f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"

                recent_pitches.append(NoteEvent(
                    start_beat=time.monotonic(),
                    pitch=midi_note,
                    duration_beats=0.5,
                    velocity=80,
                ))
                if len(recent_pitches) > 10:
                    recent_pitches = recent_pitches[-10:]

                print(f"  ♪ {note_name}", end="", flush=True)

                # Update chord every 1 second for responsiveness
                now = time.monotonic()
                if now - last_chord_time >= 1.0 and len(recent_pitches) >= 2:
                    last_chord_time = now

                    # Key is locked from initial detection — don't re-detect
                    root, triad, quality = detect_chord_from_notes(
                        recent_pitches, key_root=key_root, prefer_diatonic=True
                    )

                    # Stabilize: only change chord if the new root is different
                    # AND it's been detected consistently (not a passing note)
                    if root == current_root and triad == current_triad:
                        chord_hold_count += 1
                        continue  # Same chord, don't re-trigger

                    # New chord detected — but only switch if we have enough evidence
                    # (at least 2 detections or first chord)
                    if current_root == -1:
                        # First chord — accept immediately
                        pass
                    else:
                        # Check if the new root is diatonic to the key
                        major_scale = [(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
                        if root not in major_scale:
                            continue  # Reject non-diatonic chord changes

                    current_root = root
                    current_triad = triad
                    chord_hold_count = 0

                    chord_str = f"{PITCH_NAMES[root]} {quality}"
                    current_chord = chord_str
                    print(f"\n  → {chord_str}", flush=True)

                    # Retrieve a matching pattern
                    pattern = library.retrieve(
                        target_root=root,
                        target_triad=triad,
                        prev_pattern=prev_pattern,
                        prefer_bass=True,
                        min_notes=4,
                        max_notes=15,
                    )

                    if pattern:
                        prev_pattern = pattern

                        # Stop previous notes
                        for p in playing_notes:
                            synth.note_off(p)
                        playing_notes.clear()

                        # Play pattern notes as a chord voicing
                        unique_pitches = list(set(n[2] for n in pattern.notes))
                        unique_pitches.sort()
                        if len(unique_pitches) > 6:
                            bass = unique_pitches[:2]
                            treble = unique_pitches[-3:]
                            mid = [unique_pitches[len(unique_pitches)//2]]
                            unique_pitches = bass + mid + treble

                        for pitch in unique_pitches:
                            vel = 55 if pitch < 48 else 70
                            synth.note_on(pitch, vel)
                            playing_notes.append(pitch)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        for p in playing_notes:
            synth.note_off(p)
        synth.all_notes_off()
        synth.stop()
        print("Done.")


if __name__ == "__main__":
    main()
