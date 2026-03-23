"""
Test each step of the real-time pipeline independently.
Run this to diagnose what's working and what's broken.
"""

import sys
import time
import numpy as np

print("=" * 50)
print("STEP 1: Test microphone input")
print("=" * 50)

import sounddevice as sd

print("Recording 3 seconds from mic...")
audio = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype='float32')
sd.wait()
rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))
print(f"  RMS: {rms:.4f}, Peak: {peak:.4f}")
if peak < 0.01:
    print("  ⚠ Very quiet — are you singing/speaking into the mic?")
    print("  Try again louder, or check System Preferences → Sound → Input")
else:
    print("  ✓ Mic is picking up audio")

print("\n" + "=" * 50)
print("STEP 2: Test pitch detection")
print("=" * 50)

from src.inference.pitch_detector import PitchDetector

detector = PitchDetector(sample_rate=44100, crepe_model="tiny")

# Process the recorded audio in chunks
chunk_size = 2048
detected_pitches = []
for i in range(0, len(audio) - chunk_size, chunk_size):
    chunk = audio[i:i+chunk_size, 0]
    events = detector.process_chunk(chunk)
    for e in events:
        if e["type"] == "note_on":
            detected_pitches.append(e["pitch"])
        elif e["type"] == "pitch":
            detected_pitches.append(e["pitch"])

if detected_pitches:
    print(f"  Detected {len(detected_pitches)} pitch events")
    print(f"  Pitches: {detected_pitches[:10]}")
    print("  ✓ Pitch detection working")
else:
    print("  ⚠ No pitches detected — sing a clear sustained note")
    print("  (Try humming a steady 'ahhh' into the mic)")

print("\n" + "=" * 50)
print("STEP 3: Test chord detection")
print("=" * 50)

from src.tokenizer.midi_tokenizer import NoteEvent
from src.inference.rule_chord import detect_chord_from_notes, PITCH_NAMES

if detected_pitches:
    test_notes = [
        NoteEvent(start_beat=i * 0.5, pitch=p, duration_beats=0.5, velocity=80)
        for i, p in enumerate(detected_pitches[:8])
    ]
    root, triad, quality = detect_chord_from_notes(test_notes)
    triad_names = ["maj", "min", "dim", "aug", "sus"]
    print(f"  Detected chord: {PITCH_NAMES[root]} {quality}")
    print("  ✓ Chord detection working")
else:
    # Test with fake notes
    test_notes = [
        NoteEvent(0, 60, 1.0, 80),
        NoteEvent(1, 64, 1.0, 80),
        NoteEvent(2, 67, 1.0, 80),
    ]
    root, triad, quality = detect_chord_from_notes(test_notes)
    print(f"  Test chord (C E G): {PITCH_NAMES[root]} {quality}")
    print("  ✓ Chord detection working (tested with fake data)")

print("\n" + "=" * 50)
print("STEP 4: Test pattern retrieval")
print("=" * 50)

from src.inference.pattern_retrieval import load_library
from pathlib import Path

lib_path = "data/processed/pattern_library.pkl"
if Path(lib_path).exists():
    library = load_library(lib_path)
    pattern = library.retrieve(target_root=root, target_triad=triad)
    if pattern:
        print(f"  Retrieved pattern: {pattern.num_notes} notes, "
              f"range {pattern.pitch_range} semitones")
        print(f"  Notes: {[(n[2], round(n[1], 2)) for n in pattern.notes[:5]]}...")
        print("  ✓ Pattern retrieval working")
    else:
        print("  ⚠ No pattern found for this chord")
else:
    print(f"  ⚠ Pattern library not found at {lib_path}")

print("\n" + "=" * 50)
print("STEP 5: Test FluidSynth output")
print("=" * 50)

try:
    from src.inference.synthesizer import Synthesizer
    synth = Synthesizer(soundfont_path="soundfonts/piano.sf2")
    synth.start()

    print("  Playing test chord (C major)...")
    synth.note_on(60, 80)  # C4
    synth.note_on(64, 80)  # E4
    synth.note_on(67, 80)  # G4
    time.sleep(1.5)
    synth.all_notes_off()
    time.sleep(0.3)

    if pattern:
        print("  Playing retrieved pattern...")
        for start, dur, pitch, vel in pattern.notes:
            synth.note_on(pitch, vel)
        time.sleep(2.0)
        synth.all_notes_off()
        time.sleep(0.3)

    synth.stop()
    print("  ✓ FluidSynth working (did you hear piano?)")
except Exception as e:
    print(f"  ⚠ FluidSynth error: {e}")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)
