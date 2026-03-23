"""
Real-time inference engine — the orchestrator.

Pipeline:
    Mic → PitchDetector → melody notes
    → Rule-based chord detection → chord
    → Pattern retrieval from POP909 → piano MIDI events
    → FluidSynth → speaker

Updates chord every 2 beats (~1.5s at 80 BPM) for responsiveness.
Prints detected notes and chords so you can see it's tracking you.
"""

from __future__ import annotations

import threading
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.tokenizer.vocab import PITCH_NAMES
from src.tokenizer.midi_tokenizer import NoteEvent
from src.inference.pitch_detector import PitchDetector
from src.inference.beat_tracker import BeatTracker
from src.inference.synthesizer import Synthesizer
from src.inference.rule_chord import detect_chord_from_notes, detect_key_from_notes
from src.inference.pattern_retrieval import (
    PatternLibrary, load_library, build_library_from_pop909, save_library,
)

import pretty_midi

TRIAD_NAMES = ["maj", "min", "dim", "aug", "sus"]
LIBRARY_CACHE = "data/processed/pattern_library.pkl"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_num: int) -> str:
    return f"{NOTE_NAMES[midi_num % 12]}{midi_num // 12 - 1}"


class RealtimeEngine:
    """Real-time vocal-to-piano accompaniment."""

    def __init__(
        self,
        soundfont_path: str = "soundfonts/piano.sf2",
        pop909_dir: str = "data/raw/pop909",
        bpm: float = 80.0,
        sample_rate: int = 44100,
        chunk_size: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.pitch_detector = PitchDetector(
            sample_rate=sample_rate,
            crepe_model="tiny",
            confidence_threshold=0.4,
            median_filter_size=3,
        )
        self.beat_tracker = BeatTracker(bpm=bpm)
        self.synth = Synthesizer(soundfont_path=soundfont_path)

        # Pattern library
        if Path(LIBRARY_CACHE).exists():
            self.library = load_library(LIBRARY_CACHE)
        else:
            print("Building pattern library (first time)...")
            self.library = build_library_from_pop909(pop909_dir)
            Path(LIBRARY_CACHE).parent.mkdir(parents=True, exist_ok=True)
            save_library(self.library, LIBRARY_CACHE)

        # Queues
        self._melody_queue: queue.Queue[dict] = queue.Queue(maxsize=128)

        # State
        self._running = False
        self._threads: list[threading.Thread] = []
        self._melody_notes: list[NoteEvent] = []
        self._prev_pattern = None
        self._key_root: int = 0
        self._last_chord_time: float = 0.0
        self._current_chord_str: str = ""

    def start(self):
        print(f"Starting engine at {self.beat_tracker.bpm:.0f} BPM")
        print(f"Pattern library: {len(self.library.patterns)} patterns")

        self._running = True
        self.synth.start()
        self.beat_tracker.start()

        # Chord + retrieval + playback thread (combined for timing)
        engine_thread = threading.Thread(
            target=self._engine_loop, name="engine", daemon=True
        )
        engine_thread.start()
        self._threads.append(engine_thread)

        # Audio input
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._audio_stream.start()

        print("\nEngine running! Sing into your mic.")
        print("You should see detected notes below.\n")

    def stop(self):
        print("\nStopping engine...")
        self._running = False

        if hasattr(self, "_audio_stream"):
            self._audio_stream.stop()
            self._audio_stream.close()

        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

        self.synth.stop()
        self.pitch_detector.reset()
        self.beat_tracker.reset()
        self._melody_notes.clear()
        self._prev_pattern = None
        print("Engine stopped.")

    def tap_tempo(self):
        self.beat_tracker.tap()
        print(f"  BPM: {self.beat_tracker.bpm:.1f}")

    # --- Audio input callback ---

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        audio = indata[:, 0]
        events = self.pitch_detector.process_chunk(audio)
        for event in events:
            try:
                self._melody_queue.put_nowait(event)
            except queue.Full:
                pass

    # --- Main engine loop (chord detection + retrieval + playback) ---

    def _engine_loop(self):
        """Single loop handling chord detection, pattern retrieval, and playback timing."""
        while self._running:
            # Drain melody queue
            got_new_note = False
            while True:
                try:
                    event = self._melody_queue.get_nowait()
                except queue.Empty:
                    break

                if event["type"] == "note_on":
                    note_name = midi_to_note_name(event["pitch"])
                    print(f"  ♪ {note_name}", end="", flush=True)
                    got_new_note = True

                    self._melody_notes.append(NoteEvent(
                        start_beat=time.monotonic(),
                        pitch=event["pitch"],
                        duration_beats=0.5,
                        velocity=event.get("velocity", 80),
                    ))
                    if len(self._melody_notes) > 16:
                        self._melody_notes = self._melody_notes[-16:]

            # Check if we should update the chord (every 1.5 seconds)
            now = time.monotonic()
            if now - self._last_chord_time >= 1.5 and len(self._melody_notes) >= 2:
                self._last_chord_time = now

                # Update key
                if len(self._melody_notes) >= 6:
                    self._key_root = detect_key_from_notes(self._melody_notes)

                # Detect chord
                root, triad, quality = detect_chord_from_notes(
                    self._melody_notes[-8:],
                    key_root=self._key_root,
                    prefer_diatonic=True,
                )

                chord_str = f"{PITCH_NAMES[root]} {quality}"
                if chord_str != self._current_chord_str:
                    self._current_chord_str = chord_str
                    print(f"\n  → Chord: {chord_str}", flush=True)

                # Retrieve and play pattern
                pattern = self.library.retrieve(
                    target_root=root,
                    target_triad=triad,
                    prev_pattern=self._prev_pattern,
                    prefer_bass=True,
                )

                if pattern:
                    self._prev_pattern = pattern
                    self._play_pattern(pattern)

            time.sleep(0.02)  # 20ms loop rate

    def _play_pattern(self, pattern):
        """Play a pattern with proper timing — notes spaced out over the measure."""
        beat_dur = self.beat_tracker.beat_duration

        # Stop previous notes
        self.synth.all_notes_off()

        # Sort notes by start time
        sorted_notes = sorted(pattern.notes, key=lambda n: n[0])

        # Play notes with timing
        base_time = time.monotonic()
        for start_beat, dur_beat, pitch, velocity in sorted_notes:
            # Wait until the right time for this note
            target_time = base_time + (start_beat * beat_dur)
            wait = target_time - time.monotonic()
            if wait > 0 and wait < 4.0:  # Don't wait more than 4 seconds
                time.sleep(wait)

            if not self._running:
                break

            self.synth.note_on(pitch, velocity)

            # Schedule note-off after duration
            # (We just let notes ring and cut them on next pattern)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
