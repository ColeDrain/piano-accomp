"""
Real-time inference engine — the orchestrator.

Pipeline:
    Mic → PitchDetector → melody notes
    → Rule-based chord detection → chord (root, triad, quality)
    → Pattern retrieval from POP909 → piano MIDI events
    → FluidSynth → speaker

Three threads:
1. Audio input: reads mic → pitch detection → melody note queue
2. Chord + retrieval: reads melody notes → chord detection → pattern retrieval → MIDI queue
3. Main thread: MIDI playback via FluidSynth
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

TRIAD_NAMES = ["maj", "min", "dim", "aug", "sus"]
LIBRARY_CACHE = "data/processed/pattern_library.pkl"


class RealtimeEngine:
    """Real-time vocal-to-piano accompaniment using rule-based chords + pattern retrieval."""

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

        # Components
        self.pitch_detector = PitchDetector(
            sample_rate=sample_rate,
            crepe_model="tiny",
            confidence_threshold=0.5,
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
        self._melody_queue: queue.Queue[dict] = queue.Queue(maxsize=64)
        self._midi_queue: queue.Queue[list[dict]] = queue.Queue(maxsize=64)

        # State
        self._running = False
        self._threads: list[threading.Thread] = []
        self._melody_notes: list[NoteEvent] = []
        self._last_chord_beat: float = -1.0
        self._prev_pattern = None
        self._key_root: int = 0  # Detected key, updated periodically

    def start(self):
        print(f"Starting engine at {self.beat_tracker.bpm:.0f} BPM")
        print(f"Pattern library: {len(self.library.patterns)} patterns")

        self._running = True
        self.synth.start()
        self.beat_tracker.start()

        # Chord + retrieval thread
        inference_thread = threading.Thread(
            target=self._inference_loop, name="inference", daemon=True
        )
        inference_thread.start()
        self._threads.append(inference_thread)

        # Playback thread
        playback_thread = threading.Thread(
            target=self._playback_loop, name="playback", daemon=True
        )
        playback_thread.start()
        self._threads.append(playback_thread)

        # Audio input
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._audio_stream.start()

        print("Engine running. Sing into your mic!")
        print("Controls: t=tap tempo, b <num>=set BPM, q=quit")

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

    # --- Chord detection + pattern retrieval thread ---

    def _inference_loop(self):
        while self._running:
            # Collect melody events
            try:
                event = self._melody_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if event["type"] == "note_on":
                beat_pos = self.beat_tracker.get_beat_position()
                self._melody_notes.append(NoteEvent(
                    start_beat=beat_pos,
                    pitch=event["pitch"],
                    duration_beats=0.5,
                    velocity=event.get("velocity", 80),
                ))
                # Keep last 16 notes
                if len(self._melody_notes) > 16:
                    self._melody_notes = self._melody_notes[-16:]

            # Update key estimate periodically (every 8 notes)
            if len(self._melody_notes) >= 8 and len(self._melody_notes) % 8 == 0:
                self._key_root = detect_key_from_notes(self._melody_notes)

            # Generate accompaniment on each beat
            current_beat = self.beat_tracker.get_beat_position()
            beats_since_last = current_beat - self._last_chord_beat

            if beats_since_last >= 4.0 and len(self._melody_notes) >= 3:
                self._last_chord_beat = current_beat
                midi_events = self._generate_accompaniment()
                if midi_events:
                    try:
                        self._midi_queue.put_nowait(midi_events)
                    except queue.Full:
                        pass

    def _generate_accompaniment(self) -> list[dict]:
        """Detect chord from recent melody notes, retrieve a matching pattern."""
        if len(self._melody_notes) < 2:
            return []

        # Rule-based chord detection
        root, triad, quality = detect_chord_from_notes(
            self._melody_notes[-8:],
            key_root=self._key_root,
            prefer_diatonic=True,
        )

        # Retrieve pattern
        pattern = self.library.retrieve(
            target_root=root,
            target_triad=triad,
            prev_pattern=self._prev_pattern,
            prefer_bass=True,
        )

        if not pattern:
            return []

        self._prev_pattern = pattern

        # Convert pattern notes to MIDI events for FluidSynth
        midi_events = []
        beat_dur = self.beat_tracker.beat_duration

        for start_beat, dur_beat, pitch, velocity in pattern.notes:
            delay_sec = start_beat * beat_dur
            dur_sec = dur_beat * beat_dur
            midi_events.append({
                "type": "note_on",
                "pitch": pitch,
                "velocity": velocity,
                "delay": delay_sec,
                "duration": dur_sec,
            })

        return midi_events

    # --- MIDI playback thread ---

    def _playback_loop(self):
        pending_offs: list[tuple[float, int]] = []

        while self._running:
            now = time.monotonic()

            # Process pending note-offs
            still_pending = []
            for off_time, pitch in pending_offs:
                if now >= off_time:
                    self.synth.note_off(pitch)
                else:
                    still_pending.append((off_time, pitch))
            pending_offs = still_pending

            # Check for new events
            try:
                events = self._midi_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Clear previous accompaniment
            self.synth.all_notes_off()
            pending_offs.clear()

            # Schedule new notes with timing
            base_time = time.monotonic()
            for event in events:
                if event["type"] == "note_on":
                    delay = event.get("delay", 0)
                    duration = event.get("duration", 0.5)

                    # Schedule note-on (approximate timing via sleep in a helper)
                    # For simplicity, play all notes immediately but schedule note-offs
                    self.synth.note_on(event["pitch"], event.get("velocity", 80))
                    pending_offs.append((base_time + delay + duration, event["pitch"]))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
