"""
Real-time inference engine — the orchestrator.

Coordinates three threads:
1. Audio input thread: reads mic → pitch detection → melody note queue
2. Model inference thread: reads melody notes → chord prediction + texture generation → MIDI event queue
3. Main thread: reads MIDI events → FluidSynth playback

Pipeline:
    Mic → PitchDetector → [melody_queue] → Models → [midi_queue] → Synthesizer → Speaker
"""

from __future__ import annotations

import threading
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import yaml

from src.tokenizer.vocab import Vocabulary
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator
from src.inference.pitch_detector import PitchDetector
from src.inference.beat_tracker import BeatTracker
from src.inference.synthesizer import Synthesizer


class RealtimeEngine:
    """Orchestrates real-time vocal-to-piano accompaniment.

    Usage:
        engine = RealtimeEngine(chord_ckpt="...", texture_ckpt="...")
        engine.start()
        # Sing into your mic...
        engine.stop()
    """

    def __init__(
        self,
        chord_checkpoint: str,
        texture_checkpoint: str,
        soundfont_path: str = "soundfonts/salamander_grand_piano.sf2",
        bpm: float = 80.0,
        temperature: float = 0.8,
        device: torch.device | None = None,
        sample_rate: int = 44100,
        chunk_size: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.temperature = temperature

        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Load configs
        with open("configs/model.yaml") as f:
            model_cfg = yaml.safe_load(f)
        with open("configs/inference.yaml") as f:
            self.infer_cfg = yaml.safe_load(f)

        # Vocabulary and tokenizer
        self.vocab = Vocabulary()
        self.tokenizer = MidiTokenizer(self.vocab)

        # Load models
        chord_cfg = model_cfg["chord_predictor"]
        tex_cfg = model_cfg["texture_generator"]

        self.chord_model = ChordPredictor(
            vocab_size=self.vocab.size,
            num_chord_classes=chord_cfg["num_chord_classes"],
            embed_dim=chord_cfg["embed_dim"],
            num_layers=chord_cfg["num_layers"],
            num_heads=chord_cfg["num_heads"],
            ffn_dim=chord_cfg["ffn_dim"],
            max_melody_tokens=chord_cfg["max_melody_tokens"],
            dropout=0.0,
        ).to(self.device)
        self.chord_model.load_state_dict(
            torch.load(chord_checkpoint, map_location=self.device)
        )
        self.chord_model.eval()

        self.texture_model = TextureGenerator(
            vocab_size=self.vocab.size,
            num_chord_classes=chord_cfg["num_chord_classes"],
            embed_dim=tex_cfg["embed_dim"],
            num_layers=tex_cfg["num_layers"],
            num_heads=tex_cfg["num_heads"],
            ffn_dim=tex_cfg["ffn_dim"],
            max_seq_len=tex_cfg["max_seq_len"],
            melody_context_dim=chord_cfg["embed_dim"],
            chord_embed_dim=tex_cfg["chord_embed_dim"],
            style_embed_dim=tex_cfg["style_embed_dim"],
            dropout=0.0,
        ).to(self.device)
        self.texture_model.load_state_dict(
            torch.load(texture_checkpoint, map_location=self.device)
        )
        self.texture_model.eval()

        # Components
        self.pitch_detector = PitchDetector(
            sample_rate=sample_rate,
            crepe_model=self.infer_cfg["pitch_detection"]["crepe_model_size"],
            confidence_threshold=self.infer_cfg["pitch_detection"]["confidence_threshold"],
            median_filter_size=self.infer_cfg["pitch_detection"]["median_filter_size"],
            device=self.device,
        )
        self.beat_tracker = BeatTracker(bpm=bpm)
        self.synth = Synthesizer(soundfont_path=soundfont_path)

        # Inter-thread queues
        self._melody_queue: queue.Queue[dict] = queue.Queue(maxsize=64)
        self._midi_queue: queue.Queue[list[dict]] = queue.Queue(maxsize=64)

        # Thread control
        self._running = False
        self._threads: list[threading.Thread] = []

        # Melody note accumulator for building model input
        self._melody_notes: list[NoteEvent] = []
        self._max_melody_notes = 32

        # State
        self._last_chord_beat: float = -1.0

    def start(self):
        """Start the real-time accompaniment engine."""
        print(f"Starting engine on {self.device} at {self.beat_tracker.bpm:.0f} BPM")
        print(f"Vocab size: {self.vocab.size}, Chunk: {self.chunk_size} samples")

        self._running = True

        # Start FluidSynth
        self.synth.start()

        # Start beat tracker
        self.beat_tracker.start()

        # Start model inference thread
        inference_thread = threading.Thread(
            target=self._inference_loop, name="inference", daemon=True
        )
        inference_thread.start()
        self._threads.append(inference_thread)

        # Start MIDI playback thread
        playback_thread = threading.Thread(
            target=self._playback_loop, name="playback", daemon=True
        )
        playback_thread.start()
        self._threads.append(playback_thread)

        # Start audio input stream (runs in sounddevice's own thread)
        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._audio_stream.start()

        print("Engine running. Sing into your mic!")
        print("Press Ctrl+C to stop.")

    def stop(self):
        """Stop the engine and clean up."""
        print("\nStopping engine...")
        self._running = False

        # Stop audio stream
        if hasattr(self, "_audio_stream"):
            self._audio_stream.stop()
            self._audio_stream.close()

        # Wait for threads
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

        # Stop synth
        self.synth.stop()

        # Reset state
        self.pitch_detector.reset()
        self.beat_tracker.reset()
        self._melody_notes.clear()

        print("Engine stopped.")

    def tap_tempo(self):
        """Register a tap for tap-tempo BPM estimation."""
        self.beat_tracker.tap()
        print(f"  BPM: {self.beat_tracker.bpm:.1f}")

    # ------------------------------------------------------------------
    # Audio input callback (runs in sounddevice's thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called by sounddevice for each audio chunk from the mic."""
        if status:
            pass  # Could log xruns here

        # Run pitch detection on this chunk
        audio = indata[:, 0]  # Mono
        events = self.pitch_detector.process_chunk(audio)

        # Push pitch events to the melody queue
        for event in events:
            try:
                self._melody_queue.put_nowait(event)
            except queue.Full:
                pass  # Drop if queue is full (inference thread is behind)

    # ------------------------------------------------------------------
    # Model inference thread
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Runs chord prediction and accompaniment generation."""
        while self._running:
            # Collect melody events from the queue
            try:
                event = self._melody_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Track melody notes for model input
            if event["type"] == "note_on":
                beat_pos = self.beat_tracker.get_beat_position()
                self._melody_notes.append(NoteEvent(
                    start_beat=beat_pos,
                    pitch=event["pitch"],
                    duration_beats=0.5,  # Estimate; will be refined on note_off
                    velocity=event.get("velocity", 80),
                ))
                # Keep window bounded
                if len(self._melody_notes) > self._max_melody_notes:
                    self._melody_notes = self._melody_notes[-self._max_melody_notes:]

            # Check if we should generate accompaniment (on each beat)
            current_beat = self.beat_tracker.get_beat_position()
            beats_since_last = current_beat - self._last_chord_beat

            if beats_since_last >= 1.0 and len(self._melody_notes) >= 2:
                self._last_chord_beat = current_beat
                midi_events = self._generate_accompaniment()
                if midi_events:
                    try:
                        self._midi_queue.put_nowait(midi_events)
                    except queue.Full:
                        pass

    def _generate_accompaniment(self) -> list[dict]:
        """Run the two-stage model to generate accompaniment for the current beat."""
        # Tokenize recent melody
        # Normalize start beats relative to the window
        if not self._melody_notes:
            return []

        min_beat = self._melody_notes[0].start_beat
        relative_notes = [
            NoteEvent(
                start_beat=n.start_beat - min_beat,
                pitch=n.pitch,
                duration_beats=n.duration_beats,
                velocity=n.velocity,
            )
            for n in self._melody_notes
        ]

        mel_tokens = self.tokenizer.encode_note_events(relative_notes)
        mel_tensor = torch.tensor(
            [mel_tokens[:32]], device=self.device, dtype=torch.long
        )

        with torch.no_grad():
            # Stage 1: predict chord
            chord_logits, melody_context = self.chord_model(
                mel_tensor, return_embedding=True
            )
            chord_id = chord_logits.argmax(dim=-1)

            # Stage 2: generate accompaniment texture
            accomp_tokens = self.texture_model.generate(
                melody_context=melody_context,
                chord_ids=chord_id,
                max_tokens=self.infer_cfg["generation"]["max_tokens_per_beat"],
                temperature=self.temperature,
                top_k=self.infer_cfg["generation"]["top_k"],
                top_p=self.infer_cfg["generation"]["top_p"],
                bos_id=self.vocab.bos_id,
                eos_id=self.vocab.eos_id,
            )

        # Decode tokens to note events
        full_tokens = [self.vocab.bos_id] + accomp_tokens + [self.vocab.eos_id]
        note_events = self.tokenizer.decode_to_events(full_tokens)

        # Convert to MIDI events for FluidSynth
        midi_events = []
        for ne in note_events:
            midi_events.append({
                "type": "note_on",
                "pitch": ne.pitch,
                "velocity": ne.velocity,
            })
            # Schedule note_off (we'll handle timing in the playback loop)
            midi_events.append({
                "type": "note_off",
                "pitch": ne.pitch,
                "delay_beats": ne.duration_beats,
            })

        return midi_events

    # ------------------------------------------------------------------
    # MIDI playback thread
    # ------------------------------------------------------------------

    def _playback_loop(self):
        """Reads MIDI events from the queue and plays them through FluidSynth."""
        pending_offs: list[tuple[float, int]] = []  # (off_time, pitch)

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

            # Turn off previous accompaniment notes
            self.synth.all_notes_off()
            pending_offs.clear()

            # Play new events
            beat_dur = self.beat_tracker.beat_duration
            for event in events:
                if event["type"] == "note_on":
                    self.synth.note_on(event["pitch"], event.get("velocity", 80))
                elif event["type"] == "note_off":
                    delay = event.get("delay_beats", 0.5) * beat_dur
                    pending_offs.append((now + delay, event["pitch"]))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
