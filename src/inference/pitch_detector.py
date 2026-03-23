"""
Real-time vocal pitch detection using torchcrepe.

Converts raw audio chunks from the microphone into MIDI note events.
Handles onset/offset detection, vibrato smoothing, and confidence thresholding.

The pitch detector runs in its own processing step within the inference pipeline:
    Audio chunk (50ms, 44.1kHz) → resample to 16kHz → torchcrepe → MIDI note
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torchaudio.functional as AF

# Patch resampy's pkg_resources dependency before importing torchcrepe
import importlib
import types
_fake_pkg = types.ModuleType("pkg_resources")
_fake_pkg.resource_filename = lambda *a: ""
import sys
if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = _fake_pkg

import torchcrepe


# MIDI note number <-> frequency conversion
def freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


class PitchDetector:
    """Real-time monophonic pitch detector for vocal input.

    Uses torchcrepe for pitch estimation with confidence-based onset/offset
    detection and median filtering for vibrato smoothing.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        crepe_model: str = "tiny",
        confidence_threshold: float = 0.5,
        median_filter_size: int = 3,
        device: torch.device | None = None,
    ):
        self.sample_rate = sample_rate
        self.crepe_sample_rate = 16000  # torchcrepe expects 16kHz
        self.crepe_model = crepe_model
        self.confidence_threshold = confidence_threshold
        self.median_filter_size = median_filter_size
        self.device = device or torch.device("cpu")

        # State for onset/offset detection
        self.current_note: int | None = None  # Currently sounding MIDI note
        self.current_velocity: int = 80       # Default velocity

        # Median filter buffer for pitch smoothing
        self._pitch_buffer: deque[float] = deque(maxlen=median_filter_size)

        # Accumulate short audio chunks for minimum CREPE input (1024 samples at 16kHz = 64ms)
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._min_samples_16k = 1024  # Minimum samples needed at 16kHz

    def process_chunk(self, audio: np.ndarray) -> list[dict]:
        """Process a chunk of audio and return MIDI events.

        Args:
            audio: Raw audio samples (float32, mono, at self.sample_rate)

        Returns:
            List of MIDI events, each a dict:
                {"type": "note_on", "pitch": int, "velocity": int}
                {"type": "note_off", "pitch": int}
                {"type": "pitch", "pitch": int, "confidence": float}  # continuous pitch info
        """
        # Resample 44.1kHz -> 16kHz
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if self.sample_rate != self.crepe_sample_rate:
            audio_tensor = AF.resample(
                audio_tensor, self.sample_rate, self.crepe_sample_rate
            )

        # Accumulate into buffer until we have enough for CREPE
        resampled = audio_tensor.cpu().numpy()
        self._audio_buffer = np.concatenate([self._audio_buffer, resampled])

        if len(self._audio_buffer) < self._min_samples_16k:
            return []  # Not enough audio yet

        # Run CREPE on the buffered audio
        audio_16k = torch.from_numpy(self._audio_buffer).float().unsqueeze(0).to(self.device)
        self._audio_buffer = np.zeros(0, dtype=np.float32)  # Clear buffer

        # torchcrepe.predict returns (pitch_hz, confidence)
        pitch_hz, confidence = torchcrepe.predict(
            audio_16k,
            sample_rate=self.crepe_sample_rate,
            model=self.crepe_model,
            hop_length=160,  # 10ms steps at 16kHz
            batch_size=1,
            device=self.device,
            return_periodicity=True,
            pad=False,
        )

        # Average pitch and confidence over the chunk
        pitch_hz = pitch_hz.squeeze()
        confidence = confidence.squeeze()

        if pitch_hz.dim() == 0:
            pitch_hz = pitch_hz.unsqueeze(0)
            confidence = confidence.unsqueeze(0)

        # Use the frame with highest confidence
        if len(confidence) == 0:
            return []

        best_idx = confidence.argmax().item()
        best_freq = pitch_hz[best_idx].item()
        best_conf = confidence[best_idx].item()

        return self._update_state(best_freq, best_conf)

    def _update_state(self, freq: float, confidence: float) -> list[dict]:
        """Update internal state and generate MIDI events based on pitch detection.

        Implements onset/offset detection with hysteresis:
        - Note ON: confidence rises above threshold AND pitch is stable
        - Note OFF: confidence drops below threshold * 0.8 (hysteresis)
        """
        events = []

        if confidence < self.confidence_threshold * 0.8:
            # Silence — turn off current note
            if self.current_note is not None:
                events.append({"type": "note_off", "pitch": self.current_note})
                self.current_note = None
            self._pitch_buffer.clear()
            return events

        if confidence < self.confidence_threshold:
            # In the hysteresis zone — keep current state
            return events

        # We have a confident pitch detection
        midi_note = freq_to_midi(freq)

        # Median filter for vibrato smoothing
        self._pitch_buffer.append(midi_note)
        if len(self._pitch_buffer) >= self.median_filter_size:
            smoothed = int(np.median(list(self._pitch_buffer)))
        else:
            smoothed = midi_note

        # Estimate velocity from signal energy (placeholder — could use RMS)
        velocity = min(127, max(40, int(confidence * 127)))

        # Generate events
        if self.current_note is None:
            # New note onset
            self.current_note = smoothed
            self.current_velocity = velocity
            events.append({"type": "note_on", "pitch": smoothed, "velocity": velocity})
        elif abs(smoothed - self.current_note) >= 2:
            # Pitch changed significantly — new note
            events.append({"type": "note_off", "pitch": self.current_note})
            self.current_note = smoothed
            self.current_velocity = velocity
            events.append({"type": "note_on", "pitch": smoothed, "velocity": velocity})

        # Always emit continuous pitch info for the model
        events.append({"type": "pitch", "pitch": smoothed, "confidence": confidence})

        return events

    def reset(self):
        """Reset detector state (e.g., between songs)."""
        self.current_note = None
        self._pitch_buffer.clear()
        self._audio_buffer = np.zeros(0, dtype=np.float32)
