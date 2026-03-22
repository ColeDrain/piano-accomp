"""
Real-time beat tracking for synchronizing accompaniment with the vocalist.

Phase 2 implementation: Fixed BPM with tap-tempo and manual beat phase.
The beat tracker tells the inference engine "where we are in the bar"
so chord predictions and accompaniment generation align with musical beats.

Future: adaptive beat tracking from vocal onset detection.
"""

from __future__ import annotations

import time


class BeatTracker:
    """Tracks musical time (beats and bars) against wall-clock time.

    In fixed-BPM mode, the user sets the tempo and optionally taps to align
    the beat phase. The tracker then provides a continuous beat position
    that the model uses to know when to trigger chord changes and
    accompaniment generation.
    """

    def __init__(
        self,
        bpm: float = 80.0,
        beats_per_bar: int = 4,
    ):
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar

        self._start_time: float | None = None
        self._beat_offset: float = 0.0  # Phase alignment from tap tempo

        # Tap tempo state
        self._tap_times: list[float] = []
        self._max_tap_history = 8

    @property
    def beat_duration(self) -> float:
        """Duration of one beat in seconds."""
        return 60.0 / self.bpm

    @property
    def bar_duration(self) -> float:
        """Duration of one bar in seconds."""
        return self.beat_duration * self.beats_per_bar

    def start(self):
        """Start the beat clock."""
        self._start_time = time.monotonic()

    def get_beat_position(self) -> float:
        """Get the current position in beats since start.

        Returns:
            Fractional beat number (e.g., 4.5 = halfway through beat 5)
        """
        if self._start_time is None:
            return 0.0

        elapsed = time.monotonic() - self._start_time
        return (elapsed / self.beat_duration) + self._beat_offset

    def get_bar_position(self) -> tuple[int, float]:
        """Get current bar number and beat within bar.

        Returns:
            (bar_number, beat_in_bar) where beat_in_bar is in [0, beats_per_bar)
        """
        total_beats = self.get_beat_position()
        bar = int(total_beats // self.beats_per_bar)
        beat_in_bar = total_beats % self.beats_per_bar
        return bar, beat_in_bar

    def is_on_beat(self, tolerance: float = 0.1) -> bool:
        """Check if we're currently on a beat boundary.

        Args:
            tolerance: How close to a beat boundary (in beats) counts as "on beat"
        """
        beat_pos = self.get_beat_position()
        frac = beat_pos % 1.0
        return frac < tolerance or frac > (1.0 - tolerance)

    def is_on_bar(self, tolerance: float = 0.1) -> bool:
        """Check if we're currently on a bar boundary."""
        _, beat_in_bar = self.get_bar_position()
        return beat_in_bar < tolerance or beat_in_bar > (self.beats_per_bar - tolerance)

    def beats_until_next(self) -> float:
        """Time in beats until the next beat boundary."""
        beat_pos = self.get_beat_position()
        return 1.0 - (beat_pos % 1.0)

    def seconds_until_next_beat(self) -> float:
        """Time in seconds until the next beat boundary."""
        return self.beats_until_next() * self.beat_duration

    # --- Tap tempo ---

    def tap(self):
        """Register a tap for tap-tempo BPM estimation.

        Call this each time the user taps. After 2+ taps, BPM is estimated
        from the inter-tap intervals. The beat phase is also aligned to
        the most recent tap.
        """
        now = time.monotonic()
        self._tap_times.append(now)

        # Keep only recent taps
        if len(self._tap_times) > self._max_tap_history:
            self._tap_times = self._tap_times[-self._max_tap_history:]

        # Need at least 2 taps to estimate tempo
        if len(self._tap_times) < 2:
            return

        # Estimate BPM from inter-tap intervals
        intervals = [
            self._tap_times[i] - self._tap_times[i - 1]
            for i in range(1, len(self._tap_times))
        ]

        # Filter out obvious outliers (> 2 seconds between taps = reset)
        valid = [iv for iv in intervals if iv < 2.0]
        if not valid:
            self._tap_times = [now]
            return

        avg_interval = sum(valid) / len(valid)
        self.bpm = 60.0 / avg_interval

        # Align beat phase to this tap
        if self._start_time is not None:
            elapsed = now - self._start_time
            current_beat = elapsed / self.beat_duration
            # Snap to nearest whole beat
            self._beat_offset = round(current_beat) - current_beat

    def set_bpm(self, bpm: float):
        """Manually set BPM."""
        self.bpm = max(20.0, min(300.0, bpm))  # Clamp to reasonable range
        self._tap_times.clear()

    def reset(self):
        """Reset the beat tracker."""
        self._start_time = None
        self._beat_offset = 0.0
        self._tap_times.clear()
