"""
FluidSynth wrapper for real-time MIDI-to-audio synthesis.

Accepts MIDI note events and renders them as piano audio using a SoundFont.
Designed for low-latency playback in the real-time pipeline.

Requires:
    - FluidSynth installed: brew install fluidsynth
    - A SoundFont file (e.g., Salamander Grand Piano)
    - pyfluidsynth Python package
"""

from __future__ import annotations

from pathlib import Path


class Synthesizer:
    """Real-time MIDI synthesizer using FluidSynth.

    Plays MIDI note events through a SoundFont with minimal latency.
    """

    def __init__(
        self,
        soundfont_path: str = "soundfonts/salamander_grand_piano.sf2",
        gain: float = 0.8,
        sample_rate: int = 44100,
        reverb: bool = True,
        chorus: bool = False,
    ):
        self.soundfont_path = soundfont_path
        self.gain = gain
        self.sample_rate = sample_rate
        self.reverb = reverb
        self.chorus = chorus

        self._fs = None
        self._sfid = None
        self._active_notes: set[int] = set()  # Track active notes for cleanup

    def start(self):
        """Initialize FluidSynth and load the SoundFont."""
        import fluidsynth

        self._fs = fluidsynth.Synth(gain=self.gain, samplerate=float(self.sample_rate))

        # Start audio driver (uses system default output)
        self._fs.start(driver="coreaudio")  # macOS

        # Configure effects
        if not self.reverb:
            self._fs.set_reverb(0.0, 0.0, 0.0, 0.0)
        if not self.chorus:
            self._fs.set_chorus(0, 0, 0.0, 0.0, 0)

        # Load SoundFont
        sf_path = Path(self.soundfont_path)
        if not sf_path.exists():
            raise FileNotFoundError(
                f"SoundFont not found at {sf_path}. "
                f"Download Salamander Grand Piano from: "
                f"https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
            )

        self._sfid = self._fs.sfload(str(sf_path))
        self._fs.program_select(0, self._sfid, 0, 0)  # Channel 0, bank 0, program 0 (piano)

    def note_on(self, pitch: int, velocity: int = 80, channel: int = 0):
        """Play a note.

        Args:
            pitch: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            channel: MIDI channel (default 0)
        """
        if self._fs is None:
            return
        self._fs.noteon(channel, pitch, velocity)
        self._active_notes.add(pitch)

    def note_off(self, pitch: int, channel: int = 0):
        """Stop a note.

        Args:
            pitch: MIDI note number (0-127)
            channel: MIDI channel (default 0)
        """
        if self._fs is None:
            return
        self._fs.noteoff(channel, pitch)
        self._active_notes.discard(pitch)

    def play_chord(self, pitches: list[int], velocity: int = 80, channel: int = 0):
        """Play multiple notes simultaneously (a chord).

        Turns off any currently active notes first, then plays the new chord.
        """
        self.all_notes_off(channel)
        for pitch in pitches:
            self.note_on(pitch, velocity, channel)

    def all_notes_off(self, channel: int = 0):
        """Turn off all currently active notes."""
        for pitch in list(self._active_notes):
            self.note_off(pitch, channel)
        self._active_notes.clear()

    def play_events(self, events: list[dict], channel: int = 0):
        """Play a list of MIDI events.

        Args:
            events: List of dicts with keys:
                {"type": "note_on", "pitch": int, "velocity": int}
                {"type": "note_off", "pitch": int}
        """
        for event in events:
            if event["type"] == "note_on":
                self.note_on(event["pitch"], event.get("velocity", 80), channel)
            elif event["type"] == "note_off":
                self.note_off(event["pitch"], channel)

    def set_reverb(self, roomsize: float = 0.6, damping: float = 0.4,
                   width: float = 0.5, level: float = 0.3):
        """Adjust reverb parameters."""
        if self._fs:
            self._fs.set_reverb(roomsize, damping, width, level)

    def stop(self):
        """Shut down FluidSynth."""
        if self._fs:
            self.all_notes_off()
            self._fs.delete()
            self._fs = None
            self._sfid = None

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
