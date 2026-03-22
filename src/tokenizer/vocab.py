"""
Vocabulary definitions for the true REMI MIDI tokenizer.

The vocabulary is organized into token groups:
- Special tokens (PAD, BOS, EOS, SEP, BAR, style markers)
- Position tokens (0-15: position within bar at 4 subdivisions per beat in 4/4)
- Pitch tokens (0-127)
- Duration tokens (16 quantized levels)
- Velocity tokens (32 quantized levels)
- Time-shift tokens (17 quantized levels) — kept for backward compat, but Position is preferred
- Chord tokens (~360 root × quality combinations, gospel-extended)
"""

from dataclasses import dataclass, field

# --- Position grid ---
# 4 subdivisions per beat × 4 beats per bar = 16 positions in 4/4
# Position 0 = beat 1 (downbeat), Position 4 = beat 2, etc.
NUM_POSITIONS = 16  # For 4/4 time with 16th-note resolution
BEATS_PER_BAR = 4
SUBDIVISIONS_PER_BEAT = 4

# --- Duration grid (in beats) ---
DURATION_BINS = [
    0.125,    # 32nd
    0.167,    # 16th triplet
    0.1875,   # dotted 32nd
    0.25,     # 16th
    0.333,    # 8th triplet
    0.375,    # dotted 16th
    0.5,      # 8th
    0.667,    # quarter triplet
    0.75,     # dotted 8th
    1.0,      # quarter
    1.333,    # half triplet
    1.5,      # dotted quarter
    2.0,      # half
    2.667,    # whole triplet
    3.0,      # dotted half
    4.0,      # whole
]

# --- Time shift grid (in beats) --- kept for backward compat
TIME_SHIFT_BINS = [
    0.03125, 0.0625, 0.125, 0.167, 0.1875, 0.25, 0.333, 0.375,
    0.5, 0.667, 0.75, 1.0, 1.333, 1.5, 2.0, 3.0, 4.0,
]

# --- Velocity bins ---
VELOCITY_BINS = list(range(4, 128, 4))  # [4, 8, 12, ..., 124]

# --- Chord qualities (gospel-extended) ---
CHORD_QUALITIES = [
    # Basic triads
    "maj", "min", "dim", "aug",
    # Suspended
    "sus2", "sus4",
    # Seventh chords
    "7", "maj7", "min7", "dim7", "min7b5", "aug7",
    # Extended (common in gospel)
    "9", "maj9", "min9",
    "11", "min11",
    "13",
    # Gospel-specific voicings
    "7#9", "7b9", "7#11", "maj7#11",
    "13sus4", "9sus4", "min7add11",
    # Added tone chords
    "add9", "add11",
    "6", "min6", "6/9",
]

PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# --- Chord decomposition maps (for structured prediction) ---
# Maps each quality string to its triad and seventh components

TRIAD_GROUPS = {
    "maj": 0, "maj7": 0, "maj9": 0, "maj7#11": 0, "6": 0, "6/9": 0, "add9": 0, "add11": 0, "9": 0, "13": 0, "7": 0, "7#9": 0, "7b9": 0, "7#11": 0,
    "min": 1, "min7": 1, "min9": 1, "min6": 1, "min11": 1, "min7b5": 1, "min7add11": 1,
    "dim": 2, "dim7": 2,
    "aug": 3, "aug7": 3,
    "sus2": 4, "sus4": 4, "9sus4": 4, "13sus4": 4,
}
TRIAD_NAMES = ["maj", "min", "dim", "aug", "sus"]
NUM_TRIADS = 5

SEVENTH_GROUPS = {
    "maj": 0, "min": 0, "dim": 0, "aug": 0, "sus2": 0, "sus4": 0, "add9": 0, "add11": 0, "6": 0, "min6": 0, "6/9": 0,  # no seventh
    "7": 1, "9": 1, "13": 1, "7#9": 1, "7b9": 1, "7#11": 1, "aug7": 1, "9sus4": 1, "13sus4": 1,  # dominant 7th
    "maj7": 2, "maj9": 2, "maj7#11": 2,  # major 7th
    "min7": 3, "min9": 3, "min11": 3, "min7b5": 3, "min7add11": 3, "dim7": 3,  # minor 7th
}
SEVENTH_NAMES = ["none", "dom7", "maj7", "min7"]
NUM_SEVENTHS = 4


def quality_to_triad(quality: str) -> int:
    """Map chord quality string to triad group index (0-4)."""
    return TRIAD_GROUPS.get(quality, 0)


def quality_to_seventh(quality: str) -> int:
    """Map chord quality string to seventh group index (0-3)."""
    return SEVENTH_GROUPS.get(quality, 0)


@dataclass
class Vocabulary:
    """Manages the full token vocabulary and provides id <-> token mapping."""

    token_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    id_to_token: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build()

    def _build(self):
        tokens = []

        # Special tokens (0-11)
        self.special_tokens = [
            "<PAD>", "<BOS>", "<EOS>", "<SEP>",
            "<BAR>", "<BEAT>", "<FILL>",
            "<STRIDE>", "<WALKING>", "<BLOCK>", "<ARPEGGIO>", "<FILL_PATTERN>",
        ]
        tokens.extend(self.special_tokens)

        # Position tokens: Position_0 through Position_15
        # Position within bar at 16th-note resolution (4 subdivisions × 4 beats)
        self.position_offset = len(tokens)
        for p in range(NUM_POSITIONS):
            tokens.append(f"Position_{p}")

        # Pitch tokens: Pitch_0 through Pitch_127
        self.pitch_offset = len(tokens)
        for p in range(128):
            tokens.append(f"Pitch_{p}")

        # Duration tokens
        self.duration_offset = len(tokens)
        for i in range(len(DURATION_BINS)):
            tokens.append(f"Duration_{i}")

        # Velocity tokens
        self.velocity_offset = len(tokens)
        for i in range(len(VELOCITY_BINS)):
            tokens.append(f"Velocity_{i}")

        # Time shift tokens (kept for backward compat)
        self.time_shift_offset = len(tokens)
        for i in range(len(TIME_SHIFT_BINS)):
            tokens.append(f"TimeShift_{i}")

        # Chord tokens
        self.chord_offset = len(tokens)
        for root in PITCH_NAMES:
            for quality in CHORD_QUALITIES:
                tokens.append(f"Chord_{root}_{quality}")

        # Build mappings
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(tokens)}
        self._size = len(tokens)

    @property
    def size(self) -> int:
        return self._size

    # --- Convenience accessors ---

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<SEP>"]

    @property
    def bar_id(self) -> int:
        return self.token_to_id["<BAR>"]

    def encode_position(self, beat_in_bar: float) -> int:
        """Encode position within bar to token ID.

        Args:
            beat_in_bar: Position in beats within the bar (0.0 to 3.75 for 4/4)

        Returns:
            Token ID for the nearest 16th-note position
        """
        pos = int(round(beat_in_bar * SUBDIVISIONS_PER_BEAT))
        pos = max(0, min(NUM_POSITIONS - 1, pos))
        return self.position_offset + pos

    def decode_position(self, token_id: int) -> float:
        """Convert position token ID back to beat position within bar."""
        idx = token_id - self.position_offset
        return idx / SUBDIVISIONS_PER_BEAT

    def encode_pitch(self, midi_pitch: int) -> int:
        assert 0 <= midi_pitch <= 127
        return self.pitch_offset + midi_pitch

    def decode_pitch(self, token_id: int) -> int:
        return token_id - self.pitch_offset

    def encode_duration(self, beats: float) -> int:
        idx = _nearest_bin(beats, DURATION_BINS)
        return self.duration_offset + idx

    def decode_duration(self, token_id: int) -> float:
        idx = token_id - self.duration_offset
        return DURATION_BINS[idx]

    def encode_velocity(self, velocity: int) -> int:
        idx = _nearest_bin(velocity, VELOCITY_BINS)
        return self.velocity_offset + idx

    def decode_velocity(self, token_id: int) -> int:
        idx = token_id - self.velocity_offset
        return VELOCITY_BINS[idx]

    def encode_time_shift(self, beats: float) -> int:
        idx = _nearest_bin(beats, TIME_SHIFT_BINS)
        return self.time_shift_offset + idx

    def decode_time_shift(self, token_id: int) -> float:
        idx = token_id - self.time_shift_offset
        return TIME_SHIFT_BINS[idx]

    def encode_chord(self, root: str, quality: str) -> int:
        token = f"Chord_{root}_{quality}"
        return self.token_to_id[token]

    def decode_chord(self, token_id: int) -> tuple[str, str]:
        token = self.id_to_token[token_id]
        _, root, quality = token.split("_", 2)
        return root, quality

    def is_position(self, token_id: int) -> bool:
        return self.position_offset <= token_id < self.pitch_offset

    def is_pitch(self, token_id: int) -> bool:
        return self.pitch_offset <= token_id < self.duration_offset

    def is_duration(self, token_id: int) -> bool:
        return self.duration_offset <= token_id < self.velocity_offset

    def is_velocity(self, token_id: int) -> bool:
        return self.velocity_offset <= token_id < self.time_shift_offset

    def is_time_shift(self, token_id: int) -> bool:
        return self.time_shift_offset <= token_id < self.chord_offset

    def is_chord(self, token_id: int) -> bool:
        return self.chord_offset <= token_id < self._size

    def is_special(self, token_id: int) -> bool:
        return token_id < self.position_offset


def _nearest_bin(value: float, bins: list[float]) -> int:
    """Find the index of the nearest bin value."""
    min_dist = float("inf")
    best_idx = 0
    for i, b in enumerate(bins):
        dist = abs(value - b)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx
