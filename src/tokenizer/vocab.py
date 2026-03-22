"""
Vocabulary definitions for the REMI-style MIDI tokenizer.

The vocabulary is organized into token groups:
- Special tokens (PAD, BOS, EOS, SEP, etc.)
- Pitch tokens (0-127)
- Duration tokens (32 quantized levels)
- Velocity tokens (32 quantized levels)
- Time-shift tokens (64 quantized levels)
- Chord tokens (~200 root x quality combinations, gospel-extended)
- Style tokens (stride, walking, block, etc.)
"""

from dataclasses import dataclass, field

# --- Duration grid (in beats) ---
# From 1/8 of a beat (32nd note) to 4 beats (whole note)
# Includes standard, dotted, and triplet durations
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

# --- Time shift grid (in beats) ---
# From 1/32 beat to 4 beats, logarithmically spaced at fine end
TIME_SHIFT_BINS = [
    0.03125,  # 1/32 beat
    0.0625,   # 1/16 beat
    0.125,    # 1/8 beat (32nd note)
    0.167,    # 16th triplet
    0.1875,   # dotted 32nd
    0.25,     # 16th note
    0.333,    # 8th triplet
    0.375,    # dotted 16th
    0.5,      # 8th note
    0.667,    # quarter triplet
    0.75,     # dotted 8th
    1.0,      # quarter note
    1.333,    # half triplet
    1.5,      # dotted quarter
    2.0,      # half note
    3.0,      # dotted half
    4.0,      # whole note
]

# --- Velocity bins ---
# 32 linear bins from pp to fff
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
    "7#9",       # Hendrix chord / gospel crunch
    "7b9",
    "7#11",
    "maj7#11",
    "13sus4",    # Very common gospel suspension
    "9sus4",
    "min7add11",
    # Slash chord indicator (root will be specified separately)
    "add9",
    "add11",
    "6", "min6",
    "6/9",
]

PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Vocabulary:
    """Manages the full token vocabulary and provides id <-> token mapping."""

    # Token -> ID mapping (built in __post_init__)
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

        # Pitch tokens: Pitch_0 through Pitch_127
        self.pitch_offset = len(tokens)
        for p in range(128):
            tokens.append(f"Pitch_{p}")

        # Duration tokens: Duration_0 through Duration_{n}
        self.duration_offset = len(tokens)
        for i, d in enumerate(DURATION_BINS):
            tokens.append(f"Duration_{i}")

        # Velocity tokens: Velocity_0 through Velocity_{n}
        self.velocity_offset = len(tokens)
        for i, v in enumerate(VELOCITY_BINS):
            tokens.append(f"Velocity_{i}")

        # Time shift tokens: TimeShift_0 through TimeShift_{n}
        self.time_shift_offset = len(tokens)
        for i, t in enumerate(TIME_SHIFT_BINS):
            tokens.append(f"TimeShift_{i}")

        # Chord tokens: Chord_C_maj, Chord_Db_min7, etc.
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

    def encode_pitch(self, midi_pitch: int) -> int:
        """Convert MIDI pitch (0-127) to token ID."""
        assert 0 <= midi_pitch <= 127
        return self.pitch_offset + midi_pitch

    def decode_pitch(self, token_id: int) -> int:
        """Convert token ID back to MIDI pitch."""
        return token_id - self.pitch_offset

    def encode_duration(self, beats: float) -> int:
        """Quantize a duration in beats to the nearest bin and return token ID."""
        idx = _nearest_bin(beats, DURATION_BINS)
        return self.duration_offset + idx

    def decode_duration(self, token_id: int) -> float:
        """Convert token ID back to duration in beats."""
        idx = token_id - self.duration_offset
        return DURATION_BINS[idx]

    def encode_velocity(self, velocity: int) -> int:
        """Quantize velocity (0-127) to nearest bin and return token ID."""
        idx = _nearest_bin(velocity, VELOCITY_BINS)
        return self.velocity_offset + idx

    def decode_velocity(self, token_id: int) -> int:
        """Convert token ID back to velocity."""
        idx = token_id - self.velocity_offset
        return VELOCITY_BINS[idx]

    def encode_time_shift(self, beats: float) -> int:
        """Quantize a time shift in beats to nearest bin and return token ID."""
        idx = _nearest_bin(beats, TIME_SHIFT_BINS)
        return self.time_shift_offset + idx

    def decode_time_shift(self, token_id: int) -> float:
        """Convert token ID back to time shift in beats."""
        idx = token_id - self.time_shift_offset
        return TIME_SHIFT_BINS[idx]

    def encode_chord(self, root: str, quality: str) -> int:
        """Encode a chord (root name + quality) to token ID."""
        token = f"Chord_{root}_{quality}"
        return self.token_to_id[token]

    def decode_chord(self, token_id: int) -> tuple[str, str]:
        """Decode token ID to (root, quality) tuple."""
        token = self.id_to_token[token_id]
        _, root, quality = token.split("_", 2)
        return root, quality

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
        return token_id < self.pitch_offset


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
