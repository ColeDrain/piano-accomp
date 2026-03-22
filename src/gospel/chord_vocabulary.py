"""
Gospel-specific chord vocabulary and harmonic patterns.

Gospel music uses richer harmony than pop — extended chords, chromatic movement,
and characteristic progressions. This module defines the harmonic language
that the model needs to learn.
"""

# --- Gospel chord interval definitions ---
# Each chord quality maps to a set of semitone intervals from the root.
# These are used for constrained decoding (only allow chord tones)
# and for harmonic compatibility scoring.

GOSPEL_CHORD_TONES: dict[str, list[int]] = {
    # Basic (inherited from pop, but less common in gospel)
    "maj":      [0, 4, 7],
    "min":      [0, 3, 7],
    "dim":      [0, 3, 6],
    "aug":      [0, 4, 8],

    # Suspended
    "sus2":     [0, 2, 7],
    "sus4":     [0, 5, 7],

    # Seventh chords (the bread and butter of gospel)
    "7":        [0, 4, 7, 10],         # Dominant 7th — the most important gospel chord
    "maj7":     [0, 4, 7, 11],
    "min7":     [0, 3, 7, 10],
    "dim7":     [0, 3, 6, 9],          # Passing chord, very common
    "min7b5":   [0, 3, 6, 10],         # Half-diminished
    "aug7":     [0, 4, 8, 10],

    # Extended chords (gospel's signature sound)
    "9":        [0, 4, 7, 10, 14],     # Dominant 9th
    "maj9":     [0, 4, 7, 11, 14],
    "min9":     [0, 3, 7, 10, 14],
    "11":       [0, 4, 7, 10, 14, 17], # Usually omits the 3rd
    "min11":    [0, 3, 7, 10, 14, 17],
    "13":       [0, 4, 7, 10, 14, 21], # Dominant 13th
    "13sus4":   [0, 5, 7, 10, 14, 21], # Gospel staple — suspended dominant 13

    # Gospel-specific alterations
    "7#9":      [0, 4, 7, 10, 15],     # The "gospel crunch" — sharp 9
    "7b9":      [0, 4, 7, 10, 13],
    "7#11":     [0, 4, 7, 10, 18],
    "maj7#11":  [0, 4, 7, 11, 18],     # Lydian sound
    "9sus4":    [0, 5, 7, 10, 14],

    # Added tone chords
    "add9":     [0, 4, 7, 14],
    "add11":    [0, 4, 7, 17],
    "min7add11":[0, 3, 7, 10, 17],
    "6":        [0, 4, 7, 9],
    "min6":     [0, 3, 7, 9],
    "6/9":      [0, 4, 7, 9, 14],      # Very lush, common in ballads
}


# --- Common gospel progressions ---
# Expressed as Roman numeral patterns with specific qualities.
# Key-agnostic — will be transposed to all 12 keys during data generation.

# Format: list of (scale_degree_semitones, quality) tuples
# e.g., (0, "maj7") = I maj7, (5, "7") = IV 7

GOSPEL_PROGRESSIONS = [
    # Classic gospel turnarounds
    {
        "name": "1-4-1 gospel turnaround",
        "chords": [(0, "maj9"), (5, "9"), (0, "maj9"), (5, "13sus4"), (0, "maj7")],
    },
    {
        "name": "2-5-1 gospel",
        "chords": [(2, "min9"), (7, "13"), (0, "maj9")],
    },
    {
        "name": "gospel shout progression",
        "chords": [(0, "7"), (5, "7"), (0, "7"), (7, "7"), (0, "7")],
    },
    {
        "name": "minor gospel vamp",
        "chords": [(0, "min7"), (5, "min7"), (0, "min7"), (10, "7"), (0, "min7")],
    },
    {
        "name": "worship ballad",
        "chords": [(0, "maj9"), (9, "min7"), (7, "13sus4"), (7, "7"), (0, "maj9")],
    },
    {
        "name": "gospel passing diminished",
        "chords": [(0, "maj7"), (1, "dim7"), (2, "min7"), (7, "7"), (0, "maj7")],
    },
    {
        "name": "chromatic descent",
        "chords": [(0, "maj7"), (11, "7"), (10, "maj7"), (9, "min7"), (7, "7"), (0, "maj9")],
    },
    {
        "name": "4 to minor 4 to 1 (gospel classic)",
        "chords": [(5, "maj7"), (5, "min7"), (0, "maj9")],
    },
    {
        "name": "3-6-2-5-1",
        "chords": [(4, "7"), (9, "7"), (2, "min7"), (7, "7"), (0, "maj7")],
    },
    {
        "name": "gospel tag ending",
        "chords": [(2, "min9"), (7, "13"), (0, "6/9"), (1, "dim7"),
                   (2, "min9"), (7, "13"), (0, "maj9")],
    },
]


def get_chord_pitches(root_midi: int, quality: str, octave_range: tuple[int, int] = (48, 84)) -> list[int]:
    """Get all MIDI pitches that belong to a chord within a given range.

    Args:
        root_midi: MIDI note number of the chord root (0-11 for pitch class, or absolute)
        quality: Chord quality string
        octave_range: (low, high) MIDI note range to include

    Returns:
        Sorted list of MIDI pitches belonging to this chord
    """
    intervals = GOSPEL_CHORD_TONES.get(quality, [0, 4, 7])
    root_pc = root_midi % 12

    pitches = []
    for octave_start in range(0, 128, 12):
        for interval in intervals:
            pitch = octave_start + root_pc + interval
            if octave_range[0] <= pitch <= octave_range[1]:
                pitches.append(pitch)

    return sorted(set(pitches))
