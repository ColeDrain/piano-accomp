"""
Rule-based gospel piano voicing patterns.

These rules encode how gospel pianists typically voice chords — the specific
register, spacing, and patterns that make gospel sound like gospel.

Used for:
1. Synthetic training data generation (voice chord progressions in gospel style)
2. Post-processing model output (re-voice to be more gospel-authentic)
3. Standalone rule-based accompaniment (before ML model is trained)

Gospel piano voicing principles:
- Left hand: bass notes (root, 5th) in octaves or stride pattern
- Right hand: close-position voicings with extensions (7, 9, 11, 13)
- Register: left hand C2-C4, right hand C4-C6
- Density: more notes than pop, fewer than jazz
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.gospel.chord_vocabulary import GOSPEL_CHORD_TONES
from src.tokenizer.midi_tokenizer import NoteEvent


@dataclass
class VoicingPattern:
    """A specific way to voice a chord on piano."""
    name: str
    left_hand: list[int]    # Intervals from root for left hand
    right_hand: list[int]   # Intervals from root for right hand
    lh_octave: int          # Base octave for left hand (MIDI: 36=C2, 48=C3)
    rh_octave: int          # Base octave for right hand (MIDI: 60=C4, 72=C5)


# --- Voicing templates ---

def get_block_voicing(root_pc: int, quality: str) -> tuple[list[int], list[int]]:
    """Block chord voicing: full chord in right hand, root+5th in left.

    This is the most basic gospel voicing — thick and full.
    """
    intervals = GOSPEL_CHORD_TONES.get(quality, [0, 4, 7])

    # Left hand: root and 5th in low register
    lh_root = 36 + root_pc  # C2 range
    lh = [lh_root, lh_root + 12]  # Root octave

    # Right hand: chord tones in close position, C4-C5 range
    rh_base = 60 + root_pc  # C4 range
    rh = []
    for interval in intervals:
        pitch = rh_base + interval
        # Keep in C4-C6 range
        while pitch < 60:
            pitch += 12
        while pitch > 84:
            pitch -= 12
        rh.append(pitch)

    return sorted(set(lh)), sorted(set(rh))


def get_stride_voicing(
    root_pc: int, quality: str, beat_in_bar: float
) -> tuple[list[int], list[int]]:
    """Stride pattern: bass on beats 1,3 — chord on beats 2,4.

    Classic gospel stride gives a bouncy, driving feel.
    """
    intervals = GOSPEL_CHORD_TONES.get(quality, [0, 4, 7])

    beat_int = int(beat_in_bar) % 4

    if beat_int in (0, 2):
        # Bass beats: root (low) or root+5th
        bass = 36 + root_pc
        lh = [bass]
        if beat_int == 0:
            lh.append(bass + 12)  # Octave on beat 1
        rh = []  # No right hand on bass beats
    else:
        # Chord beats: full voicing in mid register
        lh = []
        rh_base = 60 + root_pc
        rh = []
        for interval in intervals[:5]:  # Limit density
            pitch = rh_base + interval
            while pitch > 79:
                pitch -= 12
            rh.append(pitch)

    return sorted(set(lh)), sorted(set(rh))


def get_walking_bass_voicing(
    root_pc: int, quality: str, next_root_pc: int | None = None, beat_in_bar: float = 0.0
) -> tuple[list[int], list[int]]:
    """Walking bass: chromatic or scalar bass movement between chord roots.

    Left hand walks, right hand holds chord.
    """
    intervals = GOSPEL_CHORD_TONES.get(quality, [0, 4, 7])

    bass = 36 + root_pc
    beat_frac = beat_in_bar % 1.0

    # Walking bass on the "and" of beats
    if beat_frac > 0.4 and next_root_pc is not None:
        # Chromatic approach to next root
        next_bass = 36 + next_root_pc
        if next_bass > bass:
            bass = next_bass - 1  # Approach from below
        else:
            bass = next_bass + 1  # Approach from above

    lh = [bass]

    # Right hand: held chord voicing
    rh_base = 60 + root_pc
    rh = [rh_base + i for i in intervals[:4] if 58 <= rh_base + i <= 84]

    return sorted(set(lh)), sorted(set(rh))


def get_gospel_fill(
    root_pc: int, quality: str, fill_type: str = "run"
) -> list[NoteEvent]:
    """Generate a gospel fill pattern (runs, octave jumps, tremolo).

    Fills happen during vocal pauses — the piano "responds" to the singer.
    """
    events = []
    intervals = GOSPEL_CHORD_TONES.get(quality, [0, 4, 7])

    if fill_type == "run":
        # Ascending scale run from root, 16th notes
        scale = [0, 2, 4, 5, 7, 9, 11, 12]  # Major scale
        base_pitch = 60 + root_pc
        for i, degree in enumerate(scale):
            events.append(NoteEvent(
                start_beat=i * 0.25,
                pitch=base_pitch + degree,
                duration_beats=0.25,
                velocity=random.randint(70, 100),
            ))

    elif fill_type == "octave_run":
        # Octave run ascending — very gospel
        base_pitch = 60 + root_pc
        for i in range(4):
            # Low note
            events.append(NoteEvent(
                start_beat=i * 0.5,
                pitch=base_pitch + i * 2,
                duration_beats=0.25,
                velocity=80,
            ))
            # Octave
            events.append(NoteEvent(
                start_beat=i * 0.5 + 0.25,
                pitch=base_pitch + i * 2 + 12,
                duration_beats=0.25,
                velocity=90,
            ))

    elif fill_type == "tremolo":
        # Tremolo between two chord tones — builds energy
        if len(intervals) >= 2:
            p1 = 72 + root_pc + intervals[0]
            p2 = 72 + root_pc + intervals[1]
            for i in range(8):
                events.append(NoteEvent(
                    start_beat=i * 0.125,
                    pitch=p1 if i % 2 == 0 else p2,
                    duration_beats=0.125,
                    velocity=random.randint(75, 95),
                ))

    elif fill_type == "grace_notes":
        # Grace note slides into chord tones — gospel ornament
        for interval in intervals[:3]:
            target = 60 + root_pc + interval
            # Slide from half-step below
            events.append(NoteEvent(
                start_beat=0.0,
                pitch=target - 1,
                duration_beats=0.0625,
                velocity=60,
            ))
            events.append(NoteEvent(
                start_beat=0.0625,
                pitch=target,
                duration_beats=0.5,
                velocity=85,
            ))

    return events


def voice_progression(
    chords: list[tuple[int, str]],
    beats_per_chord: float = 4.0,
    style: str = "block",
) -> list[NoteEvent]:
    """Voice an entire chord progression in a specific gospel style.

    Args:
        chords: List of (root_pitch_class, quality) tuples
        beats_per_chord: Beats per chord change
        style: "block", "stride", "walking"

    Returns:
        List of NoteEvents for the full accompaniment
    """
    all_events = []
    beat = 0.0

    for i, (root_pc, quality) in enumerate(chords):
        next_root = chords[i + 1][0] if i + 1 < len(chords) else None

        # Generate notes for each beat within this chord
        for sub_beat in range(int(beats_per_chord)):
            beat_in_bar = sub_beat % 4

            if style == "stride":
                lh, rh = get_stride_voicing(root_pc, quality, beat_in_bar)
            elif style == "walking":
                lh, rh = get_walking_bass_voicing(
                    root_pc, quality, next_root, beat_in_bar
                )
            else:  # block
                lh, rh = get_block_voicing(root_pc, quality)

            # Add notes
            all_pitches = lh + rh
            for pitch in all_pitches:
                velocity = random.randint(65, 95)
                dur = 1.0 if style == "block" else 0.5
                all_events.append(NoteEvent(
                    start_beat=beat + sub_beat,
                    pitch=pitch,
                    duration_beats=dur,
                    velocity=velocity,
                ))

        # Optionally add a fill at the end of the chord
        if random.random() < 0.3 and sub_beat == int(beats_per_chord) - 1:
            fill_type = random.choice(["run", "octave_run", "tremolo", "grace_notes"])
            fill = get_gospel_fill(root_pc, quality, fill_type)
            for event in fill:
                all_events.append(NoteEvent(
                    start_beat=beat + beats_per_chord - 1 + event.start_beat,
                    pitch=event.pitch,
                    duration_beats=event.duration_beats,
                    velocity=event.velocity,
                ))

        beat += beats_per_chord

    return all_events
