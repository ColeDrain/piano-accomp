"""
Download and prepare the POP909 dataset.

POP909 contains 909 pop songs with:
- Vocal melody track (MIDI)
- Piano accompaniment track (MIDI)
- Chord annotations
- Beat/tempo annotations

Source: https://github.com/music-x-lab/POP909-Dataset
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "raw" / "pop909"
REPO_URL = "https://github.com/music-x-lab/POP909-Dataset.git"


def download():
    """Clone the POP909 repository."""
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"POP909 already exists at {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning POP909 to {DATA_DIR}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)],
        check=True,
    )
    print("Done.")


def verify():
    """Verify the download by counting MIDI files."""
    midi_files = list(DATA_DIR.rglob("*.mid")) + list(DATA_DIR.rglob("*.midi"))
    print(f"Found {len(midi_files)} MIDI files in POP909")

    # Check for chord annotations
    chord_files = list(DATA_DIR.rglob("*.txt"))
    print(f"Found {len(chord_files)} annotation files")

    if len(midi_files) < 100:
        print("WARNING: Expected ~909 MIDI files. Download may be incomplete.")
        sys.exit(1)


def list_songs():
    """List all song directories with their MIDI files."""
    song_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    for song_dir in song_dirs[:10]:
        midis = list(song_dir.rglob("*.mid"))
        print(f"  {song_dir.name}: {[m.name for m in midis]}")

    if len(song_dirs) > 10:
        print(f"  ... and {len(song_dirs) - 10} more songs")


if __name__ == "__main__":
    download()
    verify()
    list_songs()
