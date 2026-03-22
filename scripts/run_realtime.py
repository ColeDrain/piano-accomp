"""
Real-time performance entry point.

Usage:
    python scripts/run_realtime.py \
        --chord-checkpoint checkpoints/chord_best.pt \
        --texture-checkpoint checkpoints/texture_best.pt \
        --bpm 80 \
        --soundfont soundfonts/salamander_grand_piano.sf2

Controls (while running):
    t       - Tap tempo (tap repeatedly to set BPM)
    b <num> - Set BPM manually (e.g., "b 120")
    q       - Quit
"""

import argparse
import sys
import threading

from src.inference.realtime_engine import RealtimeEngine


def input_loop(engine: RealtimeEngine):
    """Handle keyboard input for tap-tempo and controls."""
    print("\nControls:")
    print("  t       - Tap tempo")
    print("  b <num> - Set BPM (e.g., 'b 120')")
    print("  q       - Quit")
    print()

    while True:
        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "t":
            engine.tap_tempo()
        elif cmd.startswith("b "):
            try:
                bpm = float(cmd.split()[1])
                engine.beat_tracker.set_bpm(bpm)
                print(f"  BPM set to {bpm:.1f}")
            except (ValueError, IndexError):
                print("  Usage: b <number>")
        else:
            print("  Unknown command. Use 't' (tap), 'b <bpm>', or 'q' (quit)")


def main():
    parser = argparse.ArgumentParser(description="Real-time gospel piano accompaniment")
    parser.add_argument("--chord-checkpoint", type=str, required=True,
                        help="Path to trained chord predictor checkpoint")
    parser.add_argument("--texture-checkpoint", type=str, required=True,
                        help="Path to trained texture generator checkpoint")
    parser.add_argument("--soundfont", type=str,
                        default="soundfonts/salamander_grand_piano.sf2",
                        help="Path to piano SoundFont (.sf2)")
    parser.add_argument("--bpm", type=float, default=80.0,
                        help="Starting tempo in BPM")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more conservative)")
    args = parser.parse_args()

    engine = RealtimeEngine(
        chord_checkpoint=args.chord_checkpoint,
        texture_checkpoint=args.texture_checkpoint,
        soundfont_path=args.soundfont,
        bpm=args.bpm,
        temperature=args.temperature,
    )

    try:
        engine.start()
        input_loop(engine)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
