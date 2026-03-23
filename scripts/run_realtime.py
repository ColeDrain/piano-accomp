"""
Real-time piano accompaniment — sing into your mic, hear piano.

Uses rule-based chord detection + pattern retrieval from POP909.
No ML checkpoints needed.

Usage:
    uv run python scripts/run_realtime.py --bpm 80

Controls:
    t       - Tap tempo
    b <num> - Set BPM
    q       - Quit
"""

import argparse
from src.inference.realtime_engine import RealtimeEngine


def input_loop(engine: RealtimeEngine):
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


def main():
    parser = argparse.ArgumentParser(description="Real-time piano accompaniment")
    parser.add_argument("--soundfont", type=str, default="soundfonts/piano.sf2")
    parser.add_argument("--bpm", type=float, default=80.0)
    parser.add_argument("--pop909-dir", type=str, default="data/raw/pop909")
    args = parser.parse_args()

    engine = RealtimeEngine(
        soundfont_path=args.soundfont,
        pop909_dir=args.pop909_dir,
        bpm=args.bpm,
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
