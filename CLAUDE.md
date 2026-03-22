# Piano Accompaniment Project

Real-time ML gospel piano accompaniment system that listens to a vocalist and generates piano accompaniment.

## Architecture

Two-stage symbolic (MIDI) pipeline:
1. **Chord Predictor** (~4M params, encoder Transformer) — melody tokens → chord class
2. **Texture Generator** (~25M params, decoder Transformer + cross-attention) — chord + melody → piano MIDI tokens
3. **FluidSynth** — MIDI → audio output

Target latency: ~100-150ms end-to-end.

## Project Structure

- `src/tokenizer/` — REMI-style MIDI tokenizer (~500 token vocab with gospel chord extensions)
- `src/model/` — Transformer architectures (chord_predictor.py, texture_generator.py, transformer.py, positional.py)
- `src/training/` — PyTorch Dataset, losses, metrics
- `src/inference/` — Real-time pipeline (pitch_detector.py, beat_tracker.py, synthesizer.py, realtime_engine.py)
- `src/gospel/` — Gospel chord vocabulary, voicing rules, synthetic data augmentation
- `scripts/` — Entry points (train.py, run_offline_demo.py, run_realtime.py)
- `data/scripts/` — Data download (POP909) and preprocessing
- `configs/` — YAML configs for model, training, inference, audio

## Current Status

- All code for Phases 0-3 is written and committed
- 20/20 tests passing
- **Next step**: Download POP909 → preprocess → train on cloud GPU (Colab MCP configured)
- Training data: POP909 (909 pop songs) for pretraining, gospel synthetic data for fine-tuning

## Commands

```bash
uv sync --extra dev          # Install all deps
uv run pytest tests/ -v      # Run tests
uv run python data/scripts/download_pop909.py   # Download dataset
uv run python data/scripts/preprocess.py        # Tokenize to training data
uv run python scripts/train.py --model chord --data data/processed/train.pt
uv run python scripts/train.py --model texture --data data/processed/train.pt --chord-checkpoint checkpoints/chord_best.pt
```

## Conventions

- Use `uv` for all package management (not pip)
- No Co-Authored-By in git commits
- Pure Python (Rust for real-time engine is a future optimization)
- Gospel music style: extended chords (7ths, 9ths, 11ths), stride bass, walking bass, call-and-response fills
