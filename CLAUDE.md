# Piano Accompaniment Project

Real-time ML gospel piano accompaniment system for Nigerian church worship.
Listens to a vocalist and generates piano accompaniment.

## Target Style

Nigerian gospel worship — Sinach, Nathaniel Bassey, Mercy Chinwo, Tim Godfrey, Frank Edwards style.
Highlife/Afrobeat influenced rhythms, call-and-response, key modulations, praise & worship flow.

## Architecture

**Current working pipeline (rule-based):**
1. Audio input → torchcrepe pitch detection → MIDI notes
2. Key detection (Krumhansl-Schmuckler)
3. Rule-based chord detection (template matching, diatonic constraint)
4. Pattern retrieval from 80K library (POP909 + gospel hymn MIDIs)
5. FluidSynth → audio output

**ML pipeline (needs more data to work well):**
1. Chord Predictor (~700K params, decomposed: root+triad+seventh+bass)
2. Texture Generator (~34M params, structured chord conditioning)
3. Trained on POP909 (909 songs) — insufficient for good quality

## Current Status

- Pipeline works end-to-end: vocal audio → pitch → chords → piano patterns → audio
- Pitch detection accurate (torchcrepe)
- Chord detection reasonable for offline processing
- **Quality is poor** — patterns sound mechanical, not musical
- Root cause: insufficient training data (909 pop songs, no Nigerian gospel)

## Next Steps (Priority Order)

### 1. Massive Data Collection
- **GigaMIDI** (2.1M MIDIs) — filter for gospel/R&B/soul: huggingface.co/datasets/Metacreation/GigaMIDI
- **Los Angeles MIDI Dataset** (600K MIDIs with genre labels): huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset
- **Chordonomicon** (666K chord progressions): huggingface.co/datasets/ailsntua/Chordonomicon
- **Cyber Hymnal** (16,700 hymn MIDIs): hymntime.com/tch/misc/download.htm
- **Lakh MIDI** (176K MIDIs): colinraffel.com/projects/lmd/
- **Free-Chord-Progressions** (3,200 MIDI): github.com/BenLeon2001/Free-Chord-Progressions

### 2. Nigerian Gospel Data (THE MOAT)
- YouTube → MIDI pipeline: yt-dlp + Basic Pitch/Aria-AMT
- Target channels: Nigerian gospel piano tutorials (clean solo piano)
- Search: "how to play [Sinach/Nathaniel Bassey song] on piano"
- God's Gospel MIDI collection: godsgospel.com/midis.htm

### 3. Re-train with Full Data
- Pre-train chord predictor on Lakh + Chordonomicon (200K+ examples)
- Fine-tune on gospel/worship filtered subset
- Re-train texture generator on filtered gospel patterns
- Or: keep retrieval approach but with massive gospel-specific pattern library

## Project Structure

- `src/tokenizer/` — True REMI tokenizer with Bar + Position tokens
- `src/model/` — Chord predictor (decomposed heads) + Texture generator (structured chord conditioning)
- `src/training/` — PyTorch Dataset, losses, metrics
- `src/inference/` — Pitch detector, beat tracker, rule-based chords, pattern retrieval, FluidSynth
- `src/gospel/` — Gospel chord vocabulary, voicing rules, augmentation
- `scripts/` — train, evaluate, accompany_audio, run_realtime, finetune_gospel
- `data/raw/pop909/` — POP909 dataset (gitignored)
- `data/raw/gospel/real/` — 234 gospel/hymn MIDI files
- `data/processed/` — Tokenized data + pattern library (80K patterns)

## Commands

```bash
uv sync --extra dev                      # Install deps
uv run pytest tests/ -v                  # Run tests (23 passing)

# Offline accompaniment (works now)
uv run python scripts/accompany_audio.py --input vocals.wav --output output.wav

# Real-time (works but quality needs improvement)
uv run python scripts/run_simple_realtime.py

# Training on Modal
uv run modal run scripts/train_modal.py

# Evaluation
uv run python scripts/evaluate.py --chord-checkpoint checkpoints/chord_best.pt --texture-checkpoint checkpoints/texture_best.pt --render
```

## Conventions

- Use `uv` for all package management (not pip)
- No Co-Authored-By in git commits
- Pure Python (Rust for real-time engine is a future optimization)
- Target: Nigerian gospel worship style
