"""
Fine-tune the texture generator on gospel data.

Mixes synthetic gospel data with POP909 to prevent catastrophic forgetting.
Uses lower learning rate and fewer epochs.

Usage:
    uv run python scripts/finetune_gospel.py \
        --gospel-dir data/raw/gospel/synthetic \
        --pop-data data/processed/train.pt \
        --chord-checkpoint checkpoints/chord_best.pt \
        --texture-checkpoint checkpoints/texture_best.pt
"""

import argparse
import math
import random
import time
from pathlib import Path

import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import yaml

from src.tokenizer.vocab import Vocabulary, quality_to_triad, quality_to_seventh, PITCH_NAMES
from src.tokenizer.midi_tokenizer import MidiTokenizer, NoteEvent
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator
from src.training.dataset import TextureDataset, collate_texture
from src.training.losses import TextureGenerationLoss
from src.training.metrics import token_perplexity


def preprocess_gospel_midi(midi_dir: Path, tokenizer: MidiTokenizer, vocab: Vocabulary) -> list[dict]:
    """Preprocess gospel MIDI files into training windows."""
    windows = []

    for midi_path in sorted(midi_dir.glob("*.mid")):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            continue

        if len(midi.instruments) < 2:
            continue

        bpm = 80.0
        tempos = midi.get_tempo_changes()
        if len(tempos[1]) > 0:
            bpm = tempos[1][0]
        beat_dur = 60.0 / bpm

        # Track 0 = piano accompaniment, Track 1 = melody (in our generated files)
        accomp_inst = midi.instruments[0]
        melody_inst = midi.instruments[1] if len(midi.instruments) > 1 else midi.instruments[0]

        mel_events = [
            NoteEvent(n.start / beat_dur, n.pitch, (n.end - n.start) / beat_dur, n.velocity)
            for n in melody_inst.notes
        ]
        acc_events = [
            NoteEvent(n.start / beat_dur, n.pitch, (n.end - n.start) / beat_dur, n.velocity)
            for n in accomp_inst.notes
        ]

        if not mel_events or not acc_events:
            continue

        mel_tokens = tokenizer.encode_note_events(mel_events)[:48]
        acc_tokens = tokenizer.encode_note_events(acc_events)[:128]

        # Default chord label (C major for key-normalized data)
        chord_label = vocab.encode_chord("C", "maj")

        windows.append({
            "melody_tokens": torch.tensor(mel_tokens, dtype=torch.long),
            "chord_label": chord_label,
            "accomp_tokens": torch.tensor(acc_tokens, dtype=torch.long),
            "song_id": 9999,  # Gospel marker
        })

    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gospel-dir", type=str, required=True)
    parser.add_argument("--pop-data", type=str, required=True)
    parser.add_argument("--chord-checkpoint", type=str, required=True)
    parser.add_argument("--texture-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/texture_gospel.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gospel-ratio", type=float, default=0.6,
                        help="Fraction of each batch that is gospel data (rest is POP909)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    # Load configs
    with open("configs/model.yaml") as f:
        cfg = yaml.safe_load(f)
    chord_cfg = cfg["chord_predictor"]
    tex_cfg = cfg["texture_generator"]

    # Preprocess gospel data
    print("Preprocessing gospel MIDI files...")
    gospel_windows = preprocess_gospel_midi(Path(args.gospel_dir), tokenizer, vocab)
    print(f"  Gospel windows: {len(gospel_windows)}")

    # Save gospel processed data
    gospel_pt = Path(args.gospel_dir).parent / "gospel_processed.pt"
    torch.save(gospel_windows, gospel_pt)

    # Load POP909 data and subsample (if available)
    pop_sample = []
    if Path(args.pop_data).exists():
        pop_data = torch.load(args.pop_data, weights_only=False)
        pop_sample_size = int(len(gospel_windows) * (1 - args.gospel_ratio) / args.gospel_ratio)
        pop_sample = random.sample(pop_data, min(pop_sample_size, len(pop_data)))
        print(f"  POP909 subsample: {len(pop_sample)}")
    else:
        print(f"  POP909 data not found at {args.pop_data} — using gospel only")

    # Combine
    combined = gospel_windows + pop_sample
    random.shuffle(combined)
    print(f"  Combined: {len(combined)} windows ({args.gospel_ratio*100:.0f}% gospel)")

    # Save combined
    combined_pt = Path("data/processed/gospel_finetune.pt")
    combined_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, combined_pt)

    # Dataset
    finetune_ds = TextureDataset(combined_pt, chord_cfg["max_melody_tokens"], tex_cfg["max_seq_len"])
    loader = DataLoader(
        finetune_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=lambda b: collate_texture(b, vocab.pad_id),
        num_workers=0,
    )

    # Load models
    chord_model = ChordPredictor(
        vocab_size=vocab.size,
        embed_dim=chord_cfg["embed_dim"],
        num_layers=chord_cfg["num_layers"],
        num_heads=chord_cfg["num_heads"],
        ffn_dim=chord_cfg["ffn_dim"],
        max_melody_tokens=chord_cfg["max_melody_tokens"],
        dropout=0.0,
    ).to(device)
    chord_model.load_state_dict(torch.load(args.chord_checkpoint, map_location=device))
    chord_model.eval()
    print(f"Loaded chord predictor from {args.chord_checkpoint}")

    texture_model = TextureGenerator(
        vocab_size=vocab.size,
        embed_dim=tex_cfg["embed_dim"],
        num_layers=tex_cfg["num_layers"],
        num_heads=tex_cfg["num_heads"],
        ffn_dim=tex_cfg["ffn_dim"],
        max_seq_len=tex_cfg["max_seq_len"],
        melody_context_dim=chord_cfg["embed_dim"],
        chord_embed_dim=tex_cfg["chord_embed_dim"],
        style_embed_dim=tex_cfg["style_embed_dim"],
        dropout=tex_cfg["dropout"],
    ).to(device)
    texture_model.load_state_dict(torch.load(args.texture_checkpoint, map_location=device))
    print(f"Loaded texture generator from {args.texture_checkpoint}")

    # Fine-tune with lower LR
    optimizer = torch.optim.AdamW(texture_model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = TextureGenerationLoss(vocab.size, vocab.pad_id)

    print(f"\nFine-tuning for {args.epochs} epochs at lr={args.lr}...")
    best_loss = float("inf")
    patience = 0

    for epoch in range(args.epochs):
        texture_model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            melody = batch["melody_tokens"].to(device)
            accomp_in = batch["accomp_input"].to(device)
            accomp_tgt = batch["accomp_target"].to(device)

            with torch.no_grad():
                result, melody_context = chord_model(melody, return_embedding=True)
                chord_components = {
                    "root": result["root_logits"].argmax(dim=-1),
                    "triad": result["triad_logits"].argmax(dim=-1),
                    "seventh": result["seventh_logits"].argmax(dim=-1),
                    "bass": result["bass_logits"].argmax(dim=-1),
                }

            logits, _ = texture_model(accomp_in, melody_context, chord_components)
            loss = loss_fn(logits, accomp_tgt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(texture_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / n
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(texture_model.state_dict(), args.output)
        else:
            patience += 1
            if patience >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Gospel-finetuned model saved to {args.output}")


if __name__ == "__main__":
    main()
