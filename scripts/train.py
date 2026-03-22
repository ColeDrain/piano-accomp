"""
Training entry point for both the Chord Predictor and Texture Generator.

Usage:
    # Train chord predictor
    python scripts/train.py --model chord --data data/processed/train.pt

    # Train texture generator
    python scripts/train.py --model texture --data data/processed/train.pt

    # Train texture generator with pretrained chord predictor
    python scripts/train.py --model texture --data data/processed/train.pt \
        --chord-checkpoint checkpoints/chord_best.pt
"""

import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from src.tokenizer.vocab import Vocabulary
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator
from src.training.dataset import (
    ChordDataset, TextureDataset, collate_chord, collate_texture
)
from src.training.losses import ChordPredictionLoss, TextureGenerationLoss
from src.training.metrics import chord_accuracy, token_perplexity


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_chord_predictor(args):
    """Train the decomposed Chord Predictor (root + quality heads)."""
    device = get_device()
    print(f"Device: {device}")

    # Load config
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)["chord_predictor"]
    with open("configs/training.yaml") as f:
        train_cfg = yaml.safe_load(f)["chord_predictor"]

    vocab = Vocabulary()

    # Dataset
    collate_fn = partial(collate_chord, pad_id=vocab.pad_id)
    train_ds = ChordDataset(args.data, model_cfg["max_melody_tokens"], vocab.chord_offset)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True,
    )

    val_ds = None
    val_loader = None
    val_path = Path(args.data).parent / "val.pt"
    if val_path.exists():
        val_ds = ChordDataset(val_path, model_cfg["max_melody_tokens"], vocab.chord_offset)
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=8, persistent_workers=True,
        )

    # Model — decomposed into root (12) + quality (7) heads
    model = ChordPredictor(
        vocab_size=vocab.size,
        num_chord_classes=model_cfg["num_chord_classes"],
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        max_melody_tokens=model_cfg["max_melody_tokens"],
        dropout=model_cfg["dropout"],
    ).to(device)

    print(f"Chord Predictor: {model.get_num_params():,} parameters")
    print(f"  Root classes: {model.num_roots}, Quality classes: {model.num_qualities}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps)

    # Two losses — one per head
    smoothing = train_cfg.get("label_smoothing", 0.1)
    root_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    quality_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    # Optional: wandb
    if args.wandb:
        import wandb
        wandb.init(project="piano-accomp", name=f"chord-{time.strftime('%m%d-%H%M')}")

    # Training loop
    best_val_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        total_root_acc = 0.0
        total_qual_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            melody = batch["melody_tokens"].to(device)
            root = batch["root_label"].to(device)
            quality = batch["quality_label"].to(device)

            result = model(melody)
            loss_root = root_loss_fn(result["root_logits"], root)
            loss_qual = quality_loss_fn(result["quality_logits"], quality)
            loss = loss_root + loss_qual

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_root_acc += (result["root_logits"].argmax(-1) == root).float().mean().item()
            total_qual_acc += (result["quality_logits"].argmax(-1) == quality).float().mean().item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_root = total_root_acc / num_batches
        avg_qual = total_qual_acc / num_batches

        # Validation
        val_root = val_qual = 0.0
        if val_loader:
            model.eval()
            vr_sum = vq_sum = 0.0
            v_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    melody = batch["melody_tokens"].to(device)
                    root = batch["root_label"].to(device)
                    quality = batch["quality_label"].to(device)
                    result = model(melody)
                    vr_sum += (result["root_logits"].argmax(-1) == root).float().mean().item()
                    vq_sum += (result["quality_logits"].argmax(-1) == quality).float().mean().item()
                    v_total += 1
            val_root = vr_sum / v_total
            val_qual = vq_sum / v_total

        combined_val = val_root * val_qual  # Joint accuracy estimate

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Root: {avg_root:.3f}/{val_root:.3f} | "
            f"Quality: {avg_qual:.3f}/{val_qual:.3f} | "
            f"Combined: {combined_val:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if args.wandb:
            wandb.log({
                "chord/train_loss": avg_loss,
                "chord/train_root_acc": avg_root,
                "chord/train_quality_acc": avg_qual,
                "chord/val_root_acc": val_root,
                "chord/val_quality_acc": val_qual,
                "chord/val_combined": combined_val,
                "chord/lr": scheduler.get_last_lr()[0],
            })

        # Save best (based on combined root * quality)
        if combined_val > best_val_acc:
            best_val_acc = combined_val
            torch.save(model.state_dict(), ckpt_dir / "chord_best.pt")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"chord_epoch{epoch+1}.pt")

    print(f"\nBest combined val accuracy: {best_val_acc:.3f}")
    print(f"Best model saved to {ckpt_dir / 'chord_best.pt'}")


def train_texture_generator(args):
    """Train the Texture Generator model."""
    device = get_device()
    print(f"Device: {device}")

    # Load configs
    with open("configs/model.yaml") as f:
        full_cfg = yaml.safe_load(f)
    model_cfg = full_cfg["texture_generator"]
    chord_cfg = full_cfg["chord_predictor"]

    with open("configs/training.yaml") as f:
        train_cfg = yaml.safe_load(f)["texture_generator"]

    vocab = Vocabulary()

    # Dataset
    collate_fn = partial(collate_texture, pad_id=vocab.pad_id)
    train_ds = TextureDataset(args.data, chord_cfg["max_melody_tokens"], model_cfg["max_seq_len"], vocab.chord_offset)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True,
    )

    val_path = Path(args.data).parent / "val.pt"
    val_loader = None
    if val_path.exists():
        val_ds = TextureDataset(val_path, chord_cfg["max_melody_tokens"], model_cfg["max_seq_len"], vocab.chord_offset)
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=8, persistent_workers=True,
        )

    # Load pretrained chord predictor for melody encoding
    chord_model = ChordPredictor(
        vocab_size=vocab.size,
        num_chord_classes=chord_cfg["num_chord_classes"],
        embed_dim=chord_cfg["embed_dim"],
        num_layers=chord_cfg["num_layers"],
        num_heads=chord_cfg["num_heads"],
        ffn_dim=chord_cfg["ffn_dim"],
        max_melody_tokens=chord_cfg["max_melody_tokens"],
        dropout=0.0,  # No dropout during inference
    ).to(device)

    if args.chord_checkpoint:
        chord_model.load_state_dict(torch.load(args.chord_checkpoint, map_location=device))
        print(f"Loaded chord predictor from {args.chord_checkpoint}")
    chord_model.eval()

    # Texture generator
    model = TextureGenerator(
        vocab_size=vocab.size,
        num_chord_classes=84,  # 12 roots × 7 quality groups (decomposed)
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        melody_context_dim=chord_cfg["embed_dim"],
        chord_embed_dim=model_cfg["chord_embed_dim"],
        style_embed_dim=model_cfg["style_embed_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    print(f"Texture Generator: {model.get_num_params():,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = (len(train_loader) // train_cfg["gradient_accumulation_steps"]) * train_cfg["epochs"]
    scheduler = get_cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps)

    loss_fn = TextureGenerationLoss(vocab.size, vocab.pad_id)
    accum_steps = train_cfg["gradient_accumulation_steps"]

    if args.wandb:
        import wandb
        wandb.init(project="piano-accomp", name=f"texture-{time.strftime('%m%d-%H%M')}")

    best_val_ppl = float("inf")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            melody = batch["melody_tokens"].to(device)
            chord = batch["chord_label"].to(device)
            accomp_in = batch["accomp_input"].to(device)
            accomp_tgt = batch["accomp_target"].to(device)

            # Get melody encoding from frozen chord predictor
            with torch.no_grad():
                result, melody_context = chord_model(melody, return_embedding=True)
                # Combine root + quality into a single chord ID for texture generator
                pred_root = result["root_logits"].argmax(dim=-1)
                pred_qual = result["quality_logits"].argmax(dim=-1)
                chord_ids = pred_root * 7 + pred_qual  # 0..83

            # Forward through texture generator
            logits, _ = model(
                accomp_tokens=accomp_in,
                melody_context=melody_context,
                chord_ids=chord_ids,
            )

            loss = loss_fn(logits, accomp_tgt) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        val_ppl = float("inf")
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    melody = batch["melody_tokens"].to(device)
                    chord = batch["chord_label"].to(device)
                    accomp_in = batch["accomp_input"].to(device)
                    accomp_tgt = batch["accomp_target"].to(device)

                    result, melody_context = chord_model(melody, return_embedding=True)
                    pred_root = result["root_logits"].argmax(dim=-1)
                    pred_qual = result["quality_logits"].argmax(dim=-1)
                    chord_ids = pred_root * 7 + pred_qual
                    logits, _ = model(accomp_in, melody_context, chord_ids)
                    val_ppl_batch = token_perplexity(logits, accomp_tgt, vocab.pad_id)
                    val_loss_sum += val_ppl_batch
                    val_batches += 1
            val_ppl = val_loss_sum / val_batches

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Loss: {avg_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if args.wandb:
            wandb.log({
                "texture/train_loss": avg_loss,
                "texture/val_perplexity": val_ppl,
                "texture/lr": scheduler.get_last_lr()[0],
            })

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), ckpt_dir / "texture_best.pt")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"texture_epoch{epoch+1}.pt")

    print(f"\nBest validation perplexity: {best_val_ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train piano accompaniment models")
    parser.add_argument("--model", choices=["chord", "texture"], required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to train.pt")
    parser.add_argument("--chord-checkpoint", type=str, default=None,
                        help="Pretrained chord predictor (for texture training)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    if args.model == "chord":
        train_chord_predictor(args)
    else:
        train_texture_generator(args)


if __name__ == "__main__":
    main()
