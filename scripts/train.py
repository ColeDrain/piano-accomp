"""
Training entry point for both the Chord Predictor and Texture Generator.

Usage:
    python scripts/train.py --model chord --data data/processed/train.pt
    python scripts/train.py --model texture --data data/processed/train.pt \
        --chord-checkpoint checkpoints/chord_best.pt
"""

import argparse
import math
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
from src.training.losses import TextureGenerationLoss
from src.training.metrics import token_perplexity


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Chord Predictor Training (4 decomposed heads)
# =====================================================================

def train_chord_predictor(args):
    """Train the decomposed Chord Predictor (root + triad + seventh + bass)."""
    device = get_device()
    print(f"Device: {device}")

    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)["chord_predictor"]
    with open("configs/training.yaml") as f:
        train_cfg = yaml.safe_load(f)["chord_predictor"]

    vocab = Vocabulary()

    # Data
    collate_fn = partial(collate_chord, pad_id=vocab.pad_id)
    train_ds = ChordDataset(args.data, model_cfg["max_melody_tokens"])
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    val_loader = None
    val_path = Path(args.data).parent / "val.pt"
    if val_path.exists():
        val_ds = ChordDataset(val_path, model_cfg["max_melody_tokens"])
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=2,
            persistent_workers=True,
        )

    # Model
    model = ChordPredictor(
        vocab_size=vocab.size,
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        max_melody_tokens=model_cfg["max_melody_tokens"],
        dropout=model_cfg["dropout"],
    ).to(device)

    print(f"Chord Predictor: {model.get_num_params():,} parameters")
    print(f"  Heads: root({model.num_roots}) + triad({model.num_triads}) + "
          f"seventh({model.num_sevenths}) + bass({model.num_bass})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps)

    # Losses — one per head
    smoothing = train_cfg.get("label_smoothing", 0.1)
    root_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    triad_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    seventh_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    bass_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    bass_weight = model.bass_loss_weight

    if args.wandb:
        import wandb
        wandb.init(project="piano-accomp", name=f"chord-{time.strftime('%m%d-%H%M')}")

    best_val_score = 0.0
    patience_counter = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        t_loss = t_root = t_triad = t_sev = t_bass = 0.0
        n_batches = 0

        for batch in train_loader:
            melody = batch["melody_tokens"].to(device)
            root = batch["root_label"].to(device)
            triad = batch["triad_label"].to(device)
            seventh = batch["seventh_label"].to(device)
            bass = batch["bass_label"].to(device)

            result = model(melody)

            loss = (
                root_loss_fn(result["root_logits"], root)
                + triad_loss_fn(result["triad_logits"], triad)
                + seventh_loss_fn(result["seventh_logits"], seventh)
                + bass_weight * bass_loss_fn(result["bass_logits"], bass)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()

            t_loss += loss.item()
            t_root += (result["root_logits"].argmax(-1) == root).float().mean().item()
            t_triad += (result["triad_logits"].argmax(-1) == triad).float().mean().item()
            t_sev += (result["seventh_logits"].argmax(-1) == seventh).float().mean().item()
            t_bass += (result["bass_logits"].argmax(-1) == bass).float().mean().item()
            n_batches += 1

        # Averages
        t_loss /= n_batches
        t_root /= n_batches
        t_triad /= n_batches
        t_sev /= n_batches
        t_bass /= n_batches

        # Validation
        v_root = v_triad = v_sev = v_bass = 0.0
        if val_loader:
            model.eval()
            vr = vt = vs = vb = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    melody = batch["melody_tokens"].to(device)
                    root = batch["root_label"].to(device)
                    triad = batch["triad_label"].to(device)
                    seventh = batch["seventh_label"].to(device)
                    bass = batch["bass_label"].to(device)

                    result = model(melody)
                    vr += (result["root_logits"].argmax(-1) == root).float().mean().item()
                    vt += (result["triad_logits"].argmax(-1) == triad).float().mean().item()
                    vs += (result["seventh_logits"].argmax(-1) == seventh).float().mean().item()
                    vb += (result["bass_logits"].argmax(-1) == bass).float().mean().item()
                    vn += 1
            v_root = vr / vn
            v_triad = vt / vn
            v_sev = vs / vn
            v_bass = vb / vn

        # Combined = product of individual accuracies
        val_combined = v_root * v_triad * v_sev

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Loss: {t_loss:.3f} | "
            f"Root: {t_root:.3f}/{v_root:.3f} | "
            f"Triad: {t_triad:.3f}/{v_triad:.3f} | "
            f"7th: {t_sev:.3f}/{v_sev:.3f} | "
            f"Bass: {t_bass:.3f}/{v_bass:.3f} | "
            f"Comb: {val_combined:.3f}"
        )

        if args.wandb:
            wandb.log({
                "chord/loss": t_loss,
                "chord/train_root": t_root, "chord/val_root": v_root,
                "chord/train_triad": t_triad, "chord/val_triad": v_triad,
                "chord/train_seventh": t_sev, "chord/val_seventh": v_sev,
                "chord/train_bass": t_bass, "chord/val_bass": v_bass,
                "chord/val_combined": val_combined,
            })

        if val_combined > best_val_score:
            best_val_score = val_combined
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "chord_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for 5 epochs)")
                break

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"chord_epoch{epoch+1}.pt")

    print(f"\nBest combined val score: {best_val_score:.3f}")
    print(f"Best model saved to {ckpt_dir / 'chord_best.pt'}")


# =====================================================================
# Texture Generator Training
# =====================================================================

def train_texture_generator(args):
    """Train the Texture Generator with structured chord conditioning."""
    device = get_device()
    print(f"Device: {device}")

    with open("configs/model.yaml") as f:
        full_cfg = yaml.safe_load(f)
    model_cfg = full_cfg["texture_generator"]
    chord_cfg = full_cfg["chord_predictor"]

    with open("configs/training.yaml") as f:
        train_cfg = yaml.safe_load(f)["texture_generator"]

    vocab = Vocabulary()

    # Data
    collate_fn = partial(collate_texture, pad_id=vocab.pad_id)
    train_ds = TextureDataset(args.data, chord_cfg["max_melody_tokens"], model_cfg["max_seq_len"])
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    val_loader = None
    val_path = Path(args.data).parent / "val.pt"
    if val_path.exists():
        val_ds = TextureDataset(val_path, chord_cfg["max_melody_tokens"], model_cfg["max_seq_len"])
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=2,
            persistent_workers=True,
        )

    # Load pretrained chord predictor (frozen, for melody encoding)
    chord_model = ChordPredictor(
        vocab_size=vocab.size,
        embed_dim=chord_cfg["embed_dim"],
        num_layers=chord_cfg["num_layers"],
        num_heads=chord_cfg["num_heads"],
        ffn_dim=chord_cfg["ffn_dim"],
        max_melody_tokens=chord_cfg["max_melody_tokens"],
        dropout=0.0,
    ).to(device)

    if args.chord_checkpoint:
        chord_model.load_state_dict(torch.load(args.chord_checkpoint, map_location=device))
        print(f"Loaded chord predictor from {args.chord_checkpoint}")
    chord_model.eval()

    # Texture generator
    model = TextureGenerator(
        vocab_size=vocab.size,
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
    accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_loader) // accum) * train_cfg["epochs"]
    scheduler = get_cosine_schedule(optimizer, train_cfg["warmup_steps"], total_steps)

    loss_fn = TextureGenerationLoss(vocab.size, vocab.pad_id)

    if args.wandb:
        import wandb
        wandb.init(project="piano-accomp", name=f"texture-{time.strftime('%m%d-%H%M')}")

    best_val_ppl = float("inf")
    tex_patience = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            melody = batch["melody_tokens"].to(device)
            accomp_in = batch["accomp_input"].to(device)
            accomp_tgt = batch["accomp_target"].to(device)

            # Get melody encoding + chord predictions from frozen chord predictor
            with torch.no_grad():
                result, melody_context = chord_model(melody, return_embedding=True)
                chord_components = {
                    "root": result["root_logits"].argmax(dim=-1),
                    "triad": result["triad_logits"].argmax(dim=-1),
                    "seventh": result["seventh_logits"].argmax(dim=-1),
                    "bass": result["bass_logits"].argmax(dim=-1),
                }

            # Or use ground-truth chord labels (teacher forcing for chords)
            # Uncomment to use ground truth instead of predicted chords:
            # chord_components = {
            #     "root": batch["root_label"].to(device),
            #     "triad": batch["triad_label"].to(device),
            #     "seventh": batch["seventh_label"].to(device),
            #     "bass": batch["bass_label"].to(device),
            # }

            logits, _ = model(
                accomp_tokens=accomp_in,
                melody_context=melody_context,
                chord_components=chord_components,
            )

            loss = loss_fn(logits, accomp_tgt) / accum
            loss.backward()

            if (step + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        val_ppl = float("inf")
        if val_loader:
            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    melody = batch["melody_tokens"].to(device)
                    accomp_in = batch["accomp_input"].to(device)
                    accomp_tgt = batch["accomp_target"].to(device)

                    result, melody_context = chord_model(melody, return_embedding=True)
                    chord_components = {
                        "root": result["root_logits"].argmax(dim=-1),
                        "triad": result["triad_logits"].argmax(dim=-1),
                        "seventh": result["seventh_logits"].argmax(dim=-1),
                        "bass": result["bass_logits"].argmax(dim=-1),
                    }

                    logits, _ = model(accomp_in, melody_context, chord_components)
                    val_sum += token_perplexity(logits, accomp_tgt, vocab.pad_id)
                    val_n += 1
            val_ppl = val_sum / val_n

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"Loss: {avg_loss:.4f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if args.wandb:
            wandb.log({
                "texture/loss": avg_loss,
                "texture/val_ppl": val_ppl,
                "texture/lr": scheduler.get_last_lr()[0],
            })

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            tex_patience = 0
            torch.save(model.state_dict(), ckpt_dir / "texture_best.pt")
            # Copy to extra save dir (e.g. Modal volume) for persistence
            if hasattr(args, 'save_dir') and args.save_dir:
                import shutil
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt_dir / "texture_best.pt", save_dir / "texture_best.pt")
                print(f"  Saved to {save_dir / 'texture_best.pt'}", flush=True)
        else:
            tex_patience += 1
            if tex_patience >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"texture_epoch{epoch+1}.pt")

    print(f"\nBest val perplexity: {best_val_ppl:.2f}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train piano accompaniment models")
    parser.add_argument("--model", choices=["chord", "texture"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--chord-checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Extra directory to copy best checkpoints to (e.g. Modal volume)")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.model == "chord":
        train_chord_predictor(args)
    else:
        train_texture_generator(args)


if __name__ == "__main__":
    main()
