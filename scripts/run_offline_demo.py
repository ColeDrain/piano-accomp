"""
Offline demo: Generate piano accompaniment from a MIDI melody file.

Usage:
    python scripts/run_offline_demo.py \
        --melody input_melody.mid \
        --output output_with_accomp.mid \
        --chord-checkpoint checkpoints/chord_best.pt \
        --texture-checkpoint checkpoints/texture_best.pt \
        --play  # Optional: play through FluidSynth
"""

import argparse
from pathlib import Path

import torch
import pretty_midi
import yaml

from src.tokenizer.midi_tokenizer import MidiTokenizer
from src.tokenizer.vocab import Vocabulary
from src.model.chord_predictor import ChordPredictor
from src.model.texture_generator import TextureGenerator


def load_models(args, vocab: Vocabulary, device: torch.device):
    """Load pretrained chord predictor and texture generator."""
    with open("configs/model.yaml") as f:
        cfg = yaml.safe_load(f)

    chord_cfg = cfg["chord_predictor"]
    tex_cfg = cfg["texture_generator"]

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
        dropout=0.0,
    ).to(device)
    texture_model.load_state_dict(torch.load(args.texture_checkpoint, map_location=device))
    texture_model.eval()

    return chord_model, texture_model


def generate_accompaniment(
    melody_midi: pretty_midi.PrettyMIDI,
    chord_model: ChordPredictor,
    texture_model: TextureGenerator,
    tokenizer: MidiTokenizer,
    vocab: Vocabulary,
    device: torch.device,
    window_beats: float = 8.0,
    hop_beats: float = 4.0,
    temperature: float = 0.8,
) -> pretty_midi.PrettyMIDI:
    """Generate accompaniment for an entire melody."""
    bpm = 120.0
    tempos = melody_midi.get_tempo_changes()
    if len(tempos[1]) > 0:
        bpm = tempos[1][0]
    beat_dur = 60.0 / bpm

    melody_notes = melody_midi.instruments[0].notes
    max_time = max(n.end for n in melody_notes)
    max_beat = max_time / beat_dur

    all_accomp_tokens = []
    beat = 0.0

    while beat < max_beat:
        end_beat = beat + window_beats

        # Get melody notes in this window
        window_notes = [
            n for n in melody_notes
            if beat * beat_dur <= n.start < end_beat * beat_dur
        ]

        if not window_notes:
            beat += hop_beats
            continue

        # Tokenize melody window
        from src.tokenizer.midi_tokenizer import NoteEvent
        mel_events = [
            NoteEvent(
                start_beat=(n.start / beat_dur) - beat,
                pitch=n.pitch,
                duration_beats=(n.end - n.start) / beat_dur,
                velocity=n.velocity,
            )
            for n in window_notes
        ]
        mel_tokens = tokenizer.encode_note_events(mel_events)
        mel_tensor = torch.tensor([mel_tokens[:32]], device=device)

        # Predict chord (decomposed heads)
        with torch.no_grad():
            result, melody_context = chord_model(mel_tensor, return_embedding=True)
            chord_components = {
                "root": result["root_logits"].argmax(dim=-1),
                "triad": result["triad_logits"].argmax(dim=-1),
                "seventh": result["seventh_logits"].argmax(dim=-1),
                "bass": result["bass_logits"].argmax(dim=-1),
            }

        # Generate accompaniment tokens
        with torch.no_grad():
            accomp_tokens = texture_model.generate(
                melody_context=melody_context,
                chord_components=chord_components,
                max_tokens=30,
                temperature=temperature,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
            )

        # Offset the accompaniment events to the current window position
        all_accomp_tokens.extend(accomp_tokens)

        beat += hop_beats

    # Decode all accompaniment tokens to MIDI
    # Wrap with BOS/EOS for the decoder
    full_tokens = [vocab.bos_id] + all_accomp_tokens + [vocab.eos_id]
    accomp_midi = tokenizer.decode_to_midi(full_tokens, bpm=bpm, program=0)

    # Combine melody + accompaniment
    output = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    # Melody track
    melody_inst = pretty_midi.Instrument(program=0, name="Melody")
    melody_inst.notes = list(melody_midi.instruments[0].notes)
    output.instruments.append(melody_inst)
    # Accompaniment track
    accomp_inst = accomp_midi.instruments[0]
    accomp_inst.name = "Accompaniment"
    output.instruments.append(accomp_inst)

    return output


def play_midi(midi_path: str):
    """Play a MIDI file through FluidSynth."""
    try:
        import subprocess
        sf_path = "soundfonts/salamander_grand_piano.sf2"
        if not Path(sf_path).exists():
            print(f"SoundFont not found at {sf_path}. Install it first.")
            print("Download from: https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/")
            return
        subprocess.run(
            ["fluidsynth", "-ni", sf_path, midi_path, "-F", "/dev/stdout"],
            check=True,
        )
    except FileNotFoundError:
        print("FluidSynth not found. Install with: brew install fluidsynth")


def main():
    parser = argparse.ArgumentParser(description="Generate piano accompaniment from melody")
    parser.add_argument("--melody", type=str, required=True, help="Input MIDI melody file")
    parser.add_argument("--output", type=str, default="output.mid", help="Output MIDI file")
    parser.add_argument("--chord-checkpoint", type=str, required=True)
    parser.add_argument("--texture-checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--play", action="store_true", help="Play output through FluidSynth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    vocab = Vocabulary()
    tokenizer = MidiTokenizer(vocab)

    print(f"Loading models on {device}...")
    chord_model, texture_model = load_models(args, vocab, device)

    print(f"Loading melody from {args.melody}...")
    melody_midi = pretty_midi.PrettyMIDI(args.melody)

    print("Generating accompaniment...")
    output = generate_accompaniment(
        melody_midi, chord_model, texture_model, tokenizer, vocab, device,
        temperature=args.temperature,
    )

    output.write(args.output)
    print(f"Output saved to {args.output}")

    if args.play:
        play_midi(args.output)


if __name__ == "__main__":
    main()
