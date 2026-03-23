"""
Download Nigerian gospel piano tutorials from YouTube and transcribe to MIDI.

Pipeline: YouTube search → yt-dlp download → Basic Pitch transcription → MIDI files

Usage:
    uv run modal run scripts/collect_youtube_gospel_modal.py
"""

import modal

app = modal.App("piano-youtube-gospel")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "yt-dlp", "torch", "torchaudio", "torchcrepe",
        "numpy", "pretty_midi", "soundfile", "resampy", "setuptools",
    )
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"

# Nigerian gospel piano tutorial search queries
SEARCH_QUERIES = [
    # Nigerian worship songs - piano tutorials
    "how to play Way Maker piano tutorial",
    "how to play Excess Love piano tutorial",
    "how to play This is Not Ordinary piano Sinach",
    "how to play Great Are You Lord piano",
    "how to play Nathaniel Bassey piano",
    "how to play Mercy Chinwo piano",
    "how to play Tim Godfrey piano",
    "how to play Frank Edwards piano",
    "how to play Dunsin Oyekan piano",
    "how to play Judikay piano",
    "how to play Ada Ehi piano",
    "how to play Tasha Cobbs piano",
    "how to play Travis Greene piano",
    "Nigerian gospel piano tutorial",
    "naija worship piano tutorial",
    "african worship piano chords",
    "gospel piano chords tutorial Nigerian",
    "how to play Nigerian praise songs piano",
    "worship piano tutorial for beginners gospel",
    "gospel piano runs and chords tutorial",
    # Specific popular songs
    "how to play Imela piano tutorial",
    "how to play Onise Iyanu piano tutorial",
    "how to play You Are Yahweh piano",
    "how to play No One Like You Eben piano",
    "how to play Blessed Be Your Name piano gospel",
    "how to play Holy Spirit piano gospel",
    "how to play Good Good Father piano",
    "how to play Oceans piano gospel",
    "how to play 10000 Reasons piano gospel",
    "how to play What A Beautiful Name piano",
    # Gospel piano technique
    "gospel piano chords and progressions",
    "gospel piano worship style tutorial",
    "church piano accompaniment tutorial",
    "gospel piano left hand patterns",
    "gospel piano stride and runs tutorial",
    "african gospel piano style",
    "gospel piano 2 5 1 progression",
    "shouting music piano tutorial gospel",
    "slow worship piano tutorial",
    "praise break piano tutorial",
]


@app.function(
    image=image,
    timeout=14400,  # 4 hours
    volumes={VOLUME_PATH: volume},
    cpu=4,
)
def download_and_transcribe(query: str, max_videos: int = 3):
    """Download videos for one search query and transcribe to MIDI."""
    import subprocess
    import os
    import json
    from pathlib import Path

    midi_dir = f"{VOLUME_PATH}/midi_collection/nigerian_gospel"
    audio_dir = "/tmp/gospel_audio"
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    safe_query = query.replace(" ", "_")[:50]
    print(f"Searching: {query}", flush=True)

    # Download audio from YouTube search results
    try:
        result = subprocess.run([
            "yt-dlp",
            "--no-playlist",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--max-downloads", str(max_videos),
            "--match-filter", "duration < 600",  # Max 10 min
            "--match-filter", "duration > 30",   # Min 30 sec
            "-o", f"{audio_dir}/{safe_query}_%(autonumber)s.%(ext)s",
            f"ytsearch{max_videos}:{query}",
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"  yt-dlp error: {result.stderr[-200:]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  yt-dlp timeout", flush=True)
    except Exception as e:
        print(f"  yt-dlp error: {e}", flush=True)

    # Find downloaded WAV files
    wav_files = list(Path(audio_dir).glob(f"{safe_query}*.wav"))
    print(f"  Downloaded {len(wav_files)} audio files", flush=True)

    # Transcribe each to MIDI using torchcrepe
    transcribed = 0
    for wav_path in wav_files:
        try:
            import torch
            import soundfile as sf
            import resampy
            import torchcrepe
            import pretty_midi
            import numpy as np

            print(f"  Transcribing {wav_path.name}...", flush=True)

            # Use soundfile instead of torchaudio (avoids torchcodec)
            audio_np, sr = sf.read(str(wav_path), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            if sr != 16000:
                audio_np = resampy.resample(audio_np, sr, 16000)
                sr = 16000
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)

            # Run torchcrepe
            pitch_hz, confidence = torchcrepe.predict(
                waveform, sample_rate=sr, model="tiny",
                hop_length=160, batch_size=2048, device="cpu",
                return_periodicity=True, pad=True,
            )

            pitch_hz = pitch_hz.squeeze().numpy()
            confidence = confidence.squeeze().numpy()

            # Convert to MIDI notes
            midi = pretty_midi.PrettyMIDI(initial_tempo=80)
            inst = pretty_midi.Instrument(program=0)
            frame_dur = 160 / sr

            current_note = None
            current_start = 0

            for i in range(len(pitch_hz)):
                if confidence[i] > 0.5 and 50 < pitch_hz[i] < 2000:
                    midi_num = int(round(69 + 12 * np.log2(pitch_hz[i] / 440.0)))
                    midi_num = max(21, min(108, midi_num))
                    t = i * frame_dur

                    if current_note is None:
                        current_note = midi_num
                        current_start = t
                    elif abs(midi_num - current_note) >= 2:
                        dur = t - current_start
                        if dur > 0.05:
                            inst.notes.append(pretty_midi.Note(80, current_note, current_start, t))
                        current_note = midi_num
                        current_start = t
                else:
                    if current_note is not None:
                        t = i * frame_dur
                        dur = t - current_start
                        if dur > 0.05:
                            inst.notes.append(pretty_midi.Note(80, current_note, current_start, t))
                        current_note = None

            if current_note is not None:
                t = len(pitch_hz) * frame_dur
                if t - current_start > 0.05:
                    inst.notes.append(pretty_midi.Note(80, current_note, current_start, t))

            # Quality filter: skip if too few notes or too short
            if len(inst.notes) < 20:
                print(f"    Skipped (only {len(inst.notes)} notes)", flush=True)
                os.remove(str(wav_path))
                continue

            midi.instruments.append(inst)
            midi_name = wav_path.stem + ".mid"
            midi_path = f"{midi_dir}/{midi_name}"
            midi.write(midi_path)
            transcribed += 1
            print(f"    → {midi_name} ({len(inst.notes)} notes)", flush=True)

            os.remove(str(wav_path))
        except Exception as e:
            print(f"    Transcription error: {e}", flush=True)

    print(f"  {query}: {transcribed} MIDI files saved", flush=True)
    volume.commit()
    return transcribed


@app.function(
    image=image,
    timeout=14400,
    volumes={VOLUME_PATH: volume},
)
def summarize():
    """Count total files collected."""
    from pathlib import Path
    midi_dir = f"{VOLUME_PATH}/midi_collection/nigerian_gospel"
    count = len(list(Path(midi_dir).glob("*.mid")))
    print(f"\nTotal Nigerian gospel MIDI files: {count}", flush=True)

    # Also count all MIDI files on volume
    total = 0
    for subdir in Path(f"{VOLUME_PATH}/midi_collection").iterdir():
        if subdir.is_dir():
            c = len(list(subdir.rglob("*.mid")) + list(subdir.rglob("*.MID")))
            print(f"  {subdir.name}: {c}", flush=True)
            total += c
    print(f"\n  TOTAL MIDI FILES ON VOLUME: {total}", flush=True)
    return count


@app.local_entrypoint()
def main():
    # Launch all queries in parallel (Modal handles concurrency)
    print(f"Launching {len(SEARCH_QUERIES)} search queries in parallel...")
    handles = []
    for query in SEARCH_QUERIES:
        handles.append(download_and_transcribe.spawn(query, max_videos=3))

    # Collect results
    total = 0
    for h in handles:
        try:
            count = h.get()
            total += count
        except Exception as e:
            print(f"Query failed: {e}")

    print(f"\nTranscribed {total} new MIDI files from YouTube")

    # Final summary
    summarize.remote()
