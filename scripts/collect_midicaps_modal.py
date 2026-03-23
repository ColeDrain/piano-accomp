"""
Download gospel-filtered MidiCaps with HF authentication.

Usage:
    uv run modal run scripts/collect_midicaps_modal.py
"""

import modal

app = modal.App("piano-midicaps")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "datasets")
)

volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=image,
    timeout=7200,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
def download_midicaps():
    import os
    import json
    from pathlib import Path
    from huggingface_hub import hf_hub_download, login

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("HF authenticated", flush=True)

    midi_dir = f"{VOLUME_PATH}/midi_collection/midicaps_gospel"
    os.makedirs(midi_dir, exist_ok=True)

    existing = len([f for f in os.listdir(midi_dir) if f.endswith(".mid")])
    if existing > 100:
        print(f"Already have {existing} MidiCaps files", flush=True)
        return

    # Stream the dataset and filter
    print("Loading MidiCaps (streaming)...", flush=True)
    from datasets import load_dataset

    ds = load_dataset("amaai-lab/MidiCaps", split="train", streaming=True)

    gospel_keywords = [
        "gospel", "worship", "hymn", "church", "spiritual", "praise",
        "soul", "r&b", "choir", "christian", "religious", "sacred",
        "blues", "folk", "african",
    ]

    count = 0
    total = 0
    for item in ds:
        total += 1
        if total % 10000 == 0:
            print(f"  Scanned {total}, found {count}", flush=True)

        caption = str(item.get("caption", "")).lower()
        if any(kw in caption for kw in gospel_keywords):
            # Try to get MIDI data
            midi_bytes = item.get("midi")
            if midi_bytes is None:
                # Try other possible field names
                for key in ["midi_data", "audio", "file", "content"]:
                    midi_bytes = item.get(key)
                    if midi_bytes is not None:
                        break

            if midi_bytes is not None:
                fname = f"{midi_dir}/mc_{count:06d}.mid"
                try:
                    if isinstance(midi_bytes, bytes):
                        with open(fname, "wb") as f:
                            f.write(midi_bytes)
                    elif isinstance(midi_bytes, dict) and "bytes" in midi_bytes:
                        with open(fname, "wb") as f:
                            f.write(midi_bytes["bytes"])
                    elif isinstance(midi_bytes, str):
                        with open(fname, "wb") as f:
                            f.write(midi_bytes.encode("latin-1"))
                    count += 1
                except Exception:
                    pass

            if count >= 10000:
                break

        if total >= 170000:
            break

    print(f"\nMidiCaps: {count} gospel files from {total} scanned", flush=True)

    # Also print what fields are available
    if total > 0:
        ds2 = load_dataset("amaai-lab/MidiCaps", split="train", streaming=True)
        sample = next(iter(ds2))
        print(f"Fields in dataset: {list(sample.keys())}", flush=True)
        for k, v in sample.items():
            print(f"  {k}: {type(v).__name__} ({str(v)[:100]})", flush=True)

    volume.commit()


@app.local_entrypoint()
def main():
    download_midicaps.remote()
