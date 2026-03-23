"""
Download the large MIDI datasets that failed in v1.
Separate functions so they can retry independently.

Usage:
    uv run modal run scripts/collect_data_modal_v2.py
"""

import modal

app = modal.App("piano-data-v2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "datasets", "pretty_midi", "numpy")
    .apt_install("git", "wget", "unzip", "p7zip-full", "curl")
)

volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(image=image, timeout=7200, volumes={VOLUME_PATH: volume})
def download_maestro():
    """MAESTRO — small, fast, high quality piano."""
    import subprocess, os
    from pathlib import Path

    midi_dir = f"{VOLUME_PATH}/midi_collection/maestro"
    if os.path.exists(midi_dir) and len(list(Path(midi_dir).rglob("*.mid*"))) > 100:
        print(f"Already have MAESTRO", flush=True)
        return

    os.makedirs(midi_dir, exist_ok=True)
    print("Downloading MAESTRO v3 MIDI-only (57MB)...", flush=True)

    result = subprocess.run([
        "wget", "--no-check-certificate", "-O", "/tmp/maestro.zip",
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    ], timeout=600, capture_output=True, text=True)

    if os.path.exists("/tmp/maestro.zip") and os.path.getsize("/tmp/maestro.zip") > 1000:
        subprocess.run(["unzip", "-o", "/tmp/maestro.zip", "-d", midi_dir], timeout=120)
        count = len(list(Path(midi_dir).rglob("*.mid*")))
        print(f"MAESTRO: {count} files", flush=True)
    else:
        print(f"Download failed: {result.stderr[-200:]}", flush=True)

    volume.commit()


@app.function(image=image, timeout=7200, volumes={VOLUME_PATH: volume})
def download_lakh():
    """Lakh MIDI — 176K files, ~1.5GB."""
    import subprocess, os
    from pathlib import Path

    midi_dir = f"{VOLUME_PATH}/midi_collection/lakh"
    if os.path.exists(midi_dir) and len(list(Path(midi_dir).rglob("*.mid"))) > 1000:
        print(f"Already have Lakh", flush=True)
        return

    os.makedirs(midi_dir, exist_ok=True)

    # Try the clean MIDI subset first (smaller, cleaner)
    print("Downloading Lakh Clean MIDI (~800MB)...", flush=True)
    result = subprocess.run([
        "wget", "--no-check-certificate", "-O", "/tmp/lmd_clean.tar.gz",
        "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
    ], timeout=3600, capture_output=True, text=True)

    if os.path.exists("/tmp/lmd_clean.tar.gz") and os.path.getsize("/tmp/lmd_clean.tar.gz") > 10000:
        print("Extracting...", flush=True)
        subprocess.run(["tar", "-xzf", "/tmp/lmd_clean.tar.gz", "-C", midi_dir], timeout=600)
        count = len(list(Path(midi_dir).rglob("*.mid")))
        print(f"Lakh Clean: {count} MIDI files", flush=True)
    else:
        print(f"Clean MIDI failed, trying full dataset...", flush=True)
        result = subprocess.run([
            "wget", "--no-check-certificate", "-O", "/tmp/lmd_full.tar.gz",
            "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
        ], timeout=7200, capture_output=True, text=True)

        if os.path.exists("/tmp/lmd_full.tar.gz") and os.path.getsize("/tmp/lmd_full.tar.gz") > 100000:
            print("Extracting full...", flush=True)
            subprocess.run(["tar", "-xzf", "/tmp/lmd_full.tar.gz", "-C", midi_dir], timeout=1200)
            count = len(list(Path(midi_dir).rglob("*.mid")))
            print(f"Lakh Full: {count} MIDI files", flush=True)
        else:
            print(f"Both downloads failed", flush=True)

    volume.commit()


@app.function(image=image, timeout=7200, volumes={VOLUME_PATH: volume})
def download_hymnal():
    """Cyber Hymnal — 16.7K hymns in 7z archives."""
    import subprocess, os
    from pathlib import Path

    midi_dir = f"{VOLUME_PATH}/midi_collection/cyber_hymnal"
    if os.path.exists(midi_dir) and len(list(Path(midi_dir).rglob("*.mid"))) > 100:
        print(f"Already have Cyber Hymnal", flush=True)
        return

    os.makedirs(midi_dir, exist_ok=True)
    total = 0

    for i in range(1, 8):
        url = f"http://hymntime.com/tch/misc/download/tch{i}.7z"
        fname = f"/tmp/tch{i}.7z"
        print(f"Downloading part {i}/7...", flush=True)

        result = subprocess.run(
            ["wget", "--no-check-certificate", "-O", fname, url],
            timeout=600, capture_output=True, text=True
        )

        if os.path.exists(fname) and os.path.getsize(fname) > 1000:
            subprocess.run(
                ["7z", "x", "-y", f"-o{midi_dir}", fname],
                timeout=300, capture_output=True
            )
            count = len(list(Path(midi_dir).rglob("*.mid")))
            print(f"  After part {i}: {count} files total", flush=True)
            total = count
        else:
            print(f"  Part {i} failed: size={os.path.getsize(fname) if os.path.exists(fname) else 0}", flush=True)

    print(f"Cyber Hymnal total: {total} files", flush=True)
    volume.commit()


@app.function(image=image, timeout=7200, volumes={VOLUME_PATH: volume})
def download_midicaps_filtered():
    """MidiCaps — download and filter for gospel-relevant content."""
    import os, json
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    midi_dir = f"{VOLUME_PATH}/midi_collection/midicaps_gospel"
    if os.path.exists(midi_dir) and len(os.listdir(midi_dir)) > 10:
        print(f"Already have MidiCaps filtered", flush=True)
        return

    os.makedirs(midi_dir, exist_ok=True)

    # Download the metadata/captions to filter first
    print("Downloading MidiCaps metadata...", flush=True)
    try:
        meta_path = hf_hub_download(
            "amaai-lab/MidiCaps",
            filename="train.json",
            repo_type="dataset",
        )

        gospel_keywords = [
            "gospel", "worship", "hymn", "church", "spiritual", "praise",
            "soul", "r&b", "choir", "christian", "religious", "sacred",
            "blues", "folk"
        ]

        # Read and filter
        print("Filtering for gospel-relevant files...", flush=True)
        relevant_files = []
        with open(meta_path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                except:
                    continue
                caption = str(item.get("caption", "")).lower()
                if any(kw in caption for kw in gospel_keywords):
                    relevant_files.append(item)

        print(f"Found {len(relevant_files)} gospel-relevant entries", flush=True)

        # Download the actual MIDI files for relevant entries
        count = 0
        for item in relevant_files[:5000]:  # Cap at 5K
            location = item.get("location", "")
            if location:
                try:
                    midi_path = hf_hub_download(
                        "amaai-lab/MidiCaps",
                        filename=location,
                        repo_type="dataset",
                    )
                    import shutil
                    dest = f"{midi_dir}/midicaps_{count:06d}.mid"
                    shutil.copy2(midi_path, dest)
                    count += 1
                    if count % 100 == 0:
                        print(f"  Downloaded {count} MIDI files", flush=True)
                except Exception:
                    pass

        print(f"MidiCaps gospel: {count} files saved", flush=True)
    except Exception as e:
        print(f"MidiCaps error: {e}", flush=True)

    volume.commit()


@app.local_entrypoint()
def main():
    # Run all downloads in parallel
    handles = [
        download_maestro.spawn(),
        download_lakh.spawn(),
        download_hymnal.spawn(),
        download_midicaps_filtered.spawn(),
    ]

    # Wait for all to complete
    for h in handles:
        try:
            h.get()
        except Exception as e:
            print(f"One download failed: {e}")

    print("\nAll downloads complete!")
