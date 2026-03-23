"""
Download and filter massive MIDI datasets on Modal.

Downloads:
1. MidiCaps (168K with text captions — filter for gospel/worship/soul)
2. Lakh MIDI (176K — pre-training)
3. MAESTRO (1.2K high-quality piano)
4. Cyber Hymnal (16.7K hymns)
5. 251 Gospel Songs (archive.org)
6. Free-Chord-Progressions (3.2K)

Filters gospel/worship content and saves to Modal volume.

Usage:
    uv run modal run scripts/collect_data_modal.py
"""

import modal

app = modal.App("piano-data-collect")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "datasets", "pretty_midi", "numpy", "tqdm")
    .apt_install("git", "wget", "unzip", "p7zip-full")
)

volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=image,
    timeout=14400,  # 4 hours
    volumes={VOLUME_PATH: volume},
    ephemeral_disk=512 * 1024,  # 512GB scratch disk
)
def collect_all():
    import os
    import subprocess
    import shutil
    import json
    from pathlib import Path

    midi_dir = f"{VOLUME_PATH}/midi_collection"
    os.makedirs(midi_dir, exist_ok=True)

    # ============================================================
    # 1. MidiCaps — 168K MIDIs with text captions (filter for gospel)
    # ============================================================
    print("=" * 60, flush=True)
    print("STEP 1: MidiCaps (168K MIDIs, filtering for gospel/worship)", flush=True)
    print("=" * 60, flush=True)

    midicaps_dir = f"{midi_dir}/midicaps_gospel"
    if os.path.exists(midicaps_dir) and len(os.listdir(midicaps_dir)) > 10:
        print(f"Already have MidiCaps gospel subset: {len(os.listdir(midicaps_dir))} files", flush=True)
    else:
        os.makedirs(midicaps_dir, exist_ok=True)
        try:
            from datasets import load_dataset

            print("Loading MidiCaps dataset (streaming)...", flush=True)
            ds = load_dataset("amaai-lab/MidiCaps", split="train", streaming=True)

            gospel_keywords = [
                "gospel", "worship", "hymn", "church", "spiritual", "praise",
                "soul", "r&b", "choir", "christian", "religious", "sacred",
                "african", "folk", "blues",  # Adjacent genres
            ]

            count = 0
            total_scanned = 0
            for item in ds:
                total_scanned += 1
                if total_scanned % 10000 == 0:
                    print(f"  Scanned {total_scanned} / found {count} gospel-relevant", flush=True)

                caption = item.get("caption", "").lower()
                if any(kw in caption for kw in gospel_keywords):
                    # Save the MIDI data
                    midi_data = item.get("midi", None)
                    if midi_data:
                        fname = f"midicaps_{count:06d}.mid"
                        with open(f"{midicaps_dir}/{fname}", "wb") as f:
                            if isinstance(midi_data, bytes):
                                f.write(midi_data)
                            else:
                                f.write(midi_data.encode() if isinstance(midi_data, str) else bytes(midi_data))
                        count += 1

                if total_scanned >= 168000:
                    break

            print(f"  MidiCaps: {count} gospel-relevant files from {total_scanned} scanned", flush=True)
        except Exception as e:
            print(f"  MidiCaps error: {e}", flush=True)

    volume.commit()

    # ============================================================
    # 2. Lakh MIDI (from HuggingFace mirror)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Lakh MIDI (176K files)", flush=True)
    print("=" * 60, flush=True)

    lakh_dir = f"{midi_dir}/lakh"
    if os.path.exists(lakh_dir) and len(list(Path(lakh_dir).rglob("*.mid"))) > 1000:
        count = len(list(Path(lakh_dir).rglob("*.mid")))
        print(f"Already have Lakh: {count} MIDI files", flush=True)
    else:
        os.makedirs(lakh_dir, exist_ok=True)
        try:
            print("Downloading Lakh MIDI from Kaggle/HF...", flush=True)
            from huggingface_hub import hf_hub_download, snapshot_download

            # Try HuggingFace mirror first
            snapshot_download(
                "mimbres/lakh_full",
                repo_type="dataset",
                local_dir=f"/tmp/lakh_download",
                ignore_patterns=["*.md", "*.txt"],
            )
            # Move MIDIs to volume
            for f in Path("/tmp/lakh_download").rglob("*.mid"):
                dest = f"{lakh_dir}/{f.name}"
                if not os.path.exists(dest):
                    shutil.copy2(str(f), dest)

            count = len(list(Path(lakh_dir).rglob("*.mid")))
            print(f"  Lakh: {count} MIDI files saved", flush=True)
        except Exception as e:
            print(f"  Lakh download error: {e}", flush=True)
            print("  Trying direct download...", flush=True)
            subprocess.run([
                "wget", "-q", "-O", "/tmp/lmd_full.tar.gz",
                "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
            ], timeout=3600)
            if os.path.exists("/tmp/lmd_full.tar.gz"):
                subprocess.run(["tar", "-xzf", "/tmp/lmd_full.tar.gz", "-C", lakh_dir], timeout=600)
                count = len(list(Path(lakh_dir).rglob("*.mid")))
                print(f"  Lakh: {count} MIDI files extracted", flush=True)

    volume.commit()

    # ============================================================
    # 3. MAESTRO (high-quality piano, small but excellent)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: MAESTRO (1.2K high-quality piano performances)", flush=True)
    print("=" * 60, flush=True)

    maestro_dir = f"{midi_dir}/maestro"
    if os.path.exists(maestro_dir) and len(list(Path(maestro_dir).rglob("*.mid*"))) > 100:
        count = len(list(Path(maestro_dir).rglob("*.mid*")))
        print(f"Already have MAESTRO: {count} files", flush=True)
    else:
        os.makedirs(maestro_dir, exist_ok=True)
        subprocess.run([
            "wget", "-q", "-O", "/tmp/maestro.zip",
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        ], timeout=300)
        if os.path.exists("/tmp/maestro.zip"):
            subprocess.run(["unzip", "-o", "/tmp/maestro.zip", "-d", maestro_dir], timeout=120)
            count = len(list(Path(maestro_dir).rglob("*.mid*")))
            print(f"  MAESTRO: {count} files", flush=True)

    volume.commit()

    # ============================================================
    # 4. Cyber Hymnal (16.7K hymns)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Cyber Hymnal (16.7K hymn MIDIs)", flush=True)
    print("=" * 60, flush=True)

    hymnal_dir = f"{midi_dir}/cyber_hymnal"
    if os.path.exists(hymnal_dir) and len(list(Path(hymnal_dir).rglob("*.mid"))) > 100:
        count = len(list(Path(hymnal_dir).rglob("*.mid")))
        print(f"Already have Cyber Hymnal: {count} files", flush=True)
    else:
        os.makedirs(hymnal_dir, exist_ok=True)
        # Cyber Hymnal has split 7z archives
        for i in range(1, 8):
            url = f"http://hymntime.com/tch/misc/download/tch{i}.7z"
            fname = f"/tmp/tch{i}.7z"
            print(f"  Downloading part {i}/7...", flush=True)
            subprocess.run(["wget", "-q", "-O", fname, url], timeout=300)
            if os.path.exists(fname):
                subprocess.run(["7z", "x", "-y", f"-o{hymnal_dir}", fname], timeout=120, capture_output=True)

        count = len(list(Path(hymnal_dir).rglob("*.mid")))
        print(f"  Cyber Hymnal: {count} files", flush=True)

    volume.commit()

    # ============================================================
    # 5. 251 Gospel Songs (archive.org)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: 251 Gospel Songs", flush=True)
    print("=" * 60, flush=True)

    gospel_dir = f"{midi_dir}/gospel_251"
    if os.path.exists(gospel_dir) and len(list(Path(gospel_dir).rglob("*.mid"))) > 50:
        count = len(list(Path(gospel_dir).rglob("*.mid")))
        print(f"Already have 251 Gospel: {count} files", flush=True)
    else:
        os.makedirs(gospel_dir, exist_ok=True)
        subprocess.run([
            "wget", "-q", "-O", "/tmp/gospel251.zip",
            "https://archive.org/compress/251GospelSongs"
        ], timeout=300)
        if os.path.exists("/tmp/gospel251.zip"):
            subprocess.run(["unzip", "-o", "/tmp/gospel251.zip", "-d", gospel_dir], timeout=60)
            count = len(list(Path(gospel_dir).rglob("*.mid")))
            print(f"  251 Gospel: {count} files", flush=True)

    volume.commit()

    # ============================================================
    # 6. Free-Chord-Progressions (3.2K MIDI)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: Free-Chord-Progressions (3.2K)", flush=True)
    print("=" * 60, flush=True)

    chords_dir = f"{midi_dir}/free_chords"
    if os.path.exists(chords_dir) and len(list(Path(chords_dir).rglob("*.mid"))) > 50:
        count = len(list(Path(chords_dir).rglob("*.mid")))
        print(f"Already have Free Chords: {count} files", flush=True)
    else:
        os.makedirs(chords_dir, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/BenLeon2001/Free-Chord-Progressions.git",
            "/tmp/free_chords"
        ], timeout=120)
        # Extract all zips
        for zf in Path("/tmp/free_chords").rglob("*.zip"):
            subprocess.run(["unzip", "-o", str(zf), "-d", chords_dir], timeout=60, capture_output=True)
        count = len(list(Path(chords_dir).rglob("*.mid")))
        print(f"  Free Chords: {count} MIDI files", flush=True)

    volume.commit()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total = 0
    for subdir in Path(midi_dir).iterdir():
        if subdir.is_dir():
            count = len(list(subdir.rglob("*.mid")) + list(subdir.rglob("*.MID")))
            print(f"  {subdir.name}: {count} MIDI files", flush=True)
            total += count

    print(f"\n  TOTAL: {total} MIDI files on volume", flush=True)
    print("=" * 60, flush=True)


@app.local_entrypoint()
def main():
    collect_all.remote()
