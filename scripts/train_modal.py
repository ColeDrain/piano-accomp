"""
Train piano accompaniment models on Modal GPUs.

Usage:
    uv run modal run scripts/train_modal.py
"""

import modal

app = modal.App("piano-accomp-train")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchaudio",
        "pretty_midi", "mido", "pyyaml", "numpy",
    )
    .apt_install("git")
)

# Volume to persist checkpoints and data across runs
volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,  # 4 hours max
    volumes={VOLUME_PATH: volume},
)
def train_all():
    """Download data, preprocess, train both models."""
    import subprocess
    import os
    import shutil

    work_dir = "/root/piano-accomp"
    data_cache = f"{VOLUME_PATH}/pop909"
    checkpoints_dir = f"{VOLUME_PATH}/checkpoints"

    # Clone repo
    print("=" * 60)
    print("STEP 1: Clone repo")
    print("=" * 60)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ColeDrain/piano-accomp.git", work_dir],
        check=True,
    )
    os.chdir(work_dir)

    # Download POP909 (cache on volume so we don't re-download)
    print("\n" + "=" * 60)
    print("STEP 2: Download POP909")
    print("=" * 60)
    pop909_dir = f"{work_dir}/data/raw/pop909"
    if os.path.exists(f"{data_cache}/POP909"):
        print("Using cached POP909 from volume...")
        os.makedirs(f"{work_dir}/data/raw", exist_ok=True)
        os.symlink(data_cache, pop909_dir)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/music-x-lab/POP909-Dataset.git", pop909_dir],
            check=True,
        )
        # Cache to volume
        print("Caching POP909 to volume...")
        shutil.copytree(pop909_dir, data_cache, dirs_exist_ok=True)
        volume.commit()

    # Check for cached processed data
    print("\n" + "=" * 60)
    print("STEP 3: Preprocess")
    print("=" * 60)
    processed_cache = f"{VOLUME_PATH}/processed"
    processed_dir = f"{work_dir}/data/processed"

    if os.path.exists(f"{processed_cache}/train.pt"):
        print("Using cached processed data from volume...")
        os.makedirs(processed_dir, exist_ok=True)
        for f in ["train.pt", "val.pt", "test.pt", "metadata.json"]:
            src = f"{processed_cache}/{f}"
            if os.path.exists(src):
                shutil.copy2(src, f"{processed_dir}/{f}")
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = work_dir
        subprocess.run(
            ["python", "data/scripts/preprocess.py"],
            check=True, env=env,
        )
        # Cache to volume
        os.makedirs(processed_cache, exist_ok=True)
        for f in os.listdir(processed_dir):
            shutil.copy2(f"{processed_dir}/{f}", f"{processed_cache}/{f}")
        volume.commit()

    env = os.environ.copy()
    env["PYTHONPATH"] = work_dir
    ckpt_dir = f"{work_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Reuse cached chord checkpoint if available
    chord_ckpt = f"{ckpt_dir}/chord_best.pt"
    if os.path.exists(f"{checkpoints_dir}/chord_best.pt"):
        print("\n" + "=" * 60)
        print("STEP 4: Using cached Chord Predictor checkpoint")
        print("=" * 60)
        shutil.copy2(f"{checkpoints_dir}/chord_best.pt", chord_ckpt)
    else:
        print("\n" + "=" * 60)
        print("STEP 4: Train Chord Predictor")
        print("=" * 60)
        subprocess.run(
            ["python", "scripts/train.py", "--model", "chord",
             "--data", f"{processed_dir}/train.pt"],
            check=True, env=env,
        )
        # Cache chord checkpoint immediately
        os.makedirs(checkpoints_dir, exist_ok=True)
        if os.path.exists(chord_ckpt):
            shutil.copy2(chord_ckpt, f"{checkpoints_dir}/chord_best.pt")
            volume.commit()

    # Train texture generator
    print("\n" + "=" * 60)
    print("STEP 5: Train Texture Generator")
    print("=" * 60)
    subprocess.run(
        ["python", "scripts/train.py", "--model", "texture",
         "--data", f"{processed_dir}/train.pt",
         "--chord-checkpoint", chord_ckpt],
        check=True, env=env,
    )

    # Save all checkpoints to volume
    print("\n" + "=" * 60)
    print("STEP 6: Save checkpoints")
    print("=" * 60)
    os.makedirs(checkpoints_dir, exist_ok=True)
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            shutil.copy2(f"{ckpt_dir}/{f}", f"{checkpoints_dir}/{f}")
            print(f"  Saved: {f}")
    volume.commit()

    print("\n" + "=" * 60)
    print("DONE — checkpoints saved to Modal volume 'piano-accomp-data'")
    print("=" * 60)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def download_checkpoints():
    """Download trained checkpoints from the Modal volume."""
    import os
    checkpoints_dir = f"{VOLUME_PATH}/checkpoints"
    files = []
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".pt"):
                path = f"{checkpoints_dir}/{f}"
                size_mb = os.path.getsize(path) / 1e6
                files.append(f"{f} ({size_mb:.1f} MB)")
    return files if files else ["No checkpoints found"]


@app.local_entrypoint()
def main():
    train_all.remote()
