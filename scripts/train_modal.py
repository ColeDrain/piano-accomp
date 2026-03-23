"""
Train piano accompaniment models on Modal GPUs.

Usage:
    uv run modal run scripts/train_modal.py
"""

import modal

app = modal.App("piano-accomp-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchaudio",
        "pretty_midi", "mido", "pyyaml", "numpy",
    )
    .apt_install("git")
)

volume = modal.Volume.from_name("piano-accomp-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,
    volumes={VOLUME_PATH: volume},
)
def train_all():
    import subprocess, os, sys, shutil

    work_dir = "/root/piano-accomp"
    data_cache = f"{VOLUME_PATH}/pop909"
    processed_cache = f"{VOLUME_PATH}/processed_v2"
    checkpoints_dir = f"{VOLUME_PATH}/checkpoints_v2"

    # Step 1: Clone repo
    print("=" * 60, flush=True)
    print("STEP 1: Clone repo", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ColeDrain/piano-accomp.git", work_dir],
        check=True,
    )
    os.chdir(work_dir)
    sys.path.insert(0, work_dir)

    # Step 2: POP909
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Download POP909", flush=True)
    pop909_dir = f"{work_dir}/data/raw/pop909"
    if os.path.exists(f"{data_cache}/POP909"):
        print("Using cached POP909...", flush=True)
        os.makedirs(f"{work_dir}/data/raw", exist_ok=True)
        os.symlink(data_cache, pop909_dir)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/music-x-lab/POP909-Dataset.git", pop909_dir],
            check=True,
        )
        shutil.copytree(pop909_dir, data_cache, dirs_exist_ok=True)
        volume.commit()

    # Step 3: Preprocess
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Preprocess", flush=True)
    processed_dir = f"{work_dir}/data/processed"

    if os.path.exists(f"{processed_cache}/train.pt"):
        print("Using cached processed data...", flush=True)
        os.makedirs(processed_dir, exist_ok=True)
        for f in ["train.pt", "val.pt", "test.pt", "metadata.json"]:
            src = f"{processed_cache}/{f}"
            if os.path.exists(src):
                shutil.copy2(src, f"{processed_dir}/{f}")
    else:
        os.environ["PYTHONPATH"] = work_dir
        from data.scripts.preprocess import preprocess_all
        preprocess_all()
        os.makedirs(processed_cache, exist_ok=True)
        for f in os.listdir(processed_dir):
            shutil.copy2(f"{processed_dir}/{f}", f"{processed_cache}/{f}")
        volume.commit()

    # Step 4: Train chord predictor
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Train Chord Predictor", flush=True)
    ckpt_dir = f"{work_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    chord_ckpt = f"{ckpt_dir}/chord_best.pt"

    if os.path.exists(f"{checkpoints_dir}/chord_best.pt"):
        print("Using cached chord checkpoint...", flush=True)
        shutil.copy2(f"{checkpoints_dir}/chord_best.pt", chord_ckpt)
    else:
        # Import and run directly — no subprocess, immediate output
        import torch
        import yaml
        from scripts.train import train_chord_predictor
        import argparse

        args = argparse.Namespace(
            model="chord",
            data=f"{processed_dir}/train.pt",
            chord_checkpoint=None,
            wandb=False,
        )
        train_chord_predictor(args)

        # Cache
        os.makedirs(checkpoints_dir, exist_ok=True)
        if os.path.exists(chord_ckpt):
            shutil.copy2(chord_ckpt, f"{checkpoints_dir}/chord_best.pt")
            volume.commit()

    # Step 5: Train texture generator
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Train Texture Generator", flush=True)
    print("(This is the big one — ~30-60 min on L4)", flush=True)

    import torch
    import yaml
    from scripts.train import train_texture_generator
    import argparse

    args = argparse.Namespace(
        model="texture",
        data=f"{processed_dir}/train.pt",
        chord_checkpoint=chord_ckpt,
        wandb=False,
    )
    train_texture_generator(args)

    # Step 6: Save checkpoints
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: Save checkpoints", flush=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            shutil.copy2(f"{ckpt_dir}/{f}", f"{checkpoints_dir}/{f}")
            print(f"  Saved: {f}", flush=True)
    volume.commit()

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)


@app.local_entrypoint()
def main():
    train_all.remote()
