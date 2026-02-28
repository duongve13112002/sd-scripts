"""
Train separate per-artist mini LoRAs from a training dataset directory.

Scans the training directory, groups image files by artist prefix (strip trailing
digits from filename), and trains a separate LoRA for each artist group.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def extract_artist(filename: str) -> str:
    """Extract artist prefix by stripping trailing digits from the stem.

    Examples:
        asou1.png -> asou
        pref10.txt -> pref
        sweetonedollar3.png -> sweetonedollar
    """
    stem = Path(filename).stem
    return re.sub(r"\d+$", "", stem)


def group_files_by_artist(train_dir: str) -> dict[str, list[str]]:
    """Group image files (and their caption pairs) by artist prefix."""
    groups: dict[str, list[str]] = defaultdict(list)
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    for fname in sorted(os.listdir(train_dir)):
        fpath = os.path.join(train_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in image_exts:
            artist = extract_artist(fname)
            groups[artist].append(fname)

    return dict(sorted(groups.items()))


def create_symlink_dir(train_dir: str, artist: str, image_files: list[str], by_artist_dir: str) -> str:
    """Create a directory with symlinks to the artist's images and captions."""
    artist_dir = os.path.join(by_artist_dir, artist)
    os.makedirs(artist_dir, exist_ok=True)

    for img_file in image_files:
        # Symlink image
        src = os.path.abspath(os.path.join(train_dir, img_file))
        dst = os.path.join(artist_dir, img_file)
        if not os.path.exists(dst):
            os.symlink(src, dst)

        # Symlink caption if it exists
        stem = os.path.splitext(img_file)[0]
        for cap_ext in [".txt", ".caption"]:
            cap_file = stem + cap_ext
            cap_src = os.path.abspath(os.path.join(train_dir, cap_file))
            if os.path.exists(cap_src):
                cap_dst = os.path.join(artist_dir, cap_file)
                if not os.path.exists(cap_dst):
                    os.symlink(cap_src, cap_dst)

    return artist_dir


def generate_dataset_config(image_dir: str) -> str:
    """Generate a temporary dataset config TOML file for a single artist directory."""
    content = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 3

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = '{image_dir}'
  num_repeats = 1
"""
    fd, path = tempfile.mkstemp(suffix=".toml", prefix="dataset_config_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def train_artist(
    artist: str,
    config_file: str,
    dataset_config_path: str,
    output_dir: str,
) -> int:
    """Run accelerate launch for a single artist LoRA."""
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "3",
        "--mixed_precision", "bf16",
        "anima_train_network.py",
        "--config_file", config_file,
        "--dataset_config", dataset_config_path,
        "--output_dir", output_dir,
        "--output_name", artist,
    ]

    print(f"\n{'='*60}")
    print(f"Training LoRA for artist: {artist}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train per-artist mini LoRAs")
    parser.add_argument("--config", type=str, default="training_mini_config.toml", help="Training config TOML (without dataset_config/output_name)")
    parser.add_argument("--train_dir", type=str, default="train_datasets", help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="output_mini", help="Output directory for trained LoRAs")
    parser.add_argument("--count", type=int, default=None, help="Number of artists to train (default: all)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Group files by artist
    groups = group_files_by_artist(args.train_dir)
    artists = list(groups.keys())

    if args.count is not None:
        artists = artists[: args.count]

    print(f"Found {len(groups)} artists total, training {len(artists)}: {', '.join(artists)}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up .by_artist directory for symlinks
    by_artist_dir = os.path.join(args.train_dir, ".by_artist")
    os.makedirs(by_artist_dir, exist_ok=True)

    temp_configs = []
    failed = []

    try:
        for artist in artists:
            # Create symlink dir
            artist_dir = create_symlink_dir(args.train_dir, artist, groups[artist], by_artist_dir)

            # Generate temp dataset config
            dataset_config = generate_dataset_config(artist_dir)
            temp_configs.append(dataset_config)

            # Train
            ret = train_artist(artist, args.config, dataset_config, args.output_dir)
            if ret != 0:
                print(f"Warning: Training failed for {artist} (exit code {ret})")
                failed.append(artist)
    finally:
        # Clean up temp dataset configs
        for cfg in temp_configs:
            try:
                os.unlink(cfg)
            except OSError:
                pass

        # Clean up symlink dirs
        if os.path.exists(by_artist_dir):
            shutil.rmtree(by_artist_dir, ignore_errors=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Trained: {len(artists) - len(failed)}/{len(artists)} artists")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
