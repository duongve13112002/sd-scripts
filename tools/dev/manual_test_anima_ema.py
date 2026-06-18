"""GPU smoke test for the Anima EMA feature and async disk-write caching.

Runs anima_train.py (full finetune) and anima_train_network.py (LoRA) for a couple of
steps with --ema enabled and latent / text-encoder caching turned on, then checks that an
ema_ prefixed checkpoint was written. This needs real model weights and a GPU, so it is meant
to be run manually on a server, not in CI.

Usage:
    python tools/dev/manual_test_anima_ema.py \
        --image_dir /path/to/images_with_txt \
        --dit_path /path/to/dit.safetensors \
        --qwen3_path /path/to/qwen3 \
        --vae_path /path/to/vae.safetensors \
        [--t5_tokenizer_path /path/to/t5] \
        [--resolution 512]
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import shutil


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def create_dataset_toml(image_dir: str, resolution: int, toml_path: str):
    content = f"""[general]
resolution = {resolution}
enable_bucket = true
bucket_reso_steps = 8
min_bucket_reso = 256
max_bucket_reso = 1024

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  num_repeats = 1
  caption_extension = ".txt"
"""
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return toml_path


def run_test(test_name: str, cmd: list, output_dir: str, timeout: int) -> dict:
    print(f"\n{'#' * 70}")
    print(f"TEST: {test_name}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": f"exceeded {timeout}s"}
    except Exception as e:  # noqa: BLE001 - report whatever went wrong
        return {"status": "ERROR", "detail": str(e)}

    lines = (result.stdout + "\n" + result.stderr).strip().split("\n")
    print("--- Last 30 lines of output ---")
    for line in lines[-30:]:
        print(f"  {line}")
    print("--- End output ---")

    if result.returncode != 0:
        return {"status": "FAIL", "detail": f"exit code {result.returncode}"}

    ema_files = glob.glob(os.path.join(output_dir, "ema_*"))
    if not ema_files:
        return {"status": "FAIL", "detail": "training finished but no ema_ checkpoint was written"}

    return {"status": "PASS", "detail": f"ema checkpoint written: {os.path.basename(ema_files[0])}"}


def main():
    parser = argparse.ArgumentParser(description="GPU smoke test for Anima EMA + async caching")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with image+txt pairs")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to Anima DiT safetensors")
    parser.add_argument("--qwen3_path", type=str, required=True, help="Path to Qwen3 model")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to WanVAE safetensors")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per test in seconds")
    parser.add_argument("--only", type=str, default=None, choices=["finetune", "lora"])
    args = parser.parse_args()

    for name, path in [
        ("image_dir", args.image_dir),
        ("dit_path", args.dit_path),
        ("qwen3_path", args.qwen3_path),
        ("vae_path", args.vae_path),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} does not exist: {path}")
            sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="anima_ema_test_")
    toml_path = create_dataset_toml(args.image_dir, args.resolution, os.path.join(tmp_dir, "dataset.toml"))
    python = sys.executable

    # max_train_steps == save_every_n_steps so exactly one save (and one EMA save) happens.
    common_args = [
        "--dit_path", args.dit_path,
        "--qwen3_path", args.qwen3_path,
        "--vae_path", args.vae_path,
        "--pretrained_model_name_or_path", args.dit_path,
        "--dataset_config", toml_path,
        "--max_train_steps", "2",
        "--save_every_n_steps", "2",
        "--learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--max_data_loader_n_workers", "0",
        "--logging_dir", os.path.join(tmp_dir, "logs"),
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        # EMA on CPU keeps GPU VRAM free; --ema_sample also exercises the EMA sampling path.
        "--ema",
        "--ema_decay", "0.99",
        "--ema_device", "cpu",
    ]
    if args.t5_tokenizer_path:
        common_args += ["--t5_tokenizer_path", args.t5_tokenizer_path]

    results = {}

    if args.only in (None, "finetune"):
        out = os.path.join(tmp_dir, "ft")
        os.makedirs(out, exist_ok=True)
        cmd = [python, "anima_train.py"] + common_args + [
            "--output_dir", out,
            "--output_name", "ema_ft",
            "--optimizer_type", "AdamW8bit",
        ]
        results["finetune_ema"] = run_test("anima_train.py full finetune with --ema", cmd, out, args.timeout)

    if args.only in (None, "lora"):
        out = os.path.join(tmp_dir, "lora")
        os.makedirs(out, exist_ok=True)
        cmd = [python, "anima_train_network.py"] + common_args + [
            "--output_dir", out,
            "--output_name", "ema_lora",
            "--optimizer_type", "AdamW8bit",
            "--network_module", "networks.lora_anima",
            "--network_dim", "4",
            "--network_alpha", "1",
        ]
        results["lora_ema"] = run_test("anima_train_network.py LoRA with --ema", cmd, out, args.timeout)

    print(f"\n{'#' * 70}\nSUMMARY")
    all_pass = True
    for test_name, result in results.items():
        if result["status"] != "PASS":
            all_pass = False
        print(f"  [{result['status']:7s}] {test_name}: {result['detail']}")

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        print(f"Note: could not clean up {tmp_dir}")

    if not all_pass:
        sys.exit(1)
    print("\nAll EMA smoke tests PASSED!")


if __name__ == "__main__":
    main()
