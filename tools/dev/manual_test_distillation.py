"""GPU smoke test for output distillation (LoRA / network training).

Runs anima_train_network.py for a couple of steps with the distillation weights
enabled (--distillation_weight_high / --distillation_weight_low) and latent /
text-encoder caching on, then checks the run finished and a LoRA checkpoint was
written. Distillation adds a teacher forward (adapter disabled via multiplier 0),
so this exercises that path end to end. It needs real model weights and a GPU,
so run it manually on a server, not in CI.

Usage:
    python tools/dev/manual_test_distillation.py \
        --image_dir /path/to/images_with_txt \
        --dit_path /path/to/dit.safetensors \
        --qwen3_path /path/to/qwen3 \
        --vae_path /path/to/vae.safetensors \
        [--t5_tokenizer_path /path/to/t5] \
        [--resolution 512] \
        [--loss_type l2|huber]

The same distillation flags work on every network trainer
(flux_train_network.py, sd3_train_network.py, lumina_train_network.py,
hunyuan_image_train_network.py, train_network.py for SD1.x/SDXL); only the model
paths differ. Anima is used here because it is the smallest to set up.
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

    ckpts = glob.glob(os.path.join(output_dir, "*.safetensors"))
    if not ckpts:
        return {"status": "FAIL", "detail": "training finished but no checkpoint was written"}

    return {"status": "PASS", "detail": f"checkpoint written: {os.path.basename(ckpts[0])}"}


def main():
    parser = argparse.ArgumentParser(description="GPU smoke test for output distillation (LoRA)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with image+txt pairs")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to Anima DiT safetensors")
    parser.add_argument("--qwen3_path", type=str, required=True, help="Path to Qwen3 model")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to WanVAE safetensors")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "huber"])
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per test in seconds")
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

    tmp_dir = tempfile.mkdtemp(prefix="distill_test_")
    toml_path = create_dataset_toml(args.image_dir, args.resolution, os.path.join(tmp_dir, "dataset.toml"))
    python = sys.executable

    out = os.path.join(tmp_dir, "lora")
    os.makedirs(out, exist_ok=True)
    cmd = [
        python, "anima_train_network.py",
        "--dit_path", args.dit_path,
        "--qwen3_path", args.qwen3_path,
        "--vae_path", args.vae_path,
        "--pretrained_model_name_or_path", args.dit_path,
        "--dataset_config", toml_path,
        "--output_dir", out,
        "--output_name", "distill_lora",
        "--max_train_steps", "2",
        "--save_every_n_steps", "2",
        "--learning_rate", "1e-4",
        "--mixed_precision", "bf16",
        "--max_data_loader_n_workers", "0",
        "--logging_dir", os.path.join(tmp_dir, "logs"),
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--optimizer_type", "AdamW8bit",
        "--network_module", "networks.lora_anima",
        "--network_dim", "4",
        "--network_alpha", "1",
        # distillation: anchor concepts at high noise, free detail at low noise
        "--distillation_weight_high", "1.0",
        "--distillation_weight_low", "0.0",
        "--distillation_loss_type", args.loss_type,
    ]
    if args.t5_tokenizer_path:
        cmd += ["--t5_tokenizer_path", args.t5_tokenizer_path]

    result = run_test("anima_train_network.py LoRA with distillation", cmd, out, args.timeout)

    print(f"\n{'#' * 70}\nSUMMARY")
    print(f"  [{result['status']:7s}] lora_distillation: {result['detail']}")

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        print(f"Note: could not clean up {tmp_dir}")

    if result["status"] != "PASS":
        sys.exit(1)
    print("\nDistillation smoke test PASSED!")


if __name__ == "__main__":
    main()
