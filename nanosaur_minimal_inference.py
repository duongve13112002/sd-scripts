"""
Minimal NanoSaur inference script.

Generates images using a NanoSaur diffusion model checkpoint.
Supports LoRA injection (ComfyUI-compatible or sd-scripts format).

Usage (basic):
    python nanosaur_minimal_inference.py \\
      --model  nanosaur_diffusion_model.safetensors \\
      --text_encoder nanosaur_text_encoder.safetensors \\
      --vae nanosaur_vae_decoder.safetensors \\
      --prompt "a photo of a cat sitting on a red sofa" \\
      --output output.png

Usage (with LoRA):
    python nanosaur_minimal_inference.py \\
      --model  nanosaur_diffusion_model.safetensors \\
      --text_encoder nanosaur_text_encoder.safetensors \\
      --vae nanosaur_vae_decoder.safetensors \\
      --lora my_lora.safetensors \\
      --lora_scale 0.8 \\
      --prompt "a painting of a futuristic city" \\
      --output city.png

Usage (SPRINT disabled for debugging):
    ... --no_sprint ...
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from library.nanosaur_models import NanoSaurTransformer2DModel
from library.nanosaur_utils import (
    load_nanosaur_model,
    load_nanosaur_text_encoder,
    load_nanosaur_vae,
)
from library.nanosaur_train_util import rectified_flow_sample
from library import nanosaur_models


# LoRA injection


def _apply_lora(model: NanoSaurTransformer2DModel, lora_path: str, scale: float = 1.0) -> None:
    """
    Apply a NanoSaur LoRA to the model by directly merging weights.

    Supports ComfyUI format (``diffusion_model.{path}.lora_{up|down}.weight``)
    and sd-scripts format (``lora_unet_{path}.lora_{up|down}.weight``).

    The LoRA is applied by adding ``scale * (up @ down) * alpha/rank`` to the
    corresponding weight in the model.
    """
    sd = load_file(lora_path, device="cpu")

    # Collect (base_key, down_key, up_key, alpha_key) groups
    groups: dict = {}
    for key in sd.keys():
        if "lora_down.weight" not in key:
            continue
        base = key.replace(".lora_down.weight", "")
        up_key = base + ".lora_up.weight"
        alpha_key = base + ".alpha"
        if up_key not in sd:
            continue
        groups[base] = (key, up_key, alpha_key)

    model_sd = dict(model.named_parameters())

    applied = 0
    for base, (down_key, up_key, alpha_key) in groups.items():
        # Resolve model parameter path from base key
        if base.startswith("diffusion_model."):
            param_path = base[len("diffusion_model."):]
        elif base.startswith("lora_unet_"):
            # convert lora_unet_blocks_0_attn_qkv_x → need module lookup
            param_path = base[len("lora_unet_"):].replace("_", ".", 1)
            # This is ambiguous; prefer ComfyUI format
            param_path = None
        else:
            param_path = None

        if param_path is None:
            continue

        weight_key = param_path + ".weight"
        if weight_key not in model_sd:
            continue

        down = sd[down_key].float()
        up = sd[up_key].float()
        alpha = sd.get(alpha_key, torch.tensor(float(down.shape[0]))).float().item()
        rank = down.shape[0]
        lora_scale = (alpha / rank) * scale

        weight = model_sd[weight_key].data.float()

        if down.ndim == 2 and up.ndim == 2:
            delta = (up @ down) * lora_scale
        elif down.shape[2:] == (1, 1):
            delta = (
                (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            ) * lora_scale
        else:
            # Skip conv3x3 for now (NanoSaur uses only linear layers in LoRA targets)
            continue

        model_sd[weight_key].data = (weight + delta).to(model_sd[weight_key].dtype)
        applied += 1

    print(f"Applied {applied} LoRA modules from {lora_path} (scale={scale})")


# Main inference function


@torch.no_grad()
def generate(
    model: NanoSaurTransformer2DModel,
    vae,
    tokenizer,
    text_encoder,
    prompts: list,
    negative_prompts: list,
    height: int = 1024,
    width: int = 1024,
    steps: int = 40,
    guidance_scale: float = 7.0,
    sample_shift: float = 4.0,
    cfg_start: float = 0.03,
    cfg_end: float = 0.80,
    use_sprint: bool = True,
    seed: int = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> list:
    """
    Generate images for a batch of prompts.

    Args:
        model: NanoSaurTransformer2DModel.
        vae: NanoSaurVAEWrapper.
        tokenizer: NanoSaurSentencePieceTokenizer.
        text_encoder: Gemma3ForCausalLM.
        prompts: List of positive prompts.
        negative_prompts: List of negative prompts (must match length of prompts).
        height / width: Output image resolution (multiple of 16).
        steps: Number of Euler denoising steps.
        guidance_scale: CFG scale.
        sample_shift: Timestep schedule shift.
        cfg_start / cfg_end: Fraction of steps where CFG is active.
        use_sprint: If True, use SPRINT for alternating uncond steps.
        seed: Random seed (None = random).
        device: Target device (auto-detected if None).
        dtype: Model compute dtype.

    Returns:
        List of PIL.Image objects.
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    batch = len(prompts)
    assert len(negative_prompts) == batch

    # Encode prompts
    def encode(texts: list) -> torch.Tensor:
        tokens = tokenizer(texts, device=device)
        out = text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1].to(dtype)  # (B, L, D)

    cond = encode(prompts)
    null_cond = encode(negative_prompts)

    # Initial noise
    lat_h = height // 16
    lat_w = width // 16
    lat_c = nanosaur_models.MODEL_CHANNELS
    z = torch.randn(batch, lat_c, lat_h, lat_w, device=device, dtype=dtype, generator=generator)

    # Denoising
    model.eval()
    denoised = rectified_flow_sample(
        model=model,
        z=z,
        cond=cond,
        null_cond=null_cond,
        steps=steps,
        guidance_scale=guidance_scale,
        sample_shift=sample_shift,
        cfg_start=cfg_start,
        cfg_end=cfg_end,
        path_drop_guidance=use_sprint,
        show_progress=True,
    )

    # Decode
    images = vae.decode(denoised)  # (B, 3, H, W) in [-1, 1]

    pil_images = []
    for img in images:
        img_np = ((img.clamp(-1, 1).float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))

    return pil_images


# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NanoSaur minimal inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to diffusion model safetensors")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to text encoder safetensors")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE safetensors")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA safetensors (optional)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale multiplier")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape", help="Positive prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="nanosaur_output.png", help="Output image path")
    parser.add_argument("--height", type=int, default=1024, help="Output height (multiple of 16)")
    parser.add_argument("--width", type=int, default=1024, help="Output width (multiple of 16)")
    parser.add_argument("--steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="CFG guidance scale")
    parser.add_argument("--sample_shift", type=float, default=4.0, help="Timestep schedule shift")
    parser.add_argument("--cfg_start", type=float, default=0.03, help="CFG start fraction")
    parser.add_argument("--cfg_end", type=float, default=0.80, help="CFG end fraction")
    parser.add_argument("--no_sprint", action="store_true", help="Disable SPRINT (path drop guidance)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Model dtype")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, cuda:0, ...)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device and dtype
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Device: {device}, dtype: {dtype}")

    # Align resolution to multiples of 16
    height = max(16, args.height - args.height % 16)
    width = max(16, args.width - args.width % 16)

    # Load models
    print("Loading diffusion model...")
    model = load_nanosaur_model(args.model, dtype, device)
    model.eval()

    print("Loading text encoder...")
    tokenizer, text_encoder = load_nanosaur_text_encoder(args.text_encoder, dtype, device)
    text_encoder.eval()

    print("Loading VAE...")
    vae = load_nanosaur_vae(args.vae, dtype, device)

    # Apply LoRA if provided
    if args.lora:
        print(f"Applying LoRA from {args.lora}...")
        _apply_lora(model, args.lora, scale=args.lora_scale)

    # Generate
    print(f"Generating {width}×{height} image for prompt: {args.prompt!r}")
    images = generate(
        model=model,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompts=[args.prompt],
        negative_prompts=[args.negative_prompt],
        height=height,
        width=width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        sample_shift=args.sample_shift,
        cfg_start=args.cfg_start,
        cfg_end=args.cfg_end,
        use_sprint=not args.no_sprint,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
