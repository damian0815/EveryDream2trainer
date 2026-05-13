"""
Integration Test: Interposer Round-Trip Visual Validation
==========================================================

Tests the bidirectional latent-space interposer by performing a full round-trip:

    image → src VAE encode → src→dst interposer → dst VAE decode
                           → dst→src interposer → src VAE decode

Supports **all** latent-space types recognised by SD-Latent-Interposer:

    v1  (SD 1.x / SD 2.x)
    xl  (SDXL)
    v3  (SD 3)
    fx  (Flux.1)
    ca  (Stable Cascade)

For each input image, saves a JPEG with four side-by-side panels:

    LEFT  : original pixel image (ground truth)
    REF   : src VAE encode → src VAE decode   (baseline quality reference)
    MID   : src→dst interposer → dst VAE decode   (forward direction)
    RIGHT : src→dst→src interposer round-trip → src VAE decode

When ``--dst_vae`` is omitted (or identical to ``--src_vae``) both sides use
the **same** VAE, which isolates interposer round-trip errors from VAE
domain-shift artefacts.

Usage
-----
    # SDXL ↔ SD2 (separate VAEs)
    python test/test_interposer_roundtrip.py \\
        --images_dir /path/to/images \\
        --src_vae    /path/to/sdxl_model \\
        --src_type   xl \\
        --dst_vae    /path/to/sd2_model \\
        --dst_type   v1

    # Same-VAE round-trip (isolates interposer errors, no VAE domain shift)
    python test/test_interposer_roundtrip.py \\
        --images_dir /path/to/images \\
        --src_vae    /path/to/sdxl_model \\
        --src_type   xl \\
        --dst_type   v1

    # All optional flags
        [--output_dir      ./roundtrip_output]
        [--interposer_dir  /path/to/interposer_models]
        [--max_images      16]
        [--size            512]
        [--src_scale       0.13025]   # override VAE config scaling_factor
        [--dst_scale       0.18215]
        [--device          cuda]

Interpreting results
--------------------
  - REF   should look like the original (mild VAE softness is normal).
  - MID   (forward pass) should look like the original if the interposer is
          working; colour shift or heavy blur indicates a forward-direction issue.
  - RIGHT (round-trip) should be very close to REF.  Obvious colour shift or
          blur = interposer round-trip error.
  - Round-trip MSE (reported in stdout) should be small; a value much larger
          than the REF MSE indicates interposer loss.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Make sure we can import from the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.latent_interposer import LatentInterposer, LatentSpaceType, INTERPOSER_CONFIGS


# ---------------------------------------------------------------------------
# Well-known default scaling factors per latent-space type
# (used as fallback when the VAE config does not provide one)
# ---------------------------------------------------------------------------

_DEFAULT_SCALE: dict[str, float] = {
    LatentSpaceType.SD1:     0.18215,   # SD 1.x / 2.x
    LatentSpaceType.SDXL:    0.13025,   # SDXL
    LatentSpaceType.SD3:     1.5305,    # SD3
    LatentSpaceType.FLUX:    0.3611,    # Flux.1
    LatentSpaceType.CASCADE: 0.3764,    # Stable Cascade
}

_TYPE_LABEL: dict[str, str] = {
    LatentSpaceType.SD1:     "SD1/SD2 (v1)",
    LatentSpaceType.SDXL:    "SDXL (xl)",
    LatentSpaceType.SD3:     "SD3 (v3)",
    LatentSpaceType.FLUX:    "Flux (fx)",
    LatentSpaceType.CASCADE: "Cascade (ca)",
}

_ALL_TYPES = sorted({LatentSpaceType.SD1, LatentSpaceType.SDXL,
                     LatentSpaceType.SD3, LatentSpaceType.FLUX,
                     LatentSpaceType.CASCADE})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vae(model_path: str, device: str, dtype: torch.dtype):
    """Load an AutoencoderKL from a model directory or HF hub ID."""
    from diffusers import AutoencoderKL

    # If the path contains a vae/ sub-folder use that; otherwise treat the
    # whole path as a standalone VAE checkpoint.
    vae_subdir = os.path.join(model_path, "vae")
    if os.path.isdir(vae_subdir):
        print(f"  Loading VAE from {vae_subdir}")
        vae = AutoencoderKL.from_pretrained(vae_subdir, torch_dtype=dtype)
    else:
        print(f"  Loading VAE from {model_path}")
        vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=dtype)
    vae.eval().to(device)
    return vae


def get_scaling_factor(vae, latent_type: str, override: float | None) -> float:
    """Return the VAE scaling factor, honouring an explicit CLI override."""
    if override is not None:
        return override
    cfg_val = getattr(vae.config, "scaling_factor", None)
    if cfg_val is not None:
        return float(cfg_val)
    default = _DEFAULT_SCALE.get(latent_type, 1.0)
    print(f"  WARNING: could not read scaling_factor from VAE config; "
          f"using default {default} for type '{latent_type}'")
    return default


def load_image(path: str, size: int) -> torch.Tensor:
    """Load an image as a [1, 3, size, size] float32 tensor in [-1, 1]."""
    img  = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    left  = (w - short) // 2
    top   = (h - short) // 2
    img   = img.crop((left, top, left + short, top + short))
    img   = img.resize((size, size), Image.LANCZOS)
    arr   = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a [1, 3, H, W] or [3, H, W] float tensor in [-1, 1] → [H, W, 3] uint8."""
    t = t.float().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.permute(1, 2, 0).clamp(-1.0, 1.0)
    return ((t + 1.0) / 2.0 * 255.0).numpy().astype(np.uint8)


def decode_latents(vae, latents: torch.Tensor, scaling_factor: float) -> np.ndarray:
    """
    Decode scaled latents ([1, C, h, w]) → [H, W, 3] uint8.
    Divides by scaling_factor before passing to the VAE decoder.
    """
    with torch.no_grad():
        unscaled = latents.to(vae.dtype) / scaling_factor
        decoded  = vae.decode(unscaled, return_dict=False)[0]   # [1, 3, H, W]
    return tensor_to_uint8(decoded)


def make_strip(panels: list[np.ndarray], labels: list[str],
               label_height: int = 22) -> Image.Image:
    """Horizontally concatenate uint8 numpy panels with text labels above each."""
    assert len(panels) == len(labels)
    h, w    = panels[0].shape[:2]
    total_w = w * len(panels)
    strip   = np.ones((h + label_height, total_w, 3), dtype=np.uint8) * 220
    for i, panel in enumerate(panels):
        strip[label_height:, i * w : (i + 1) * w] = panel
    img  = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((i * w + 4, 3), label, fill=(40, 40, 40), font=font)
    return img


def collect_images(images_dir: str, max_images: int) -> list[str]:
    exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [os.path.join(images_dir, f)
             for f in sorted(os.listdir(images_dir))
             if os.path.splitext(f)[1].lower() in exts]
    return paths[:max_images]


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_roundtrip_test(args):
    src_type: str = args.src_type
    dst_type: str = args.dst_type
    device:   str = args.device
    dtype         = torch.float16 if "cuda" in device else torch.float32

    # Validate the requested conversion pair
    fwd_key = f"{src_type}-to-{dst_type}"
    rev_key = f"{dst_type}-to-{src_type}"
    if src_type != dst_type:
        missing = [k for k in (fwd_key, rev_key) if k not in INTERPOSER_CONFIGS]
        if missing:
            print(f"ERROR: No interposer model for: {missing}")
            print(f"Supported conversions: {sorted(INTERPOSER_CONFIGS.keys())}")
            sys.exit(1)
    else:
        print("WARNING: src_type == dst_type — the interposer is a no-op; "
              "round-trip MSE will be 0.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load VAEs
    # ------------------------------------------------------------------
    print(f"\n=== Loading src VAE  [{_TYPE_LABEL.get(src_type, src_type)}] ===")
    src_vae   = load_vae(args.src_vae, device, dtype)
    src_scale = get_scaling_factor(src_vae, src_type, args.src_scale)
    print(f"  src VAE scaling_factor = {src_scale}")

    dst_vae_path = args.dst_vae or args.src_vae
    same_vae     = args.dst_vae is None or (dst_vae_path == args.src_vae)

    if same_vae:
        print(f"\n=== dst VAE is the same as src VAE (shared) ===")
        dst_vae   = src_vae
        dst_scale = get_scaling_factor(dst_vae, dst_type, args.dst_scale)
    else:
        print(f"\n=== Loading dst VAE  [{_TYPE_LABEL.get(dst_type, dst_type)}] ===")
        dst_vae   = load_vae(dst_vae_path, device, dtype)
        dst_scale = get_scaling_factor(dst_vae, dst_type, args.dst_scale)
    print(f"  dst VAE scaling_factor = {dst_scale}")

    # ------------------------------------------------------------------
    # Load interposer
    # ------------------------------------------------------------------
    print("\n=== Loading Interposer ===")
    interposer = LatentInterposer(model_dir=args.interposer_dir)

    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------
    images = collect_images(args.images_dir, args.max_images)
    if not images:
        print(f"ERROR: No images found in {args.images_dir}")
        sys.exit(1)
    print(f"\n=== Processing {len(images)} image(s) ===\n")
    print(f"  Conversion:  {src_type} → {dst_type} → {src_type}  (round-trip)")

    mse_scores: list[float] = []

    for idx, img_path in enumerate(images):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx+1}/{len(images)}] {basename}")

        # ---- Load & encode with src VAE ----
        pixel_tensor = load_image(img_path, args.size).to(device=device, dtype=dtype)

        with torch.no_grad():
            dist = src_vae.encode(pixel_tensor, return_dict=False)[0]
        src_latents = dist.sample() * src_scale          # scaled src latents [1,C,h,w]
        print(f"  src latents : shape={list(src_latents.shape)}  "
              f"mean={src_latents.mean().item():.4f}  std={src_latents.std().item():.4f}")

        # ---- Forward: src → dst ----
        dst_latents = interposer.convert(
            src_latents.float(), src=src_type, dst=dst_type
        ).to(device=device, dtype=dtype)
        print(f"  dst latents : shape={list(dst_latents.shape)}  "
              f"mean={dst_latents.mean().item():.4f}  std={dst_latents.std().item():.4f}")

        # ---- Reverse: dst → src (round-trip) ----
        src_latents_rt = interposer.convert(
            dst_latents.float(), src=dst_type, dst=src_type
        ).to(device=device, dtype=dtype)
        print(f"  src rt      : shape={list(src_latents_rt.shape)}  "
              f"mean={src_latents_rt.mean().item():.4f}  std={src_latents_rt.std().item():.4f}")

        # ---- Round-trip MSE on scaled latents ----
        mse = ((src_latents.float() - src_latents_rt.float()) ** 2).mean().item()
        mse_scores.append(mse)
        print(f"  Round-trip MSE (scaled latents): {mse:.6f}")

        # ---- Build panels ----
        src_label = _TYPE_LABEL.get(src_type, src_type)
        dst_label = _TYPE_LABEL.get(dst_type, dst_type)

        # LEFT : original pixel image
        panel_orig = tensor_to_uint8(pixel_tensor)

        # REF  : src VAE encode → src VAE decode  (shows pure VAE quality baseline)
        panel_ref  = decode_latents(src_vae, src_latents, src_scale)

        # MID  : dst VAE decodes the interposed dst latents  (forward validation)
        panel_fwd  = decode_latents(dst_vae, dst_latents, dst_scale)

        # RIGHT: src VAE decodes the round-trip latents  (full cycle validation)
        panel_rt   = decode_latents(src_vae, src_latents_rt, src_scale)

        strip = make_strip(
            [panel_orig, panel_ref, panel_fwd, panel_rt],
            [
                "Input image",
                f"{src_label}\nencode→decode (ref)",
                f"{src_label}→{dst_label}\nforward",
                f"{src_label}→{dst_label}→{src_label}\nround-trip",
            ],
        )

        out_path = os.path.join(args.output_dir, f"{basename}_roundtrip.jpg")
        strip.save(out_path, quality=92)
        print(f"  Saved → {out_path}\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Round-trip MSE summary over {len(mse_scores)} image(s):")
    print(f"  min  : {min(mse_scores):.6f}")
    print(f"  max  : {max(mse_scores):.6f}")
    print(f"  mean : {sum(mse_scores)/len(mse_scores):.6f}")
    print("=" * 60)
    print("\nInspect the JPEGs in:", args.output_dir)
    print("What to look for:")
    print("  REF   — should look like the original (mild VAE softness is normal)")
    print("  MID   — should resemble the original (forward interposer working)")
    print("  RIGHT — should be very close to REF (any difference = round-trip loss)")
    print("  Large colour shift or heavy blur in MID/RIGHT → interposer error")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generic visual round-trip test for the SD-Latent-Interposer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported latent-space type codes: {_ALL_TYPES}\n"
            f"Supported conversions:\n"
            + "\n".join(f"  {k}" for k in sorted(INTERPOSER_CONFIGS.keys()))
        ),
    )

    # Required
    p.add_argument("--images_dir", required=True,
                   help="Folder containing test images")
    p.add_argument("--src_vae",    required=True,
                   help="Path or HF hub ID for the *source* model / VAE")
    p.add_argument("--src_type",   required=True, choices=_ALL_TYPES,
                   help="Latent-space type of the source VAE "
                        "(v1=SD1/SD2, xl=SDXL, v3=SD3, fx=Flux, ca=Cascade)")

    # Optional destination — defaults to same VAE / same type
    p.add_argument("--dst_vae",    default=None,
                   help="Path or HF hub ID for the *destination* model / VAE.  "
                        "Defaults to --src_vae (shared VAE, isolates interposer error).")
    p.add_argument("--dst_type",   default=None, choices=_ALL_TYPES,
                   help="Latent-space type of the destination VAE.  "
                        "Defaults to --src_type.")

    # Scaling factor overrides
    p.add_argument("--src_scale",  type=float, default=None,
                   help="Override the src VAE scaling_factor "
                        "(auto-detected from VAE config by default)")
    p.add_argument("--dst_scale",  type=float, default=None,
                   help="Override the dst VAE scaling_factor "
                        "(auto-detected from VAE config by default)")

    # General
    p.add_argument("--output_dir",     default="./roundtrip_output",
                   help="Where to write the comparison JPEGs (default: ./roundtrip_output)")
    p.add_argument("--interposer_dir", default=None,
                   help="Local directory containing interposer .safetensors files "
                        "(auto-downloads from HF Hub if omitted)")
    p.add_argument("--max_images",     type=int, default=16,
                   help="Maximum number of images to process (default: 16)")
    p.add_argument("--size",           type=int, default=512,
                   help="Resize/crop images to this square size before encoding (default: 512)")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device: cuda / cpu / mps (auto-detected by default)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply defaults that depend on other args
    if args.dst_type is None:
        args.dst_type = args.src_type

    run_roundtrip_test(args)

