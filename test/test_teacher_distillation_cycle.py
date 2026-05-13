"""
Integration Test 2: Full Distillation-Cycle Visual Validation
=============================================================

Exercises the complete SD2-FM-teacher → SDXL-FM-student distillation pipeline
described in ``sdxl-sd2-vae-teacher.plan.md``, showing every intermediate phase
as a side-by-side image strip for manual inspection.

For each test image and each tested timestep σ the output strip shows:

  Col 1 ORIGINAL         – the source image (ground truth)
  Col 2 NOISY (SDXL)     – x_t decoded with SDXL VAE at the tested noise level
  Col 3 TEACHER CLEAN v1 – teacher's predicted clean endpoint, decoded with SD2 VAE
  Col 4 TEACHER TARGET   – teacher clean endpoint back in SDXL space (interposer v1→xl),
                           decoded with SDXL VAE → this is the distillation target
  Col 5 STUDENT CLEAN    – student's predicted clean endpoint, decoded with SDXL VAE
  Col 6 DIFF             – absolute pixel difference between cols 4 and 5 (×3 for visibility)

If distillation is working correctly:
  - Cols 4 and 5 should look similar.
  - Col 6 should be dark / featureless.

Usage
-----
    python test/test_teacher_distillation_cycle.py \\
        --sdxl_model /path/to/sdxl-fm-model \\
        --sd2_model  /path/to/sd2-fm-model \\
        --images_dir /path/to/images \\
        --count 4 \\
        --timesteps 250 500 750 \\
        --output_dir ./distillation_output \\
        [--interposer_dir /path/to/interposer_models] \\
        [--prompt "a photo of a cat"] \\
        [--size 512] \\
        [--device cuda]

Notes
-----
* The script uses the *same noise instance* (x_0_vS) for the student and a *fresh*
  independent sample (Option A from the plan) for the teacher, exactly as
  ``core/loss.py:_teacher_target_via_interposer`` does.
* Timesteps are expressed as integers in [0, 999]; σ = timestep / 1000 is used as
  the noise fraction for the linear flow-matching schedule.
* If no --prompt is given the models are run with empty-string conditioning.
  For SD2's UNet the embedding is shape [1, 77, 1024]; for SDXL it is
  [1, 77, 2048] (concatenated clip-L + openclip-G).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Make sure we can import from the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.latent_interposer import LatentInterposer, LatentSpaceType
from model.training_model import TrainingModel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str, device: str, dtype: torch.dtype) -> TrainingModel:
    """Load any SD-family model as a TrainingModel via AutoPipelineForText2Image."""
    from diffusers import AutoPipelineForText2Image
    print(f"  Loading pipeline from: {path}")
    pipe = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype).to(device)
    model = TrainingModel.from_pipeline(pipe)
    del pipe
    return model


def vae_scale(model: TrainingModel) -> float:
    return float(getattr(model.vae.config, 'scaling_factor',
                         0.13025 if model.is_sdxl else 0.18215))


# ---------------------------------------------------------------------------
# Latent & image utilities
# ---------------------------------------------------------------------------

def load_image(path: str, size: int) -> torch.Tensor:
    """Load image → [1, 3, H, W] float32 in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    left, top = (w - short) // 2, (h - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def encode_image(model: TrainingModel, pixel_tensor: torch.Tensor) -> torch.Tensor:
    """Returns scaled latents [1, 4, h, w]."""
    scale = vae_scale(model)
    device, dtype = model.device, model.vae.dtype
    with torch.no_grad():
        dist = model.vae.encode(pixel_tensor.to(device=device, dtype=dtype), return_dict=False)[0]
        return dist.sample() * scale


def decode_latents_to_numpy(model: TrainingModel, scaled_latents: torch.Tensor) -> np.ndarray:
    """Returns [H, W, 3] uint8."""
    scale = vae_scale(model)
    with torch.no_grad():
        unscaled = scaled_latents.to(model.vae.dtype) / scale
        decoded  = model.vae.decode(unscaled, return_dict=False)[0].float().cpu()
    arr = (decoded.squeeze(0).permute(1, 2, 0).clamp(-1, 1).numpy() + 1.0) / 2.0 * 255
    return arr.clip(0, 255).astype(np.uint8)


def pixel_tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] → [H, W, 3] uint8."""
    arr = (t.float().squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255
    return arr.clip(0, 255).astype(np.uint8)


def make_diff_panel(a: np.ndarray, b: np.ndarray, amplify: float = 3.0) -> np.ndarray:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) * amplify
    return diff.clip(0, 255).astype(np.uint8)


def make_strip(panels: list, labels: list, label_height: int = 24) -> Image.Image:
    assert len(panels) == len(labels)
    h, w = panels[0].shape[:2]
    total_w = w * len(panels)
    strip = np.ones((h + label_height, total_w, 3), dtype=np.uint8) * 220
    for i, panel in enumerate(panels):
        strip[label_height:, i * w: (i + 1) * w] = panel
    img = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        draw.text((i * w + 3, 3), label, fill=(30, 30, 30), font=font)
    return img


def make_grid(strips: list) -> Image.Image:
    total_h = sum(s.height for s in strips)
    total_w = max(s.width for s in strips)
    grid = Image.new("RGB", (total_w, total_h), (200, 200, 200))
    y = 0
    for s in strips:
        grid.paste(s, (0, y))
        y += s.height
    return grid


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def get_conditioning(prompt: str, model: TrainingModel, size: int) -> dict:
    """
    Build UNet conditioning for *model*.

    Returns a dict with:
      'encoder_hidden_states' : [1, 77, D]
      'added_cond_kwargs'      : dict (SDXL only, else None)
    """
    device = model.device
    dtype  = model.unet.dtype

    with torch.no_grad():
        def _encode(tokenizer, encoder):
            tokens = tokenizer(
                [prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = encoder(tokens, output_hidden_states=True, return_dict=True)
            return out.hidden_states[-2], getattr(out, "text_embeds", None)

        h1, _     = _encode(model.tokenizer, model.text_encoder)  # [1, 77, D1]

        if model.is_sdxl:
            h2, pool  = _encode(model.tokenizer_2, model.text_encoder_2)  # [1, 77, D2], [1, 1280]
            encoder_hidden_states = torch.cat([h1, h2], dim=-1).to(dtype)  # [1, 77, D1+D2]
            pooled = pool.to(dtype) if pool is not None else torch.zeros(1, 1280, device=device, dtype=dtype)
            add_time_ids = torch.tensor(
                [[size, size, 0, 0, size, size]], dtype=dtype, device=device
            )  # [1, 6]
            added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}
        else:
            # SD1/SD2: single encoder, use last hidden state directly
            tokens = model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            enc_out = model.text_encoder(tokens, return_dict=True)
            encoder_hidden_states = enc_out.last_hidden_state.to(dtype)
            added_cond_kwargs = None

    return {"encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": added_cond_kwargs}


def run_unet(model: TrainingModel, noisy_latents: torch.Tensor,
             t_tensor: torch.Tensor, cond: dict) -> torch.Tensor:
    """Run model.unet and return the velocity prediction as float32."""
    return model.unet(
        noisy_latents.to(dtype=model.unet.dtype),
        t_tensor.to(model.device, dtype=model.unet.dtype),
        encoder_hidden_states=cond["encoder_hidden_states"].to(model.device, dtype=model.unet.dtype),
        added_cond_kwargs=(
            {k: v.to(model.device, dtype=model.unet.dtype)
             for k, v in cond["added_cond_kwargs"].items()}
            if cond["added_cond_kwargs"] is not None else None
        ),
    ).sample.float().to(model.device)


# ---------------------------------------------------------------------------
# Core cycle logic
# ---------------------------------------------------------------------------

def run_one_sample(
    *,
    pixel_tensor,           # [1, 3, H, W] float32 in [-1, 1]
    timestep: int,          # integer in [0, 999]
    sdxl_model: TrainingModel,
    sd2_model: TrainingModel,
    sdxl_cond: dict,
    sd2_cond: dict,
    interposer: LatentInterposer,
    device: str,
    dtype: torch.dtype,
):
    """
    Run the full distillation cycle for one image at one timestep.
    Returns a dict of named numpy images (uint8) for each phase.
    """
    sigma = max(0.01, min(0.99, timestep / 1000.0))
    print(f"    σ = {sigma:.4f} (timestep={timestep})")

    sdxl_scale = vae_scale(sdxl_model)
    sd2_scale  = vae_scale(sd2_model)

    # ---- SDXL: encode clean image ----
    x_1_xl = encode_image(sdxl_model, pixel_tensor)   # scaled [1,4,h,w]

    # ---- Student noise (shared) ----
    x_0_xl = torch.randn_like(x_1_xl)

    # ---- Build student's noisy latent ----
    x_t_xl = (1.0 - sigma) * x_1_xl + sigma * x_0_xl

    # ========= TEACHER PATH (no grad) =========
    with torch.no_grad():

        # 1. Map clean student latent → teacher VAE space.
        #    Unscale before interposer (expects raw VAE latents), re-scale after.
        x_1_v1 = interposer.convert(
            (x_1_xl / sdxl_scale).float(), src=LatentSpaceType.SDXL, dst=LatentSpaceType.SD1
        ).to(device=device, dtype=dtype) * sd2_scale

        # 2. Option A: independent noise in teacher space
        x_0_v1 = torch.randn_like(x_1_v1)

        # 3. Build teacher's noisy latent
        x_t_v1 = (1.0 - sigma) * x_1_v1 + sigma * x_0_v1

        # 4. Run SD2 teacher UNet
        t_tensor = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)
        v_hat_v1 = run_unet(sd2_model, x_t_v1, t_tensor, sd2_cond)

        # 5. Recover teacher's predicted clean endpoint in SD2 space (scaled)
        #    Convention (code): v = x_0 − x_1  ⟹  x_1 = x_t − σ·v
        x_1_hat_v1 = x_t_v1.float() - sigma * v_hat_v1

        # 6. Map teacher's predicted clean → SDXL space.
        #    Unscale before interposer, re-scale with sdxl_scale afterwards.
        x_1_hat_xl_teacher = interposer.convert(
            (x_1_hat_v1 / sd2_scale).to(dtype=torch.float32),
            src=LatentSpaceType.SD1, dst=LatentSpaceType.SDXL
        ).to(device=device, dtype=dtype) * sdxl_scale

        # 7. Compute distillation velocity target (for reference / logging)
        v_target_xl = x_0_xl.float() - x_1_hat_xl_teacher.float()
        print(f"    v_target_xl norm: {v_target_xl.norm().item():.4f}")

    # ========= STUDENT PATH (no grad for visual test) =========
    with torch.no_grad():
        t_tensor_xl = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)
        v_hat_xl = run_unet(sdxl_model, x_t_xl, t_tensor_xl, sdxl_cond)
        x_1_hat_xl_student = x_t_xl.float() - sigma * v_hat_xl
        print(f"    v_hat_xl norm:    {v_hat_xl.norm().item():.4f}")

    # ========= Decode all phases =========
    panel_orig     = pixel_tensor_to_numpy(pixel_tensor)
    panel_noisy    = decode_latents_to_numpy(sdxl_model, x_t_xl)
    panel_teach_v1 = decode_latents_to_numpy(sd2_model,  x_1_hat_v1.to(dtype=sd2_model.vae.dtype))
    panel_teach_xl = decode_latents_to_numpy(sdxl_model, x_1_hat_xl_teacher)
    panel_stud_xl  = decode_latents_to_numpy(sdxl_model, x_1_hat_xl_student.to(dtype=sdxl_model.vae.dtype))
    panel_diff     = make_diff_panel(panel_teach_xl, panel_stud_xl, amplify=3.0)

    return {
        "orig":     panel_orig,
        "noisy":    panel_noisy,
        "teach_v1": panel_teach_v1,
        "teach_xl": panel_teach_xl,
        "stud_xl":  panel_stud_xl,
        "diff":     panel_diff,
    }


def collect_images(images_dir: str, count: int) -> list:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))
             if os.path.splitext(f)[1].lower() in exts]
    return paths[:count]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_distillation_cycle_test(args):
    device = args.device
    dtype  = torch.bfloat16# if "cuda" in device else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading SDXL model ===")
    sdxl_model = load_model(args.sdxl_model, device, dtype)
    print(f"  is_sdxl={sdxl_model.is_sdxl}  VAE scale={vae_scale(sdxl_model)}")

    print("=== Loading SD2 model ===")
    sd2_model = load_model(args.sd2_model, device, dtype)
    print(f"  is_sdxl={sd2_model.is_sdxl}   VAE scale={vae_scale(sd2_model)}")

    # Sanity-check UNet input channel counts
    sdxl_in_ch = sdxl_model.unet.config.get("in_channels", 4)
    sd2_in_ch  = sd2_model.unet.config.get("in_channels",  4)
    print(f"\n  SDXL UNet in_channels={sdxl_in_ch}  |  SD2 UNet in_channels={sd2_in_ch}")
    assert sdxl_in_ch == 4, f"Expected SDXL UNet to have 4 in_channels, got {sdxl_in_ch}"
    assert sd2_in_ch  == 4, f"Expected SD2  UNet to have 4 in_channels, got {sd2_in_ch}"

    print("=== Loading Interposer ===")
    interposer = LatentInterposer(model_dir=args.interposer_dir)

    print("=== Building text conditioning ===")
    prompt = args.prompt or ""
    print(f"  Prompt: {repr(prompt)}")
    sdxl_cond = get_conditioning(prompt, sdxl_model, args.size)
    sd2_cond  = get_conditioning(prompt, sd2_model,  args.size)
    print(f"  SDXL encoder_hidden_states: {sdxl_cond['encoder_hidden_states'].shape}")
    print(f"  SD2  encoder_hidden_states: {sd2_cond['encoder_hidden_states'].shape}")

    images = collect_images(args.images_dir, args.count)
    if not images:
        print(f"ERROR: No images in {args.images_dir}")
        sys.exit(1)
    print(f"\n=== Processing {len(images)} image(s) × {len(args.timesteps)} timestep(s) ===\n")

    for img_idx, img_path in enumerate(images):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{img_idx+1}/{len(images)}] {basename}")

        pixel_tensor = load_image(img_path, args.size)
        strips = []

        for ts in args.timesteps:
            print(f"  timestep={ts}")
            phases = run_one_sample(
                pixel_tensor=pixel_tensor,
                timestep=ts,
                sdxl_model=sdxl_model,
                sd2_model=sd2_model,
                sdxl_cond=sdxl_cond,
                sd2_cond=sd2_cond,
                interposer=interposer,
                device=device,
                dtype=dtype,
            )

            strip = make_strip(
                panels=[
                    phases["orig"],
                    phases["noisy"],
                    phases["teach_v1"],
                    phases["teach_xl"],
                    phases["stud_xl"],
                    phases["diff"],
                ],
                labels=[
                    "1. Original",
                    f"2. Noisy SDXL (σ={ts/1000:.2f})",
                    "3. Teacher clean\n(SD2 VAE)",
                    "4. Teacher target\n(SDXL VAE, distil target)",
                    "5. Student predict\n(SDXL VAE)",
                    "6. |4−5|×3 diff\n(should be dark)",
                ],
            )
            strips.append(strip)

        grid = make_grid(strips)
        out_path = os.path.join(args.output_dir, f"{basename}_distil_cycle.jpg")
        grid.save(out_path, quality=92)
        print(f"  Saved → {out_path}\n")

    print("=" * 60)
    print("What to look for in the output images:")
    print("  Col 3 (Teacher SD2):   should resemble the original, perhaps higher-level / softer")
    print("  Col 4 (Teacher→SDXL):  same content as col 3 but in SDXL color/detail space")
    print("  Col 5 (Student):       at early training → often noisy/random; after training → matches col 4")
    print("  Col 6 (Diff ×3):       should be dark/featureless when cols 4 & 5 agree")
    print()
    print("Common failure signatures:")
    print("  • Col 3 looks like random noise → teacher UNet not loading correctly")
    print("  • Col 4 has severe color shift vs col 3 → v1→xl interposer issue")
    print("  • Col 4 is sharp but student produces blurry col 5 → need more training (Stage 1/2)")
    print("  • Col 5 matches col 4 but wrong style vs col 3 → interposer accuracy issue")
    print("=" * 60)
    print("Results saved to:", args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual distillation-cycle test: SD2-FM teacher → SDXL-FM student via interposer"
    )
    p.add_argument("--sdxl_model",     required=True,
                   help="Path or HF ID for the SDXL flow-matching model")
    p.add_argument("--sd2_model",      required=True,
                   help="Path or HF ID for the SD2 flow-matching teacher model")
    p.add_argument("--images_dir",     required=True,
                   help="Folder containing test images")
    p.add_argument("--count",          type=int, default=4,
                   help="Number of images to process (default: 4)")
    p.add_argument("--timesteps",      type=int, nargs="+", default=[250, 500, 750],
                   help="Timestep(s) to test as integers in [0,999] (default: 250 500 750)")
    p.add_argument("--output_dir",     default="./distillation_output",
                   help="Where to write the output grids (default: ./distillation_output)")
    p.add_argument("--interposer_dir", default=None,
                   help="Local dir with interposer .safetensors; auto-downloads from HF if omitted")
    p.add_argument("--prompt",         default="",
                   help="Text prompt for conditioning (default: empty string = unconditional)")
    p.add_argument("--size",           type=int, default=512,
                   help="Square image / latent resolution to use (default: 512)")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device (default: cuda if available, else cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_distillation_cycle_test(args)

