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


# ---------------------------------------------------------------------------
# Model-loading helpers
# ---------------------------------------------------------------------------

def _vae_subdir(model_path: str) -> str:
    """Return the path to use for AutoencoderKL, handling both full pipelines and bare VAEs."""
    candidate = os.path.join(model_path, "vae")
    return candidate if os.path.isdir(candidate) else model_path


def load_vae(model_path: str, device: str, dtype: torch.dtype):
    from diffusers import AutoencoderKL
    path = _vae_subdir(model_path)
    print(f"  Loading VAE from: {path}")
    vae = AutoencoderKL.from_pretrained(path, torch_dtype=dtype)
    vae.eval().to(device)
    return vae


def load_unet(model_path: str, device: str, dtype: torch.dtype):
    from diffusers import UNet2DConditionModel
    candidate = os.path.join(model_path, "unet")
    path = candidate if os.path.isdir(candidate) else model_path
    print(f"  Loading UNet from: {path}")
    unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=dtype)
    unet.eval().to(device)
    return unet


def load_text_encoder_sd2(model_path: str, device: str, dtype: torch.dtype):
    """Load a single CLIP text encoder (SD2 uses a single ViT-H encoder, hidden_size=1024)."""
    from transformers import CLIPTextModel, CLIPTokenizer
    te_dir = os.path.join(model_path, "text_encoder")
    te_path = te_dir if os.path.isdir(te_dir) else model_path
    tok_path = te_path
    print(f"  Loading SD2 text encoder from: {te_path}")
    tokenizer = CLIPTokenizer.from_pretrained(tok_path)
    text_encoder = CLIPTextModel.from_pretrained(te_path, torch_dtype=dtype)
    text_encoder.eval().to(device)
    return tokenizer, text_encoder


def load_text_encoders_sdxl(model_path: str, device: str, dtype: torch.dtype):
    """Load both SDXL text encoders (CLIP-L + OpenCLIP-ViT-G)."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    te1_dir = os.path.join(model_path, "text_encoder")
    te2_dir = os.path.join(model_path, "text_encoder_2")
    tok1_dir = te1_dir
    tok2_dir = te2_dir

    print(f"  Loading SDXL text_encoder from: {te1_dir}")
    tokenizer1 = CLIPTokenizer.from_pretrained(tok1_dir)
    te1 = CLIPTextModel.from_pretrained(te1_dir, torch_dtype=dtype)
    te1.eval().to(device)

    print(f"  Loading SDXL text_encoder_2 from: {te2_dir}")
    tokenizer2 = CLIPTokenizer.from_pretrained(tok2_dir)
    te2 = CLIPTextModelWithProjection.from_pretrained(te2_dir, torch_dtype=dtype)
    te2.eval().to(device)

    return (tokenizer1, te1), (tokenizer2, te2)


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def get_sd2_conditioning(prompt: str, tokenizer, text_encoder, device, dtype):
    """Returns encoder_hidden_states [1, 77, 1024] for SD2."""
    with torch.no_grad():
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        enc_out = text_encoder(tokens, return_dict=True)
        hidden_states = enc_out.last_hidden_state.to(dtype=dtype)
    return hidden_states


def get_sdxl_conditioning(
    prompt: str,
    tokenizer1, te1,
    tokenizer2, te2,
    device, dtype,
    target_size=(1024, 1024),
    original_size=(1024, 1024),
    crop_top_left=(0, 0),
):
    """
    Returns (encoder_hidden_states, pooled_embeds, add_time_ids) for SDXL.
    encoder_hidden_states : [1, 77, 2048]  (concat of hidden1 + hidden2)
    pooled_embeds          : [1, 1280]
    add_time_ids           : [1, 6]
    """
    with torch.no_grad():
        def encode(tok, enc, prompt):
            tokens = tok(
                [prompt],
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = enc(tokens, output_hidden_states=True, return_dict=True)
            # Use penultimate layer hidden state (standard diffusers SDXL practice)
            return out.hidden_states[-2], getattr(out, "text_embeds", None)

        h1, _    = encode(tokenizer1, te1, prompt)  # [1, 77, 768]
        h2, pool = encode(tokenizer2, te2, prompt)  # [1, 77, 1280], [1, 1280]

        encoder_hidden_states = torch.cat([h1, h2], dim=-1).to(dtype=dtype)  # [1, 77, 2048]
        pooled_embeds = pool.to(dtype=dtype) if pool is not None else torch.zeros(1, 1280, device=device, dtype=dtype)

        # Build add_time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
        add_time_ids = torch.tensor(
            [list(original_size) + list(crop_top_left) + list(target_size)],
            dtype=dtype, device=device,
        )  # [1, 6]

    return encoder_hidden_states, pooled_embeds, add_time_ids


# ---------------------------------------------------------------------------
# Latent & image utilities
# ---------------------------------------------------------------------------

def load_image(path: str, size: int) -> torch.Tensor:
    """Load image → [1, 3, H, W] float32 in [-1, 1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    left, top = (w - short) // 2, (h - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def encode_image(vae, pixel_tensor, scaling_factor: float, device: str, dtype: torch.dtype):
    """pixel_tensor: [1,3,H,W] in [-1,1]. Returns scaled latents [1,4,h,w]."""
    with torch.no_grad():
        dist = vae.encode(pixel_tensor.to(device=device, dtype=dtype), return_dict=False)[0]
        return dist.sample() * scaling_factor


def decode_latents_to_numpy(vae, scaled_latents, scaling_factor: float) -> np.ndarray:
    """Returns [H, W, 3] uint8."""
    with torch.no_grad():
        unscaled = scaled_latents.to(vae.dtype) / scaling_factor
        decoded  = vae.decode(unscaled, return_dict=False)[0].float().cpu()
    arr = (decoded.squeeze(0).permute(1, 2, 0).clamp(-1, 1).numpy() + 1.0) / 2.0 * 255
    return arr.clip(0, 255).astype(np.uint8)


def pixel_tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] → [H, W, 3] uint8."""
    arr = (t.float().squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255
    return arr.clip(0, 255).astype(np.uint8)


def make_diff_panel(a: np.ndarray, b: np.ndarray, amplify: float = 3.0) -> np.ndarray:
    """Compute amplified absolute difference between two uint8 images."""
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
    """Vertically stack a list of PIL strips."""
    total_h = sum(s.height for s in strips)
    total_w = max(s.width for s in strips)
    grid = Image.new("RGB", (total_w, total_h), (200, 200, 200))
    y = 0
    for s in strips:
        grid.paste(s, (0, y))
        y += s.height
    return grid


# ---------------------------------------------------------------------------
# Core cycle logic
# ---------------------------------------------------------------------------

def run_one_sample(
    *,
    pixel_tensor,       # [1, 3, H, W] float32 in [-1, 1]
    timestep: int,      # integer in [0, 999]
    sdxl_vae,
    sd2_vae,
    sdxl_unet,
    sd2_unet,
    sdxl_cond: tuple,   # (encoder_hidden_states, pooled_embeds, add_time_ids)
    sd2_cond,           # encoder_hidden_states [1, 77, 1024]
    sdxl_scale: float,
    sd2_scale: float,
    interposer: LatentInterposer,
    device: str,
    dtype: torch.dtype,
    size: int,
):
    """
    Run the full distillation cycle for one image at one timestep.
    Returns a dict of named numpy images (uint8) for each phase.
    """
    # σ = t/T, linear FM schedule; clamp slightly away from endpoints
    sigma = max(0.01, min(0.99, timestep / 1000.0))
    print(f"    σ = {sigma:.4f} (timestep={timestep})")

    # ---- SDXL: encode clean image ----
    x_1_xl = encode_image(sdxl_vae, pixel_tensor, sdxl_scale, device, dtype)  # [1,4,h,w]

    # ---- Student noise (shared) ----
    x_0_xl = torch.randn_like(x_1_xl)

    # ---- Build student's noisy latent ----
    x_t_xl = (1.0 - sigma) * x_1_xl + sigma * x_0_xl  # [1,4,h,w]

    # ========= TEACHER PATH (no grad) =========
    with torch.no_grad():

        # 1. Map clean student latent → teacher VAE space
        x_1_v1 = interposer.convert(
            x_1_xl.float(), src=LatentSpaceType.SDXL, dst=LatentSpaceType.SD1
        ).to(device=device, dtype=dtype)

        # 2. Option A: independent noise in teacher space
        x_0_v1 = torch.randn_like(x_1_v1)

        # 3. Build teacher's noisy latent
        x_t_v1 = (1.0 - sigma) * x_1_v1 + sigma * x_0_v1  # [1,4,h,w]

        # 4. Run SD2 teacher UNet
        #    SD2 expects timestep as a float tensor in the range the scheduler's timesteps cover.
        #    Flow-matching schedulers use timesteps like 999.0 → 1.0 (high=noisy).
        #    Here sigma ∈ [0,1] from clean→noise, so teacher timestep = sigma * 1000.
        t_tensor = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)

        v_hat_v1 = sd2_unet(
            x_t_v1.to(dtype=sd2_unet.dtype),
            t_tensor.to(dtype=sd2_unet.dtype),
            encoder_hidden_states=sd2_cond.to(device=device, dtype=sd2_unet.dtype),
        ).sample.float().to(device)

        # 5. Recover teacher's predicted clean endpoint in SD2 space
        #    Convention (code): v = x_0 − x_1  ⟹  x_1 = x_t − σ·v
        x_1_hat_v1 = x_t_v1.float() - sigma * v_hat_v1

        # 6. Map teacher's predicted clean → SDXL space
        x_1_hat_xl_teacher = interposer.convert(
            x_1_hat_v1.to(dtype=torch.float32), src=LatentSpaceType.SD1, dst=LatentSpaceType.SDXL
        ).to(device=device, dtype=dtype)

        # 7. Compute distillation velocity target (for reference / logging)
        v_target_xl = x_0_xl.float() - x_1_hat_xl_teacher.float()
        print(f"    v_target_xl norm: {v_target_xl.norm().item():.4f}")

    # ========= STUDENT PATH (no grad for visual test) =========
    with torch.no_grad():
        enc_hs, pooled, add_time_ids = sdxl_cond
        t_tensor_xl = torch.tensor([sigma * 1000.0], device=device, dtype=dtype)

        v_hat_xl = sdxl_unet(
            x_t_xl.to(dtype=sdxl_unet.dtype),
            t_tensor_xl.to(dtype=sdxl_unet.dtype),
            encoder_hidden_states=enc_hs.to(device=device, dtype=sdxl_unet.dtype),
            added_cond_kwargs={
                "text_embeds": pooled.to(device=device, dtype=sdxl_unet.dtype),
                "time_ids":    add_time_ids.to(device=device, dtype=sdxl_unet.dtype),
            },
        ).sample.float().to(device)

        # Recover student's predicted clean endpoint
        x_1_hat_xl_student = x_t_xl.float() - sigma * v_hat_xl
        print(f"    v_hat_xl norm:    {v_hat_xl.norm().item():.4f}")

    # ========= Decode all phases =========
    panel_orig      = pixel_tensor_to_numpy(pixel_tensor)
    panel_noisy     = decode_latents_to_numpy(sdxl_vae,  x_t_xl,                sdxl_scale)
    panel_teach_v1  = decode_latents_to_numpy(sd2_vae,   x_1_hat_v1.to(dtype=sd2_vae.dtype),   sd2_scale)
    panel_teach_xl  = decode_latents_to_numpy(sdxl_vae,  x_1_hat_xl_teacher,    sdxl_scale)
    panel_stud_xl   = decode_latents_to_numpy(sdxl_vae,  x_1_hat_xl_student.to(dtype=sdxl_vae.dtype), sdxl_scale)
    panel_diff      = make_diff_panel(panel_teach_xl, panel_stud_xl, amplify=3.0)

    return {
        "orig":      panel_orig,
        "noisy":     panel_noisy,
        "teach_v1":  panel_teach_v1,
        "teach_xl":  panel_teach_xl,
        "stud_xl":   panel_stud_xl,
        "diff":      panel_diff,
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
    dtype  = torch.float16 if "cuda" in device else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading SDXL components ===")
    sdxl_vae  = load_vae(args.sdxl_model,  device, dtype)
    sdxl_unet = load_unet(args.sdxl_model, device, dtype)
    (tok1, te1), (tok2, te2) = load_text_encoders_sdxl(args.sdxl_model, device, dtype)
    sdxl_scale = getattr(sdxl_vae.config, "scaling_factor", 0.13025)
    print(f"  SDXL VAE scaling_factor = {sdxl_scale}")

    print("=== Loading SD2 components ===")
    sd2_vae  = load_vae(args.sd2_model,  device, dtype)
    sd2_unet = load_unet(args.sd2_model, device, dtype)
    tok_sd2, te_sd2 = load_text_encoder_sd2(args.sd2_model, device, dtype)
    sd2_scale = getattr(sd2_vae.config, "scaling_factor", 0.18215)
    print(f"  SD2  VAE scaling_factor = {sd2_scale}")

    # Sanity-check UNet input channel counts
    sdxl_in_ch = sdxl_unet.config.get("in_channels", 4)
    sd2_in_ch  = sd2_unet.config.get("in_channels",  4)
    print(f"\n  SDXL UNet in_channels={sdxl_in_ch}  |  SD2 UNet in_channels={sd2_in_ch}")
    assert sdxl_in_ch == 4, f"Expected SDXL UNet to have 4 in_channels, got {sdxl_in_ch}"
    assert sd2_in_ch  == 4, f"Expected SD2  UNet to have 4 in_channels, got {sd2_in_ch}"

    print("=== Loading Interposer ===")
    interposer = LatentInterposer(model_dir=args.interposer_dir)

    print("=== Building text conditioning ===")
    prompt = args.prompt or ""
    print(f"  Prompt: {repr(prompt)}")
    # SDXL size conditioning
    target_sz = (args.size, args.size)
    sdxl_cond = get_sdxl_conditioning(
        prompt, tok1, te1, tok2, te2, device, dtype,
        target_size=target_sz, original_size=target_sz,
    )
    sd2_cond = get_sd2_conditioning(prompt, tok_sd2, te_sd2, device, dtype)
    print(f"  SDXL hidden_states: {sdxl_cond[0].shape}  pooled: {sdxl_cond[1].shape}")
    print(f"  SD2  hidden_states: {sd2_cond.shape}")

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
                sdxl_vae=sdxl_vae,
                sd2_vae=sd2_vae,
                sdxl_unet=sdxl_unet,
                sd2_unet=sd2_unet,
                sdxl_cond=sdxl_cond,
                sd2_cond=sd2_cond,
                sdxl_scale=sdxl_scale,
                sd2_scale=sd2_scale,
                interposer=interposer,
                device=device,
                dtype=dtype,
                size=args.size,
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

