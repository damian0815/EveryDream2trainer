# Implementation Guide: Distilling SD2-FM into a CosXL-FM Student via Bidirectional Interposer

## What you're building

A flow-matching student in SDXL VAE space that inherits aesthetics from an SD2 flow-matching teacher. The teacher and student live in different VAE latent spaces, so we use a bidirectional interposer (clean-latent-only MLP) to bridge them. The teacher provides a predicted clean endpoint per training step; the student learns to flow toward that endpoint.

## Notation (stick to this throughout)

- `vS` = student VAE space (SDXL). `vT` = teacher VAE space (SD2).
- `t ∈ [0, 1]` with **t=0 → noise**, **t=1 → clean**.
- `x_0` = noise sample, `x_1` = clean latent. `x_t = (1−t)·x_0 + t·x_1`.
- Velocity convention: `v = x_1 − x_0`. So `x_1 = x_t + (1−t)·v` and `x_0 = x_t − t·v`.
- `^` denotes a model prediction (e.g. `v_hat`, `x_1_hat`).

If your codebase already uses the opposite t convention (t=0 clean, t=1 noise — common in diffusers), pick one and stick with it; don't mix. Below assumes t=0 noise, t=1 clean.

## Prerequisites

- Frozen, loaded SD2 FM teacher UNet + its tokenizer/text encoder.
- Frozen SDXL VAE (for encoding training images and decoding samples).
- Two interposer MLPs: `interp_S2T` (vS → vT) and `interp_T2S` (vT → vS). Both frozen.
- Initialized CosXL UNet for the student (your starting weights — likely the CosXL v-pred checkpoint).
- A dataset of (image, caption) pairs.
- Both text encoders (SD2's CLIP and SDXL's CLIP-L + OpenCLIP-G). Frozen. Captions get tokenized for both.

## Step 1: Sanity-check the interposers before anything else

Don't skip this. Everything downstream depends on the interposers being clean.

```python
# Pseudocode
images = load_batch(32 real images)
x_1_vS = sdxl_vae.encode(images) * sdxl_vae.config.scaling_factor

# Round-trip
x_1_vT = interp_S2T(x_1_vS)
x_1_vS_roundtrip = interp_T2S(x_1_vT)

mse = ((x_1_vS - x_1_vS_roundtrip) ** 2).mean()
print(f"Round-trip MSE: {mse}")

# Also decode both and eyeball
img_orig = sdxl_vae.decode(x_1_vS / scaling_factor)
img_rt = sdxl_vae.decode(x_1_vS_roundtrip / scaling_factor)
```

Acceptable round-trip MSE is hard to give a number for without knowing the interposer; what matters is that decoded `img_rt` looks like `img_orig` without obvious color shift or blur. If it doesn't, fix the interposer first — no training will recover what's lost here.

Also verify direction: `interp_S2T` should take SDXL-shaped latents (4 channels at SDXL scale) and produce SD2-shaped latents (4 channels at SD2 scale). Print shapes and value ranges. SD2 latents typically have std ~1 after scaling by 0.18215; SDXL latents have std ~1 after scaling by 0.13025. Make sure your interposer expects scaled or unscaled inputs and you're feeding it the right thing. **This is the most common source of silent bugs.**

## Step 2: Data pipeline

Per training sample, you need:
- `image` → encoded to `x_1_vS` (SDXL VAE, scaled).
- `caption` → tokenized for both the teacher's text encoder and the student's two text encoders.
- SDXL conditioning: `original_size`, `crop_top_left`, `target_size` for the `add_embedding` pathway. Use real values from your dataset or sensible defaults (e.g. (1024, 1024), (0, 0), (1024, 1024)).

Cache `x_1_vS` to disk if dataset is fixed; VAE encoding is expensive to repeat.

## Step 3: The training step

Here's the full forward pass in pseudocode:

```python
def training_step(batch):
    x_1_vS = batch["latents_vS"]              # (B, 4, H, W) SDXL latents, scaled
    caption_T = batch["tokens_teacher"]        # for SD2 text encoder
    caption_S = batch["tokens_student"]        # for SDXL text encoders (both)
    add_cond = batch["sdxl_add_cond"]          # size/crop tuple

    B = x_1_vS.shape[0]

    # ---- Sample noise and time ----
    x_0_vS = torch.randn_like(x_1_vS)

    # Logit-normal time sampling, centered at 0.5, std ~1
    # (avoids degenerate endpoints near t=0 and t=1)
    t = torch.sigmoid(torch.randn(B, device=x_1_vS.device))
    t_b = t.view(B, 1, 1, 1)  # broadcastable

    # ---- TEACHER PATH (no grad) ----
    with torch.no_grad():
        # Map clean and noise to teacher space
        x_1_vT = interp_S2T(x_1_vS)
        x_0_vT = interp_S2T(x_0_vS)  # NOTE: see caveat below

        # Build teacher's noisy input
        x_t_vT = (1 - t_b) * x_0_vT + t_b * x_1_vT

        # Teacher's time convention may differ — convert if needed
        # If teacher uses t_teacher = 1 - t (clean→noise convention), flip here.
        t_teacher = convert_time_if_needed(t)

        # Run teacher UNet
        text_emb_T = teacher_text_encoder(caption_T)
        v_hat_vT = teacher_unet(x_t_vT, t_teacher, text_emb_T)

        # Recover teacher's clean-endpoint prediction
        # x_1 = x_t + (1-t)*v  (under t=0 noise, t=1 clean, v=x_1-x_0)
        x_1_hat_vT = x_t_vT + (1 - t_b) * v_hat_vT

        # Map back to student space (interposer is clean-latent-only — OK here)
        x_1_hat_vS = interp_T2S(x_1_hat_vT)

        # Build student's distillation target
        v_target_vS = x_1_hat_vS - x_0_vS

    # ---- STUDENT PATH (with grad) ----
    x_t_vS = (1 - t_b) * x_0_vS + t_b * x_1_vS

    text_emb_S = student_text_encoder(caption_S)  # both CLIPs concatenated
    add_emb = student_add_embedding(add_cond)

    v_hat_vS = student_unet(x_t_vS, t, text_emb_S, added_cond=add_emb)

    # ---- Loss ----
    loss = ((v_hat_vS - v_target_vS) ** 2).mean()
    return loss
```

### Caveat on `interp_S2T(x_0_vS)`

The interposer was trained on clean latents. Pure Gaussian noise is not clean latents. Two options:

**Option A (cleanest):** Sample `x_0_vT` independently as fresh Gaussian noise in teacher space, *not* by interposing student noise. Then `x_t_vT` doesn't correspond to the same noise instance as `x_t_vS`, but that's fine — the teacher's job is just to predict its own clean endpoint, and the noise is interchangeable.

**Option B:** Interpose the noise anyway. In practice the interposer applied to Gaussian noise produces something roughly Gaussian-like, and SD2 trained on a wide noise distribution may tolerate it. But it's out-of-distribution for the interposer.

**Recommendation: use Option A.** Sample teacher-space noise fresh.

```python
x_0_vT = torch.randn_like(x_1_vT)  # fresh, not interposed
```

You lose nothing — the noise samples don't need to correspond across spaces.

### Time convention conversion

Diffusers' SD2 v-pred typically uses discrete `timestep ∈ [0, 999]` with `0` = clean, `999` = noise. If you're using a converted FM teacher trained per the earlier curriculum, it should already accept continuous `t ∈ [0, 1]`. **Verify which convention your teacher expects.** A common mistake is feeding `t` with the wrong polarity and getting a network that "almost works" — outputs are coherent but stylistically wrong, and it's hard to debug.

Quick test: feed teacher `(pure_noise, t=0)` and `(pure_noise, t=1)`, predict endpoints, decode. The one that gives a plausible image is the "noise time"; the other gives garbage.

## Step 4: Apply the layer-unfreezing curriculum

Follow the CosXL curriculum from earlier. Starting weights are CosXL v-pred. Stage 0 sanity is now: confirm one training step runs end-to-end and loss is finite. Then proceed:

- **Stage 1 (~3-6k steps):** unfreeze `conv_out`, time MLP, `add_embedding`. Everything else frozen. LR 1e-4.
- **Stage 2 (~8-20k):** add `attn2.to_k`, `attn2.to_v`, all norm affines. LR 1e-4.
- **Stage 3 (~15-30k):** add mid-block transformer fully + deepest down/up blocks. LR 5e-5 new, 1e-4 prior.
- **Stage 4 (remainder):** full UNet except VAE/text encoders. LR 5e-5 with 0.5× multiplier on high-res blocks.

Implement as a parameter group setup. Each stage transition: rebuild optimizer with new param groups, *keep* the optimizer state for previously-unfrozen params (Adam moments matter), reset only for newly-unfrozen params.

## Step 5: Logging and validation

Log per step:
- Training loss (raw and EMA-smoothed).
- Mean of `t` sampled (sanity — should be ~0.5 for logit-normal).
- Norms of `v_target_vS` and `v_hat_vS` (should be similar magnitude; if target norm explodes at small/large t, you have an interposer issue at extreme noise levels).

Validation every N steps (N = 500-2000):
- Fixed set of 16-32 held-out captions.
- Generate samples with 8-step Euler from `x_0 ~ N(0, I)`, decode with SDXL VAE.
- Compute student FM loss on a held-out batch of real latents.
- Visually inspect samples. This is the single most important signal; loss curves lie, eyeballs don't.

Triggers for stage transitions (preferred over fixed step counts):
- Validation loss plateaus for ~1k steps.
- For Stage 2→3: track norm of cross-attention K/V outputs on a fixed eval batch — wait until stable.
- For Stage 3→4: mid-time (t ∈ [0.3, 0.7]) validation loss catches up to extreme-time loss.

## Step 6: Things that will go wrong, in rough order of likelihood

1. **Wrong VAE scaling factor.** SDXL is 0.13025, SD2 is 0.18215. Apply on encode, divide on decode. Get this wrong and everything looks plausibly trained but generates garbage.
2. **Time convention mismatch.** Teacher expects opposite t polarity. Symptom: training loss decreases but samples are blurry or wrong-style.
3. **Interposer input not scaled correctly.** Interposer expects scaled or unscaled latents; check its training code.
4. **Add_embedding fed wrong shape.** SDXL `add_embedding` expects a specific concatenation order of size/crop/target embeddings. Use the diffusers reference implementation.
5. **Mixed precision blowing up at extreme t.** At t very close to 0 or 1, target velocity has small components that fp16 rounds to zero. Use bf16 if available, or clamp t to [0.01, 0.99].
6. **Forgetting `torch.no_grad()` on teacher path.** Memory explosion and slow training.
7. **Reusing same noise for teacher and student.** With Option A above (independent noise) this isn't an issue. With Option B (interposed noise) it's "correct" but OOD for the interposer.

## Step 7: When training is done

Generate a comparison grid: same caption, samples from teacher (decode through SD2 VAE) vs samples from student (decode through SDXL VAE). The student should look like the teacher's aesthetic but at SDXL's resolution and fidelity. If it looks like a generic SDXL output, distillation didn't take — most likely Stage 1/2 didn't run long enough and the head/conditioning didn't realign. If it looks like the teacher but blurrier or with color shift, that's interposer error — improve the interposer.

Optionally, do a ReFlow pass on the trained student: generate (x_0, x_1) pairs using the student's own ODE, then train another pass with those as coupled pairs. This straightens trajectories and enables few-step or 1-step sampling.

That's the full pipeline. Build it incrementally: get Step 1 working before Step 2, get a single training step running before any training loop, get 100 steps clean before launching a full run.