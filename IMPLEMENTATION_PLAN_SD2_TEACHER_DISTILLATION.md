# Implementation Plan: SD2-FM → CosXL-FM Teacher Distillation

Covers **Steps 2 and 3** of `sdxl-sd2-vae-teacher.plan.md`.  
Steps 1 (interposer sanity-check) and 4–7 (unfreezing curriculum, logging, post-training) are out of scope.

> **ODE removal**: All multi-step Euler ODE code paths and the `TeacherLatentBridge` class have been
> removed.  Teacher distillation is done entirely via the existing `get_teacher_target` path in
> `core/loss.py`: the teacher runs a **single forward pass** at the current training timestep,
> its velocity prediction is converted to the student's prediction type, and this is used as the
> distillation target alongside (or instead of) the standard FM target.

---

## What exists and is correct

| Concern | Code Location | Status |
|---|---|---|
| `LatentInterposer` — v1↔xl weight loading and conversion | `core/latent_interposer.py` | ✅ complete |
| `get_teacher_target` — same-space FM→FM single-step distillation | `core/loss.py:1115` | ✅ complete |
| v-pred→FM cross-prediction-type conversion | `core/loss.py:1185` | ✅ complete |
| Teacher conditioning: SD2 text encoder loaded separately from student | `model/teacher.py` | ✅ correct |
| Teacher conditioning: `add_time_ids` excluded for non-SDXL teacher | `core/step.py:885` | ✅ correct |
| VAE scaling factors: SDXL=0.13025, SD2/SD1=0.18215 | `core/step.py:761`, `core/loss.py:47` | ✅ correct |
| `teacher_model.is_flow_matching` property | `model/training_model.py:270` | ✅ defined |
| `--teacher`, `--teacher_p` (default 1.0), `--teacher_lambda` CLI args | `train.py` | ✅ exist |
| SDXL `add_time_ids` generated from real crop/size metadata | `data/every_dream.py:238–243` | ✅ correct |
| Logit-normal timestep sampling (flag-gated) | `core/loss.py:1063` | ✅ available |
| Teacher loaded via `load_teacher_model()` in `model/teacher.py` | `model/teacher.py` | ✅ complete |

---

## Distillation approach: single-step teacher prediction (get_teacher_target)

At each training step, for samples in `teacher_mask`, the loss has two terms:

1. **Standard FM loss** using the student's own prediction against the real target (`noise - latents`).
2. **Distillation loss** (`teacher_lambda` weighted): teacher UNet runs once at the same timestep,
   its velocity prediction is (optionally) converted to the student's prediction type via
   `get_teacher_target`, and the student is trained to match this teacher velocity.

When teacher and student share the same latent space (e.g. both SD2-FM) no interposer is needed.  
When they differ (e.g. SD2-FM teacher → SDXL-FM student) `LatentInterposer` converts latents
between spaces before the teacher forward pass.

---

## Bugs fixed

### BUG 1 — Teacher scheduler `set_timesteps` never called (critical)

**File**: `model/teacher.py`  
**Symptom**: `get_sigmas_for_timesteps` (used by `get_teacher_target`) accesses `self.timesteps` and
`self.sigmas` on the teacher's `TrainFlowMatchEulerDiscreteScheduler`, which are only populated by
`set_timesteps()`.

**Fix** (in `model/teacher.py`): After constructing `teacher_model`:
```python
if isinstance(teacher_model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
    teacher_model.set_noise_scheduler_shift(args.flow_match_shift)
```

---

### BUG 2 — Teacher loaded only via `StableDiffusionPipeline` (partial coverage)

**File**: `model/teacher.py`  
**Fix**: Use `AutoPipelineForText2Image.from_pretrained` (handles SD1/SD2/SDXL).  
`--teacher_prediction_type` arg (values: `auto`, `flow_prediction`, `v_prediction`, `epsilon`)
forces the scheduler type when saved with wrong config.

---

## Features implemented

### IMPL 1 — `latent_bridge` removed end-to-end

All `latent_bridge` wiring has been removed from:
- `core/latent_interposer.py` — `TeacherLatentBridge` class deleted
- `core/loss.py` — `latent_bridge` param removed from `get_model_prediction_and_target`
- `core/step.py` — `latent_bridge` param removed from `train_step` and `_do_model_forward`
- `train.py` — `TeacherLatentBridge` import and `latent_bridge` variable removed

### IMPL 2 — Timestep clamping at extremes (bf16 stability)

**File**: `core/step.py`, after `get_shifted_timesteps`  
`--flow_match_t_clamp_min` and `--flow_match_t_clamp_max` clamp sampled timestep indices
after the FM shift is applied, keeping away from t=0/t=999 extremes where bf16 loses precision.

### IMPL 3 — Example config JSON for SD2-FM → CosXL-FM distillation

**File**: `train_sdxl_from_sd2_teacher.json` (create as needed)

### IMPL 4 — Teacher loading in `model/teacher.py`

`load_teacher_model(args, device, student_model)` — complete implementation.

---

## Step 2 (Data pipeline) — no code changes required

- **Tokenization**: CLIP-L BPE vocabulary is shared with SD2's ViT-H encoder — tokens are directly compatible.
- **`tokens_2`**: OpenCLIP-G tokens for SDXL encoder-2. SD2 teacher ignores these (`is_sdxl=False`).
- **`add_time_ids`**: Real crop/size metadata; teacher conditioning sets `add_time_ids=None` for non-SDXL. ✅

---

## Step 3 (Training step) — convention note

The plan pseudocode uses `t=0 → noise, t=1 → clean`.  
This codebase uses diffusers convention: **`t=0 → clean, t=999 → noise`**, `target = noise - latents`.  
Internally consistent for both teacher and student — no sign-flip needed. ✅

---

## Quick verification checklist

- `teacher_model.noise_scheduler.timesteps` is not None after `train.py` setup → BUG 1 fix
- `get_teacher_target` called for `teacher_mask` samples → distillation active
- Teacher forward output shape matches student latent shape (via interposer if cross-space)
- `teacher_lambda`-weighted distillation loss finite after step 1 with `teacher_p=1.0`
