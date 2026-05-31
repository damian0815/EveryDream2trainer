# SANA Training with EveryDream2trainer

This document explains how to fine-tune a [SANA](https://github.com/NVlabs/Sana) model using the EveryDream2trainer infrastructure.

---

## Overview

`train_sana.py` is a SANA-specific training entry point that reuses the battle-tested ED2 machinery:

| ED2 component | Role in SANA training |
|---|---|
| `core/step.py` → `run_accumulation_loop()` | Nibble slicing, OOM-robust accumulation, backward pass, optimizer step |
| `data/every_dream.py` → `EveryDreamBatch` | Multi-aspect image loading, caption handling |
| `optimizer/optimizers.py` → `EveryDreamOptimizer` | Optimizer wrapping (AdamW, CAME, Prodigy, …) |
| `core/log.py` | TensorBoard logging utilities |
| `utils/inference_context.py` | Eval/train guard for sample generation |

SANA-specific modules:

| Module | Responsibility |
|---|---|
| `model/sana_training_model.py` | `SanaTrainingModel` dataclass, `load_sana_model()`, `save_sana_model()` |
| `model/sana_text_encoder.py` | `encode_sana_text()` supporting T5, Gemma, and Qwen encoders |
| `core/loss_sana.py` | `sample_sana_timesteps()`, `compute_sana_loss()` |
| `train_sana.py` | Entry point: argument parsing, data wiring, training loop, sample generation |

---

## Prerequisites

### 1. Install the SANA library

SANA is not on PyPI. Clone and install it from NVIDIA's repo:

```bash
git clone https://github.com/NVlabs/Sana.git
cd Sana
pip install -e .
cd ..
```

### 2. Install EveryDream2trainer dependencies

```bash
pip install -r requirements.txt
```

Key additions for SANA (already in `requirements.txt`):

```
pyrallis
termcolor
```

### 3. Download a SANA checkpoint

Follow the instructions in the [SANA model card](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e) to download a pretrained checkpoint (e.g. `Sana_1600M_1024px`).

---

## Data preparation

SANA training uses the standard ED2 data layout — the same folder structure you already use for SD/SDXL fine-tuning:

```
my_dataset/
    image1.jpg
    image1.txt          # caption (one per image)
    image2.png
    image2.txt
    subfolder/
        image3.webp
        image3.txt
```

Captions can also be stored as sidecar `.caption` files. All the existing ED2 tagging and caption-shuffle features are supported.

> **Important:** The SANA DC-AE VAE expects pixel values in **[0, 255]**, not the usual [-1, 1].  
> `EveryDreamBatch` handles this automatically when `image_output_range="[0,255]"` is set — which `train_sana.py` does by default.

---

## Configuration

SANA uses [pyrallis](https://github.com/eladrich/pyrallis) YAML config files. Copy one from the SANA repo and adjust it, or start from the example below.

### Minimal `my_sana_config.yaml`

```yaml
work_dir: output/sana-finetune

model:
  model: SanaMS_1600M_P1_D20
  image_size: 1024
  mixed_precision: bf16
  load_from: /path/to/Sana_1600M_1024px.pth   # pretrained checkpoint

vae:
  vae_type: dc-ae
  vae_config: /path/to/Sana/configs/vae/dc-ae-f32c32-sana-1.0.yaml
  vae_pretrained: /path/to/dc-ae-f32c32-sana-1.0-diffusers
  scale_factor: 0.41407
  vae_latent_dim: 32
  vae_downsample_rate: 32
  sample_posterior: true

text_encoder:
  text_encoder_name: gemma-2-2b-it
  model_max_length: 300
  chi_prompt: []

scheduler:
  train_sampling_steps: 1000
  noise_schedule: linear_flow
  weighting_scheme: logit_normal
  logit_mean: 0.0
  logit_std: 1.0
  vis_sampler: flow_euler
  cfg_scale: 4.5

train:
  train_batch_size: 2
  num_steps: 10000
  save_model_steps: 1000
  gradient_accumulation_steps: 1
  seed: 42
  optimizer:
    lr: 2e-5

data:
  data_dir: /path/to/my_dataset
```

---

## Training

```bash
python train_sana.py \
    --sana_config my_sana_config.yaml \
    --project_name my-sana-finetune \
    --logdir logs \
    --data_root /path/to/my_dataset \
    --batch_size 2 \
    --max_steps 10000 \
    --save_every 1000 \
    --sample_every 500
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--sana_config` | *(required)* | Path to the SanaConfig YAML |
| `--project_name` | from YAML `work_dir` | Name of the run (used as log subdirectory) |
| `--logdir` | from YAML | Root directory for TensorBoard logs and checkpoints |
| `--data_root` | from YAML `data.data_dir` | Training data directory |
| `--batch_size` | from YAML | Data loading batch size — also the aspect-ratio bucket size. See [Batch size vs. optimizer batch size](#batch-size-vs-optimizer-batch-size) |
| `--optimizer_batch_size` | = batch\_size | Effective optimizer batch size (samples between optimizer steps). See [Batch size vs. optimizer batch size](#batch-size-vs-optimizer-batch-size) |
| `--resolution` | from YAML `model.image_size` | One or more training resolutions — see [Multi-resolution training](#multi-resolution-training) |
| `--resolution_multiplier` | — | Per-resolution dataset weight — see [Multi-resolution training](#multi-resolution-training) |
| `--forward_slice_size` | = batch\_size | Max images per forward-pass slice — see [Memory management](#memory-management-forward_slice_size-and-max_backward_slice_size) |
| `--max_backward_slice_size` | = batch\_size | Max images accumulated before backward — see [Memory management](#memory-management-forward_slice_size-and-max_backward_slice_size) |
| `--skip_undersized_images` | false | Drop images smaller than the target resolution bucket |
| `--lr` | from YAML | Learning rate |
| `--max_steps` | from YAML | Total number of optimizer steps |
| `--grad_accum` | from YAML | Gradient accumulation steps |
| `--mixed_precision` | from YAML | `bf16` / `fp16` / `no` |
| `--seed` | from YAML | Random seed |
| `--resume_from` | — | Path to a checkpoint to resume from |
| `--save_every` | 1000 | Save transformer weights every N steps |
| `--sample_every` | 500 | Generate validation samples every N steps |
| `--num_workers` | 4 | DataLoader worker count |
| `--amp` | true | Enable AMP for the backward pass |
| `--max_epochs` | 100 | Maximum number of data epochs |

CLI flags take **precedence** over the YAML config.

---

## Multi-resolution training

SANA's `SanaConfig` has a single `model.image_size` field, but `train_sana.py` supports multiple simultaneous training resolutions — the same way SD/SDXL training works in EveryDream2trainer.

### How it works

When you pass multiple values to `--resolution`, the data pipeline scans your dataset **once per resolution**. Each scan creates aspect-ratio bucketed `ImageTrainItem` objects at that resolution. All items from all resolutions are merged into a single `DataLoaderMultiAspect`. On every batch the trainer inspects the actual image pixel count and selects the matching resolution bucket to apply the correct slice sizes.

```bash
python train_sana.py \
    --sana_config my_sana_config.yaml \
    --resolution 512 1024 \
    --batch_size 4 \
    --max_steps 10000
```

`model.image_size` in the YAML is used **only** as the resolution for sample generation (inference). Training can cover any number of resolutions regardless of that value.

### `--resolution_multiplier`

Control how much each resolution contributes to the epoch. Pass one multiplier per `--resolution` entry:

```bash
# Train twice as much on 1024px images as on 512px images
python train_sana.py \
    --resolution 512 1024 \
    --resolution_multiplier 1.0 2.0 \
    ...
```

Omitting `--resolution_multiplier` treats all resolutions equally (multiplier = 1.0 each).

---

## Memory management: `--forward_slice_size` and `--max_backward_slice_size`

These flags are identical in meaning to their SD/SDXL counterparts and are fully supported.

### `--forward_slice_size`

Splits the forward pass into smaller chunks to reduce peak VRAM usage. The **loss batch size is unaffected** — slices are accumulated before the backward pass. The trainer halves the slice size automatically when an OOM is detected mid-forward.

```bash
# At 1024px only 1 image fits in VRAM for the forward pass
python train_sana.py \
    --resolution 1024 \
    --forward_slice_size 1 \
    --batch_size 4 \
    ...
```

Pass one value per resolution for mixed setups:

```bash
# 4 images per forward pass at 512px, 1 image at 1024px
python train_sana.py \
    --resolution 512 1024 \
    --forward_slice_size 4 1 \
    --batch_size 4 \
    ...
```

Or pass a single value to apply it to all resolutions:

```bash
--forward_slice_size 2   # applied to every resolution
```

### `--max_backward_slice_size`

Caps how many loss images accumulate in the graph before a backward pass fires. Lowering this reduces the peak gradient memory footprint independently of `--batch_size`.

```bash
python train_sana.py \
    --resolution 512 1024 \
    --max_backward_slice_size 8 2 \
    --batch_size 16 \
    ...
```

### Typical VRAM setups

| GPU VRAM | Suggested flags |
|---|---|
| 24 GB (RTX 4090) | `--resolution 1024 --batch_size 4` |
| 16 GB (RTX 4080) | `--resolution 1024 --batch_size 2 --forward_slice_size 1` |
| 12 GB (RTX 4080S) | `--resolution 1024 --batch_size 2 --forward_slice_size 1 --max_backward_slice_size 1` |
| 8 GB | `--resolution 512 --batch_size 2 --forward_slice_size 1` |

---

---

## Batch size vs. optimizer batch size

EveryDream2trainer makes a deliberate distinction between two batch-size concepts, and SANA inherits it in full.

### `--batch_size` — data loading / aspect-ratio bucketing

`--batch_size` controls how many images are loaded and processed together in a single data batch. `DataLoaderMultiAspect` fills each batch exclusively with images that share the same aspect-ratio bucket, so a batch of `batch_size=4` will contain 4 images at (e.g.) 1024×1024. This is the minimum granularity of a forward pass.

### `--optimizer_batch_size` — effective training batch size

`--optimizer_batch_size` is the number of images that must be processed before `optimizer.step()` is called. When it is larger than `--batch_size`, gradients are accumulated across multiple data batches (potentially from different aspect-ratio buckets) before stepping.

The accumulation is **not** a naïve `grad_accum` multiplier. The nibble machinery in `run_accumulation_loop()` counts actual images via `TrainingVariables.backwarded_images_count`, so the optimizer step fires after the correct number of samples regardless of how the memory slicing divides the data.

```
batch_size=4, optimizer_batch_size=16:

 data batch 1 → [img_A 1024×1024  ×4]  → forward → backward  (4 images)
 data batch 2 → [img_B  768×1024  ×4]  → forward → backward  (4 images)
 data batch 3 → [img_C 1024×1024  ×4]  → forward → backward  (4 images)
 data batch 4 → [img_D  512× 512  ×4]  → forward → backward  (4 images)
                                                              ── 16 images ──▶ optimizer.step()
```

The DataLoader is told about the effective window size (`grad_accum = optimizer_batch_size // batch_size`) so that it can keep the same-batch-id images adjacent where possible.

### Example

```bash
# Load 2 images at a time (aspect-ratio bucketing at batch_size=2),
# but step the optimizer only every 16 images.
python train_sana.py \
    --sana_config my_sana_config.yaml \
    --batch_size 2 \
    --optimizer_batch_size 16 \
    --forward_slice_size 1 \
    ...
```

### Sample prompts

Place a `sample_prompts.txt` file next to your YAML config (one prompt per line). The trainer generates images at every `--sample_every` step and writes them to:

- `{logdir}/{project_name}/samples-sana/gs{N}/` — WebP tiles on disk  
- TensorBoard under the `samples-sana/` tag

---

## Outputs

```
logs/my-sana-finetune/
    events.out.tfevents.*       # TensorBoard
    null_embed_cache.pt         # cached null-text embedding (reused across runs)
    gs1000/
        transformer_gs1000.safetensors
        config.yaml
    gs2000/
        transformer_gs2000.safetensors
        config.yaml
    ...
    final/
        transformer_gs10000.safetensors
        config.yaml
    samples-sana/
        gs0/
            000_a photo of a cat.webp
        gs500/
            000_a photo of a cat.webp
```

Only the **transformer** is saved — the text encoder and VAE are frozen and unchanged from the base model.

---

## Resuming a run

```bash
python train_sana.py \
    --sana_config my_sana_config.yaml \
    --resume_from logs/my-sana-finetune/gs5000/transformer_gs5000.safetensors \
    ...
```

Or set `model.resume_from` in your YAML.

---

## Optimizer configuration

`train_sana.py` uses ED2's `EveryDreamOptimizer`, so you can use any optimizer supported by ED2. Pass an optimizer JSON via the standard ED2 mechanism, or rely on defaults (AdamW).

Example — using a cosine schedule at 2e-5:

```bash
python train_sana.py \
    --sana_config my_sana_config.yaml \
    --lr 2e-5 \
    ...
```

For advanced schedules, copy one of the `optimizer-*.json` files from the repo root and refer to `doc/ADVANCED_TWEAKING.md`.

---

## Architecture notes

### Why `run_accumulation_loop()`?

`train_step()` in `core/step.py` was refactored into two layers:

1. **`run_accumulation_loop()`** — model-agnostic. Handles nibble slicing, OOM retry on the backward pass, gradient accumulation, and the optimizer step.
2. **`nibble_loss_fn`** — a closure passed into the loop. Contains all model-specific forward-pass logic.

`train_sana_step()` builds a SANA-specific `nibble_loss_fn` (which calls `compute_sana_loss()`) and passes it to `run_accumulation_loop()`. This means SANA gets the same OOM-robustness as SD/SDXL for free.

### Text encoders

Three branches are supported, selected by `config.text_encoder.text_encoder_name`:

| Name contains | Branch |
|---|---|
| `T5` | Standard T5 tokenize → encode → `(B, 1, N, C)` |
| `gemma` or `Qwen` | Gemma/Qwen with optional system prompt (`chi_prompt`), select-index reshaping |

### VAE

SANA's DC-AE expects **[0, 255]** pixel values. `EveryDreamBatch` is initialised with `image_output_range="[0,255]"` which converts the standard [-1, 1] output after normalisation:

```
image = (image * 0.5 + 0.5) * 255.0
```

### What is NOT yet implemented

| Feature | Status |
|---|---|
| EMA (`transformer_ema` field) | Reserved slot, not wired up |
| Pre-computed VAE / text-encoder feature caching | Deferred — add to `data/latent_cache.py` |
| Multi-GPU via `accelerate` | Described in `plan-sanaRefactor.prompt.md` Phase 7.6 |
| Online metrics / FSDP | Out of scope |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'diffusion'`**  
Install the SANA library first (see Prerequisites §1).

**OOM on forward pass**  
Reduce `--batch_size` or let the nibble machinery handle it — it halves the slice size automatically on OOM.

**OOM on backward pass**  
Increase `--grad_accum` to accumulate more nibbles before stepping, which amortises memory across steps.

**Loss is NaN from the first step**  
Check that your checkpoint path is correct and that `vae.scale_factor` matches the DC-AE variant you are using.

