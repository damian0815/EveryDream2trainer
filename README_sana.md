# SANA Training with EveryDream2trainer

Fine-tune a [SANA](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e)
model using the EveryDream2trainer infrastructure and standard 🤗 diffusers.
No SANA repo clone required.

---

## Overview

`train_sana.py` is a SANA-specific training entry point that reuses the battle-tested
ED2 machinery:

| ED2 component | Role in SANA training |
|---|---|
| `core/step.py` → `run_accumulation_loop()` | Nibble slicing, OOM-robust accumulation, backward pass, optimizer step |
| `data/every_dream.py` → `EveryDreamBatch` | Multi-aspect image loading, caption handling |
| `optimizer/optimizers.py` → `EveryDreamOptimizer` | Optimizer wrapping (AdamW, CAME, Prodigy, …) |
| `utils/inference_context.py` | Eval/train guard for sample generation |

SANA-specific modules:

| Module | Responsibility |
|---|---|
| `model/sana_training_model.py` | `SanaTrainingModel` dataclass, `load_sana_model()`, `save_sana_model()` |
| `model/sana_text_encoder.py` | `encode_prompts()` via Gemma (diffusers) |
| `core/loss_sana.py` | `sample_flow_sigmas()`, `compute_sana_loss()` — self-contained flow-matching |
| `train_sana.py` | Entry point: argument parsing, data wiring, training loop, sample generation |

---

## Prerequisites

### 1. Install EveryDream2trainer dependencies

```bash
pip install -r requirements.txt
```

`diffusers>=0.33` is already listed.  No additional packages are required for SANA.

### 2. Accept the Gemma 2 licence

SANA uses Gemma 2 as its text encoder.  You must accept the licence on HuggingFace
before the weights can be downloaded:

1. Visit <https://huggingface.co/google/gemma-2-2b-it> and click **Agree**.
2. Log in locally: `huggingface-cli login`

---

## Data preparation

SANA training uses the standard ED2 data layout — the same folder structure you
already use for SD/SDXL fine-tuning:

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

Captions can also be stored as sidecar `.caption` files. All existing ED2 tagging and
caption-shuffle features are supported.

> **Note:** Images are fed to the diffusers `AutoencoderDC` VAE in the standard
> `[-1, 1]` range — no special pre-processing is needed.  
> `EveryDreamBatch` handles this automatically with its default `image_output_range`.

---

## Training

```bash
python train_sana.py \
    --model_id Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --data_root /path/to/my_dataset \
    --project_name my-sana-finetune \
    --logdir logs \
    --batch_size 2 \
    --max_steps 10000 \
    --save_every 1000 \
    --sample_every 500
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model_id` | *(required)* | HuggingFace hub model ID |
| `--data_root` | *(required)* | Training data directory |
| `--resume_from` | — | Path to a saved `transformer_gsNNNN.safetensors` checkpoint |
| `--project_name` | `sana-finetune` | Run name (used as log subdirectory) |
| `--logdir` | `logs` | Root directory for TensorBoard logs and checkpoints |
| `--batch_size` | `2` | Data loading batch size — also the aspect-ratio bucket size |
| `--optimizer_batch_size` | = batch\_size | Effective optimizer batch size (samples between optimizer steps) |
| `--resolution` | `[1024]` | One or more training resolutions — see [Multi-resolution training](#multi-resolution-training) |
| `--resolution_multiplier` | — | Per-resolution dataset weight — see [Multi-resolution training](#multi-resolution-training) |
| `--forward_slice_size` | = batch\_size | Max images per forward-pass slice — see [Memory management](#memory-management) |
| `--max_backward_slice_size` | = batch\_size | Max images accumulated before backward — see [Memory management](#memory-management) |
| `--skip_undersized_images` | false | Drop images smaller than the target resolution bucket |
| `--lr` | `2e-5` | Learning rate |
| `--max_steps` | `10000` | Total number of optimizer steps |
| `--mixed_precision` | `bf16` | `bf16` / `fp16` / `no` |
| `--seed` | `42` | Random seed |
| `--save_every` | `1000` | Save transformer weights every N optimizer steps |
| `--sample_every` | `500` | Generate validation samples every N steps |
| `--guidance_scale` | `4.5` | CFG scale for sample generation |
| `--sample_height` | = first resolution | Height of generated samples |
| `--sample_width` | = first resolution | Width of generated samples |
| `--max_sequence_length` | `300` | Gemma token budget |
| `--num_train_timesteps` | `1000` | Flow-matching T |
| `--weighting_scheme` | `logit_normal` | `uniform` or `logit_normal` |
| `--logit_mean` | `0.0` | Logit-normal mean (ignored for uniform) |
| `--logit_std` | `1.0` | Logit-normal std (ignored for uniform) |
| `--num_workers` | `4` | DataLoader worker count |
| `--amp` | true | Enable AMP for the backward pass |
| `--max_epochs` | `100` | Maximum number of data epochs |

---

## Multi-resolution training

Pass multiple values to `--resolution` to train across several resolutions
simultaneously.  The data pipeline scans your dataset **once per resolution**, creating
aspect-ratio bucketed `ImageTrainItem` objects at each resolution, then merges them
into a single `DataLoaderMultiAspect`.

```bash
python train_sana.py \
    --model_id Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --data_root /path/to/my_dataset \
    --resolution 512 1024 \
    --batch_size 4 \
    --max_steps 10000
```

### `--resolution_multiplier`

Control how much each resolution contributes to the epoch. Pass one multiplier per
`--resolution` entry:

```bash
# Train twice as much on 1024 px images as on 512 px images
python train_sana.py \
    --resolution 512 1024 \
    --resolution_multiplier 1.0 2.0 \
    ...
```

Omitting `--resolution_multiplier` treats all resolutions equally (multiplier = 1.0
each).

---

## Memory management

### `--forward_slice_size`

Splits the forward pass into smaller chunks to reduce peak VRAM usage. The **loss
batch size is unaffected** — slices are accumulated before the backward pass. The
trainer halves the slice size automatically when an OOM is detected mid-forward.

```bash
# Only 1 image fits in VRAM for the forward pass at 1024 px
python train_sana.py \
    --resolution 1024 \
    --forward_slice_size 1 \
    --batch_size 4 \
    ...
```

Pass one value per resolution for mixed setups:

```bash
# 4 images at 512 px, 1 image at 1024 px
python train_sana.py \
    --resolution 512 1024 \
    --forward_slice_size 4 1 \
    ...
```

### `--max_backward_slice_size`

Caps how many loss images accumulate in the graph before a backward pass fires.
Lowering this reduces peak gradient memory independently of `--batch_size`.

### Typical VRAM setups

| GPU VRAM | Suggested flags |
|---|---|
| 24 GB (RTX 4090) | `--resolution 1024 --batch_size 4` |
| 16 GB (RTX 4080) | `--resolution 1024 --batch_size 2 --forward_slice_size 1` |
| 12 GB (RTX 4080S) | `--resolution 1024 --batch_size 2 --forward_slice_size 1 --max_backward_slice_size 1` |
| 8 GB | `--resolution 512 --batch_size 2 --forward_slice_size 1` |

---

## Batch size vs. optimizer batch size

EveryDream2trainer makes a deliberate distinction between two batch-size concepts,
and SANA inherits it in full.

### `--batch_size` — data loading / aspect-ratio bucketing

`--batch_size` controls how many images are loaded and processed together in a single
data batch. `DataLoaderMultiAspect` fills each batch exclusively with images that
share the same aspect-ratio bucket.

### `--optimizer_batch_size` — effective training batch size

`--optimizer_batch_size` is the number of images that must be processed before
`optimizer.step()` is called. When it is larger than `--batch_size`, gradients are
accumulated across multiple data batches before stepping.

The accumulation is **not** a naïve `grad_accum` multiplier. The nibble machinery in
`run_accumulation_loop()` counts actual images, so the optimizer step fires after the
correct number of samples regardless of how the memory slicing divides the data.

```
batch_size=4, optimizer_batch_size=16:

 data batch 1 → [img_A 1024×1024  ×4]  → forward → backward  (4 images)
 data batch 2 → [img_B  768×1024  ×4]  → forward → backward  (4 images)
 data batch 3 → [img_C 1024×1024  ×4]  → forward → backward  (4 images)
 data batch 4 → [img_D  512× 512  ×4]  → forward → backward  (4 images)
                                                               ── 16 images ──▶ optimizer.step()
```

---

## Sample prompts

Place a `sample_prompts.txt` file in the repo root (one prompt per line). The trainer
generates images at every `--sample_every` step and writes them to:

- `{logdir}/{project_name}/samples-sana/gs{N}/` — WebP tiles on disk
- TensorBoard under the `samples-sana/` tag

---

## Outputs

```
logs/my-sana-finetune/
    events.out.tfevents.*       # TensorBoard
    gs1000/
        transformer_gs1000.safetensors
        model_id.txt            # records the HF hub ID for later reconstruction
    gs2000/
        ...
    final/
        transformer_gsFINAL.safetensors
        model_id.txt
    samples-sana/
        gs0/
            000_a photo of a cat.webp
        gs500/
            000_a photo of a cat.webp
```

Only the **transformer** is saved — text encoder and VAE are frozen and unchanged.

### Reconstructing a full pipeline from a checkpoint

```python
from diffusers import SanaPipeline
from safetensors.torch import load_model

model_id = open("logs/my-sana-finetune/gs1000/model_id.txt").read().strip()
pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
load_model(pipe.transformer, "logs/my-sana-finetune/gs1000/transformer_gs1000.safetensors")
pipe.to("cuda")

image = pipe("a beautiful landscape", height=1024, width=1024).images[0]
```

---

## Resuming a run

```bash
python train_sana.py \
    --model_id Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --data_root /path/to/my_dataset \
    --resume_from logs/my-sana-finetune/gs5000/transformer_gs5000.safetensors \
    ...
```

---

## Optimizer configuration

`train_sana.py` uses ED2's `EveryDreamOptimizer`, so any optimizer supported by ED2
is available. For advanced schedules, copy one of the `optimizer-*.json` files from
the repo root and refer to `doc/ADVANCED_TWEAKING.md`.

---

## Architecture notes

### Why `run_accumulation_loop()`?

`train_step()` in `core/step.py` was refactored into two layers:

1. **`run_accumulation_loop()`** — model-agnostic. Handles nibble slicing, OOM retry
   on the backward pass, gradient accumulation, and the optimizer step.
2. **`nibble_loss_fn`** — a closure passed into the loop. Contains all model-specific
   forward-pass logic.

`train_sana_step()` builds a SANA-specific `nibble_loss_fn` (which calls
`compute_sana_loss()`) and passes it to `run_accumulation_loop()`. SANA gets the
same OOM-robustness as SD/SDXL for free.

### Flow-matching loss

SANA uses rectified flow matching. The training loss in `core/loss_sana.py`:

```
z_t = (1 - σ) · z₀  +  σ · ε     # noise the latent
v   = ε − z₀                       # velocity target
loss = MSE(transformer(z_t, σ·T), v)
```

`σ` is sampled from a logit-normal distribution by default (`--weighting_scheme
logit_normal`), which up-weights mid-range noise levels.

### Text encoder

The Gemma 2 text encoder (loaded from the HF hub as part of `SanaPipeline`) returns
embeddings of shape `(B, N, C)` with an attention mask of shape `(B, N)`.  These are
passed directly to `SanaTransformer2DModel.forward(encoder_hidden_states=...,
encoder_attention_mask=...)`.

### VAE

The `AutoencoderDC` (DC-AE) encoder is deterministic — it returns a plain latent
tensor, not a distribution. Access it with `.latent` (not `.latent_dist.sample()`):

```python
latents = vae.encode(images).latent * vae.config.scaling_factor
```

Images should be in `[-1, 1]` — the standard diffusers convention.

### What is NOT yet implemented

| Feature | Status |
|---|---|
| EMA (`transformer_ema` field) | Reserved slot, not wired up |
| Pre-computed VAE / text-encoder feature caching | Deferred — add to `data/latent_cache.py` |
| Multi-GPU via `accelerate` | Out of scope for this PR |

---

## Troubleshooting

**`GatedRepoError` / `401 Unauthorized`**  
You need to accept the Gemma 2 licence and run `huggingface-cli login`.

**OOM on forward pass**  
Reduce `--batch_size` or set `--forward_slice_size 1`. The nibble machinery halves
the slice size automatically on OOM.

**OOM on backward pass**  
Lower `--max_backward_slice_size` to fire backward more frequently.

**Loss is NaN from the first step**  
Check that `--mixed_precision bf16` matches the model dtype. Try `--mixed_precision no`
to rule out dtype issues.
