# Refactor SANA Training to Use `diffusers` Instead of the SANA Repo Clone

## Goal

Replace all imports from the cloned SANA codebase (`diffusion.*`, `pyrallis`,
`SanaConfig`) with standard [🤗 diffusers](https://github.com/huggingface/diffusers)
APIs. The ED2 training infrastructure (`run_accumulation_loop`, `DataLoaderMultiAspect`,
`EveryDreamOptimizer`) is unchanged.

---

## Background — why now?

The previous implementation required the user to:

1. Clone the NVIDIA SANA repo (`github.com/NVlabs/Sana`) and `pip install -e` it.
2. Download a `.pth` checkpoint plus a DC-AE config YAML.
3. Write a `pyrallis` YAML config (`SanaConfig`) to drive training.

`diffusers` 0.33+ ships `SanaPipeline`, `SanaTransformer2DModel`, `AutoencoderDC`
and `FlowMatchEulerDiscreteScheduler`. Models are available on the HuggingFace Hub
(e.g. `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`). This makes the manual
SANA repo clone entirely redundant.

---

## Files changed

| File | Action |
|---|---|
| `model/sana_training_model.py` | Rewrite — load via `SanaPipeline.from_pretrained` |
| `model/sana_text_encoder.py` | Rewrite — use diffusers Gemma encoding logic |
| `core/loss_sana.py` | Rewrite — direct flow-matching loss using diffusers scheduler |
| `train_sana.py` | Rewrite — replace `--sana_config`/`pyrallis` with `--model_id` |
| `README_sana.md` | Update docs; remove SANA-repo-clone prerequisites |
| `requirements.txt` | Remove `pyrallis`, `termcolor`; add `diffusers>=0.33.0` note |

---

## Phase 1 — `model/sana_training_model.py`

### What changes

| Old | New |
|---|---|
| `from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae` | `from diffusers import SanaPipeline` |
| `from diffusion import Scheduler` | `from diffusers import FlowMatchEulerDiscreteScheduler` |
| `from diffusion.utils.checkpoint import load_checkpoint` | `safetensors.torch.load_model` |
| `SanaConfig` (pyrallis) fields throughout | `transformer.config`, `vae.config`, `scheduler.config` |

### New `SanaTrainingModel` dataclass

```python
@dataclass
class SanaTrainingModel:
    transformer: SanaTransformer2DModel   # sole trained component
    text_encoder: nn.Module               # Gemma — frozen
    tokenizer: Any                        # GemmaTokenizerFast — frozen
    vae: AutoencoderDC                    # DC-AE — frozen
    scheduler: FlowMatchEulerDiscreteScheduler
    model_id: str                         # HF hub ID, recorded for save/resume
    max_sequence_length: int = 300        # Gemma token budget
    complex_human_instruction: list       # optional system prompt prefix
    transformer_ema: Optional[nn.Module]  # reserved, not wired
```

### `load_sana_model(args)`

```python
pipe = SanaPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
# freeze text encoder and VAE
for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
for p in pipe.vae.parameters():          p.requires_grad_(False)
# optionally load a fine-tuned transformer checkpoint
if args.resume_from:
    load_model(pipe.transformer, args.resume_from)
return SanaTrainingModel(transformer=pipe.transformer, ...)
```

### `save_sana_model(path, model, global_step)`

Save only the transformer (unchanged component), using `safetensors`. The `model_id`
string is saved as `model_id.txt` so a user can reconstruct the full pipeline later.

---

## Phase 2 — `model/sana_text_encoder.py`

### What changes

The old code handled T5, Gemma, and Qwen branches via
`config.text_encoder.text_encoder_name`. Diffusers' `SanaPipeline` only supports
Gemma (via `_get_gemma_prompt_embeds`). The branch dispatch logic is removed.

### New API

```python
def encode_prompts(
    tokenizer,
    text_encoder: nn.Module,
    captions: list[str],
    device: torch.device,
    *,
    max_sequence_length: int = 300,
    complex_human_instruction: list[str] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        prompt_embeds        : (B, N, C)  — ready for SanaTransformer2DModel.forward
        prompt_attention_mask: (B, N)
    """
```

### Shape difference from old code

| | old (SANA repo) | new (diffusers) |
|---|---|---|
| `y` shape | `(B, 1, N, C)` | `(B, N, C)` |
| `y_mask` shape | `(B, 1, 1, N)` | `(B, N)` |

The diffusers `SanaTransformer2DModel.forward` accepts the simpler shapes directly.
The transformer's `caption_projection` linear layer handles the channel mapping.

### Null-text caching

`encode_null_prompt(...)` replaces `encode_sana_null_text(...)`. Same on-disk caching
logic, same function contract.

---

## Phase 3 — `core/loss_sana.py`

### What changes

Old code called `model.train_diffusion.training_losses(model.transformer, ...)` from
the SANA repo's `Scheduler`. That is replaced with a self-contained flow-matching
training step using the diffusers scheduler's API.

### Flow-matching training loss (theory)

Flow matching defines a forward (noising) process:

```
z_t = (1 - σ) · z₀  +  σ · ε     where ε ~ N(0,I),  σ ∈ [0,1]
```

The model is trained to predict the **velocity** `v = ε − z₀` (the derivative of `z_t`
with respect to `σ`):

```
loss = MSE(transformer(z_t, σ·T), ε − z₀)
```

where `T = num_train_timesteps`.

### Timestep sampling

| Scheme | Formula |
|---|---|
| `uniform` | `σ ~ U(0, 1)` |
| `logit_normal` | `σ = sigmoid(N(logit_mean, logit_std²))` |

### New API

```python
def sample_flow_sigmas(batch_size, weighting_scheme, logit_mean, logit_std, device)
    -> tuple[torch.Tensor, torch.Tensor]:   # sigma (B,), timestep_t (B,)

def compute_sana_loss(transformer, z, y, y_mask, sigma, timestep_t) -> torch.Tensor
```

`timestep_t = sigma * num_train_timesteps` — a float tensor passed directly to
`SanaTransformer2DModel.forward(timestep=...)`.

---

## Phase 4 — `train_sana.py`

### Argument changes

| Old | New |
|---|---|
| `--sana_config <path>` (required, pyrallis YAML) | `--model_id <hf-hub-id>` (required) |
| `--grad_accum` (sana_config field) | removed (use `--optimizer_batch_size`) |
| `--mixed_precision` | kept (bf16/fp16/no, passed to `torch.autocast`) |
| (implicit from YAML) `train_sampling_steps`, `weighting_scheme`, etc. | `--num_train_timesteps`, `--weighting_scheme`, `--logit_mean`, `--logit_std` |

Full new arg list:

```
--model_id                   HF hub ID (e.g. Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers)
--resume_from                path to a saved transformer .safetensors
--project_name
--logdir
--data_root
--batch_size
--optimizer_batch_size
--resolution                 (nargs=+, default=[1024])
--resolution_multiplier
--forward_slice_size
--max_backward_slice_size
--skip_undersized_images
--lr
--max_steps
--mixed_precision            (bf16 / fp16 / no)
--seed
--save_every
--sample_every
--num_workers
--amp
--max_epochs
--max_sequence_length        (default 300, Gemma token budget)
--num_train_timesteps        (default 1000)
--weighting_scheme           (uniform | logit_normal, default logit_normal)
--logit_mean                 (default 0.0)
--logit_std                  (default 1.0)
--guidance_scale             (default 4.5, for sample generation only)
```

### VAE image range

The diffusers `AutoencoderDC` expects `[-1, 1]` normalised pixel values (standard
diffusers convention — same as the standard SD/SDXL pipeline in ED2). Remove the
`image_output_range="[0,255]"` parameter from the `EveryDreamBatch` call; use the
default `[-1, 1]` range instead.

### VAE encoding

```python
# images: (B, C, H, W)  float, range [-1, 1]
latents = vae.encode(images).latent * vae.config.scaling_factor
```

`AutoencoderDC.encode` returns `EncoderOutput(latent=...)` — note `.latent` (not
`.latent_dist` — DC-AE is deterministic, not variational).

### Sample generation

Replace the manual `FlowEuler`/`DPMS` sampler loop with a `SanaPipeline` constructed
from the live model components:

```python
with inference_guard(model.transformer):
    pipe = SanaPipeline(
        transformer=model.transformer,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        vae=model.vae,
        scheduler=copy.deepcopy(model.scheduler),
    )
    images = pipe(prompt=prompt, height=h, width=w, guidance_scale=args.guidance_scale)[0]
```

Using a deepcopy of the scheduler avoids mutating `model.scheduler.timesteps` in-place.

---

## Phase 5 — `README_sana.md`

1. Replace the "Install SANA library" prerequisite with just `pip install diffusers>=0.33`.
2. Replace the `pyrallis` YAML config section with `--model_id`.
3. Update the CLI flag table.
4. Remove the `pyrallis` and `termcolor` mentions.
5. Update the Architecture notes section.

---

## Phase 6 — `requirements.txt`

- Remove `pyrallis`
- Remove `termcolor`
- Add comment noting `diffusers>=0.33.0` required (already installed transitively by
  ED2's existing diffusers dependency).

---

## What is NOT changed

- `core/step.py` — `run_accumulation_loop`, `train_step`, etc.
- `data/` — `DataLoaderMultiAspect`, `EveryDreamBatch`, `resolver`, etc.
- `optimizer/` — `EveryDreamOptimizer`
- `utils/inference_context.py`
- All existing SD/SDXL training paths

---

## Success criteria

1. `python -c "from model.sana_training_model import load_sana_model"` runs without
   importing anything from `diffusion.*` or `pyrallis`.
2. `pytest test/` passes (no regressions in existing SD/SDXL tests).
3. `train_sana.py --help` lists all new flags.
4. `pyrallis` and `termcolor` are no longer imported anywhere in the SANA modules.

