"""
SanaTrainingModel: dataclass holding all SANA model components plus factory and save functions.
Uses 🤗 diffusers (SanaPipeline, SanaTransformer2DModel, AutoencoderDC,
TrainFlowMatchEulerDiscreteScheduler) — no SANA repo clone required.
"""
from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Literal

import safetensors
from diffusers import SanaPipeline, SanaTransformer2DModel

import torch
import torch.nn as nn

from optimizer.optimizers import EveryDreamOptimizer


def _is_dispatched(module: nn.Module) -> bool:
    """Check if *module* is under accelerate dispatch (device_map or BitsAndBytes).

    When a module is dispatched, calling .to() raises:
        "You shouldn't move a model that is dispatched using accelerate hooks"
    """
    return hasattr(module, '_hf_hook') or getattr(module, 'hf_quantizer', None) is not None


@dataclass
class SanaTrainingModel:
    """Holds all SANA model components for training."""

    transformer: SanaTransformer2DModel      # SanaTransformer2DModel — sole trained component
    text_encoder: nn.Module                  # Gemma2 — frozen, not trained
    tokenizer: Any                           # GemmaTokenizerFast — frozen
    vae: nn.Module                           # AutoencoderDC — frozen, not trained
    noise_scheduler: Any                     # TrainFlowMatchEulerDiscreteScheduler
    model_id: str                            # HF hub ID, recorded for save/resume

    max_sequence_length: int = 300           # Gemma token budget
    complex_human_instruction: list = field(default_factory=list)  # optional system-prompt prefix

    transformer_ema: Optional[nn.Module] = None  # main EMA shadow weights
    ema_working_dir: Optional[str] = None         # set by _init_sana_ema when ema_device='disk'
    teacher_transformer: Optional[nn.Module] = None  # frozen snapshot teacher for distillation
    self_flow_proj_head: Optional[nn.Module] = None

    is_video: bool = False  # set to True for SanaVideoPipeline training

    # ---- EveryDreamOptimizer duck-type adapter properties ----------------

    @property
    def unet(self) -> nn.Module:
        """Alias for transformer — satisfies EveryDreamOptimizer's model.unet access."""
        return self.transformer

    @property
    def text_encoder_2(self):
        """SANA has only one text encoder. Returns None to satisfy EveryDreamOptimizer."""
        return None

    @property
    def is_sdxl(self) -> bool:
        """SANA is not SDXL. Returns False to satisfy EveryDreamOptimizer."""
        return False

    # ---- Core properties -------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def is_flow_matching(self) -> bool:
        """Signals to SampleGenerator that this model uses a flow-matching scheduler."""
        return True

    def build_inference_pipeline(self, scheduler=None):
        """
        Returns a SanaPipeline (or SanaVideoPipeline) built from the live model
        components.  Called by SampleGenerator.create_inference_pipe().

        scheduler: if None, constructs an SDPipelineInferenceFlowMatchEulerDiscreteScheduler
                   from the training scheduler's config. SampleGenerator always passes an
                   already-constructed inference scheduler here.
        """
        from diffusers import SanaPipeline
        from diffusers import SanaVideoPipeline
        from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler

        inf_scheduler = scheduler or SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
            self.noise_scheduler.config
        )
        if self.is_video:
            return SanaVideoPipeline(
                transformer=self.transformer,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                vae=self.vae,
                scheduler=inf_scheduler,
            )
        return SanaPipeline(
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            scheduler=inf_scheduler,
        )

    def build_ema_inference_pipeline(self, scheduler=None):
        """Like build_inference_pipeline but uses the EMA transformer weights.

        Returns None when no EMA weights are available.
        """
        if self.transformer_ema is None and self.ema_working_dir is None:
            return None
        from diffusers import SanaPipeline, SanaVideoPipeline
        from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler

        transformer = self.transformer_ema if self.transformer_ema is not None else self.transformer
        inf_scheduler = scheduler or SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
            self.noise_scheduler.config
        )
        cls = SanaVideoPipeline if self.is_video else SanaPipeline
        return cls(
            transformer=transformer,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            scheduler=inf_scheduler,
        )

    @contextmanager
    def ema_inplace_swap(self):
        """Context manager for disk-offload EMA inference.

        Saves live transformer weights to a temporary backup, loads the EMA
        weights from disk into the live module in-place, yields for inference,
        then restores the live weights and cleans up the backup.
        """
        if self.ema_working_dir is None:
            raise RuntimeError("ema_inplace_swap requires disk-offload EMA (ema_device='disk')")

        import safetensors.torch as _st

        backup_path = os.path.join(self.ema_working_dir, "transformer_live_backup.safetensors")
        ema_path = os.path.join(self.ema_working_dir, "transformer_ema.safetensors")
        restored = False

        try:
            if os.path.isfile(ema_path):
                # Step 1: persist live weights
                logging.info(f"ema_inplace_swap: backing up live transformer → {backup_path}")
                live_sd = {
                    k: v.detach().cpu().contiguous()
                    for k, v in self.transformer.state_dict().items()
                }
                _st.save_file(live_sd, backup_path)
                del live_sd

                # Step 2: load EMA weights in-place
                target_dtype = next(self.transformer.parameters()).dtype
                target_device = next(self.transformer.parameters()).device
                logging.info(f"ema_inplace_swap: applying EMA transformer ({target_dtype} on {target_device})")
                ema_sd = _st.load_file(ema_path, device="cpu")
                named_params = dict(self.transformer.named_parameters())
                with torch.no_grad():
                    for k, ema_v in ema_sd.items():
                        if k in named_params:
                            named_params[k].data.copy_(
                                ema_v.to(dtype=target_dtype, device=target_device)
                            )
                del ema_sd, named_params

            yield  # inference runs here

        finally:
            # Restore live weights from backup
            if os.path.isfile(backup_path):
                target_dtype = next(self.transformer.parameters()).dtype
                target_device = next(self.transformer.parameters()).device
                logging.info(f"ema_inplace_swap: restoring live transformer from {backup_path}")
                live_sd = _st.load_file(backup_path, device="cpu")
                named_params = dict(self.transformer.named_parameters())
                with torch.no_grad():
                    for k, p in named_params.items():
                        if k in live_sd:
                            p.data.copy_(live_sd[k].to(dtype=target_dtype, device=target_device))
                del live_sd, named_params
                os.unlink(backup_path)

    # ---- Device offload helpers -------------------------------------------

    def load_vae_to_device(self, device):
        """Move VAE to *device* (e.g. 'cuda' or 'cpu'). No-op if dispatched by accelerate."""
        if not _is_dispatched(self.vae):
            self.vae.to(device)

    def load_textenc_to_device(self, device):
        """Move text encoder to *device* (e.g. 'cuda' or 'cpu'). No-op if dispatched by accelerate."""
        if not _is_dispatched(self.text_encoder):
            self.text_encoder.to(device)


def load_sana_model(args: Namespace) -> SanaTrainingModel:
    """
    Loads all SANA components via SanaPipeline.from_pretrained (or SanaVideoPipeline
    when args.is_video is set) and wraps them in a SanaTrainingModel.
    Freezes the text encoder and VAE (requires_grad = False).
    Converts the pipeline's stock FlowMatchEulerDiscreteScheduler to a
    TrainFlowMatchEulerDiscreteScheduler so the training loop can use the same
    noising/timestep utilities as SD2/SDXL flow-matching training.

    The transformer is kept in float32 for numerical stability (SANA's linear
    attention is prone to NaN gradients in bf16).  The frozen text encoder and
    VAE are cast to bfloat16 to save VRAM (except for video mode where the VAE
    must stay in float32 to avoid NaN errors with 3D latents).
    """
    from diffusers import SanaPipeline
    from diffusers import SanaVideoPipeline
    from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler

    is_video = getattr(args, 'is_video', False)
    #dtype_vae = torch.float32 if is_video else torch.bfloat16

    pipe = _load_sana_pipeline(args.model_id, dtype=torch.bfloat16, te_quantization=args.te_quantization, is_video=is_video)

    if args.te_quantization == 'none':
        pipe.text_encoder.to(dtype=torch.bfloat16)
    #if not _is_dispatched(pipe.vae):
    #    pipe.vae.to(dtype=dtype_vae)

    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # Replace the inference scheduler with the training-aware subclass.
    noise_scheduler = TrainFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    model = SanaTrainingModel(
        transformer=pipe.transformer,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        vae=pipe.vae,
        noise_scheduler=noise_scheduler,
        model_id=args.model_id,
        max_sequence_length=getattr(args, "max_sequence_length", 300),
        complex_human_instruction=getattr(args, "complex_human_instruction", []) or [],
        is_video=is_video,
    )
    if is_video:
        try:
            import imageio_ffmpeg
        except ImportError:
            raise ImportError(
                "SanaVideoPipeline requires imageio-ffmpeg, or you'll get green frames when generating. Install it with: pip install imageio-ffmpeg"
            )

    if args.anchor_reg_alpha > 0:
        from core.anchor_reg import capture_base_params
        device_capture = 'cpu' if args.anchor_reg_cpu_offload else pipe.device
        dtype_capture = torch.float32 if args.anchor_reg_cpu_offload else pipe.dtype
        model.anchor_base_params = capture_base_params(pipe.transformer, device=device_capture, dtype=dtype_capture)
        logging.info(f"Captured anchor base params ({len(model.anchor_base_params)} tensors, "
                     f"device={device_capture}, dtype={dtype_capture})")

    # Teacher transformer (frozen snapshot for distillation)
    import copy
    if args.teacher is not None and len(args.teacher) > 0:
        model.teacher_transformer = copy.deepcopy(pipe.transformer)
        print(" * loading teacher from", args.teacher[0])
        _load_transformer_checkpoint(model.teacher_transformer, args.teacher[0])

    if model.teacher_transformer is not None:
        model.teacher_transformer.requires_grad_(False)
        model.teacher_transformer.eval()
        model.teacher_transformer.to(device=pipe.device, dtype=pipe.dtype)

    if args.resume_from is not None:
        logging.info(f" * Resuming from {args.resume_from}")
        _load_transformer_checkpoint(model.transformer, args.resume_from)

    if args.lora_resume is not None:
        logging.info(f" * Loading LoRA adapter from {args.lora_resume}")
        pipe.load_lora_weights(args.lora_resume)

    if args.self_flow_p > 0:
        logging.info(f" * Initializing Self-Flow components (p={args.self_flow_p})")
        _inject_self_flow(model=model, pipe=pipe)
        if args.resume_from is not None:
            _try_load_self_flow_state(model, os.path.dirname(args.resume_from))

    # ── Main EMA resume hint ──────────────────────────────────────────────
    from model.training_model import get_use_ema_decay_training
    if get_use_ema_decay_training(args) and args.resume_from is not None:
        ema_sidecar = os.path.join(os.path.dirname(args.resume_from), "transformer_ema.safetensors")
        if os.path.isfile(ema_sidecar):
            logging.info(f" * Found EMA sidecar at {ema_sidecar} — will load in _init_sana_ema")
        else:
            logging.info(" * No EMA sidecar found — initialising main EMA from current weights")

    return model


def _load_sana_pipeline(repo_id, dtype, te_quantization: Literal['none', 'int8', 'int4']= 'none', is_video: bool = False) -> SanaPipeline:
    """
    Load a SanaPipeline (or SanaVideoPipeline when is_video=True) with optional
    bitsandbytes quantization.

    SANA dtype requirements:
      - The full pipeline is loaded in float32 so the VAE stays stable.
      - text_encoder and transformer are then cast to bfloat16.
    With quantization (int8 / int4):
      - Both text_encoder and transformer are loaded with BitsAndBytesConfig.
      - device_map="balanced" handles placement automatically (no manual .to()).
    """
    import torch
    from diffusers import SanaPipeline, SanaTransformer2DModel
    from diffusers import SanaVideoPipeline, SanaVideoTransformer3DModel

    pipeline_cls = SanaVideoPipeline if is_video else SanaPipeline
    transformer_cls = SanaVideoTransformer3DModel if is_video else SanaTransformer2DModel

    if te_quantization in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig as TransformersBnBConfig, AutoModel
            from diffusers import BitsAndBytesConfig as DiffusersBnBConfig, SanaTransformer2DModel
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes is required for quantized SANA loading. "
                "Install it with: pip install bitsandbytes"
            ) from exc

        load_in_8bit = te_quantization == "int8"
        load_in_4bit = te_quantization == "int4"

        bnb_kwargs_te = dict(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        bnb_kwargs_tr = dict(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

        te_quant_cfg = TransformersBnBConfig(**bnb_kwargs_te)
        text_encoder = AutoModel.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            quantization_config=te_quant_cfg,
            torch_dtype=dtype,
        )

        #tr_quant_cfg = DiffusersBnBConfig(**bnb_kwargs_tr)
        transformer = transformer_cls.from_pretrained(
            repo_id,
            subfolder="transformer",
            #quantization_config=tr_quant_cfg,
            torch_dtype=torch.bfloat16,
        )

        pipeline = pipeline_cls.from_pretrained(
            repo_id,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    else:
        # Standard (non-quantized) SANA loading
        pipeline = pipeline_cls.from_pretrained(repo_id, torch_dtype=dtype)

    return pipeline


def _inject_self_flow(model: SanaTrainingModel, pipe: SanaPipeline):
    import types
    from core.self_flow import SelfFlowMLPProjectionHead

    # Teacher IS the main EMA — no separate deepcopy needed.

    # Initialize Projection Head (SANA 1.6B hidden size is 2240)
    embed_dim = getattr(pipe.transformer.config, 'hidden_size', 2240)
    model.self_flow_proj_head = SelfFlowMLPProjectionHead(
        in_channels=embed_dim,
        hidden_channels=embed_dim,
        out_channels=embed_dim
    ).to(device=pipe.device, dtype=pipe.dtype)

    # 3. Monkey-Patch SANA for Dual-Timestep (B, N) support
    def patch_sana_for_self_flow(transformer):
        # ---------------------------------------------------------
        # Patch 1: The Time Embedder
        # ---------------------------------------------------------
        old_te_forward = transformer.time_embed.forward

        def new_te_forward(self, timestep, *args, **kwargs):
            if timestep.ndim == 2:
                B, N = timestep.shape
                t_flat = timestep.reshape(-1)

                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = B * N

                out_t, emb_t = old_te_forward(t_flat, *args, **kwargs)

                if out_t is not None:
                    out_t = out_t.reshape(B, N, *out_t.shape[1:])
                    # Wrap out_t in the magic subclass to protect the transformer blocks
                    out_t = out_t.as_subclass(BypassSanaBlockTensor)

                if emb_t is not None:
                    # Keep as standard tensor for norm_out
                    emb_t = emb_t.reshape(B, N, *emb_t.shape[1:])

                return out_t, emb_t

            return old_te_forward(timestep, *args, **kwargs)

        transformer.time_embed.forward = types.MethodType(new_te_forward, transformer.time_embed)

        # ---------------------------------------------------------
        # Patch 2: The Final norm_out block (No Flattening!)
        # ---------------------------------------------------------
        if hasattr(transformer, 'norm_out') and transformer.norm_out is not None:
            old_norm_out_fwd = transformer.norm_out.forward

            def new_norm_out_fwd(self, hidden_states, temb, scale_shift_table, *args, **kwargs):
                if temb.ndim == 3:  # (B, N, D)
                    # scale_shift_table is shape (2, D).
                    # We pull out the shift/scale biases and add them to temb safely.
                    # PyTorch will perfectly broadcast (D) + (B, N, D) -> (B, N, D)
                    shift = scale_shift_table[0] + temb.to(scale_shift_table.device)
                    scale = scale_shift_table[1] + temb.to(scale_shift_table.device)

                    # Apply the final layer norm
                    hidden_states = self.norm(hidden_states)

                    # Element-wise modulation (No crazy broadcasting OOM!)
                    hidden_states = hidden_states * (1 + scale) + shift
                    return hidden_states

                return old_norm_out_fwd(hidden_states, temb, scale_shift_table, *args, **kwargs)

            transformer.norm_out.forward = types.MethodType(new_norm_out_fwd, transformer.norm_out)

    # Apply patch
    patch_sana_for_self_flow(pipe.transformer)

# Create a magical Tensor subclass to bypass hardcoded unsqueezes
class BypassSanaBlockTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}

        # A. Intercept reshape() for the block's inline modulation
        if func.__name__ in ('reshape', 'view'):
            shape_args = args[1:] if len(args) > 1 else (kwargs.get('shape') or kwargs.get('size', ()))
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list, torch.Size)):
                shape_args = shape_args[0]

            # Intercept SANA's hardcoded: timestep.reshape(batch_size, 6, -1)
            if len(shape_args) == 3 and shape_args[1] == 6 and shape_args[2] == -1:
                if args[0].ndim == 3:  # (B, N, 6*D)
                    B, N, _ = args[0].shape
                    new_shape = (B, N, 6, -1)
                    ret = super().__torch_function__(func, types, (args[0], *new_shape), kwargs)
                    return ret.as_subclass(cls)

        # B. Intercept chunk() for the block's inline scale_shift
        if func.__name__ in ('chunk', 'split'):
            dim = kwargs.get('dim', args[2] if len(args) > 2 else 0)
            chunks = kwargs.get('chunks', args[1] if len(args) > 1 else 1)

            # ONLY intercept SANA's specific chunk(6, dim=1) for the timestep modulation!
            # This ensures we don't break the GLU's chunk(2) inside the feed-forward layer.
            if chunks == 6 and dim == 1 and args[0].ndim == 4:
                if 'dim' in kwargs:
                    kwargs['dim'] = 2
                else:
                    l_args = list(args)
                    if len(l_args) > 2:
                        l_args[2] = 2
                    else:
                        l_args.append(2)
                    args = tuple(l_args)

                ret = super().__torch_function__(func, types, args, kwargs)

                # IMPORTANT: Strip the subclass! Once chunking is done, the block
                # operates on standard (B, N, D) tensors safely without interference.
                return tuple(r.squeeze(2).as_subclass(torch.Tensor) for r in ret)

        # C. Fallback for math operations
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, torch.Tensor) and func.__name__ not in ('size', 'shape', 'ndim', 'dim', '__repr__',
                                                                   '__str__', 'squeeze', 'chunk', 'split'):
            return ret.as_subclass(cls)

        return ret


def save_sana_model(to_folder: str, model: SanaTrainingModel, optimizer: EveryDreamOptimizer, global_step: int, num_samples: int, lora: bool = False) -> None:
    """
    Saves only the transformer (the trained component) as a safetensors file.
    Also writes model_id.txt so the full pipeline can be reconstructed later:

        pipe = SanaPipeline.from_pretrained(model_id)
        load_model(pipe.transformer, "transformer_gsNNNN.safetensors")

    When *lora* is True only the LoRA adapter weights are saved instead.
    """
    if lora:
        save_sana_lora(model=model, save_path=to_folder)
        return

    from safetensors.torch import save_file

    os.makedirs(to_folder, exist_ok=True)

    weights_path = os.path.join(to_folder, f"transformer_gs{global_step:05}_n{num_samples:05}.safetensors")
    logging.info(f" * Saving transformer checkpoint to {weights_path}")
    _sd = model.transformer.state_dict()
    save_file(_sd, weights_path)
    del _sd

    if optimizer is not None:
        logging.info(f" Saving optimizer state to {to_folder}")
        optimizer.save(to_folder)

    if model.self_flow_proj_head is not None:
        proj_head_path = os.path.join(to_folder, "self_flow_proj_head.pt")
        logging.info(f" * Saving Self-Flow projection head to {proj_head_path}")
        _sd = model.self_flow_proj_head.state_dict()
        torch.save(_sd, proj_head_path)
        del _sd

    # ── Main EMA sidecar (separate from self-flow EMA above) ──────────────
    if model.transformer_ema is not None:
        ema_path = os.path.join(to_folder, "transformer_ema.safetensors")
        logging.info(f" * Saving EMA transformer to {ema_path}")
        _sd = model.transformer_ema.state_dict()
        safetensors.torch.save_file(_sd, ema_path)
        del _sd

    if model.ema_working_dir is not None:
        _src = os.path.join(model.ema_working_dir, "transformer_ema.safetensors")
        if os.path.isfile(_src):
            _dst = os.path.join(to_folder, "transformer_ema.safetensors")
            shutil.copy2(_src, _dst)
            logging.info(f" * Copied EMA sidecar (disk-offload): transformer_ema.safetensors")

    model_id_path = os.path.join(to_folder, "model_id.txt")
    with open(model_id_path, "w") as f:
        f.write(model.model_id)


@torch.no_grad()
def save_sana_lora(model: SanaTrainingModel, save_path: str) -> None:
    """
    Save only the LoRA adapter weights for the SANA transformer.

    Uses PEFT's built-in save_lora_weights(), which produces a diffusers-
    compatible adapter checkpoint that can be loaded via load_lora_weights().
    Also writes model_id.txt so the base model can be reconstructed later.
    """
    if not hasattr(model.transformer, "peft_config"):
        logging.warning("No LoRA adapters found on transformer — skipping LoRA save")
        return

    os.makedirs(save_path, exist_ok=True)
    logging.info(f" * Saving LoRA adapter to {save_path}")

    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers

    transformer_lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(model.transformer)
    )

    model.build_inference_pipeline().save_lora_weights(
        save_directory=save_path,
        transformer_lora_layers=transformer_lora_layers,
    )

    model_id_path = os.path.join(save_path, "model_id.txt")
    with open(model_id_path, "w") as f:
        f.write(model.model_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_self_flow_state(model: SanaTrainingModel, checkpoint_folder: str) -> None:
    """
    Attempts to load the Self-Flow projection head from a checkpoint folder.
    Teacher weights are now the main EMA — no separate loading needed.
    """
    import os

    proj_head_path = os.path.join(checkpoint_folder, "self_flow_proj_head.pt")

    if os.path.exists(proj_head_path):
        logging.info(f" * Loading Self-Flow projection head from {proj_head_path}")
        model.self_flow_proj_head.load_state_dict(torch.load(proj_head_path))
    else:
        logging.warning(f"Self-Flow projection head checkpoint not found at {proj_head_path}. Continuing without it.")


def _load_transformer_checkpoint(transformer: nn.Module, checkpoint_path: str) -> None:
    """Loads a safetensors checkpoint into the transformer in-place."""
    from safetensors.torch import load_model
    load_model(transformer, checkpoint_path)
