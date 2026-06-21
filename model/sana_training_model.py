"""
SanaTrainingModel: dataclass holding all SANA model components plus factory and save functions.
Uses 🤗 diffusers (SanaPipeline, SanaTransformer2DModel, AutoencoderDC,
TrainFlowMatchEulerDiscreteScheduler) — no SANA repo clone required.
"""
from __future__ import annotations

import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional, Literal

import safetensors
from diffusers import SanaPipeline

import torch
import torch.nn as nn

from optimizer.optimizers import EveryDreamOptimizer


@dataclass
class SanaTrainingModel:
    """Holds all SANA model components for training."""

    transformer: nn.Module                   # SanaTransformer2DModel — sole trained component
    text_encoder: nn.Module                  # Gemma2 — frozen, not trained
    tokenizer: Any                           # GemmaTokenizerFast — frozen
    vae: nn.Module                           # AutoencoderDC — frozen, not trained
    noise_scheduler: Any                     # TrainFlowMatchEulerDiscreteScheduler
    model_id: str                            # HF hub ID, recorded for save/resume

    max_sequence_length: int = 300           # Gemma token budget
    complex_human_instruction: list = field(default_factory=list)  # optional system-prompt prefix

    transformer_ema: Optional[nn.Module] = None  # reserved for future EMA support
    self_flow_teacher_transformer: Optional[nn.Module] = None  # initialized if self-flow enabled; used as teacher UNet
    self_flow_proj_head: Optional[nn.Module] = None

    is_video: bool = False  # set to True for SanaVideoPipeline training

    # ---- EveryDreamOptimizer duck-type adapter properties ----------------

    @property
    def unet(self) -> nn.Module:
        """Alias for transformer — satisfies EveryDreamOptimizer's model.unet access."""
        return self.transformer

    # Add this right under your @property def unet(self):
    @property
    def self_flow_teacher_unet(self) -> Optional[nn.Module]:
        """Alias for self_flow_teacher_transformer — satisfies ED2's Self-Flow logic."""
        return self.self_flow_teacher_transformer

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

    # ---- Device offload helpers -------------------------------------------

    def load_vae_to_device(self, device):
        """Move VAE to *device* (e.g. 'cuda' or 'cpu')."""
        self.vae.to(device)

    def load_textenc_to_device(self, device):
        """Move text encoder to *device* (e.g. 'cuda' or 'cpu')."""
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
    dtype_vae = torch.float32 if is_video else torch.bfloat16

    pipe = _load_sana_pipeline(args.model_id, dtype=torch.bfloat16, te_quantization=args.te_quantization, is_video=is_video)

    if args.te_quantization == 'none':
        pipe.text_encoder.to(dtype=torch.bfloat16)
    pipe.vae.to(dtype=dtype_vae)

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

    if args.resume_from is not None:
        logging.info(f" * Resuming from {args.resume_from}")
        _load_transformer_checkpoint(model.transformer, args.resume_from)

    if args.self_flow_p > 0:
        _inject_self_flow(model=model, pipe=pipe)
        if args.resume_from is not None:
            _try_load_self_flow_state(model, os.path.dirname(args.resume_from))

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
        load_kwargs: dict = {"torch_dtype": torch.float32}
        pipeline = pipeline_cls.from_pretrained(repo_id, **load_kwargs)
        # Cast the heavy compute modules to bfloat16 to halve VRAM usage
        pipeline.vae.to(dtype)
        pipeline.text_encoder.to(dtype)
        pipeline.transformer = pipeline.transformer.to(dtype)

    return pipeline


def _inject_self_flow(model: SanaTrainingModel, pipe: SanaPipeline):
    import copy
    import types
    from core.self_flow import SelfFlowMLPProjectionHead

    # 1. Initialize EMA Teacher
    model.self_flow_teacher_transformer = copy.deepcopy(pipe.transformer)
    model.self_flow_teacher_transformer.requires_grad_(False)
    model.self_flow_teacher_transformer.eval()

    # 2. Initialize Projection Head (SANA 1.6B hidden size is 2240)
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


def save_sana_model(to_folder: str, model: SanaTrainingModel, optimizer: EveryDreamOptimizer, global_step: int, num_samples: int) -> None:
    """
    Saves only the transformer (the trained component) as a safetensors file.
    Also writes model_id.txt so the full pipeline can be reconstructed later:

        pipe = SanaPipeline.from_pretrained(model_id)
        load_model(pipe.transformer, "transformer_gsNNNN.safetensors")
    """
    from safetensors.torch import save_file

    os.makedirs(to_folder, exist_ok=True)

    weights_path = os.path.join(to_folder, f"transformer_gs{global_step:05}_n{num_samples:05}.safetensors")
    logging.info(f" * Saving transformer checkpoint to {weights_path}")
    save_file(model.transformer.state_dict(), weights_path)

    if optimizer is not None:
        logging.info(f" Saving optimizer state to {to_folder}")
        optimizer.save(to_folder)

    if model.self_flow_proj_head is not None:
        proj_head_path = os.path.join(to_folder, "self_flow_proj_head.pt")
        logging.info(f" * Saving Self-Flow projection head to {proj_head_path}")
        torch.save(model.self_flow_proj_head.state_dict(), proj_head_path)

    if model.self_flow_teacher_transformer is not None:
        teacher_path = os.path.join(to_folder, "self_flow_teacher_module.safetensors")
        logging.info(f" * Saving Self-Flow teacher UNet to {teacher_path}")
        state_dict = {k: v.cpu().contiguous() for k, v in model.self_flow_teacher_transformer.state_dict().items()}
        safetensors.torch.save_file(state_dict, teacher_path)

    model_id_path = os.path.join(to_folder, "model_id.txt")
    with open(model_id_path, "w") as f:
        f.write(model.model_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_self_flow_state(model: SanaTrainingModel, checkpoint_folder: str) -> None:
    """
    Attempts to load the Self-Flow teacher UNet and projection head from a checkpoint folder.
    If the files are not found, logs a warning and continues without raising an error.
    """
    import os

    teacher_path = os.path.join(checkpoint_folder, "self_flow_teacher_module.safetensors")
    proj_head_path = os.path.join(checkpoint_folder, "self_flow_proj_head.pt")

    if os.path.exists(teacher_path):
        logging.info(f" * Loading Self-Flow teacher UNet from {teacher_path}")
        _load_transformer_checkpoint(model.self_flow_teacher_transformer, teacher_path)
    else:
        logging.warning(f"Self-Flow teacher UNet checkpoint not found at {teacher_path}. Continuing without it.")

    if os.path.exists(proj_head_path):
        logging.info(f" * Loading Self-Flow projection head from {proj_head_path}")
        model.self_flow_proj_head.load_state_dict(torch.load(proj_head_path))
    else:
        logging.warning(f"Self-Flow projection head checkpoint not found at {proj_head_path}. Continuing without it.")


def _load_transformer_checkpoint(transformer: nn.Module, checkpoint_path: str) -> None:
    """Loads a safetensors checkpoint into the transformer in-place."""
    from safetensors.torch import load_model
    load_model(transformer, checkpoint_path)
