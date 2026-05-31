"""
SanaTrainingModel: dataclass holding all SANA model components plus factory and save functions.
Uses 🤗 diffusers (SanaPipeline, SanaTransformer2DModel, AutoencoderDC,
TrainFlowMatchEulerDiscreteScheduler) — no SANA repo clone required.
"""
from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


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
    self_flow_proj_head: Optional[Any] = None    # always None for SANA; satisfies EveryDreamOptimizer interface

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
        Returns a SanaPipeline built from the live model components.
        Called by SampleGenerator.create_inference_pipe().

        scheduler: if None, constructs an SDPipelineInferenceFlowMatchEulerDiscreteScheduler
                   from the training scheduler's config. SampleGenerator always passes an
                   already-constructed inference scheduler here.
        """
        from diffusers import SanaPipeline
        from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler

        inf_scheduler = scheduler or SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
            self.noise_scheduler.config
        )
        return SanaPipeline(
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            scheduler=inf_scheduler,
        )


def load_sana_model(args: Namespace) -> SanaTrainingModel:
    """
    Loads all SANA components via SanaPipeline.from_pretrained and wraps them in
    a SanaTrainingModel.  Freezes the text encoder and VAE (requires_grad = False).
    Converts the pipeline's stock FlowMatchEulerDiscreteScheduler to a
    TrainFlowMatchEulerDiscreteScheduler so the training loop can use the same
    noising/timestep utilities as SD2/SDXL flow-matching training.
    Optionally loads fine-tuned transformer weights from args.resume_from.
    """
    from diffusers import SanaPipeline
    from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler

    torch_dtype = _parse_dtype(getattr(args, "mixed_precision", "bf16"))

    pipe = SanaPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)

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
    )

    if getattr(args, "resume_from", None):
        _load_transformer_checkpoint(model.transformer, args.resume_from)

    return model


def save_sana_model(path: str, model: SanaTrainingModel, global_step: int) -> None:
    """
    Saves only the transformer (the trained component) as a safetensors file.
    Also writes model_id.txt so the full pipeline can be reconstructed later:

        pipe = SanaPipeline.from_pretrained(model_id)
        load_model(pipe.transformer, "transformer_gsNNNN.safetensors")
    """
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)

    weights_path = os.path.join(path, f"transformer_gs{global_step}.safetensors")
    save_file(model.transformer.state_dict(), weights_path)

    model_id_path = os.path.join(path, "model_id.txt")
    with open(model_id_path, "w") as f:
        f.write(model.model_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(mixed_precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }.get(mixed_precision, torch.bfloat16)


def _load_transformer_checkpoint(transformer: nn.Module, checkpoint_path: str) -> None:
    """Loads a safetensors checkpoint into the transformer in-place."""
    from safetensors.torch import load_model
    load_model(transformer, checkpoint_path)
