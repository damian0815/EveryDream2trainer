"""
SanaTrainingModel: dataclass holding all SANA model components plus factory and save functions.
Uses 🤗 diffusers (SanaPipeline, SanaTransformer2DModel, AutoencoderDC,
FlowMatchEulerDiscreteScheduler) — no SANA repo clone required.
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
    scheduler: Any                           # FlowMatchEulerDiscreteScheduler
    model_id: str                            # HF hub ID, recorded for save/resume

    max_sequence_length: int = 300           # Gemma token budget
    complex_human_instruction: list = field(default_factory=list)  # optional system-prompt prefix
    guidance_scale: float = 4.5             # default cfg scale for sample generation

    transformer_ema: Optional[nn.Module] = None  # reserved for future EMA support

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype


def load_sana_model(args: Namespace) -> SanaTrainingModel:
    """
    Loads all SANA components via SanaPipeline.from_pretrained and wraps them in
    a SanaTrainingModel.  Freezes the text encoder and VAE (requires_grad = False).
    Optionally loads fine-tuned transformer weights from args.resume_from.
    """
    from diffusers import SanaPipeline

    torch_dtype = _parse_dtype(getattr(args, "mixed_precision", "bf16"))

    pipe = SanaPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)

    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    model = SanaTrainingModel(
        transformer=pipe.transformer,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        vae=pipe.vae,
        scheduler=pipe.scheduler,
        model_id=args.model_id,
        max_sequence_length=getattr(args, "max_sequence_length", 300),
        complex_human_instruction=getattr(args, "complex_human_instruction", []) or [],
        guidance_scale=getattr(args, "guidance_scale", 4.5),
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
