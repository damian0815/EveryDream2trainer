"""
SanaTrainingModel: dataclass holding all SANA model components plus factory and save functions.
"""
from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class SanaTrainingModel:
    """Holds all SANA model components for training."""

    transformer: nn.Module          # SanaMS — the sole trained component
    text_encoder: nn.Module         # T5/Gemma/Qwen — frozen, not trained
    tokenizer: Any                  # matching tokenizer
    vae: nn.Module                  # DC-AE — frozen, not trained
    vae_config: Any                 # SanaVaeConfig (vae_type, vae_downsample_rate, etc.)
    train_diffusion: Any            # SANA Scheduler with .training_losses()
    sana_config: Any                # full SanaConfig pyrallis object

    transformer_ema: Optional[nn.Module] = None  # reserved for future EMA support

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype


def load_sana_model(args: Namespace, sana_config) -> SanaTrainingModel:
    """
    Instantiates and returns a SanaTrainingModel from sana_config.
    Loads a checkpoint if sana_config.model.resume_from or .load_from is set.
    Freezes text_encoder and vae parameters (requires_grad = False).
    """
    from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
    from diffusion import Scheduler
    from diffusion.utils.checkpoint import load_checkpoint

    transformer = build_model(sana_config)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(sana_config)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    vae = get_vae(sana_config)
    for p in vae.parameters():
        p.requires_grad_(False)

    train_diffusion = Scheduler(
        sana_config.scheduler.train_sampling_steps,
        noise_schedule=sana_config.scheduler.noise_schedule,
        predict_v=getattr(sana_config.scheduler, 'predict_v', False),
        snr_shift_scale=getattr(sana_config.scheduler, 'snr_shift_scale', 1.0),
    )

    model = SanaTrainingModel(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        vae_config=sana_config.vae,
        train_diffusion=train_diffusion,
        sana_config=sana_config,
    )

    if getattr(sana_config.model, 'resume_from', None) or getattr(sana_config.model, 'load_from', None):
        load_checkpoint(model.transformer, sana_config)

    return model


def save_sana_model(path: str, model: SanaTrainingModel, global_step: int) -> None:
    """
    Saves only the transformer (the trained component) as a safetensors file.
    Saves the config as config.yaml.
    Text encoder and VAE are NOT saved (they are frozen/unchanged).
    """
    import pyrallis
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    weights_path = os.path.join(path, f"transformer_gs{global_step}.safetensors")
    save_file(model.transformer.state_dict(), weights_path)

    config_path = os.path.join(path, "config.yaml")
    with open(config_path, "w") as f:
        pyrallis.dump(model.sana_config, f)

