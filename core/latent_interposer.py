"""
Latent Space Interposer for EveryDream2trainer
===============================================

Integrates the SD-Latent-Interposer (city96/SD-Latent-Interposer) to allow
cross-latent-space teacher → student knowledge distillation.

Primary use-case:  SD1.x / SD2  **Flow Matching** teacher
                   →  SDXL       **Flow Matching** student.

The ``get_teacher_target`` path in ``core/loss.py`` uses the teacher UNet for a
single-step velocity prediction, using the interposer to convert between latent
spaces when needed.

Public API
----------

.. code-block:: python

    from core.latent_interposer import (
        LatentSpaceType,
        LatentInterposer,
        infer_latent_space_type,
    )

    # Simple latent conversion
    interposer = LatentInterposer()
    xl_as_v1 = interposer.convert(xl_latents, src=LatentSpaceType.SDXL, dst=LatentSpaceType.SD1)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Neural network layers – ported from city96/SD-Latent-Interposer
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual block with BatchNorm and dropout."""

    def __init__(self, ch: int):
        super().__init__()
        self.join = nn.ReLU()
        self.norm = nn.BatchNorm2d(ch)
        self.long = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.join(self.long(x) + x)


class _ExtractBlock(nn.Module):
    """Channel-expansion block used at the head of the interposer."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.join = nn.ReLU()
        self.short = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.long = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.join(self.long(x) + self.short(x))


class InterposerModel(nn.Module):
    """
    Lightweight CNN that maps latents from one Stable Diffusion latent space
    to another.  Architecture from city96/SD-Latent-Interposer v4.0.

    Parameters
    ----------
    ch_in:   Number of input channels  (4 for SD1/SDXL/Cascade, 16 for SD3/Flux)
    ch_out:  Number of output channels
    ch_mid:  Width of internal feature maps  (64 by default)
    scale:   Spatial upsample factor applied inside ``core`` (1.0 = no resize)
    blocks:  Number of ``_ResBlock`` layers in ``core``
    """

    def __init__(
        self,
        ch_in: int = 4,
        ch_out: int = 4,
        ch_mid: int = 64,
        scale: float = 1.0,
        blocks: int = 12,
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_mid = ch_mid
        self.blocks = blocks
        self.scale = scale

        self.head = _ExtractBlock(ch_in, ch_mid)
        self.core = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            *[_ResBlock(ch_mid) for _ in range(blocks)],
            nn.BatchNorm2d(ch_mid),
            nn.SiLU(),
        )
        self.tail = nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)
        z = self.core(y)
        return self.tail(z)


# ---------------------------------------------------------------------------
# Model configuration registry  (mirrors comfy_latent_interposer.py)
# ---------------------------------------------------------------------------

#: Mapping from ``"{src}-to-{dst}"`` key → ``InterposerModel`` constructor kwargs.
INTERPOSER_CONFIGS: dict[str, dict] = {
    "v1-to-xl": {"ch_in": 4,  "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v1-to-v3": {"ch_in": 4,  "ch_out": 16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "xl-to-v1": {"ch_in": 4,  "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "xl-to-v3": {"ch_in": 4,  "ch_out": 16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v3-to-v1": {"ch_in": 16, "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v3-to-xl": {"ch_in": 16, "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-v1": {"ch_in": 16, "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-xl": {"ch_in": 16, "ch_out": 4,  "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-v3": {"ch_in": 16, "ch_out": 16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "ca-to-v1": {"ch_in": 4,  "ch_out": 4,  "ch_mid": 64, "scale": 0.5, "blocks": 12},
    "ca-to-xl": {"ch_in": 4,  "ch_out": 4,  "ch_mid": 64, "scale": 0.5, "blocks": 12},
    "ca-to-v3": {"ch_in": 4,  "ch_out": 16, "ch_mid": 64, "scale": 0.5, "blocks": 12},
}

#: HuggingFace Hub repository ID for pretrained weights.
_HF_REPO = "city96/SD-Latent-Interposer"

#: Model file version string (matches filenames on HF Hub).
_INTERPOSER_VERSION = "4.0"


# ---------------------------------------------------------------------------
# LatentSpaceType – string constants for readability
# ---------------------------------------------------------------------------

class LatentSpaceType:
    """
    String constants identifying each supported Stable Diffusion latent space.

    Use these when calling :meth:`LatentInterposer.convert` to make call sites
    self-documenting::

        interposer.convert(latents, src=LatentSpaceType.SDXL, dst=LatentSpaceType.SD1)

    The underlying string values (``"v1"``, ``"xl"``, …) match the naming
    convention used in the city96/SD-Latent-Interposer weight files.
    """

    #: Stable Diffusion 1.x / 2.x – 4 channels, 1/8 spatial compression.
    SD1: str = "v1"
    #: SDXL – 4 channels, 1/8 spatial compression (different VAE than SD1).
    SDXL: str = "xl"
    #: Stable Diffusion 3 – 16 channels, 1/8 spatial compression.
    SD3: str = "v3"
    #: Flux.1 – 16 channels, 1/8 spatial compression.
    FLUX: str = "fx"
    #: Stable Cascade Stage A/B – 4 channels, variable spatial compression.
    CASCADE: str = "ca"


# ---------------------------------------------------------------------------
# LatentInterposer
# ---------------------------------------------------------------------------

class LatentInterposer:
    """
    Manages loading and applying SD-Latent-Interposer models.

    Models are loaded lazily on first use and then cached in-process.
    All inference runs on CPU in ``float32`` (matching the reference
    ComfyUI implementation).

    Parameters
    ----------
    model_dir:
        Optional path to a local directory containing pre-downloaded
        ``*.safetensors`` weight files.  Three layouts are accepted:

        * Flat:       ``<model_dir>/<key>_interposer-v4.0.safetensors``
        * Versioned:  ``<model_dir>/v4.0/<key>_interposer-v4.0.safetensors``
        * Submodule:  ``SD-Latent-Interposer/models/`` next to the repo root

        Falls back to HuggingFace Hub (requires ``huggingface-hub``) when
        no local file is found.

    Example::

        interposer = LatentInterposer()
        v1_latents = interposer.convert(xl_latents, "xl", "v1")
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = model_dir
        self._cache: dict[str, InterposerModel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        latents: torch.Tensor,
        src: str,
        dst: str,
    ) -> torch.Tensor:
        """
        Convert **unscaled** *latents* from the *src* latent space to the *dst* latent space.

        The returned tensor is placed on the same device and in the same dtype
        as the input.

        Parameters
        ----------
        latents:  ``[B, C, H, W]`` latent tensor.
        src:      Source latent space code, e.g. ``"xl"`` or ``LatentSpaceType.SDXL``.
        dst:      Destination latent space code, e.g. ``"v1"`` or ``LatentSpaceType.SD1``.

        Raises
        ------
        ValueError
            If no pretrained model exists for the requested conversion.
        """
        if src == dst:
            print("returning unchanged")
            return latents

        key = f"{src}-to-{dst}"
        if key not in INTERPOSER_CONFIGS:
            raise ValueError(
                f"No interposer model exists for '{src}' → '{dst}'.  "
                f"Available conversions: {sorted(INTERPOSER_CONFIGS.keys())}"
            )
        print('get model', key)

        model = self._load_model(key)
        original_device = latents.device
        original_dtype = latents.dtype

        with torch.no_grad():
            result = model(latents.cpu().float())

        return result.to(device=original_device, dtype=original_dtype)

    def is_supported(self, src: str, dst: str) -> bool:
        """Return ``True`` if a pretrained model exists for the ``src`` → ``dst`` pair."""
        return src != dst and f"{src}-to-{dst}" in INTERPOSER_CONFIGS

    @staticmethod
    def is_supported_static(src: str, dst: str) -> bool:
        """Class-level check — does not require an instance."""
        return src != dst and f"{src}-to-{dst}" in INTERPOSER_CONFIGS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, key: str) -> InterposerModel:
        if key in self._cache:
            print('cached:', key)
            return self._cache[key]

        path = self._resolve_model_path(key)
        print(f"LatentInterposer: loading '{key}' from {path}")
        cfg = INTERPOSER_CONFIGS[key]
        model = InterposerModel(**cfg)
        model.eval()
        model.load_state_dict(load_file(path))
        self._cache[key] = model
        print(f"LatentInterposer: loaded '{key}' from {path}")
        return model

    def _resolve_model_path(self, key: str) -> str:
        fname = f"{key}_interposer-v{_INTERPOSER_VERSION}.safetensors"

        if self._model_dir:
            # 1. Flat layout: <model_dir>/<fname>
            candidate = os.path.join(self._model_dir, fname)
            if os.path.isfile(candidate):
                return candidate

            # 2. Versioned layout: <model_dir>/v4.0/<fname>
            candidate = os.path.join(self._model_dir, f"v{_INTERPOSER_VERSION}", fname)
            if os.path.isfile(candidate):
                return candidate

        # 3. Bundled submodule: SD-Latent-Interposer/models/ alongside the repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        submodule_candidate = os.path.join(repo_root, "SD-Latent-Interposer", "models", fname)
        if os.path.isfile(submodule_candidate):
            return submodule_candidate

        versioned_submodule_candidate = os.path.join(
            repo_root, "SD-Latent-Interposer", "models", f"v{_INTERPOSER_VERSION}", fname
        )
        if os.path.isfile(versioned_submodule_candidate):
            return versioned_submodule_candidate

        # 4. HuggingFace Hub fallback
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to download interposer weights automatically.  "
                "Install with:  pip install huggingface-hub\n"
                "Or download the weights manually from "
                f"https://huggingface.co/{_HF_REPO}/tree/main/v{_INTERPOSER_VERSION}"
            ) from exc

        logging.info(f"LatentInterposer: downloading '{key}' from HuggingFace Hub …")
        return str(
            hf_hub_download(
                repo_id=_HF_REPO,
                subfolder=f"v{_INTERPOSER_VERSION}",
                filename=fname,
            )
        )


# ---------------------------------------------------------------------------
# Helper: infer latent space type from a TrainingModel
# ---------------------------------------------------------------------------

def infer_latent_space_type(model) -> Optional[str]:
    """
    Infer the latent space type for *model* and return its code string.

    Returns
    -------
    ``"xl"`` for SDXL models (``text_encoder_2`` present).
    ``"v1"`` for SD1.x / SD2.x models (4-channel UNet, no second text encoder).
    ``"v3"`` if the UNet has 16 input channels (SD3/Flux territory – callers
    should verify further before relying on this).
    ``None`` if the type cannot be reliably determined.
    """
    if model is None:
        return None

    # SDXL is identified by the presence of a second text encoder
    if getattr(model, "is_sdxl", False):
        return LatentSpaceType.SDXL

    unet = getattr(model, "unet", None)
    if unet is not None:
        in_ch = unet.config.get("in_channels", 4)
        if in_ch == 16:
            return LatentSpaceType.SD3  # SD3 / Flux – caller should verify
    return LatentSpaceType.SD1

