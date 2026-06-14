"""
Latent Space Interposer for EveryDream2trainer
===============================================

Provides two backends for cross-latent-space conversion:

1. **TaesdLatentConverter** *(default)* — routes through pixel space using the
   ``taesd/`` submodule (madebyollin/taesd).  Converts:
   ``raw_src_latents → TAESD_src.decoder → pixels[0,1] → TAESD_dst.encoder → raw_dst_latents``
   No extra weight downloads needed — the ``.pth`` files ship with the submodule.

2. **LatentInterposer** *(legacy)* — uses the city96/SD-Latent-Interposer learned
   latent-to-latent CNN.  Requires separate ``.safetensors`` weight files.

Both classes expose the same public API::

    converter.convert(latents, src=LatentSpaceType.SDXL, dst=LatentSpaceType.SD1)
    ConverterClass.is_supported(src, dst)

``get_shared_interposer()`` returns the process-level :class:`TaesdLatentConverter`
singleton (created lazily).

Supported latent-space codes (``LatentSpaceType``)
---------------------------------------------------
  ``"v1"``  SD1 / SD2  →  ``taesd_*``     weights
  ``"xl"``  SDXL       →  ``taesdxl_*``   weights
  ``"v3"``  SD3        →  ``taesd3_*``    weights
  ``"fx"``  Flux.1     →  ``taef1_*``     weights
"""

from __future__ import annotations

import importlib.util
import logging
import os
import types
from typing import Optional, Literal

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from torchvision import transforms


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
# TAESD  –  tiny autoencoder-based latent converter  (primary backend)
# ---------------------------------------------------------------------------

#: Maps LatentSpaceType code → TAESD weight-file prefix and (latent_channels, arch_variant).
#: arch_variant values mirror those in taesd/taesd.py:
#:   None     → standard Encoder/Decoder  (8× spatial compression)
#:   "flux_2" → midblock GroupNorm enabled
#:   "f32"    → F32Encoder/F32Decoder     (32× spatial compression, Sana DC-AE)
_TAESD_SPECS: dict[str, tuple[str, int, str | None]] = {
    # space code  : (file prefix,  latent_channels, arch_variant)
    "v1":          ("taesd",    4,  None),
    "xl":          ("taesdxl",  4,  None),
    "v3":          ("taesd3",   16, None),
    "fx":          ("taef1",    16, None),
}

_taesd_module_cache: Optional[types.ModuleType] = None


def _import_taesd_module() -> types.ModuleType:
    """Lazily import ``taesd/taesd.py`` from the submodule directory."""
    global _taesd_module_cache
    if _taesd_module_cache is not None:
        return _taesd_module_cache

    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    taesd_py = os.path.join(repo_root, "taesd", "taesd.py")
    if not os.path.isfile(taesd_py):
        raise FileNotFoundError(
            f"TAESD submodule not found at {taesd_py}.\n"
            "Run:  git submodule update --init taesd"
        )
    spec = importlib.util.spec_from_file_location("_taesd_submodule", taesd_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _taesd_module_cache = mod
    return mod


def _load_taesd_encoder(path: str, latent_channels: int, arch_variant: str | None) -> nn.Module:
    """Build and weight-load a TAESD encoder from a ``.pth`` checkpoint."""
    taesd = _import_taesd_module()
    if arch_variant == "f32":
        net = taesd.F32Encoder(latent_channels)
    else:
        net = taesd.Encoder(latent_channels, use_midblock_gn=(arch_variant == "flux_2"))
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def _load_taesd_decoder(path: str, latent_channels: int, arch_variant: str | None) -> nn.Module:
    """Build and weight-load a TAESD decoder from a ``.pth`` checkpoint."""
    taesd = _import_taesd_module()
    if arch_variant == "f32":
        net = taesd.F32Decoder(latent_channels)
    else:
        net = taesd.Decoder(latent_channels, use_midblock_gn=(arch_variant == "flux_2"))
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def _encode_to_scaled_latent_vae(vae, image):
    scale = vae.config.scaling_factor
    x = (image * 2.0) - 1.0
    return scale * vae.encode(x)[0].sample()


def _decode_scaled_latent_vae(vae, latent):
    scale = vae.config.scaling_factor
    x = vae.decode(latent / scale).sample   # (1, 3, H, W), range [-1,1]
    return (x + 1.0) / 2.0


def convert_latents_slow_vae_roundtrip(
    latents: torch.Tensor,
    from_vae: AutoencoderKL,
    to_vae: AutoencoderKL,
    normalization: Literal['none', 'imagenet', '0.5']
) -> torch.Tensor:
    intermediate = _decode_scaled_latent_vae(from_vae, latents.to(dtype=from_vae.dtype))
    if normalization == 'imagenet':
        intermediate = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(intermediate)
    elif normalization == '0.5':
        intermediate = transforms.Normalize(mean=[0.5], std=[0.5])(intermediate)
    return _encode_to_scaled_latent_vae(to_vae, intermediate)


class TaesdLatentConverter:
    """
    Cross-latent-space converter backed by TAESD tiny autoencoders.

    Conversion path::

        raw_src_latents
            → TAESD_src.decoder  →  pixel image [0, 1]
            → TAESD_dst.encoder  →  raw_dst_latents

    TAESD operates on **raw** (unscaled) VAE latents — i.e. before the
    SD ``scaling_factor`` is applied.  The call-sites in ``core/loss.py``
    already unscale before calling ``convert()`` and re-scale the result,
    so no extra scaling is needed here.

    Supported latent-space codes
    ----------------------------
    ``"v1"``  SD1 / SD2  —  ``taesd_*``   weights
    ``"xl"``  SDXL       —  ``taesdxl_*`` weights
    ``"v3"``  SD3        —  ``taesd3_*``  weights
    ``"fx"``  Flux.1     —  ``taef1_*``   weights

    Parameters
    ----------
    taesd_dir:
        Path to the directory containing the ``*_encoder.pth`` /
        ``*_decoder.pth`` weight files.  Defaults to the ``taesd/``
        submodule directory next to the repo root.
    """

    #: Set of supported latent-space codes.
    SUPPORTED_SPACES: frozenset[str] = frozenset(_TAESD_SPECS.keys())

    def __init__(self, taesd_dir: Optional[str] = None):
        if taesd_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            taesd_dir = os.path.join(repo_root, "taesd")
        self._taesd_dir = taesd_dir
        self._encoders: dict[str, nn.Module] = {}
        self._decoders: dict[str, nn.Module] = {}

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
        Convert **raw** (unscaled) *latents* from *src* space to *dst* space
        by routing through TAESD's pixel-space representation.

        Parameters
        ----------
        latents:  ``[B, C, H, W]`` raw latent tensor (no VAE scaling_factor applied).
        src:      Source latent-space code, e.g. ``"xl"`` / ``LatentSpaceType.SDXL``.
        dst:      Destination latent-space code, e.g. ``"v1"`` / ``LatentSpaceType.SD1``.

        Returns a tensor on the **same device and dtype** as the input.
        """
        if src == dst:
            return latents

        for space in (src, dst):
            if space not in _TAESD_SPECS:
                raise ValueError(
                    f"TaesdLatentConverter: unsupported latent space '{space}'.  "
                    f"Supported: {sorted(_TAESD_SPECS.keys())}"
                )

        original_device = latents.device
        original_dtype  = latents.dtype

        # Move the cached encoder/decoder to the same device as the input.
        # .to() is a no-op when the module is already on that device, so this
        # is free after the first call.
        decoder = self._get_decoder(src).to(original_device)
        encoder = self._get_encoder(dst).to(original_device)

        with torch.no_grad():
            x      = latents.float()               # stay on original_device
            pixels = decoder(x).clamp(0.0, 1.0)   # raw src latents → pixel [0,1]
            out    = encoder(pixels)               # pixel [0,1]    → raw dst latents

        return out.to(dtype=original_dtype)        # already on original_device

    @classmethod
    def is_supported(cls, src: str, dst: str) -> bool:
        """Return ``True`` when both *src* and *dst* are supported spaces."""
        return src == dst or (src in _TAESD_SPECS and dst in _TAESD_SPECS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_decoder(self, space: str) -> nn.Module:
        if space not in self._decoders:
            prefix, ch, arch = _TAESD_SPECS[space]
            path = os.path.join(self._taesd_dir, f"{prefix}_decoder.pth")
            logging.info(f"TaesdLatentConverter: loading decoder for '{space}' from {path}")
            self._decoders[space] = _load_taesd_decoder(path, ch, arch)
        return self._decoders[space]

    def _get_encoder(self, space: str) -> nn.Module:
        if space not in self._encoders:
            prefix, ch, arch = _TAESD_SPECS[space]
            path = os.path.join(self._taesd_dir, f"{prefix}_encoder.pth")
            logging.info(f"TaesdLatentConverter: loading encoder for '{space}' from {path}")
            self._encoders[space] = _load_taesd_encoder(path, ch, arch)
        return self._encoders[space]


# ---------------------------------------------------------------------------
# LatentInterposer  (legacy backend — city96/SD-Latent-Interposer)
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
            return latents

        key = f"{src}-to-{dst}"
        if key not in INTERPOSER_CONFIGS:
            raise ValueError(
                f"No interposer model exists for '{src}' → '{dst}'.  "
                f"Available conversions: {sorted(INTERPOSER_CONFIGS.keys())}"
            )
        #print('get model', key)

        model = self._load_model(key)
        original_device = latents.device
        original_dtype = latents.dtype

        with torch.no_grad():
            result = model.to(original_device)(latents.float())

        return result.to(dtype=original_dtype)

    @staticmethod
    def is_supported(src: str, dst: str) -> bool:
        """Class-level check — does not require an instance."""
        return src == dst or f"{src}-to-{dst}" in INTERPOSER_CONFIGS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, key: str) -> InterposerModel:
        if key in self._cache:
            #print('cached:', key)
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

_shared_interposer: Optional["TaesdLatentConverter"] = None


def get_shared_interposer(taesd_dir: Optional[str] = None) -> "TaesdLatentConverter":
    """
    Return the process-level shared :class:`TaesdLatentConverter` instance,
    creating it lazily on first call.

    TAESD encoder/decoder weights are loaded on the first
    :meth:`~TaesdLatentConverter.convert` call for each conversion direction
    and cached inside the singleton, so repeated calls to this function are cheap.

    Parameters
    ----------
    taesd_dir:
        Optional path to the directory containing TAESD ``*.pth`` files.
        Defaults to the ``taesd/`` submodule next to the repo root.
        Only honoured on the very first call; subsequent calls return the
        already-created instance.
    """
    global _shared_interposer
    if _shared_interposer is None:
        _shared_interposer = TaesdLatentConverter(taesd_dir=taesd_dir)
    return _shared_interposer


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
    if model.is_sdxl:
        return LatentSpaceType.SDXL

    unet = model.unet
    if unet is not None:
        in_ch = unet.config.get("in_channels", 4)
        if in_ch == 16:
            return LatentSpaceType.SD3  # SD3 / Flux – caller should verify
    return LatentSpaceType.SD1

