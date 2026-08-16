"""
LoRA adapter EMA — in-memory EMA over the trainable (adapter) parameters only.

LoRA training attaches PEFT adapters at model-load time and only the adapter
parameters train.  An EMA over just those tiny adapter params is cheap enough
to always keep in memory (no disk-offload), and is surfaced three ways:

  * mid-training EMA sampling (in-place swap of adapter weights),
  * a ``<ckpt>_ema`` diffusers LoRA + single-file civitai export at each lora save,
  * a ``lora_ema.safetensors`` sidecar written next to the adapter dir for resume.

All functions here are DDP-agnostic: callers pass the unwrapped live module (or
we unwrap it ourselves).  PEFT imports stay lazy (inside functions) so this
module imports without peft installed.
"""
import logging
import os
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from typing import Optional

import safetensors.torch
import torch
import torch.nn as nn


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def get_adapter_params(module) -> "OrderedDict[str, torch.nn.Parameter]":
    """{name: param} from the module's trainable params (adapter params in LoRA mode).

    In LoRA mode PEFT freezes everything except the ``lora_*`` params, so
    filtering on ``requires_grad`` selects exactly the adapter weights — the
    same filter ``setup_lora_training`` uses.
    """
    return OrderedDict(
        (name, param)
        for name, param in _unwrap(module).named_parameters()
        if param.requires_grad
    )


def get_lora_components(model, args) -> list[str]:
    """Names of the model components that carry LoRA adapters."""
    if hasattr(model, "transformer"):
        return ["transformer"]
    components = []
    if not args.disable_unet_training:
        components.append("unet")
    if not args.disable_textenc_training:
        components.append("text_encoder")
        if model.is_sdxl and model.text_encoder_2 is not None:
            components.append("text_encoder_2")
    return components


def build_lora_ema_state(module) -> "OrderedDict[str, torch.Tensor]":
    """Snapshot of the adapter params as detached clones: ``{name: tensor}``."""
    return OrderedDict(
        (name, param.detach().clone())
        for name, param in get_adapter_params(module).items()
    )


def update_lora_ema(live_module, ema_state, decay, default_device, ema_device) -> None:
    """Mirror of ``model/ema.py:update_ema`` math, but only over adapter params.

    ``ema_state[name]`` lives on ``ema_device``; live params on ``default_device``.
    Cannot literally reuse ``update_ema`` because PEFT param names contain dots
    (``register_parameter`` forbids dotted names), so the math is mirrored here.
    """
    ema_device = ema_device if ema_device in ("cpu", "cuda") else "cpu"
    live_params = get_adapter_params(live_module)
    with torch.no_grad():
        for name, ema in ema_state.items():
            if name not in live_params:
                continue
            live = live_params[name]
            if torch.device(ema.device) != torch.device(default_device):
                ema.copy_(
                    ema * decay
                    + live.float().to(device=ema.device, dtype=ema.dtype) * (1.0 - decay)
                )
            else:
                ema.mul_(decay).add_(live.detach(), alpha=1.0 - decay)


@contextmanager
def apply_lora_ema_inplace(live_module, ema_state):
    """Back up the live adapter params, copy EMA values in, yield, then restore."""
    live_module = _unwrap(live_module)
    live_params = get_adapter_params(live_module)
    backups = OrderedDict(
        (name, param.detach().clone())
        for name, param in live_params.items()
        if name in ema_state
    )
    try:
        with torch.no_grad():
            for name, ema in ema_state.items():
                if name in live_params:
                    live_params[name].data.copy_(
                        ema.to(device=live_params[name].device,
                               dtype=live_params[name].dtype)
                    )
        yield
    finally:
        with torch.no_grad():
            for name, backup in backups.items():
                if name in live_params:
                    live_params[name].data.copy_(backup)


@contextmanager
def lora_ema_inplace_swap(model):
    """Swap every component's adapter params to its EMA state for inference.

    Targets ``model.<comp>`` directly — the same objects
    ``build_inference_pipeline()`` consumes.
    """
    ema_state_by_comp = getattr(model, "lora_ema", None) or {}
    with ExitStack() as stack:
        for comp, ema_state in ema_state_by_comp.items():
            stack.enter_context(apply_lora_ema_inplace(getattr(model, comp), ema_state))
        yield


def build_ema_lora_layers(live_module, ema_state, adapter_name=None) -> OrderedDict:
    """PEFT state dict of the EMA adapter weights, converted to diffusers keys."""
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    kwargs = {} if adapter_name is None else {"adapter_name": adapter_name}
    with apply_lora_ema_inplace(live_module, ema_state):
        raw = get_peft_model_state_dict(live_module, **kwargs)
    return convert_state_dict_to_diffusers(raw)


LORA_EMA_INIT_KEEP_FRACTION = 0.9


def _lora_identity_tensor(name: str, ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Standard-LoRA 'no change' start value for a single adapter param.

    Fresh LoRA starts from ``A ~ gaussian (std=1/r)`` and ``B = zeros``, so the
    merged model behaves exactly like the base model.  Non-A/B params have no
    well-defined identity start and return ``None`` (caller keeps the live value).
    """
    tokens = name.split(".")
    if "lora_A" in tokens:
        return torch.randn_like(ref) / ref.shape[0]
    if "lora_B" in tokens:
        return torch.zeros_like(ref)
    return None


def _blend_lora_ema_toward_identity(lora_ema: dict, keep_fraction: float) -> None:
    """Move each EMA tensor toward its identity start: ``keep`` live + ``1-keep`` identity."""
    for ema_state in lora_ema.values():
        for name in ema_state:
            identity = _lora_identity_tensor(name, ema_state[name])
            if identity is not None:
                ema_state[name].lerp_(identity, 1.0 - keep_fraction)


def _reset_lora_ema_to_identity(lora_ema: dict) -> None:
    """Overwrite the EMA with the pure identity start (``A~gaussian, B=zeros``)."""
    for ema_state in lora_ema.values():
        for name in ema_state:
            identity = _lora_identity_tensor(name, ema_state[name])
            if identity is not None:
                ema_state[name].copy_(identity)


def init_lora_ema(model, args, device, resume_dir) -> bool:
    """Build in-memory EMA state over adapter params, resuming from sidecar if any."""
    ema_device = args.ema_device if args.ema_device in ("cpu", "cuda") else "cpu"
    if args.ema_device == "disk":
        logging.warning(
            "disk EMA not supported for LoRA; using cpu (adapter EMA is tiny)"
        )

    model.lora_ema = {}
    for comp in get_lora_components(model, args):
        model.lora_ema[comp] = OrderedDict(
            (name, t.to(ema_device))
            for name, t in build_lora_ema_state(getattr(model, comp)).items()
        )

    if resume_dir:
        sidecar = os.path.join(resume_dir, "lora_ema.safetensors")
        if os.path.isfile(sidecar):
            logging.info(f"LoRA EMA: resuming from sidecar {sidecar}")
            sd = safetensors.torch.load_file(sidecar, device=ema_device)
            for comp, ema_state in model.lora_ema.items():
                for name in ema_state:
                    key = f"{comp}::{name}"
                    if key in sd:
                        ema_state[name].copy_(sd[key])
            del sd
        else:
            _blend_lora_ema_toward_identity(model.lora_ema, LORA_EMA_INIT_KEEP_FRACTION)
            logging.info(
                f"LoRA EMA: no sidecar found, blended adapter EMA toward identity "
                f"start ({LORA_EMA_INIT_KEEP_FRACTION} of trained weights kept)"
            )
    else:
        _reset_lora_ema_to_identity(model.lora_ema)
        logging.info("LoRA EMA: no adapter to resume; EMA initialised as identity start (A~gaussian, B=zeros)")

    logging.info(
        f"LoRA EMA enabled (in-memory, device={ema_device}, train device={device}): "
        f"decay={args.ema_decay_rate}, interval={args.ema_update_interval} | "
        f"components={list(model.lora_ema.keys())}"
    )
    return True


def save_lora_ema_sidecar(model, save_dir) -> None:
    """Flatten ``model.lora_ema`` into one small safetensors sidecar.

    Keys are ``f"{comp}::{param_name}"``.  Written inside the adapter dir as
    ``lora_ema.safetensors`` (ignored by ``load_lora_weights``).
    """
    ema_state_by_comp = getattr(model, "lora_ema", None)
    if ema_state_by_comp is None:
        return
    flattened = OrderedDict()
    for comp, ema_state in ema_state_by_comp.items():
        for name, t in ema_state.items():
            flattened[f"{comp}::{name}"] = t.detach().cpu().contiguous()
    os.makedirs(save_dir, exist_ok=True)
    sidecar_path = os.path.join(save_dir, "lora_ema.safetensors")
    safetensors.torch.save_file(flattened, sidecar_path)
