"""
EMA update utilities for EveryDream2trainer.

Moved from train.py during refactoring.
"""
import os
import logging
import torch
import safetensors.torch
from copy import deepcopy


def update_ema(model, ema_model, decay, default_device, ema_device: str):
    with torch.no_grad():
        original_model_on_proper_device = model
        need_to_delete_original = False
        if torch.device(ema_device) != torch.device(default_device):
            original_model_on_other_device = deepcopy(model)
            original_model_on_proper_device = original_model_on_other_device.to(ema_device, dtype=model.dtype)
            del original_model_on_other_device
            need_to_delete_original = True

        params: dict[str, torch.nn.Parameter] = dict(original_model_on_proper_device.named_parameters())
        ema_params: dict[str, torch.nn.Parameter] = dict(ema_model.named_parameters())

        for name in ema_params:
            ema_params[name].data = ema_params[name] * decay + params[name].data * (1.0 - decay)

        if need_to_delete_original:
            del(original_model_on_proper_device)


def update_ema_disk(model_module: torch.nn.Module, disk_path: str, decay: float) -> None:
    """
    Load the EMA state-dict from *disk_path*, apply one EMA step using the
    current *model_module* named parameters, and write the result back to disk.

    All arithmetic is done in float32 on CPU so VRAM is untouched.
    Buffers (non-parameter tensors in state_dict) are preserved unchanged.
    """
    if not os.path.isfile(disk_path):
        logging.warning(f"[EMA-disk] working file not found at {disk_path}, skipping update")
        return
    with torch.no_grad():
        ema_state   = safetensors.torch.load_file(disk_path, device="cpu")
        model_params = {k: v.detach().float().cpu() for k, v in model_module.named_parameters()}
        for k in ema_state:
            if k in model_params:
                ema_state[k] = (ema_state[k].float() * decay + model_params[k] * (1.0 - decay)).contiguous()
        del model_params
        safetensors.torch.save_file(ema_state, disk_path)
        del ema_state
