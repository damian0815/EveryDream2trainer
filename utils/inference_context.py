from contextlib import contextmanager
import torch
from utils.isolate_rng import isolate_rng


@contextmanager
def inference_guard(*modules: torch.nn.Module):
    """
    Sets each module to eval(), isolates the RNG state, then restores training
    mode for any module that was training before.

    Usage:
        with inference_guard(model.unet, model.text_encoder):
            run_inference(...)
    """
    was_training = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        with isolate_rng():
            yield
    finally:
        for m, was in zip(modules, was_training):
            if was:
                m.train()

