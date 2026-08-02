import torch


def capture_base_params(model, device=None, dtype=None):
    base_params = []
    for p in model.parameters():
        if p.requires_grad:
            base = p.detach().clone()
            if device is not None:
                base = base.to(device=device)
            if dtype is not None:
                base = base.to(dtype=dtype)
            base.requires_grad_(False)
            base_params.append(base)
    return base_params


def apply_anchor_reg(model, base_params, alpha):
    """
    Directly pulls trainable parameters toward their base values.
    Equivalent to weight decay toward a specific anchor point rather than zero.
    Must be called after the optimizer step.
    Returns the mean squared distance (scalar) for logging.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_dist = 0.0
    n = 0
    with torch.no_grad():
        for p, b in zip(trainable, base_params):
            b = b.to(p.device, dtype=p.dtype)
            diff = p.data - b
            total_dist += diff.pow(2).sum().item()
            n += diff.numel()
            p.data -= alpha * diff
            del diff
    return total_dist / max(1, n)
