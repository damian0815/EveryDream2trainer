import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import re


class ActivationLogger:
    """Simple hook-based activation magnitude logger for UNet attention layers"""

    def __init__(self, model, writer: SummaryWriter):
        self.writer = writer
        self.hooks = []
        self.activations = defaultdict(list)

        # Register hooks on attention layers
        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register forward hooks on attention-related layers"""
        for name, module in model.named_modules():
            # Target attention layers in UNet2DConditionModel
            if any(keyword in name.lower() for keyword in ["attn", "attention"]):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Hook to capture activations after linear/conv layers
                    hook = module.register_forward_hook(
                        lambda module, input, output, name=name: self._log_activation(
                            name, output
                        )
                    )
                    self.hooks.append(hook)

    def _log_activation(self, name, activation):
        """Store activation for logging"""
        if isinstance(activation, torch.Tensor):
            # Calculate magnitude statistics
            mean = activation.detach().abs().mean().item()
            min = activation.detach().min().item()
            max = activation.detach().max().item()

            self.activations[name].append((mean, min, max))

    def log_to_tensorboard(self, global_step):
        """Log accumulated activations to tensorboard"""
        if self.activations:
            for name, mags in self.activations.items():
                if mags:
                    for index, label in enumerate(['mean', 'min', 'max']):
                        values = [mag[index] for mag in mags]
                        avg = sum(values) / len(mags)
                        self.writer.add_scalar(
                            f"activations-{label}/{name}", avg, global_step=global_step
                        )

            # Clear accumulated activations
            self.activations.clear()


    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def get_attention_parameters(model, qk_only=True):
    """
    Get attention parameters for targeted weight decay

    Args:
        model: UNet2DConditionModel
        qk_only: If True, only return Q and K projection weights (not V, output)

    Returns:
        List of parameters that contribute to attention magnitude
    """
    attention_params = []

    for name, param in model.named_parameters():
        name_lower = name.lower()

        # Look for attention-related parameters
        if "attn" in name_lower or "attention" in name_lower:
            if qk_only:
                # Only Q and K projections affect attention magnitude
                if any(qk in name_lower for qk in ["to_q", "to_k", "query", "key"]):
                    attention_params.append(param)
            else:
                # All attention parameters
                if any(
                    proj in name_lower
                    for proj in [
                        "to_q",
                        "to_k",
                        "to_v",
                        "to_out",
                        "query",
                        "key",
                        "value",
                    ]
                ):
                    attention_params.append(param)

    return attention_params


def create_optimizer_with_targeted_decay(
    model,
    base_lr=1e-4,
    base_weight_decay=1e-4,
    attention_weight_decay=1e-2,
    qk_only=True,
):
    """
    Create optimizer with higher weight decay for attention parameters

    Args:
        model: Your UNet model
        base_lr: Learning rate for all parameters
        base_weight_decay: Weight decay for non-attention parameters
        attention_weight_decay: Higher weight decay for attention parameters
        qk_only: If True, only apply higher decay to Q/K projections

    Returns:
        torch.optim.AdamW optimizer with parameter groups
    """

    # Get attention parameters
    attention_params = get_attention_parameters(model, qk_only=qk_only)
    attention_param_ids = {id(p) for p in attention_params}

    # Separate parameters into groups
    attention_group = []
    regular_group = []

    for param in model.parameters():
        if param.requires_grad:
            if id(param) in attention_param_ids:
                attention_group.append(param)
            else:
                regular_group.append(param)

    # Create parameter groups
    param_groups = [
        {
            "params": regular_group,
            "lr": base_lr,
            "weight_decay": base_weight_decay,
            "name": "regular",
        },
        {
            "params": attention_group,
            "lr": base_lr,
            "weight_decay": attention_weight_decay,
            "name": "attention_high_decay",
        },
    ]

    print(f"Regular parameters: {len(regular_group)}")
    print(f"Attention parameters (high decay): {len(attention_group)}")

    return torch.optim.AdamW(param_groups)


# Example usage in your training loop:
"""
# Setup
model = UNet2DConditionModel.from_pretrained(...)
writer = SummaryWriter('runs/experiment')

# 1. Activation logging
activation_logger = ActivationLogger(model, writer, log_every=50)

# 2. Targeted weight decay optimizer
optimizer = create_optimizer_with_targeted_decay(
    model, 
    base_lr=1e-4,
    base_weight_decay=1e-4,
    attention_weight_decay=1e-2,  # 10x higher for attention
    qk_only=True  # Only Q/K projections get high decay
)

# Training loop
for step, batch in enumerate(dataloader):
    # Your forward pass
    loss = model(**batch).loss

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log activations
    activation_logger.log_to_tensorboard()

    # Log loss
    writer.add_scalar('loss/train', loss.item(), step)

# Cleanup
activation_logger.cleanup()
writer.close()
"""
