"""
Self-Flow representation learning for Flow Matching U-Net training.

Implements the self-distillation loop described in:
  "Self-Flow: Self-Supervised Feature Learning for Flow Matching Models"

The student U-Net is forward-passed with heterogeneously noised latents and
forced to predict semantically richer features of an EMA teacher U-Net that
receives a uniformly cleaner version of the same latents.

Extraction points for SD 2.1 (block_out_channels = [320, 640, 1280, 1280]):
  Student: down_blocks[1] output   -> [B, 640,  H/4, W/4]   (~30% depth)
  Teacher: up_blocks[0]  output   -> [B, 1280, H/4, W/4]   (~65% depth)
Both have identical spatial resolution, so only a 1x1 channel projection is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfFlowProjectionHead(nn.Module):
    """
    Lightweight 1×1 conv projecting student down_blocks[1] features to teacher
    up_blocks[0] channel space.  No spatial interpolation needed because both
    extraction points are at the same H/4 × W/4 resolution.

    Default channels match SD 2.1 (block_out_channels = [320, 640, 1280, 1280]):
      in_channels  = block_out_channels[1]  = 640
      out_channels = block_out_channels[-1] = 1280
    Pass the actual values from model.unet.config.block_out_channels if needed.
    """

    def __init__(self, in_channels: int = 640, out_channels: int = 1280):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def compute_self_flow_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    proj_head: SelfFlowProjectionHead,
) -> torch.Tensor:
    """
    Representation loss: negative mean cosine similarity between projected student
    features [B, Cs, H, W] and (detached) teacher features [B, Ct, H, W].

    Uses cosine similarity along the channel dimension (dim=1) to avoid the
    numerical instabilities of L1/L2 losses as feature norms grow over training.

    Returns a scalar tensor (the mean over all spatial tokens and batch entries).
    """
    # project student features to teacher channel space
    projected = proj_head(student_features.to(proj_head.proj.weight.dtype))  # [B, Ct, H, W]

    B, C, H, W = projected.shape

    student_flat = projected.view(B, C, -1)                                   # [B, C, H*W]
    teacher_flat = teacher_features.detach().view(B, C, -1).to(student_flat.dtype)  # [B, C, H*W]

    # cosine similarity along channel dim → [B, H*W]
    cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)

    return -cos_sim.mean()

