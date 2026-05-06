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
    @property
    def dtype(self):
        return self.proj.weight.dtype

    def __init__(self, in_channels: int = 640, out_channels: int = 1280):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class SelfFlowMLPProjectionHead_Old(nn.Module):
    """
    MLP projecting student down_blocks[1] features to teacher
    up_blocks[0] channel space.  No spatial interpolation needed because both
    extraction points are at the same H/4 × W/4 resolution.

    Default channels match SD 2.1 (block_out_channels = [320, 640, 1280, 1280]):
      in_channels  = block_out_channels[1]  = 640
      out_channels = block_out_channels[-1] = 1280
    Pass the actual values from model.unet.config.block_out_channels if needed.

    Self-Flow is heavily based on REPA (which it cites as its baseline). In representation alignment and contrastive learning (like MoCo, BYOL, and REPA), it is an established best practice to use a non-linear projection head (a 2-layer MLP) rather than a strict linear projection.

    Because the Student feature is from a much shallower layer than the Teacher, the Student needs a bit of non-linear "flexibility" to translate its low-level spatial features into the Teacher's high-level semantic features. A strict linear projection can sometimes act as a bottleneck.

    You can implement this effortlessly by expanding your 1×1 conv into two 1×1 convs separated by an activation function (like SiLU/Swish, which SD 2.1 uses natively).
    """
    @property
    def dtype(self):
        return self.proj[0].weight.dtype

    def __init__(self, in_channels: int = 640, hidden_channels: int = 1280, out_channels: int = 1280):
        super().__init__()
        # 2-layer 1x1 Conv (acts exactly like an MLP over the channel dimension)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SelfFlowMLPProjectionHead(nn.Module):
    """
    Projects Student bottleneck features (e.g., mid_block, 8x8, 1280c)
    to match Teacher decoder features (e.g., up_blocks.1, 16x16, 1280c).
    """
    @property
    def dtype(self):
        return self.proj[0].weight.dtype

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 1280, out_channels: int = 1280):
        super().__init__()

        # 2-layer 1x1 Conv (acts as a non-linear MLP over channels)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor, target_spatial_shape: tuple) -> torch.Tensor:
        # 1. Project channels and apply non-linearity
        x = self.proj(x)

        # 2. Upsample spatial dimensions to match the Teacher
        # target_spatial_shape should be (16, 16) based on your nodes
        x = F.interpolate(x, size=target_spatial_shape, mode='bilinear', align_corners=False)

        return x


def compute_self_flow_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    proj_head: SelfFlowProjectionHead|SelfFlowMLPProjectionHead,
) -> torch.Tensor:
    """
    Representation loss: negative mean cosine similarity between projected student
    features [B, Cs, H, W] and (detached) teacher features [B, Ct, H, W].

    Uses cosine similarity along the channel dimension (dim=1) to avoid the
    numerical instabilities of L1/L2 losses as feature norms grow over training.

    Returns a scalar tensor (the mean over all spatial tokens and batch entries).
    """

    if type(proj_head) is SelfFlowProjectionHead or type(proj_head) is SelfFlowMLPProjectionHead_Old:
        # project student features to teacher channel space
        projected = proj_head(student_features.to(proj_head.dtype))  # [B, Ct, H, W]

    elif type(proj_head) is SelfFlowMLPProjectionHead:
        # Project student to match teacher
        target_shape = teacher_features.shape[2:]  # gets (16, 16)
        projected = proj_head(student_features, target_shape)
    else:
        raise NotImplementedError(f"Missing logic for proj head type {type(proj_head)}")

    B, C, H, W = projected.shape

    student_flat = projected.view(B, C, -1)
    teacher_flat = teacher_features.detach().view(B, C, -1).to(student_flat.dtype)  # [B, C, H*W]

    # cosine similarity along channel dim → [B, H*W]
    cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)

    return -cos_sim.mean()
