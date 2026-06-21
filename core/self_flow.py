"""
# Self-Flow representation learning for Flow Matching

Implements the self-distillation loop described in:
  "Self-Flow: Self-Supervised Feature Learning for Flow Matching Models"

The student model is forward-passed with heterogeneously noised latents and
forced to predict semantically richer features of an EMA teacher model that
receives a uniformly cleaner version of the same latents.

## DiT Implementation


## U-net Implementation
Four extraction-point arrangements are supported (SD 2.1 example shapes, 512px input):

  'shallow' (default):
    Student: down_blocks[1]                -> (B, boc[1],  H/4, W/4)  e.g. [B, 640,  16, 16]
    Teacher: up_blocks[0]                  -> (B, boc[-1], H/4, W/4)  e.g. [B, 1280, 16, 16]

  'deep':
    Student: down_blocks[2].attentions[-1] -> (B, boc[-1], H/4, W/4)  e.g. [B, 1280, 16, 16]  (pre-downsampler)
    Teacher: up_blocks[1].attentions[-1]   -> (B, boc[-1], H/4, W/4)  e.g. [B, 1280, 16, 16]  (pre-upsampler)

  'semantic':
    Student: down_blocks[2].attentions[-1] -> (B, boc[-1], H/4, W/4)  e.g. [B, 1280, 16, 16]  (pre-downsampler)
    Teacher: up_blocks[0]                  -> (B, boc[-1], H/4, W/4)  e.g. [B, 1280, 16, 16]  (post-upsampler)

  'detail':
    Student: down_blocks[1].attentions[-1] -> (B, boc[1],  H/2, W/2)  e.g. [B, 640,  32, 32]  (pre-downsampler)
    Teacher: up_blocks[1]                  -> (B, boc[-1], H/2, W/2)  e.g. [B, 1280, 32, 32]  (post-upsampler)

In all modes student and teacher share the same spatial resolution, so only a
1×1 channel projection is needed (no spatial interpolation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

SELF_FLOW_MODES = ['shallow', 'deep', 'semantic', 'detail']


def get_self_flow_channels(mode: str, boc: list) -> tuple[int, int]:
    """Return (student_channels, teacher_channels) for the given mode.

    boc = block_out_channels list from the UNet config, e.g. [320, 640, 1280, 1280].

    Modes 'shallow' and 'detail' tap into boc[1] (e.g. 640) on the student side.
    Modes 'deep' and 'semantic' tap into boc[-1] (e.g. 1280) on the student side.
    The teacher always produces boc[-1] channels.
    """
    if mode in ('shallow', 'detail'):
        return boc[1], boc[-1]
    elif mode in ('deep', 'semantic'):
        return boc[-1], boc[-1]
    else:
        raise ValueError(f"Unknown self_flow_mode: {mode!r}. Choose from: {SELF_FLOW_MODES}")


def get_self_flow_spatial_divisors(mode: str) -> tuple[int, int]:
    """Return (student_spatial_divisor, teacher_spatial_divisor) for the given mode.

    The divisors apply to the latent H and W to give the feature-map resolution:
      'shallow' / 'deep' / 'semantic' → both H/4 × W/4
      'detail'                         → both H/2 × W/2
    """
    if mode in ('shallow', 'deep', 'semantic'):
        return 4, 4
    elif mode == 'detail':
        return 2, 2
    else:
        raise ValueError(f"Unknown self_flow_mode: {mode!r}. Choose from: {SELF_FLOW_MODES}")


def get_self_flow_modules(student_module, teacher_module, mode: str):
    """Return (student_module, teacher_module) to register forward hooks on."""
    # SANA (DiT) support
    if hasattr(student_module, 'transformer_blocks'):
        # For a depth of 20: Layer 6 = index 5, Layer 14 = index 13
        return student_module.transformer_blocks[5], teacher_module.transformer_blocks[13]

    # Standard UNet support
    if mode == 'shallow':
        return student_module.down_blocks[1], teacher_module.up_blocks[0]
    elif mode == 'deep':
        return student_module.down_blocks[2].attentions[-1], teacher_module.up_blocks[1].attentions[-1]
    elif mode == 'semantic':
        return student_module.down_blocks[2].attentions[-1], teacher_module.up_blocks[0]
    elif mode == 'detail':
        return student_module.down_blocks[1].attentions[-1], teacher_module.up_blocks[1]
    else:
        raise ValueError(f"Unknown self_flow_mode: {mode!r}. Choose from: {SELF_FLOW_MODES}")


class SelfFlowMLPProjectionHead(nn.Module):
    """

    UNet implementation:

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
        # Use Linear so it naturally handles DiT (B, N, D) outputs
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels, bias=False)
        )
        nn.init.zeros_(self.proj[-1].weight)
        # Only zero the bias if it actually exists!
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4: # UNet spatial logic: (B, C, H, W)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.proj(x)
            return x.permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 3: # DiT sequence logic: (B, N, D)
            return self.proj(x)
        return x


def compute_self_flow_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    proj_head: SelfFlowMLPProjectionHead,
    debug_mask: torch.Tensor
) -> torch.Tensor:
    """
    Representation loss: negative mean cosine similarity between projected student
    features and (detached) teacher features.

    Uses cosine similarity along the channel dimension (dim=1) to avoid the
    numerical instabilities of L1/L2 losses as feature norms grow over training.
    projected = proj_head(student_features.to(proj_head.dtype))

    Returns a scalar tensor (the mean over all spatial tokens and batch entries).
    """
    projected = proj_head(student_features.to(proj_head.dtype))

    # Flatten spatial/sequence dims uniformly
    if projected.ndim == 4: # (B, C, H, W)
        B, C, H, W = projected.shape
        student_flat = projected.view(B, C, -1)
        teacher_flat = teacher_features.detach().view(B, C, -1)
    else: # (B, N, D)
        B, N, D = projected.shape
        student_flat = projected.transpose(1, 2) # (B, D, N)
        teacher_flat = teacher_features.detach().transpose(1, 2)

    teacher_flat = teacher_flat.to(student_flat.dtype)
    # --- Upcast to float32
    student_flat_fp32 = student_flat.to(torch.float32)
    teacher_flat_fp32 = teacher_flat.to(torch.float32)

    cos_sim = F.cosine_similarity(student_flat_fp32, teacher_flat_fp32, dim=1)

    # --- DIAGNOSTIC LOGGING ---
    # mask == True means the token was NOISY (Student had to guess it)
    # mask == False means the token was CLEAN (Student and Teacher saw the same thing)
    #with torch.no_grad():
    #    if debug_mask.any():
    #        hard_sim = cos_sim[debug_mask].mean().item()
    #        easy_sim = cos_sim[~debug_mask].mean().item()
    #        # Print this to your console/wandb!
    #        print(f"Self-Flow Sim -> Masked(Hard): {hard_sim:.3f} | Unmasked(Easy): {easy_sim:.3f}")

    return -cos_sim.mean(dim=1)
