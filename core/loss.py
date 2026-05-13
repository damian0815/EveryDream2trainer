import logging

import line_profiler
import math
import random
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torchvision
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.cuda.amp import autocast
import torch.nn.functional as F

from scipy.stats import beta as sp_beta

from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from core.loss_softrepa import (
    text_weighted_infonce_loss,
    text_weighted_infonce_loss_with_softrepa,
)
from core.self_flow import get_self_flow_modules, get_self_flow_channels, get_self_flow_spatial_divisors
from model.training_model import TrainingModel, Conditioning

# from train import pyramid_noise_like, compute_snr


"""
alpha=2-4: Slow advance (spread hugs initial_value)
alpha=1: Linear progression
alpha<1: Quick advance (spread hugs final_value)
"""


def get_latents(image, model: TrainingModel, device, args):
    with torch.no_grad():
        with autocast(
            enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16
        ):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(
                device, dtype=model.vae.dtype
            )
            latents = model.vae.encode(pixel_values, return_dict=False)
        del pixel_values
        scaling_factor = 0.13025 if model.is_sdxl else 0.18215
        latents = latents[0].sample() * scaling_factor
        return latents


def get_loss(
    model_pred,
    target,
    is_cond_dropout,
    mask_img,
    timesteps,
    negative_loss_mask,
    noise_scheduler,
    prompt_embeds: torch.Tensor,
    do_contrastive_learning,
    contrastive_loss_scale,
    args,
    verbose=False,
):
    # logging.info(f"get_loss timesteps: {timesteps.detach().cpu().tolist()}")
    device = model_pred.device

    non_contrastive_loss = compute_basic_loss(args.loss_type, model_pred, target, timesteps, noise_scheduler)
    non_contrastive_loss *= get_timestep_weight(timesteps,
                                                loss_shape=non_contrastive_loss.shape,
                                                min_snr_gamma=args.min_snr_gamma,
                                                loss_mode_scale=args.loss_mode_scale,
                                                noise_scheduler=noise_scheduler,
                                                ).to(device)

    num_valid_contrastive_samples = (~is_cond_dropout.cpu() & ~negative_loss_mask.cpu()).sum()
    if not do_contrastive_learning or num_valid_contrastive_samples <= 1:
        return _apply_mask_image(non_contrastive_loss, mask_img, both_sides=args.use_both_mask_sides_contrastive, hinge_negative_margin=args.negative_loss_margin)

    assert negative_loss_mask.sum() == 0, "negative loss not implemented"

    def contrastive_loss_delta(model_pred, target, is_cond_dropout):
        """
        Delta-based contrastive loss: penalize when prediction deltas
        match target deltas for different captions.

        Args:
            model_pred: [B, C, H, W] - model predictions
            target: [B, C, H, W] - ground truth targets
            is_cond_dropout: [B] - boolean mask, True where conditioning was dropped

        Returns:
            per_sample_loss: [B] - contrastive loss per sample (0 for dropped samples)
        """
        B = model_pred.shape[0]
        device = model_pred.device

        # Identify valid samples (not dropped)
        valid_mask = ~is_cond_dropout

        # Initialize per-sample losses
        losses = torch.zeros(B, device=device)
        neg_counts = torch.zeros(B, device=device)

        # For each sample i
        for i in range(B):
            if not valid_mask[i]:
                continue

            # For each other sample j (negative)
            for j in range(B):
                if i == j or not valid_mask[j]:
                    continue

                # Compute delta between predictions
                pred_delta = model_pred[j] - model_pred[i]
                target_delta = target[j] - target[i]

                # Loss: how much does pred_delta match target_delta?
                # We want them to NOT match (higher loss = worse)
                delta_loss = F.mse_loss(pred_delta, target_delta, reduction="mean")

                losses[i] += delta_loss
                neg_counts[i] += 1

        # Average over number of negatives (avoid batch size dependence)
        neg_counts = torch.clamp(neg_counts, min=1.0)  # Avoid division by zero
        losses = losses / neg_counts

        return losses

    def contrastive_loss_infonce(model_pred, target, is_cond_dropout, timesteps):
        [B_full, C, H, W] = model_pred.shape
        valid_mask = ~is_cond_dropout
        pred_valid = model_pred[valid_mask]
        target_valid = target[valid_mask]

        # Contrastive loss
        # Flatten spatial dimensions for similarity computation
        pred_flat = pred_valid.reshape(num_valid_contrastive_samples, -1)  # [B, C*H*W]
        target_flat = target_valid.reshape(
            num_valid_contrastive_samples, -1
        )  # [B, C*H*W]

        # Compute similarity matrix: how similar is each prediction to each target
        # Using negative L2 distance as similarity (could also use cosine)
        # sim[i, j] = similarity between prediction_i and target_j
        dist_matrix = torch.cdist(pred_flat, target_flat, p=2)  # [B, B]
        # Normalize by the dimensionality to get reasonable scale
        dist_matrix = dist_matrix / (C * H * W) ** 0.5
        sim_matrix = -dist_matrix / args.contrastive_loss_temperature

        # Alternative: cosine similarity
        # pred_norm = F.normalize(pred_flat, p=2, dim=1)
        # target_norm = F.normalize(target_flat, p=2, dim=1)
        # sim_matrix = torch.mm(pred_norm, target_norm.t())  # [B, B]

        # InfoNCE: diagonal should be high, off-diagonal should be low
        # Loss for each sample: -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
        labels = torch.arange(num_valid_contrastive_samples, device=model_pred.device)
        contrastive_loss_valid = F.cross_entropy(
            sim_matrix, labels, reduction="none"
        ) / math.sqrt(num_valid_contrastive_samples)

        # expand to cover cond dropout samples
        contrastive_loss_full = torch.zeros(B_full, device=model_pred.device)
        contrastive_loss_full[valid_mask] = contrastive_loss_valid

        del contrastive_loss_valid

        return contrastive_loss_full

    with torch.autocast("cuda"):
        if args.contrastive_loss_type == "infonce_with_text_similarity":
            contrastive_loss = text_weighted_infonce_loss(
                model_pred,
                target,
                text_embeddings=prompt_embeds,
                is_dropped=is_cond_dropout,
                temperature=args.contrastive_loss_temperature,
                hard_negative_weight=args.contrastive_loss_hard_negative_weight,
                use_text_similarity=True,
            )
        elif args.contrastive_loss_type == "infonce_softrepa":
            contrastive_loss = text_weighted_infonce_loss_with_softrepa(
                model_pred,
                target,
                text_embeddings=prompt_embeds,
                is_dropped=is_cond_dropout,
                temperature=None,
                sigma=args.contrastive_loss_softrepa_sigma,
                hard_negative_weight=args.contrastive_loss_hard_negative_weight,
                similarity_method="softrepa",
            )
        elif args.contrastive_loss_type == "infonce":
            contrastive_loss = contrastive_loss_infonce(
                model_pred, target, is_cond_dropout, timesteps
            )
        elif args.contrastive_loss_type == "delta":
            contrastive_loss = contrastive_loss_delta(
                model_pred, target, is_cond_dropout
            )
        else:
            raise ValueError(
                f"Unrecognized contrastive loss type: {args.contrastive_loss_type}"
            )

    contrastive_loss = contrastive_loss.view(-1, 1, 1, 1).expand_as(
        non_contrastive_loss
    )
    contrastive_loss *= get_timestep_weight(timesteps,
                                            loss_shape=contrastive_loss.shape,
                                            min_snr_gamma=args.min_snr_gamma,
                                            loss_mode_scale=args.loss_mode_scale,
                                            noise_scheduler=noise_scheduler,
                                            ).to(device)

    # Combined loss
    if verbose:
        logging.info(
            f"verbose loss logging:\n"
            f" - timestep: {timesteps.detach().cpu().tolist()}\n"
            f" - contrastive loss: {contrastive_loss.mean(dim=(1, 2, 3)).detach().cpu().tolist()} mean: {contrastive_loss.mean()}\n"
            f" - non-contrastive loss: {non_contrastive_loss.mean(dim=(1, 2, 3)).detach().cpu().tolist()} mean: {non_contrastive_loss.mean()}\n"
        )
    total_loss = _apply_mask_image(
        non_contrastive_loss + contrastive_loss * contrastive_loss_scale
    , mask_img, both_sides=args.use_both_mask_sides_contrastive, hinge_negative_margin=args.negative_loss_margin)
    return total_loss


def _apply_mask_image(loss: torch.Tensor, mask_img: torch.Tensor|None, both_sides: bool=False, hinge_negative_margin=None) -> torch.Tensor:
    if mask_img is None:
        return loss
    # expand from single channel to RGB
    mask_img = mask_img.repeat(1, loss.shape[1], 1, 1).to(loss.device)
    if both_sides:
        #print("applying both sides mask with hinge negative loss")
        negative_loss = apply_negative_loss_hinge(loss, mask=torch.ones_like(loss).bool(), margin=hinge_negative_margin)
        return loss * mask_img + negative_loss * (1 - mask_img)
    else:
        return loss * mask_img


def compute_basic_loss(
    loss_type: str,
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.LongTensor,
    noise_scheduler: SchedulerMixin|ConfigMixin,
):
    device = model_pred.device

    if 'huber' in loss_type:
        early_timestep_bias = timesteps / noise_scheduler.config.num_train_timesteps
        early_timestep_bias = early_timestep_bias.float().to(device)
        early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(model_pred)
        loss_huber = F.huber_loss(
            model_pred.float(), target.float(), reduction="none", delta=1.0
        )
        loss_mse = F.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        if loss_type == "mse_huber":
            loss_mse = loss_mse * early_timestep_bias
            loss_huber = (
                loss_huber * (1.0 - early_timestep_bias)
            )
            return loss_mse + loss_huber
        elif loss_type == "huber_mse":
            loss_mse = loss_mse * (1.0 - early_timestep_bias)
            loss_huber = loss_huber * early_timestep_bias
            return loss_huber + loss_mse
        elif loss_type == "huber":
            return loss_huber
        else:
            raise ValueError(f"unhandled loss type {loss_type}")

    elif loss_type.startswith("sd3-"):
        # SD3 loss weight
        if not isinstance(noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
            raise NotImplementedError("SD3 loss weighting only implemented for TrainFlowMatchEulerDiscreteScheduler")
        sigmas = noise_scheduler.get_sigmas_for_timesteps(timesteps)
        if loss_type == "sd3-cosmap":
            weighting_scheme = "cosmap"
        elif loss_type == "sd3-sigma_sqrt":
            weighting_scheme = "sigma_sqrt"
        else:
            raise ValueError(f"unhandled loss type {loss_type}")
        weights = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
        loss_mse = F.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        return (
            loss_mse
            * weights.view(-1, 1, 1, 1).to(loss_mse.device)
        )
    elif loss_type == "cosmap-2":
        lmbd = 0.9
        sigmas = timesteps / 1000
        weights = 1 - lmbd * (
            torch.exp(-10 * (sigmas - 0) ** 2) + torch.exp(-10 * (sigmas - 1) ** 2)
        )
        loss_mse = F.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        return (
            loss_mse
            * weights.view(-1, 1, 1, 1).to(loss_mse.device)
        )
    else:
        if loss_type != "mse":
            raise ValueError(f"Unrecognized --loss_type {loss_type}")
        loss_mse = F.mse_loss(
            model_pred.float(), target.float(), reduction="none"
        )
        return loss_mse


def compute_saturation_penalty_loss(
    model_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    t_max: float = 200.0,
    mean_penalty_scale: float = 1.0,
    var_penalty_scale: float = 0.0,
    var_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Penalises all-black or all-white/washed-out predictions in flow matching.

    The model predicts velocity v ≈ noise - x₀.  For flow matching the noisy
    latent is  z_t = (1-t̄)*x₀ + t̄*noise  where t̄ = t / T ∈ [0,1].
    Inverting gives the denoised estimate:
        x̂₀ = (z_t - t̄ · v) / (1 - t̄)   [denominator clamped ≥ 1e-3]

    Empirically (VAE latent space):
      - All-black images produce large positive spatial means in some channels
        (e.g. +4.8, +3.0).
      - All-white images produce even larger spatial means (+11.3, +7.3).
      - Natural images produce near-zero means on average across the dataset.
    So penalising (spatial_mean)² reliably fires on both collapse modes and
    is essentially free on well-distributed natural image batches.

    The variance-collapse term is *disabled by default* (var_penalty_scale=0):
    empirical tests show that black images can have *high* latent variance in
    some channels (e.g. channel 2 variance ~12 for black vs ~9 for random),
    making it an unreliable signal. Enable it only if you observe true
    low-variance collapse in your latent space.

    Both terms are only active for t < t_max (low-noise regime) where x̂₀ is
    a reliable estimate of the clean image.

    Args:
        model_pred:         [B, C, H, W] velocity prediction (grad flows through)
        noisy_latents:      [B, C, H, W] z_t  (detached, no grad required)
        timesteps:          [B] integer timesteps in [0, num_train_timesteps]
        noise_scheduler:    scheduler with .config.num_train_timesteps
        t_max:              only apply to timesteps < t_max  (default 200 / 1000)
        mean_penalty_scale: weight for the (spatial_mean)² term  (default 1.0)
        var_penalty_scale:  weight for the variance-collapse hinge (default 0.0)
        var_threshold:      variance below which the hinge fires   (default 0.1)

    Returns:
        per_sample_loss: [B] — zeros for samples with timestep ≥ t_max
    """

    if type(noise_scheduler) is not TrainFlowMatchEulerDiscreteScheduler:
        raise ValueError("Saturation penalty loss only implemented for Flow Matching")

    B = model_pred.shape[0]
    device = model_pred.device
    T = noise_scheduler.config.num_train_timesteps

    # Normalised time t̄ ∈ [0, 1]: 0 = clean, 1 = pure noise
    t_bar = (timesteps.float() / T).view(B, 1, 1, 1).to(device)          # [B,1,1,1]

    with torch.no_grad():
        noisy_latents_f = noisy_latents.float().to(device)

    # Reconstruct x̂₀ = (z_t - t̄ · v) / (1 - t̄)
    denom = (1.0 - t_bar).clamp(min=1e-3)
    x0_hat = (noisy_latents_f - t_bar * model_pred.float()) / denom      # [B, C, H, W]

    # Spatial mean per sample per channel — large magnitude = degenerate output
    spatial_mean = x0_hat.mean(dim=(2, 3))                                # [B, C]
    mean_penalty = (spatial_mean ** 2).mean(dim=1)                        # [B]

    penalty = mean_penalty_scale * mean_penalty

    if var_penalty_scale > 0:
        spatial_var = x0_hat.var(dim=(2, 3), unbiased=False)              # [B, C]
        var_penalty = F.relu(var_threshold - spatial_var).mean(dim=1)     # [B]
        penalty = penalty + var_penalty_scale * var_penalty

    # Zero out samples in the high-noise regime
    active_mask = (timesteps.float() < t_max).to(device)                  # [B]
    penalty = penalty * active_mask

    return penalty


def get_timestep_weight(
    timesteps: torch.Tensor, loss_shape, min_snr_gamma, loss_mode_scale, noise_scheduler
) -> torch.Tensor:
    if min_snr_gamma is not None and min_snr_gamma > 0:
        if type(noise_scheduler) is TrainFlowMatchEulerDiscreteScheduler:
            t = timesteps / noise_scheduler.config.num_train_timesteps
            snr = (1 - t) / (t + 1e-8)  # Linear approximation
            snr_weight = torch.minimum(snr, torch.tensor(min_snr_gamma)) / (
                snr + 1e-8
            )
        else:
            snr = compute_snr(timesteps, noise_scheduler)
            v_pred = noise_scheduler.config.prediction_type in [
                "v_prediction",
                "v-prediction",
            ]
            divisor = (snr + 1) if v_pred else snr
            snr_weight = (
                torch.stack(
                    [snr, min_snr_gamma * torch.ones_like(timesteps).float()],
                    dim=1,
                ).min(dim=1)[0]
                / divisor
            )
        return snr_weight.view(-1, 1, 1, 1).expand(loss_shape)
    elif loss_mode_scale > 0:
        # Scale loss to emphasize middle timesteps (mode around 500)
        # loss_mode_scale=0 means uniform, higher values increase the peak
        t_normalized = timesteps.float() / 1000.0  # normalize to [0, 1]
        sharpness = 3
        mode_weight = (
            torch.cos(math.pi * (t_normalized - 0.5)) ** sharpness + 1
        ) - 1
        mode_weight = (
            1 - loss_mode_scale
        ) + loss_mode_scale * mode_weight
        return mode_weight.view(-1, 1, 1, 1).expand(loss_shape)
    else:
        return torch.tensor(1)


def get_midblock_out_shape(latents_shape: torch.Size, model: TrainingModel) -> torch.Size:
    return (latents_shape[0], model.unet.mid_block.out_channels, latents_shape[2]//8, latents_shape[3]//8)


def get_self_flow_shapes(latents_shape: torch.Size, model: TrainingModel, mode: str = 'shallow'):
    """Return (student_shape, teacher_shape) for self-flow feature tensors.

    Shapes depend on the extraction-point mode:
      'shallow' / 'deep' / 'semantic' → H/4 × W/4 spatial resolution
      'detail'                         → H/2 × W/2 spatial resolution
    """
    B = latents_shape[0]
    boc = model.unet.config.block_out_channels
    student_ch, teacher_ch = get_self_flow_channels(mode, boc)
    s_div, t_div = get_self_flow_spatial_divisors(mode)
    Hs, Ws = latents_shape[2] // s_div, latents_shape[3] // s_div
    Ht, Wt = latents_shape[2] // t_div, latents_shape[3] // t_div
    return (B, student_ch, Hs, Ws), (B, teacher_ch, Ht, Wt)


def build_self_flow_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    t: torch.Tensor,
    s: torch.Tensor,
    mask_ratio: float,
):
    """Build heterogeneously noised student latents and uniformly cleaner teacher latents.
    Student  x_τ    : mask_ratio of spatial locations use noise level s, rest use t.
    Teacher  x_{τ_min}: uniformly noised at τ_min = min(t, s) per sample.
    Returns (x_tau, x_tau_min, tau_min_timesteps).
    """
    B, C, H, W = latents.shape
    device = latents.device
    x_t = _get_noisy_latents(latents, noise, noise_scheduler, t, latents_perturbation=0)
    x_s = _get_noisy_latents(latents, noise, noise_scheduler, s, latents_perturbation=0)
    M = (torch.rand(B, 1, H, W, device=device) < mask_ratio).float()
    x_tau = (1.0 - M) * x_t + M * x_s
    tau_min = torch.minimum(t, s)
    x_tau_min = _get_noisy_latents(latents, noise, noise_scheduler, tau_min, latents_perturbation=0)
    return x_tau, x_tau_min, tau_min


@dataclass
class ModelPredictionAndTargetReturnType:
    model_pred: torch.Tensor
    target: torch.Tensor
    noisy_latents: torch.Tensor
    midblock_out: torch.Tensor|None = None
    teacher_target: torch.Tensor|None = None
    self_flow_student_features: torch.Tensor|None = None  # [B, Cs, H/4, W/4]
    self_flow_teacher_features: torch.Tensor|None = None  # [B, Ct, H/4, W/4]


@line_profiler.profile
def get_model_prediction_and_target(
    latents,
    conditioning: Conditioning,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    model: TrainingModel,
    args=None,
    is_cond_dropout_noise: torch.Tensor = None,
    skip_contrastive: bool = False,
    teacher_model: TrainingModel | None = None,
    teacher_mask: torch.Tensor | None = None,
    teacher_conditioning: Conditioning | None = None,
    debug_fake: bool = False,
    log_writer=None,
    global_step: int = 0,
    mask=None,
    lcf_mask=None,
    self_flow_s_timesteps: torch.Tensor | None = None,
) -> ModelPredictionAndTargetReturnType:
    """
    If mask is provided, only compute for the masked entries and return full tensors with zeros elsewhere.
    Returns model_pred, target, teacher_target (None if teacher_mask is None), noisy_latents
    """
    midblock_out_shape = get_midblock_out_shape(latents.shape, model)

    if mask is not None:
        # create full tensors
        model_pred = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        target = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        # When teacher_mask is None no separate teacher_target tensor is needed.
        teacher_target = None if teacher_mask is None else torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        noisy_latents = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        midblock_out = None if lcf_mask is None else torch.zeros(midblock_out_shape, dtype=model.unet.dtype, device=model.unet.device)

        # self-flow feature placeholders
        do_self_flow = (self_flow_s_timesteps is not None
                  and model.self_flow_teacher_unet is not None)
        if do_self_flow:
            sf_mode = getattr(args, 'self_flow_mode', 'shallow')
            sf_student_shape, sf_teacher_shape = get_self_flow_shapes(latents.shape, model, sf_mode)
            self_flow_student_features = torch.zeros(sf_student_shape, dtype=model.unet.dtype, device=model.unet.device)
            self_flow_teacher_features = torch.zeros(sf_teacher_shape, dtype=model.unet.dtype, device=model.unet.device)
        else:
            self_flow_student_features = None
            self_flow_teacher_features = None

        if mask.sum() == 0:
            # early out for empty mask
            return ModelPredictionAndTargetReturnType(
                model_pred=model_pred,
                target=target,
                teacher_target=teacher_target,
                noisy_latents=noisy_latents,
                midblock_out=midblock_out,
                self_flow_student_features=self_flow_student_features,
                self_flow_teacher_features=self_flow_teacher_features,
            )

        latents_masked = latents[mask]
        conditioning_masked = conditioning.get_masked(mask)
        noise_masked = noise[mask]
        teacher_mask_masked = teacher_mask[mask] if teacher_mask is not None else None
        teacher_conditioning_masked = (
            teacher_conditioning.get_masked(mask)
            if teacher_conditioning is not None
            else None
        )
        lcf_mask_masked = lcf_mask[mask] if lcf_mask is not None else None
        timesteps_masked = timesteps[mask]
        self_flow_s_timesteps_masked = self_flow_s_timesteps[mask] if self_flow_s_timesteps is not None else None
        masked_result = get_model_prediction_and_target(
            latents=latents_masked,
            conditioning=conditioning_masked,
            noise=noise_masked,
            timesteps=timesteps_masked,
            model=model,
            args=args,
            skip_contrastive=skip_contrastive,
            teacher_model=teacher_model,
            teacher_mask=teacher_mask_masked,
            teacher_conditioning=teacher_conditioning_masked,
            debug_fake=debug_fake,
            log_writer=log_writer,
            global_step=global_step,
            mask=None,
            lcf_mask=lcf_mask_masked,
            self_flow_s_timesteps=self_flow_s_timesteps_masked,
        )
        model_pred[mask] += masked_result.model_pred
        target[mask] += masked_result.target
        if teacher_target is not None and masked_result.teacher_target is not None:
            teacher_target[mask] += masked_result.teacher_target
        if lcf_mask is not None:
            midblock_out[mask] += masked_result.midblock_out
        if do_self_flow and masked_result.self_flow_student_features is not None:
            self_flow_student_features[mask] += masked_result.self_flow_student_features
            self_flow_teacher_features[mask] += masked_result.self_flow_teacher_features
        noisy_latents[mask] += masked_result.noisy_latents
        return ModelPredictionAndTargetReturnType(
            model_pred=model_pred,
            target=target,
            noisy_latents=noisy_latents,
            midblock_out=midblock_out,
            teacher_target=teacher_target,
            self_flow_student_features=self_flow_student_features,
            self_flow_teacher_features=self_flow_teacher_features,
        )

    if is_cond_dropout_noise is not None:
        # apply cond dropout noise probability: for samples where conditioning is dropped, replace latents with pure noise with some probability
        for sample_index in range(latents.shape[0]):
            if is_cond_dropout_noise[sample_index]:
                latents[sample_index] = torch.randn_like(latents[sample_index])


    # logging.info(f"get_model_prediction_and_target timesteps: {timesteps.detach().cpu().tolist()}")
    noisy_latents = _get_noisy_latents(
        latents, noise, model.noise_scheduler, timesteps, args.latents_perturbation
    )
    target = _get_target(latents, noise, model.noise_scheduler, timesteps).to(dtype=model.unet.dtype)

    midblock_out = None
    if debug_fake:
        model_pred = torch.ones_like(target).to(model.device)
    else:
        lcf_storage = {'midblock_out': None}
        handle = None
        try:
            # hook mid-block output for LCF
            if lcf_mask is None:
                handle = None
            else:
                def hook_fn(module, input, output):
                    assert output.shape == midblock_out_shape, f"Expected midblock output shape {midblock_out_shape} but got {output.shape}"
                    lcf_storage['midblock_out'] = output
                handle = model.unet.mid_block.register_forward_hook(hook_fn)

            model_pred = model.unet(
                noisy_latents.to(dtype=model.unet.dtype),
                timesteps.to(model.unet.device, dtype=model.unet.dtype),
                encoder_hidden_states=conditioning.prompt_embeds.to(dtype=model.unet.dtype),
                added_cond_kwargs=(
                    conditioning.get_added_cond_kwargs(dtype=model.unet.dtype)
                    if model.is_sdxl
                    else None
                ),
            ).sample
            midblock_out = None if lcf_mask is None else lcf_storage['midblock_out']
            del lcf_storage
        finally:
            if handle is not None:
                handle.remove()

    teacher_target = None
    if teacher_mask is None:
        # No supplementary teacher_target needed: no teacher guidance this step.
        pass
    elif teacher_mask.sum() == 0:
        teacher_target = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
    else:
        with torch.no_grad():
            if teacher_conditioning is None:
                teacher_conditioning = conditioning

            teacher_target = get_teacher_target(
                teacher_model=teacher_model,
                teacher_conditioning=teacher_conditioning,
                student_model=model,
                timesteps=timesteps,
                student_unet_timesteps=timesteps,
                clean_image_latents=latents,
                noise=noise,
            )


    # ---- Self-Flow representation learning ----
    self_flow_student_features = None
    self_flow_teacher_features = None
    sf_teacher_unet = model.self_flow_teacher_unet
    if self_flow_s_timesteps is not None and sf_teacher_unet is not None:
        sf_mask_ratio = args.self_flow_mask_ratio
        sf_mode = getattr(args, 'self_flow_mode', 'shallow')

        # Build heterogeneously noised latents for student and teacher
        x_tau, x_tau_min, tau_min_ts = build_self_flow_latents(
            latents=latents,
            noise=noise,
            noise_scheduler=model.noise_scheduler,
            t=timesteps,
            s=self_flow_s_timesteps,
            mask_ratio=sf_mask_ratio,
        )

        # Resolve the modules to hook for this mode
        sf_student_module, sf_teacher_module = get_self_flow_modules(model.unet, sf_teacher_unet, sf_mode)

        # --- Student forward: x_τ input, scalar timestep t, capture student module ---
        sf_student_storage = {}

        def _sf_student_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            sf_student_storage['h'] = out

        sf_student_handle = sf_student_module.register_forward_hook(_sf_student_hook)
        try:
            model.unet(
                x_tau.to(dtype=model.unet.dtype),
                timesteps.to(model.unet.device, dtype=model.unet.dtype),  # use original t
                encoder_hidden_states=conditioning.prompt_embeds.to(dtype=model.unet.dtype),
                added_cond_kwargs=(
                    conditioning.get_added_cond_kwargs(dtype=model.unet.dtype)
                    if model.is_sdxl else None
                ),
            )
        finally:
            sf_student_handle.remove()
        self_flow_student_features = sf_student_storage.get('h')

        # --- Teacher forward: x_{τ_min} input, scalar τ_min, capture teacher module ---
        sf_teacher_storage = {}

        def _sf_teacher_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            sf_teacher_storage['h'] = out

        sf_teacher_handle = sf_teacher_module.register_forward_hook(_sf_teacher_hook)
        try:
            with torch.no_grad():
                sf_teacher_unet(
                    x_tau_min.to(dtype=sf_teacher_unet.dtype),
                    tau_min_ts.to(sf_teacher_unet.device, dtype=sf_teacher_unet.dtype),
                    encoder_hidden_states=conditioning.prompt_embeds.to(dtype=sf_teacher_unet.dtype),
                    added_cond_kwargs=(
                        conditioning.get_added_cond_kwargs(dtype=sf_teacher_unet.dtype)
                        if model.is_sdxl else None
                    ),
                )
        finally:
            sf_teacher_handle.remove()
        self_flow_teacher_features = sf_teacher_storage.get('h')

    return ModelPredictionAndTargetReturnType(
        model_pred=model_pred, target=target, noisy_latents=noisy_latents,
        teacher_target=teacher_target, midblock_out=midblock_out,
        self_flow_student_features=self_flow_student_features,
        self_flow_teacher_features=self_flow_teacher_features,
    )


def _get_noisy_latents(
    latents, noise, noise_scheduler, timesteps, latents_perturbation
):
    # logging.info(f"get_noisy_latents timesteps: {timesteps.detach().cpu().tolist()}")
    if hasattr(noise_scheduler, "add_noise"):
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    elif isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
        noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)
    else:
        raise RuntimeError("Noise scheduler has no method to add noise to latents (tried .add_noise() and .scale_noise())")
    if latents_perturbation > 0:
        noisy_latents += torch.randn_like(noisy_latents) * latents_perturbation
    return noisy_latents


def _get_target(latents, noise, noise_scheduler, timesteps):
    # logging.info(f"get_target timesteps: {timesteps.detach().cpu().tolist()}")
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    elif noise_scheduler.config.prediction_type in ["flow-matching", "flow_prediction"]:
        target = noise - latents
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )
    return target


def _get_noisy_latents_and_target(
    latents,
    noise,
    noise_scheduler: SchedulerMixin|ConfigMixin,
    timesteps,
    latents_perturbation,
):
    noisy_latents = _get_noisy_latents(
        latents, noise, noise_scheduler, timesteps, latents_perturbation
    )
    target = _get_target(latents, noise, noise_scheduler, timesteps)

    return noisy_latents, target



def get_noise(
    latents_shape,
    device,
    dtype,
    pyramid_noise_discount,
    zero_frequency_noise_ratio,
    batch_share_noise,
):
    noise = torch.randn(latents_shape, dtype=dtype, device=device)
    if pyramid_noise_discount != None:
        if 0 < pyramid_noise_discount:
            noise = pyramid_noise_like(noise, discount=pyramid_noise_discount)
    if zero_frequency_noise_ratio != None:
        if zero_frequency_noise_ratio < 0:
            zero_frequency_noise_ratio = 0

        # see https://www.crosslabs.org//blog/diffusion-with-offset-noise
        zero_frequency_noise = zero_frequency_noise_ratio * torch.randn(
            latents_shape[0], latents_shape[1], 1, 1, device=device
        )
        noise = noise + zero_frequency_noise

    if batch_share_noise:
        noise = noise[:1].repeat((noise.shape[0], 1, 1, 1))

    return noise


def pyramid_noise_like(x, discount=0.8):
    b, c, w, h = (
        x.shape
    )  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    minimal_value = 1e-9
    alphas_cumprod = noise_scheduler.alphas_cumprod
    # Use .any() to check if any elements in the tensor are zero
    if (alphas_cumprod[:-1] == 0).any():
        logging.warning(
            f"Alphas cumprod has zero elements! Resetting to {minimal_value}.."
        )
        alphas_cumprod[alphas_cumprod[:-1] == 0] = minimal_value
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR, first without epsilon
    snr = (alpha / sigma) ** 2
    snr[snr < minimal_value] = minimal_value
    return snr


def snr_based_clustering(snr: "np.ndarray", k: int) -> list[tuple[int, int]]:
    """
    cf Hyojun Go et al [Neurips 2023] Addressing Negative Transfer in Diffusion Models
    https://gohyojun15.github.io/ANT_diffusion/

    Dynamic programming partition of [0, len(snr)-1] into k contiguous intervals
    minimising within-interval SNR heterogeneity.

    Cost of interval [left, right]: sum of |snr[t] - snr[center]| for t in [left, right].
    Returns k (left, right) inclusive index pairs (relative to the input snr array).
    """
    import numpy as np
    n = snr.shape[0]
    D = np.full((n, k), np.inf)   # D[i,j] = min cost to cover [0..i] with j+1 clusters
    S = np.zeros((n, k), dtype=int)

    def interval_cost(left, right):
        if left == right:
            return 0.0
        center = round((left + right + 1) / 2)
        return float(np.abs(snr[left:right + 1] - snr[center]).sum())

    for j in range(k):
        for i in reversed(range(n)):
            if j == 0:
                D[i, j] = interval_cost(0, i)
            elif i >= j:
                costs = np.full(i, np.inf)
                for L in range(j, i):
                    costs[L] = D[L, j - 1] + interval_cost(L + 1, i)
                D[i, j] = costs.min()
                S[i, j] = int(costs.argmin())

    # Backtrack to recover split boundaries
    bounds = []
    b = 0
    for j in reversed(range(k)):
        b = S[-1, k - 1] if j == k - 1 else S[int(b), j]
        bounds.append(int(b))

    # Build (left, right) inclusive index pairs
    clusters = []
    reversed_bounds = list(reversed(bounds))
    for idx in range(k):
        left  = reversed_bounds[idx]
        right = reversed_bounds[idx + 1] - 1 if idx + 1 < k else n - 1
        clusters.append((int(left), int(right)))
    return clusters


def compute_timestep_intervals(noise_scheduler, k: int,
                               t_start: int, t_end: int) -> list[tuple[int, int]]:
    """
    Compute k SNR-homogeneous contiguous intervals over [t_start, t_end] using DP.
    Works for both flow-matching and DDPM/v-pred schedulers.
    Returns absolute timestep (left, right) inclusive pairs.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler

    all_t = torch.arange(t_start, t_end)
    if isinstance(noise_scheduler, (TrainFlowMatchEulerDiscreteScheduler,
                                    FlowMatchEulerDiscreteScheduler)):
        # SNR(t) = t² / (1−t)²  for linear flow-matching schedule
        t_norm = all_t.float() / noise_scheduler.config.num_train_timesteps
        snr = (t_norm ** 2 / (1.0 - t_norm + 1e-8) ** 2).numpy()
    else:
        snr = compute_snr(all_t, noise_scheduler).numpy()

    clusters = snr_based_clustering(snr, k)
    # clusters are relative indices into all_t — shift to absolute timestep values
    return [(t_start + l, t_start + r) for l, r in clusters]


def vae_preview(latents: torch.Tensor) -> torch.Tensor:
    # adapted from https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/app/util/step_callback.py
    SD1_5_LATENT_RGB_FACTORS = torch.tensor(
        [
            #    R        G        B
            [0.3444, 0.1385, 0.0670],  # L1
            [0.1247, 0.4027, 0.1494],  # L2
            [-0.3192, 0.2513, 0.2103],  # L3
            [-0.1307, -0.1874, -0.7445],  # L4
        ]
    )

    def sample_to_lowres_estimated_image(
        samples: torch.Tensor,
        latent_rgb_factors: torch.Tensor,
        smooth_matrix: Optional[torch.Tensor] = None,
    ):
        if samples.dim() == 4:
            samples = samples[0]
        latent_image = samples.permute(1, 2, 0) @ latent_rgb_factors

        if smooth_matrix is not None:
            latent_image = latent_image.unsqueeze(0).permute(3, 0, 1, 2)
            latent_image = torch.nn.functional.conv2d(
                latent_image, smooth_matrix.reshape((1, 1, 3, 3)), padding=1
            )
            latent_image = latent_image.permute(1, 2, 3, 0).squeeze(0)

        latents_ubyte = (
            ((latent_image + 1) / 2)
            .clamp(0, 1)
            .mul(0xFF)
            .byte()  # change scale from -1..1 to 0..1  # to 0..255
        ).cpu()

        return latents_ubyte

    return torch.stack(
        [
            sample_to_lowres_estimated_image(
                latents[i], SD1_5_LATENT_RGB_FACTORS
            ).permute(2, 0, 1)
            for i in range(latents.shape[0])
        ],
        dim=0,
    )


# ref https://huggingface.co/jimmycarter/LibreFLUX
# "Beta timestep scheduling and timestep stratification"
def get_multirank_stratified_random_timesteps(
    batch_size,
    device,
    distribution: Literal[
        "uniform", "beta", "mode", "boundary-oversampling", "lognormal"
    ] = "beta",
    alpha=2,
    beta=1.6,
    mode_scale=0.5,
    scheduler=None,
    stratify=True,
):
    """
    get timesteps, with stratified distribution across batches
    distribution: 'beta' or 'mode'
    alpha, beta: parameters for beta distribution
    mode_scale: parameter for mode distribution. 0 = uniform distribution, 0.5 = ts500 is about 2.2x more likely than tails
    """
    if distribution == "boundary-oversampling":
        sigmas = _get_boundary_oversampling_sigmas(batch_size, lambda_=1e3)
    elif distribution == "uniform":
        u = torch.rand(batch_size)
        if stratify:
            indices = torch.arange(0, batch_size, dtype=torch.float64)
            u = (indices + u) / batch_size
        sigmas = u
    elif distribution == "lognormal":
        std = alpha
        mean = 0
        u = torch.randn(batch_size) * std + mean
        sigmas = torch.sigmoid(u)
    else:
        indices = torch.arange(0, batch_size, dtype=torch.float64)
        u = torch.rand(batch_size)
        p = ((indices + u) / batch_size) if stratify else u
        if distribution == "beta":
            sigmas = _get_beta_sigmas(p, alpha, beta)
        elif distribution == "mode":
            sigmas = _get_mode_sigmas(p, mode_scale)
        else:
            raise ValueError("Unrecognized distribution type: ", distribution)

    timesteps = (sigmas * 1000).to(device)
    # shuffle
    timesteps = timesteps[torch.randperm(timesteps.shape[0])]
    timesteps = timesteps.long().clamp(min=0, max=999)

    # logging.info(
    #    f"get_multirank_stratified_random_timesteps: {timesteps.detach().cpu().tolist()} for batch size {batch_size} alpha {alpha} beta {beta}"
    # )
    return timesteps


def _get_beta_sigmas(p, alpha, beta):
    return torch.from_numpy(sp_beta.ppf(p.numpy(), a=alpha, b=beta))


def _get_mode_sigmas(p, mode_scale):
    return 1 - p - mode_scale * (torch.cos(math.pi * p / 2) ** 2 - 1 + p)


def _get_boundary_oversampling_sigmas(k, lambda_=1.0):
    n_bins = 1000
    t_bins = torch.linspace(0, 1, n_bins)

    # Compute unnormalized weights
    w = 1 + lambda_ * (torch.exp(-10 * t_bins**2) + torch.exp(-10 * (t_bins - 1) ** 2))

    # Sample indices according to weights
    indices = torch.multinomial(w, k, replacement=True)

    # Add jitter within bins for continuity
    jitter = torch.rand(k) / n_bins
    t_samples = t_bins[indices] + jitter

    return t_samples.clamp(0, 1)



def get_teacher_target(
    teacher_model: TrainingModel,
    teacher_conditioning: Conditioning,
    student_model: TrainingModel,
    timesteps: torch.Tensor,
    student_unet_timesteps: torch.Tensor,
    clean_image_latents: torch.Tensor,
    noise: torch.Tensor,
):
    teacher_prediction_type = 'flow_prediction' if type(teacher_model.noise_scheduler) is TrainFlowMatchEulerDiscreteScheduler else teacher_model.noise_scheduler.config.get('prediction_type', '<unknown teacher prediction type>')
    student_prediction_type = 'flow_prediction' if type(student_model.noise_scheduler) is TrainFlowMatchEulerDiscreteScheduler else student_model.noise_scheduler.config.get('prediction_type', '<unknown student prediction type>')

    # ---- Cross-VAE-space distillation via bidirectional interposer ----
    # When teacher and student live in different VAE latent spaces (e.g. SD2 teacher →
    # SDXL student) the interposer is required to bridge them.  This path implements
    # the full Step 3 forward pass from the plan:
    #   x_1_vS → interp_S2T → x_1_vT → teacher → v_hat_vT
    #   → x_1_hat_vT → interp_T2S → x_1_hat_vS → v_target_vS
    latent_interposer = getattr(student_model, 'latent_interposer', None)
    if latent_interposer is not None and teacher_prediction_type == 'flow_prediction' and student_prediction_type == 'flow_prediction':
        from core.latent_interposer import infer_latent_space_type
        src_space = infer_latent_space_type(student_model)  # e.g. "xl"
        dst_space = infer_latent_space_type(teacher_model)  # e.g. "v1"
        if src_space is not None and dst_space is not None and src_space != dst_space:
            return _teacher_target_via_interposer(
                teacher_model=teacher_model,
                teacher_conditioning=teacher_conditioning,
                student_model=student_model,
                timesteps=student_unet_timesteps,  # already shifted floats
                clean_image_latents=clean_image_latents,
                student_noise=noise,
                latent_interposer=latent_interposer,
                src_space=src_space,
                dst_space=dst_space,
            )

    # ---- Same-VAE-space distillation (original path) ----
    if (
        teacher_prediction_type in ["v_prediction", "v-prediction"]
        and student_prediction_type == "flow_prediction"
    ):
        #raise NotImplementedError("SNR-based timestep remapping implementation not correct for new flowmap timestep scheduling.")
        #@todo use student_noisy_timesteps to compute teacher timesteps based on SNR matching
        teacher_unet_timesteps, teacher_noisy_latents = _remap_noise_v_pred_to_flow_matching(
            teacher_model, student_model, timesteps, clean_image_latents, noise
        )
    else:
        if teacher_prediction_type != student_prediction_type:
            supported_prediction_interconversion_types = {"epsilon", "v_prediction", "v-prediction"}
            present_prediction_types = {
                teacher_prediction_type,
                student_prediction_type,
            }
            if supported_prediction_interconversion_types.isdisjoint(present_prediction_types):
                raise ValueError(
                    f"Unsupported prediction type conversion: {teacher_prediction_type} to {student_prediction_type}"
                )

        # _get_noise_latents uses non-shifted integer timesteps (indices)
        teacher_noisy_latents = _get_noisy_latents(
            clean_image_latents,
            noise,
            teacher_model.noise_scheduler,
            timesteps,
            latents_perturbation=0,
        )
        # student_unet_timesteps are already shifted
        teacher_unet_timesteps = student_unet_timesteps

    teacher_model_output = teacher_model.unet(
        teacher_noisy_latents.to(teacher_model.device, dtype=teacher_model.unet.dtype),
        teacher_unet_timesteps.to(teacher_model.device, dtype=teacher_model.unet.dtype),
        teacher_conditioning.prompt_embeds.to(
            teacher_model.device, dtype=teacher_model.unet.dtype
        ),
        added_cond_kwargs=(
            teacher_conditioning.get_added_cond_kwargs(dtype=teacher_model.unet.dtype)
            if teacher_model.is_sdxl
            else None
        ),
    ).sample.float()

    if teacher_prediction_type == student_prediction_type:
        return teacher_model_output

    return _convert_model_output(
        noise=noise,
        teacher_input=teacher_noisy_latents,
        teacher_output=teacher_model_output,
        teacher_scheduler=teacher_model.noise_scheduler,
        teacher_unet_timesteps=teacher_unet_timesteps,
        student_prediction_type=student_prediction_type,
        student_timesteps=student_unet_timesteps,
    )


def _teacher_target_via_interposer(
    teacher_model: TrainingModel,
    teacher_conditioning: Conditioning,
    student_model: TrainingModel,
    timesteps: torch.Tensor,       # shifted float timesteps, shape [B]
    clean_image_latents: torch.Tensor,  # x_1_vS — scaled SDXL latents, shape [B,4,H,W]
    student_noise: torch.Tensor,   # x_0_vS, shape [B,4,H,W]
    latent_interposer,
    src_space: str,                # e.g. "xl"
    dst_space: str,                # e.g. "v1"
) -> torch.Tensor:
    """
    Cross-VAE-space velocity target via the bidirectional interposer.

    Implements the plan's Step 3 forward pass using code-convention velocity
    (v = noise − clean_latent):

      1. x_1_vT  = interp_S2T(x_1_vS)
      2. x_0_vT  = randn_like(x_1_vT)          [Option A: independent noise]
      3. x_t_vT  = (1−σ)·x_1_vT + σ·x_0_vT    [teacher's noisy input]
      4. v̂_vT   = teacher_unet(x_t_vT, t, ...)
      5. x̂_1_vT = x_t_vT − σ·v̂_vT            [recover clean endpoint]
      6. x̂_1_vS = interp_T2S(x̂_1_vT)
      7. v_tgt_vS = x_0_vS − x̂_1_vS            [student velocity target]

    Returns v_target_vS with same dtype as student_model.unet.
    """
    # ---- 1. Convert clean student latents → teacher VAE space ----
    x_1_vT = latent_interposer.convert(
        clean_image_latents.float(), src=src_space, dst=dst_space
    )  # shape [B, 4, H, W], restored to original device

    # ---- 2. Get σ for the current timestep batch ----
    # Both schedulers share the same shift config, so student's lookup is fine.
    # get_sigmas_for_timesteps requires timesteps to match the scheduler's device.
    sched = student_model.noise_scheduler
    sigmas = sched.get_sigmas_for_timesteps(
        timesteps.to(sched.timesteps.device)
    ).to(x_1_vT.device).float()  # shape [B]
    sigma_b = sigmas.view(-1, 1, 1, 1)  # broadcast over spatial dims

    # ---- 3. Option A: independent noise in teacher space ----
    x_0_vT = torch.randn_like(x_1_vT)  # fresh, not interposed — avoids OOD interposer input

    # ---- 4. Build teacher's noisy latent ----
    # Flow-matching interpolant: x_t = (1−σ)·x_1 + σ·x_0
    x_t_vT = (1.0 - sigma_b) * x_1_vT.float() + sigma_b * x_0_vT.float()

    # ---- 5. Run teacher UNet (SD2: no added_cond_kwargs) ----
    v_hat_vT = teacher_model.unet(
        x_t_vT.to(teacher_model.device, dtype=teacher_model.unet.dtype),
        timesteps.to(teacher_model.device, dtype=teacher_model.unet.dtype),
        teacher_conditioning.prompt_embeds.to(
            teacher_model.device, dtype=teacher_model.unet.dtype
        ),
        added_cond_kwargs=(
            teacher_conditioning.get_added_cond_kwargs(dtype=teacher_model.unet.dtype)
            if teacher_model.is_sdxl
            else None
        ),
    ).sample.to(x_t_vT.device).float()

    # ---- 6. Recover predicted clean endpoint in teacher space ----
    # Code convention: v = x_0 − x_1  ⟹  x_1 = x_t − σ·v
    x_1_hat_vT = x_t_vT - sigma_b * v_hat_vT

    # ---- 7. Map predicted clean endpoint back to student VAE space ----
    x_1_hat_vS = latent_interposer.convert(
        x_1_hat_vT.to(clean_image_latents.dtype), src=dst_space, dst=src_space
    ).to(student_noise.device).float()

    # ---- 8. Build student-space distillation velocity target ----
    # Code convention: target = x_0_vS − x_1_hat_vS  (noise − clean)
    v_target_vS = student_noise.float() - x_1_hat_vS

    return v_target_vS.to(dtype=student_model.unet.dtype)


def _convert_model_output(
    noise: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_output: torch.Tensor,
    teacher_scheduler: SchedulerMixin | ConfigMixin,
    teacher_unet_timesteps,
    student_prediction_type,
    student_timesteps,
):
    source_prediction_type = teacher_scheduler.config.prediction_type
    if source_prediction_type in ["v_prediction", "v-prediction"]:
        alpha_t = (
            teacher_scheduler.alphas_cumprod[teacher_unet_timesteps].view(-1, 1, 1, 1).sqrt()
        )
        sigma_t = (1 - alpha_t**2).sqrt().to(teacher_output.device)
        # Solve for x_0 from v-prediction
        x_0_pred = (alpha_t * teacher_input - sigma_t * teacher_output) / (
            alpha_t**2 + sigma_t**2
        ).sqrt()  # note that this denominator == 1 typically

        if student_prediction_type == "epsilon":
            assert student_timesteps == teacher_unet_timesteps
            # Epsilon: ε = (x_t - α_t·x_0) / σ_t
            return (teacher_input - alpha_t * x_0_pred) / sigma_t
        elif student_prediction_type == "flow_prediction":
            return noise - x_0_pred

    elif source_prediction_type == "epsilon" and student_prediction_type in [
        "v_prediction",
        "v-prediction",
    ]:
        assert student_timesteps == teacher_unet_timesteps
        # Convert epsilon to v-prediction
        alpha_t = (
            teacher_scheduler.alphas_cumprod[teacher_unet_timesteps].view(-1, 1, 1, 1).sqrt()
        )
        sigma_t = (1 - alpha_t**2).sqrt().to(teacher_output.device)
        # First get x_0 from epsilon: x_0 = (x_t - σ_t·ε) / α_t
        x_0_pred = (teacher_input - sigma_t * teacher_output) / alpha_t
        # Then compute v: v = α_t·ε - σ_t·x_0
        return alpha_t * teacher_output - sigma_t * x_0_pred
    else:
        raise ValueError(
            f"Cannot convert between teacher model prediction type {source_prediction_type} and training model prediction type {student_prediction_type}"
        )


def _remap_noise_v_pred_to_flow_matching(
    teacher_model,  # v-pred
    student_model,  # flow matching
    student_timesteps: torch.Tensor,  # integer
    latents: torch.Tensor,
    noise: torch.Tensor,
):
    if teacher_model.noise_scheduler.config.prediction_type not in [
        "v_prediction",
        "v-prediction",
    ]:
        raise ValueError(
            "Teacher model must use v-prediction for SNR-based timestep remapping."
        )
    if student_model.noise_scheduler.config.prediction_type != "flow_prediction":
        raise ValueError(
            "Student model must use flow-matching for SNR-based timestep remapping."
        )

    student_timesteps = student_timesteps.to(student_model.device)
    teacher_timesteps = _get_ddpm_timesteps_for_flowmatch_timesteps(
        flowmatch_timesteps=student_timesteps,
        ddpm_scheduler=teacher_model.noise_scheduler,
        flowmatch_scheduler=student_model.noise_scheduler,
    )

    teacher_noisy_latents = _get_noisy_latents(
        latents,
        noise,
        teacher_model.noise_scheduler,
        teacher_timesteps,
        latents_perturbation=0.0,
    )

    teacher_timesteps = teacher_timesteps.to(teacher_model.device)
    return teacher_timesteps, teacher_noisy_latents


def _get_ddpm_timesteps_for_flowmatch_timesteps(
    flowmatch_timesteps: torch.Tensor,
    ddpm_scheduler: SchedulerMixin,
    flowmatch_scheduler: TrainFlowMatchEulerDiscreteScheduler,
) -> torch.Tensor:
    assert flowmatch_timesteps.dtype not in [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.long,
    ], "expecting floating point (shifted) timesteps for flowmatch"
    t_flow = flowmatch_scheduler.get_sigmas_for_timesteps(
        flowmatch_timesteps.to(flowmatch_scheduler.timesteps.device)
    )

    # FM interpolant: z_s = (1-s)*x_0 + s*eps  → SNR_FM = (1-s)^2 / s^2
    # DDPM interpolant: x_t = α_t*x_0 + σ_t*eps → SNR_DDPM = alpha_bar/(1-alpha_bar)
    # Matching SNRs: alpha_bar = (1-s)^2 / (s^2 + (1-s)^2)
    snr_flow = (1 - t_flow) ** 2 / (t_flow ** 2 + 1e-8)

    # Find DDPM timestep where SNR matches
    # SNR_ddpm = alpha_bar / (1 - alpha_bar)
    # Solving: alpha_bar = SNR / (1 + SNR)
    alpha_bar_target = snr_flow / (1 + snr_flow)

    # Find the closest DDPM timestep
    # scheduler.alphas_cumprod contains alpha_bar values for each timestep
    alphas_cumprod = ddpm_scheduler.alphas_cumprod

    # Find timestep with closest alpha_bar
    ddpm_timesteps = torch.argmin(
        torch.abs(alphas_cumprod.cpu() - alpha_bar_target.unsqueeze(-1).cpu()), dim=-1
    )
    # print(f"Flow t={t_flow} -> DDPM timestep={ddpm_timesteps}")

    # For DDPM
    # print(f"DDPM alpha_bar at t={ddpm_timesteps}: {ddpm_scheduler.alphas_cumprod[ddpm_timestep]}")
    # print(f"DDPM sqrt(alpha_bar): {torch.sqrt(ddpm_scheduler.alphas_cumprod[ddpm_timestep])}")

    return ddpm_timesteps.to(flowmatch_timesteps.device)


def get_local_contrastive_flow_loss(
    midblock_out: torch.Tensor,
    midblock_clean_out: torch.Tensor,
    low_noise_timesteps_mask: torch.Tensor,
    unique_identifiers: list[str],
    temperature,
) -> torch.Tensor:

    # remove duplicate uids from low_noise_timesteps_mask
    for i in torch.nonzero(low_noise_timesteps_mask):
        uid = unique_identifiers[i.item()]
        for j in range(i.item() + 1, len(unique_identifiers)):
            if unique_identifiers[j] == uid:
                low_noise_timesteps_mask[j] = False
    low_noise_timesteps_count = low_noise_timesteps_mask.sum().item()
    if low_noise_timesteps_count > low_noise_timesteps_mask.sum().item():
        logging.warning(
            f" * get_local_contrastive_flow_loss: removed {low_noise_timesteps_count - low_noise_timesteps_mask.sum().item()} duplicate unique_identifiers from low_noise_timesteps_mask - had {low_noise_timesteps_count} now {low_noise_timesteps_mask.sum().item()}"
        )

    contrastive_losses_full = torch.zeros(midblock_out.shape[0], device=midblock_out.device, dtype=midblock_out.dtype)

    # can only do contrastive if >1 samples
    n_contrastive = low_noise_timesteps_mask.sum().item()
    if n_contrastive >= 2:
        # 1. Extract features

        # Noisy representations (z) requiring gradients
        features_anchor = F.normalize(midblock_out[low_noise_timesteps_mask].mean(dim=(2, 3)), p=2, dim=1) # [N_anchors, C]
        # Clean representations (h_clean) - correct labels (positive samples)
        features_clean = F.normalize(midblock_clean_out[low_noise_timesteps_mask].mean(dim=(2, 3)), p=2, dim=1) # [N_anchors, C]
        # Noisy representations from other samples  targets, detached, from all samples (including anchors themselves, but we will ignore self-matches later)
        features_negatives = F.normalize(midblock_out.mean(dim=(2, 3)).detach(), p=2, dim=1) # [B, C]

        # Noisy representations (z) requiring gradients
        #features_anchor = midblock_out[low_noise_timesteps_mask].mean(dim=(2, 3)) # [N_anchors, C]
        # Clean representations (h_clean) - correct labels (positive samples)
        #features_clean = midblock_clean_out[low_noise_timesteps_mask].mean(dim=(2, 3)) # [N_anchors, C]
        # Noisy representations from other samples  targets, detached, from all samples (including anchors themselves, but we will ignore self-matches later)
        #features_negatives = midblock_out.mean(dim=(2, 3)).detach() # [B, C]

        # --- 2. Distance Computation ---

        # Positive distances: ||z(i) - h_clean(i)||^2
        pos_dists_sq = torch.sum((features_anchor - features_clean) ** 2, dim=1)  # [N_anchors]
        # Negative distances: ||z(i) - h_noisy(j)||^2 for all j
        full_dists_sq = torch.cdist(features_anchor, features_negatives, p=2) ** 2  # [N_anchors, B]

        # --- 3. Construct Logits ---

        # We need to form a logit vector where the "correct" class is the Clean version,
        # and the "incorrect" classes are the Noisy versions of other images.

        # Get the original batch indices corresponding to the anchors
        anchor_indices = torch.where(low_noise_timesteps_mask)[0].to(full_dists_sq.device)  # [N_anchors]

        # We start with the full distance matrix (Anchor vs All Noisy)
        # Currently, full_dists_sq[k, anchor_indices[k]] is distance(Anchor_i, Noisy_i) ≈ 0.
        # We REPLACE this self-match with the distance(Anchor_i, Clean_i).

        # Create a range for row indexing
        row_indices = torch.arange(features_anchor.shape[0], device=full_dists_sq.device)

        # Inject the positive distances into the matrix at the correct indices
        # We clone to avoid in-place modification errors if this tensor is used elsewhere
        logits_dists = full_dists_sq.clone()
        logits_dists[row_indices, anchor_indices] = pos_dists_sq

        # Negate because CrossEntropy expects logits (higher is better), but we have distances (lower is better)
        logits = -logits_dists / temperature

        # --- 4. Loss Calculation ---

        # The 'target' for each anchor row is its original index in the batch
        # (because we placed the clean-distance at that specific index)
        contrastive_losses_masked = torch.nn.functional.cross_entropy(logits, anchor_indices, reduction='none')

        # Assign back to full batch loss tensor
        contrastive_losses_full[low_noise_timesteps_mask] = contrastive_losses_masked
    return contrastive_losses_full

def apply_negative_loss_hinge(loss: torch.Tensor, mask: torch.Tensor, margin):
    assert loss.shape == mask.shape
    # where loss_scale < 0, apply a "hinge" negative loss
    # hinge_loss=max(0,m−loss)
    return torch.where(
        mask,
        torch.clamp(margin - loss, min=0.0),
        loss,
    )


def get_contrastive_flow_matching_loss(target, v_pred, unique_identifiers, loss_type, timesteps, noise_scheduler, mask, amount: float) -> tuple[torch.Tensor, int]:
    B = v_pred.shape[0]

    # For stronger contrastive signal, use K negatives per sample
    K = min(mask.sum()-1, 1/amount)  # number of negatives - cap by lambda to avoid too much negative influence

    contrastive_losses = torch.zeros_like(v_pred)
    # pick K random reference indices
    references = torch.randperm(B).tolist()
    contrastive_losses_count = 0
    for ref_idx in references:
        if not mask[ref_idx].item():
            continue
        choices = [i for i in range(B)
                   if i != ref_idx and mask[i].item() and unique_identifiers[i] != unique_identifiers[ref_idx]]
        if len(choices) == 0:
            continue
        neg_idx = random.choice(choices)
        neg_dist = compute_basic_loss(
            loss_type,
            v_pred[ref_idx].unsqueeze(0),
            target[neg_idx].unsqueeze(0),
            timesteps=timesteps[ref_idx].unsqueeze(0),
            noise_scheduler=noise_scheduler,
        )
        contrastive_losses[ref_idx] += neg_dist.squeeze(0)
        # we are not always able to get K negatives (e.g. duplicate images), so count only when we do
        contrastive_losses_count += 1
        if contrastive_losses_count >= K:
            break

    return -contrastive_losses * amount, contrastive_losses_count

def get_contrastive_class_loss(clean_latents, noise, model_pred, timesteps,
                               noise_scheduler, class_labels: list[dict[str, set[str]]],
                               top_k: int=2, loss_type: str="mse") -> torch.Tensor:
    """
    Compute class-wise contrastive loss. Each model_pred is pushed away from the freshly-computed velocity of other samples in contrasting classes.

    Classes are defined by class_labels, a list of dicts - one per sample - containing maps of class category -> labels. eg for samples a, b, c, class_labels could be:
    [
        {"object": {"dog"}, "color": {"black", "brown"}},
        {"object": {"house"}, "color": {"white"}},
        {"lighting": {"high contrast", "saturated"}, "color": {"white"}}
    ]

    For each sample, score the other samples by summing the number of class categories where there is no label overlap, then subtracting the number of label overlaps. Samples with higher scores are considered more contrasting. The top N most contrasting samples are selected as negatives for contrastive loss computation.
    """
    assert isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler), "class-based contrastive loss is only implemented for flow-matching currently."

    B = model_pred.shape[0]

    if B < 2:
        # Need at least 2 samples for contrastive learning
        return torch.zeros_like(model_pred)

    contrast_scores = _compute_contrast_scores(class_labels)

    # For each sample, select top-k most contrasting samples as negatives
    contrastive_losses = torch.zeros_like(model_pred)

    for i in range(B):
        # Get top-k most contrasting samples
        _, top_k_indices = torch.topk(contrast_scores[i], min(top_k, B - 1))

        # Compute contrastive targets for selected negatives
        neg_losses = []
        for j in top_k_indices:
            if contrast_scores[i, j] <= 0:
                continue

            j = j.item()
            # Create incorrect target: noise[i] - clean_latents[j]
            neg_target = noise[i] - clean_latents[j]

            # Compute loss pushing model_pred away from this negative target
            neg_loss = compute_basic_loss(
                loss_type,
                model_pred[i].unsqueeze(0),
                neg_target.unsqueeze(0),
                timesteps[i].unsqueeze(0),
                noise_scheduler,
            )
            neg_loss = neg_loss * torch.tanh(contrast_scores[i, j])
            neg_losses.append(neg_loss)

        if len(neg_losses) > 0:
            # Average over selected negatives and negate (we want to maximize distance)
            contrastive_losses[i] = -torch.stack(neg_losses).mean(dim=0).squeeze(0)

    return contrastive_losses


def _compute_contrast_scores(class_labels: list[dict[str, set[str]]]) -> torch.Tensor:
    B = len(class_labels)
    # Compute contrast scores between all pairs of samples
    contrast_scores = torch.zeros((B, B))

    for i in range(B):
        for j in range(B):
            if i == j:
                contrast_scores[i, j] = -float('inf')  # Don't use self as negative
                continue

            labels_i = class_labels[i]
            labels_j = class_labels[j]

            # Get all class categories
            all_categories = set(labels_i.keys()).union(set(labels_j.keys()))

            score = 0
            for category in all_categories:
                labels_i_cat = labels_i.get(category, set())
                labels_j_cat = labels_j.get(category, set())

                # Count overlapping labels
                overlap = len(labels_i_cat.intersection(labels_j_cat))

                if len(labels_i_cat) > 0 and len(labels_j_cat) > 0:
                    if overlap == 0:
                        # No overlap in this category - samples are contrasting
                        score += 1
                    else:
                        # Has overlap - penalize
                        score -= overlap

            contrast_scores[i, j] = score
    return contrast_scores


# CLIP cross-entropy (InfoNCE) loss
def get_clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    model: TrainingModel,
    mask: torch.Tensor
) -> torch.Tensor:
    #with torch.no_grad():
    #    pixels = model.clip_processor(images=images, return_tensors="pt").pixel_values.to(model.device)  # [B, 3, H, W]
    #    image_embeds = model.clip_model.get_image_features(pixels)  # [B, 1024]

    #text_embeds = model.clip_model.text_projection(text_encoder_pooler_output.to(dtype=model.clip_model.text_projection.weight.dtype))  # [B, 1024]

    # normalize features
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
    logits_per_text = logits_per_text * model.clip_model.logit_scale.exp().to(text_embeds.device)
    logits_per_image = logits_per_text.t()

    # labels are indices of the correct match
    batch_size = text_embeds.shape[0]
    labels = torch.arange(batch_size, device=text_embeds.device).long()
    labels[~mask] = -1  # set ignored samples to -1
    clip_loss_1d = (
        F.cross_entropy(logits_per_image, labels, reduction="none", ignore_index=-1) +
        F.cross_entropy(logits_per_text, labels, reduction="none", ignore_index=-1)
    ) / 2
    clip_loss_1d[~mask] = 0

    return clip_loss_1d
