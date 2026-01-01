import logging
import math
import random
from typing import Tuple, Optional, Literal

import torch
import torchvision
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.cuda.amp import autocast
import torch.nn.functional as F

from scipy.stats import beta as sp_beta

from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler

from flow_match_model import TrainFlowMatchScheduler
from loss_softrepa import (
    text_weighted_infonce_loss,
    text_weighted_infonce_loss_with_softrepa,
)
from model.training_model import TrainingModel, Conditioning


# from train import pyramid_noise_like, compute_snr


def nibble_batch(batch, take_count) -> tuple[dict, dict]:
    runt_size = batch["runt_size"]
    current_batch_size = batch["image"].shape[0]
    non_runt_size = current_batch_size - runt_size
    assert non_runt_size > 0

    nibble_size = min(non_runt_size, take_count)
    nibble = _subdivide_batch_part(batch, 0, nibble_size)
    nibble["runt_size"] = 0

    remaining_size = non_runt_size - nibble_size
    if remaining_size == 0:
        remainder = None
    else:
        remainder = _subdivide_batch_part(batch, nibble_size, non_runt_size)
        remainder["runt_size"] = 0
    return nibble, remainder


def subdivide_batch(batch, current_batch_size, desired_batch_size):
    if desired_batch_size >= current_batch_size:
        yield batch
        return
    runt_size = batch["runt_size"]
    non_runt_size = current_batch_size - runt_size
    for i, offset in enumerate(range(0, non_runt_size, desired_batch_size)):
        sub_batch = _subdivide_batch_part(batch, offset, offset + desired_batch_size)
        end = min(current_batch_size, offset + desired_batch_size)
        sub_batch["runt_size"] = max(0, end - non_runt_size)
        yield sub_batch


def _subdivide_batch_part(part, start, end):
    if type(part) is list or type(part) is torch.Tensor:
        return part[start:end]
    elif type(part) is dict:
        return {k: _subdivide_batch_part(v, start, end) for k, v in part.items()}
    else:
        return part


def choose_effective_batch_size(args, train_progress_01):
    return max(
        1,
        round(
            get_exponential_scaled_value(
                train_progress_01,
                initial_value=args.batch_size
                if args.initial_batch_size is None
                else args.initial_batch_size,
                final_value=args.batch_size
                if args.final_batch_size is None
                else args.final_batch_size,
                alpha=args.batch_size_curriculum_alpha,
            )
        ),
    )


def compute_train_process_01(
    epoch, step, steps_per_epoch, max_epochs, max_global_steps
):
    total_steps = steps_per_epoch * max_epochs
    if max_global_steps is not None:
        total_steps = min(total_steps, max_global_steps)
    steps_completed = steps_per_epoch * epoch + step
    return min(1, steps_completed / total_steps)


"""
alpha=2-4: Slow advance (spread hugs initial_value)
alpha=1: Linear progression
alpha<1: Quick advance (spread hugs final_value)
"""


def get_exponential_scaled_value(progress_01, initial_value, final_value, alpha=3.0):
    # Apply non-linear scaling with alpha (higher alpha = faster early descent)
    scaled_progress = 1.0 - (1.0 - progress_01) ** alpha
    return initial_value + scaled_progress * (final_value - initial_value)


def get_timestep_curriculum_range(
    progress_01,
    t_min_initial=800,
    t_max_initial=1000,
    t_min_final=0,
    t_max_final=400,
    alpha=3.0,
):
    # Interpolate boundaries
    min_t = min(
        1000,
        max(
            0,
            get_exponential_scaled_value(
                progress_01, t_min_initial, t_min_final, alpha=alpha
            ),
        ),
    )
    max_t = min(
        1000,
        max(
            0,
            get_exponential_scaled_value(
                progress_01, t_max_initial, t_max_final, alpha=alpha
            ),
        ),
    )

    assert min_t <= max_t
    return int(min_t), int(max_t)


def get_image_from_latents(latents, model: TrainingModel, args):
    with torch.no_grad():
        with autocast(
            enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16
        ):
            scaling_factor = 0.13025 if model.is_sdxl else 0.18215
            latents = latents / scaling_factor
            pixel_values = model.vae.decode(latents, return_dict=False)[0]
        del latents
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = torch.clamp((pixel_values + 1.0) / 2.0, 0.0, 1.0)
        return pixel_values


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
    text_embeds: torch.Tensor,
    do_contrastive_learning,
    contrastive_loss_scale,
    args,
    verbose=False,
):
    # logging.info(f"get_loss timesteps: {timesteps.detach().cpu().tolist()}")
    device = model_pred.device

    if mask_img is not None:
        mask_img = mask_img.repeat(1, target.shape[1], 1, 1).to(device)
    else:
        mask_img = torch.ones_like(target).to(device)

    assert sum(negative_loss_mask) == 0, "negative loss not implemented yet"
    non_contrastive_loss = compute_basic_loss(args.loss_type, model_pred, target, timesteps, noise_scheduler)
    non_contrastive_loss *= get_timestep_weight(timesteps,
                                                loss_shape=non_contrastive_loss.shape,
                                                min_snr_gamma=args.min_snr_gamma,
                                                loss_mode_scale=args.loss_mode_scale,
                                                noise_scheduler=noise_scheduler,
                                                ).to(device)

    num_valid_contrastive_samples = (~is_cond_dropout).sum()
    if not do_contrastive_learning or num_valid_contrastive_samples <= 1:
        return non_contrastive_loss * mask_img

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
                text_embeddings=text_embeds,
                is_dropped=is_cond_dropout,
                temperature=args.contrastive_loss_temperature,
                hard_negative_weight=args.contrastive_loss_hard_negative_weight,
                use_text_similarity=True,
            )
        elif args.contrastive_loss_type == "infonce_softrepa":
            contrastive_loss = text_weighted_infonce_loss_with_softrepa(
                model_pred,
                target,
                text_embeddings=text_embeds,
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
    total_loss = (
        non_contrastive_loss + contrastive_loss * contrastive_loss_scale
    ) * mask_img
    return total_loss


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
        sigmas = timesteps / noise_scheduler.config.num_train_timesteps
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


def get_timestep_weight(
    timesteps: torch.Tensor, loss_shape, min_snr_gamma, loss_mode_scale, noise_scheduler
) -> torch.Tensor:
    if min_snr_gamma is not None and min_snr_gamma > 0:
        if type(noise_scheduler) is TrainFlowMatchScheduler:
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



def get_model_prediction_and_target(
    latents,
    conditioning: Conditioning,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    model: TrainingModel,
    args=None,
    skip_contrastive: bool = False,
    teacher_model: TrainingModel | None = None,
    teacher_mask: torch.Tensor | None = None,
    teacher_conditioning: Conditioning | None = None,
    debug_fake: bool = False,
    log_writer=None,
    global_step: int = 0,
    mask=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    If mask is provided, only compute for the masked entries and return full tensors with zeros elsewhere.
    """

    if mask is not None:
        # create full tensors
        model_pred = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        target = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        noisy_latents = torch.zeros_like(
            latents, dtype=model.unet.dtype, device=model.unet.device
        )
        if mask.sum() == 0:
            # early out for empty mask
            return model_pred, target, noisy_latents

        latents_masked = latents[mask]
        conditioning_masked = conditioning.get_masked(mask)
        noise_masked = noise[mask]
        teacher_mask_masked = teacher_mask[mask] if teacher_mask is not None else None
        teacher_conditioning_masked = (
            teacher_conditioning.get_masked(mask)
            if teacher_conditioning is not None
            else None
        )
        timesteps_masked = timesteps[mask]
        model_pred_masked, target_masked, noisy_latents_masked = (
            get_model_prediction_and_target(
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
            )
        )
        model_pred[mask] += model_pred_masked
        target[mask] += target_masked
        noisy_latents[mask] += noisy_latents_masked
        return model_pred, target, noisy_latents

    # logging.info(f"get_model_prediction_and_target timesteps: {timesteps.detach().cpu().tolist()}")
    noisy_latents = _get_noisy_latents(
        latents, noise, model.noise_scheduler, timesteps, args.latents_perturbation
    )
    target = _get_target(latents, noise, model.noise_scheduler, timesteps).to(dtype=model.unet.dtype)

    if isinstance(model.noise_scheduler, FlowMatchEulerDiscreteScheduler):
        unet_timesteps = TrainFlowMatchScheduler.get_shifted_timesteps(timestep_indices=timesteps, timestep_values=model.noise_scheduler.timesteps)
    else:
        unet_timesteps = timesteps

    if debug_fake:
        model_pred = torch.ones_like(target).to(model.device)
    else:
        model_pred = model.unet(
            noisy_latents.to(dtype=model.unet.dtype),
            unet_timesteps.to(model.unet.device, dtype=model.unet.dtype),
            encoder_hidden_states=conditioning.prompt_embeds.to(dtype=model.unet.dtype),
            added_cond_kwargs=(
                conditioning.get_added_cond_kwargs(dtype=model.unet.dtype)
                if model.is_sdxl
                else None
            ),
        ).sample

    if teacher_mask is not None and teacher_mask.sum() > 0:
        with torch.no_grad():
            if teacher_conditioning is None:
                teacher_conditioning = conditioning

            teacher_target = get_teacher_target(
                teacher_model=teacher_model,
                teacher_conditioning=teacher_conditioning,
                student_model=model,
                timesteps=timesteps,
                student_unet_timesteps=unet_timesteps,
                clean_image_latents=latents,
                noise=noise,
            )

            if log_writer is not None:
                loss_preview_image_rgb = torchvision.utils.make_grid(teacher_target)
                log_writer.add_image(
                    tag="loss/teacher target",
                    img_tensor=loss_preview_image_rgb,
                    global_step=global_step,
                )

            target = teacher_target * teacher_mask.view(-1, 1, 1, 1).expand_as(
                target
            ).to(target.device) + target * ~teacher_mask.view(-1, 1, 1, 1).expand_as(
                target
            ).to(target.device)

    return model_pred, target, noisy_latents


def _get_noisy_latents(
    latents, noise, noise_scheduler, timesteps, latents_perturbation
):
    # logging.info(f"get_noisy_latents timesteps: {timesteps.detach().cpu().tolist()}")
    if hasattr(noise_scheduler, "add_noise"):
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    elif isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
        shifted_timesteps = TrainFlowMatchScheduler.get_shifted_timesteps(timestep_indices=timesteps, timestep_values=noise_scheduler.timesteps)
        noisy_latents = noise_scheduler.scale_noise(latents, shifted_timesteps, noise)
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
    noise_scheduler: (SchedulerMixin, ConfigMixin),
    timesteps,
    latents_perturbation,
):
    noisy_latents = _get_noisy_latents(
        latents, noise, noise_scheduler, timesteps, latents_perturbation
    )
    target = _get_target(latents, noise, noise_scheduler, timesteps)

    return noisy_latents, target


def get_timesteps(
    batch_size, batch_share_timesteps, device, timesteps_ranges, scheduler
):
    """if continuous_float_timesteps is True, return float timestamps (continuous), otherwise return int timestamps (discrete)"""
    timesteps = torch.cat(
        [torch.randint(a, b, size=(1,), device=device) for a, b in timesteps_ranges]
    )
    if batch_share_timesteps:
        timesteps = timesteps[:1].repeat((batch_size,))
    timesteps = timesteps.long()
    # logging.info(f"get_timesteps: {timesteps.detach().cpu().tolist()} from ranges: {timesteps_ranges}")
    return timesteps


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
        std = 1
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
    teacher_prediction_type = teacher_model.noise_scheduler.config.prediction_type
    student_prediction_type = student_model.noise_scheduler.config.prediction_type
    if (
        teacher_prediction_type in ["v_prediction", "v-prediction"]
        and student_prediction_type == "flow_prediction"
    ):
        raise NotImplementedError("SNR-based timestep remapping implementation not correct for new flowmap timestep scheduling.")
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

    if student_prediction_type == 'flow_prediction' or teacher_prediction_type == 'flow_prediction':
        raise NotImplementedError("conversion to/from flow_prediction is untested due to unet_timesteps handling")
    return _convert_model_output(
        noise=noise,
        teacher_input=teacher_noisy_latents,
        teacher_output=teacher_model_output,
        teacher_scheduler=teacher_model.noise_scheduler,
        teacher_unet_timesteps=teacher_unet_timesteps,
        student_prediction_type=student_prediction_type,
        student_unet_timesteps=student_unet_timesteps,
    )


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
    flowmatch_scheduler: TrainFlowMatchScheduler,
) -> torch.Tensor:
    assert flowmatch_timesteps.dtype not in [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.long,
    ], "expecting floating point (shifted) timesteps for flowmatch"
    flowmatch_timestep_indices = flowmatch_scheduler.get_timestep_indices(
        flowmatch_timesteps
    ).cpu()
    t_flow = flowmatch_scheduler.sigmas[flowmatch_timestep_indices]

    snr_flow = t_flow**2 / (1 - t_flow) ** 2

    # Find DDPM timestep where SNR matches
    # SNR_ddpm = alpha_bar / (1 - alpha_bar)
    # Solving: alpha_bar = SNR / (1 + SNR)
    alpha_bar_target = snr_flow / (1 + snr_flow)

    # Find the closest DDPM timestep
    # scheduler.alphas_cumprod contains alpha_bar values for each timestep
    alphas_cumprod = ddpm_scheduler.alphas_cumprod.cpu()

    # Find timestep with closest alpha_bar
    ddpm_timesteps = torch.argmin(
        torch.abs(alphas_cumprod - alpha_bar_target.unsqueeze(-1)), dim=-1
    )
    # print(f"Flow t={t_flow} -> DDPM timestep={ddpm_timesteps}")

    # For DDPM
    # print(f"DDPM alpha_bar at t={ddpm_timesteps}: {ddpm_scheduler.alphas_cumprod[ddpm_timestep]}")
    # print(f"DDPM sqrt(alpha_bar): {torch.sqrt(ddpm_scheduler.alphas_cumprod[ddpm_timestep])}")

    return ddpm_timesteps


def get_local_contrastive_flow_loss(
    model_pred: torch.Tensor,
    model_pred_anchor: torch.Tensor,
    low_noise_timesteps_mask: torch.Tensor,
    unique_identifiers: list[str],
    temperature,
) -> torch.Tensor:

    low_noise_timesteps_count = low_noise_timesteps_mask.sum().item()
    # remove duplicate uids from low_noise_timesteps_mask
    for i in torch.nonzero(low_noise_timesteps_mask):
        uid = unique_identifiers[i.item()]
        for j in range(i.item() + 1, len(unique_identifiers)):
            if unique_identifiers[j] == uid:
                low_noise_timesteps_mask[j] = False
    if low_noise_timesteps_count > low_noise_timesteps_mask.sum().item():
        logging.warning(
            f" * get_local_contrastive_flow_loss: removed {low_noise_timesteps_count - low_noise_timesteps_mask.sum().item()} duplicate unique_identifiers from low_noise_timesteps_mask - had {low_noise_timesteps_count} now {low_noise_timesteps_mask.sum().item()}"
        )

    contrastive_losses_full = torch.zeros(model_pred.shape[0], device=model_pred.device, dtype=model_pred.dtype)

    # can only do contrastive if >1 samples
    n_contrastive = low_noise_timesteps_mask.sum().item()
    if n_contrastive >= 2:
        # 3. Flatten spatial dimensions, then normalize for cosine similarity
        features_current = model_pred[low_noise_timesteps_mask].flatten(
            1
        )  # [n_contrastive, CHW]
        features_anchor = model_pred_anchor[low_noise_timesteps_mask].flatten(
            1
        )  # [n_contrastive, CHW]

        features_current = torch.nn.functional.normalize(features_current, dim=-1)
        features_anchor = torch.nn.functional.normalize(features_anchor, dim=-1).to(
            dtype=features_current.dtype
        )

        # 4. Compute contrastive loss (InfoNCE-style)
        # Similarity matrix: [n_contrastive, n_contrastive]
        logits = torch.matmul(features_current, features_anchor.T) / temperature

        # Positives are on the diagonal (same image at different noise levels)
        labels = torch.arange(n_contrastive, device=model_pred.device)

        contrastive_losses_masked = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        contrastive_losses_full[low_noise_timesteps_mask] = contrastive_losses_masked
    return contrastive_losses_full


def get_contrastive_flow_matching_loss(target, v_pred, unique_identifiers, K, loss_type, timesteps, noise_scheduler):
    B = v_pred.shape[0]

    # For stronger contrastive signal, use K negatives per sample
    K = min(B-1, K)  # number of negatives

    contrastive_losses = torch.zeros_like(v_pred)
    # pick K random reference indices
    references = torch.randperm(B).tolist()
    contrastive_losses_count = 0
    for ref_idx in references:
        choices = [i for i in range(B) if i != ref_idx and unique_identifiers[i] != unique_identifiers[ref_idx]]
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

    return -contrastive_losses
