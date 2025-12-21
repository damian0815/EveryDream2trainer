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

from diffusers import SchedulerMixin, ConfigMixin

from flow_match_model import TrainFlowMatchScheduler
from model.training_model import TrainingModel, Conditioning


# from train import pyramid_noise_like, compute_snr

def nibble_batch(batch, take_count):
    runt_size = batch['runt_size']
    current_batch_size = batch['image'].shape[0]
    non_runt_size = current_batch_size - runt_size
    assert non_runt_size > 0

    nibble_size = min(non_runt_size, take_count)
    nibble = _subdivide_batch_part(batch, 0, nibble_size)
    nibble['runt_size'] = 0

    remaining_size = non_runt_size - nibble_size
    if remaining_size == 0:
        remainder = None
    else:
        remainder = _subdivide_batch_part(batch, nibble_size, non_runt_size)
        remainder['runt_size'] = 0
    return nibble, remainder


def subdivide_batch(batch, current_batch_size, desired_batch_size):
    if desired_batch_size >= current_batch_size:
        yield batch
        return
    runt_size = batch['runt_size']
    non_runt_size = current_batch_size - runt_size
    for i, offset in enumerate(range(0, non_runt_size, desired_batch_size)):
        sub_batch = _subdivide_batch_part(batch, offset, offset+desired_batch_size)
        end = min(current_batch_size, offset + desired_batch_size)
        sub_batch['runt_size'] = max(0, end - non_runt_size)
        yield sub_batch

def _subdivide_batch_part(part, start, end):
    if type(part) is list or type(part) is torch.Tensor:
        return part[start:end]
    elif type(part) is dict:
        return {k: _subdivide_batch_part(v, start, end) for k, v in part.items()}
    else:
        return part

def choose_effective_batch_size(args, train_progress_01):
    return max(1, round(get_exponential_scaled_value(
        train_progress_01,
        initial_value=args.batch_size if args.initial_batch_size is None else args.initial_batch_size,
        final_value=args.batch_size if args.final_batch_size is None else args.final_batch_size,
        alpha=args.batch_size_curriculum_alpha
    )))


def compute_train_process_01(epoch, step, steps_per_epoch, max_epochs, max_global_steps):
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
    scaled_progress = 1.0 - (1.0-progress_01) ** alpha
    return initial_value + scaled_progress * (final_value - initial_value)

def get_timestep_curriculum_range(progress_01,
                                  t_min_initial=800, t_max_initial=1000,
                                  t_min_final=0, t_max_final=400,
                                  alpha=3.0):
    # Interpolate boundaries
    min_t = min(1000, max(0, get_exponential_scaled_value(progress_01, t_min_initial, t_min_final, alpha=alpha)))
    max_t = min(1000, max(0, get_exponential_scaled_value(progress_01, t_max_initial, t_max_final, alpha=alpha)))

    assert min_t <= max_t
    return int(min_t), int(max_t)

def get_image_from_latents(latents, model: TrainingModel, args):
    with torch.no_grad():
        with autocast(enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16):
            scaling_factor = 0.13025 if model.is_sdxl else 0.18215
            latents = latents / scaling_factor
            pixel_values = model.vae.decode(latents, return_dict=False)[0]
        del latents
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = torch.clamp((pixel_values + 1.0) / 2.0, 0.0, 1.0)
        return pixel_values

def get_latents(image, model: TrainingModel, device, args):
    with torch.no_grad():
        with autocast(enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(device, dtype=model.vae.dtype)
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
    loss_scale,
    noise_scheduler,
    text_embeds: torch.Tensor,
    do_contrastive_learning,
    contrastive_loss_scale,
    args,
    verbose=False,
):

    #logging.info(f"get_loss timesteps: {timesteps.detach().cpu().tolist()}")
    device = model_pred.device

    if mask_img is not None:
        mask_img = mask_img.repeat(1, target.shape[1], 1, 1).to(target.device)
    else:
        mask_img = torch.ones_like(target)

    def timestep_weight_loss(loss: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:

        if args.min_snr_gamma is not None and args.min_snr_gamma > 0:
            if args.train_sampler == 'flow-matching':
                t = timesteps / noise_scheduler.config.num_train_timesteps
                snr = (1 - t) / (t + 1e-8)  # Linear approximation
                snr_weight = torch.minimum(snr, torch.tensor(args.min_snr_gamma)) / (snr + 1e-8)
            else:
                snr = compute_snr(timesteps, noise_scheduler)
                v_pred = noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]
                divisor = (snr + 1) if v_pred else snr
                snr_weight = (
                    torch.stack(
                        [snr, args.min_snr_gamma * torch.ones_like(timesteps).float()], dim=1
                    ).min(dim=1)[0]
                    / divisor
                )
            snr_weight = snr_weight.view(-1, 1, 1, 1).expand_as(loss)
            return loss * snr_weight.to(loss.device)
        elif args.loss_mode_scale > 0:
            # Scale loss to emphasize middle timesteps (mode around 500)
            # loss_mode_scale=0 means uniform, higher values increase the peak
            t_normalized = timesteps.float() / 1000.0  # normalize to [0, 1]
            sharpness = 3
            mode_weight = (
                torch.cos(math.pi * (t_normalized - 0.5)) ** sharpness + 1
            ) - 1
            mode_weight = (1 - args.loss_mode_scale) + args.loss_mode_scale * mode_weight
            mode_weight = mode_weight.view(-1, 1, 1, 1).expand_as(loss)
            return loss * mode_weight.to(loss.device)
        else:
            return loss


    def compute_loss(model_pred: torch.Tensor, target: torch.Tensor, timesteps: torch.LongTensor, loss_scale: torch.Tensor):
        reduction = "none"
        loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction=reduction)
        loss_scale = torch.ones(model_pred.shape[0], dtype=torch.float) * loss_scale
        loss_scale = loss_scale.view(-1, 1, 1, 1).expand_as(loss_mse)

        loss_type = args.loss_type
        if loss_type == "mse_huber":
            early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
            early_timestep_bias = early_timestep_bias.float().to(device)
            early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_mse = loss_mse * loss_scale.to(device) * early_timestep_bias
            loss_huber = loss_huber * loss_scale.to(device) * (1.0 - early_timestep_bias)
            loss = loss_mse + loss_huber
            del loss_huber
        elif loss_type == "huber_mse":
            early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
            early_timestep_bias = torch.tensor(early_timestep_bias, dtype=torch.float).to(device)
            early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_mse = loss_mse * loss_scale.to(device) * (1.0 - early_timestep_bias)
            loss_huber = loss_huber * loss_scale.to(device) * early_timestep_bias
            loss = loss_huber + loss_mse
            del loss_huber
        elif loss_type == "huber":
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_huber = loss_huber * loss_scale.to(device)
            loss = loss_huber
            del loss_huber
        elif loss_type.startswith('sd3-'):
            # SD3 loss weight
            sigmas = timesteps / noise_scheduler.config.num_train_timesteps
            if args.loss_type == 'sd3-cosmap':
                weighting_scheme = "cosmap"
            elif args.loss_type == 'sd3-sigma_sqrt':
                weighting_scheme = "sigma_sqrt"
            else:
                raise ValueError(f"unhandled loss type {args.loss_type}")
            weights = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            loss = loss_mse * weights.view(-1, 1, 1, 1).to(loss_mse.device) * loss_scale.to(device)
            del weights, sigmas
        elif loss_type == 'cosmap-2':
            lmbd = 0.9
            sigmas = timesteps / 1000
            weights = 1 - lmbd * (
                torch.exp(-10 * (sigmas - 0) ** 2) + torch.exp(-10 * (sigmas - 1) ** 2)
            )
            loss = loss_mse * weights.view(-1, 1, 1, 1).to(loss_mse.device) * loss_scale.to(device)
        else:
            if loss_type != 'mse':
                raise ValueError(f"Unrecognized --loss_type {args.loss_type}")
            loss = loss_mse * loss_scale.to(device)
        del loss_mse

        if torch.any(loss_scale < 0):
            distance_sq = torch.pow(model_pred.float() - target.float(), 2)
            margin_sq = args.negative_loss_margin * args.negative_loss_margin
            repulsion_loss = torch.max(torch.tensor(0), margin_sq - distance_sq)
            negative_loss_mask = ((loss_scale < 0) * 1.0).to(loss.device)
            loss = repulsion_loss * negative_loss_mask + loss * (1-negative_loss_mask)

        return loss

    non_contrastive_loss = timestep_weight_loss(compute_loss(model_pred, target, timesteps, loss_scale), timesteps=timesteps)

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
            is_dropped: [B] - boolean mask, True where conditioning was dropped

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
                delta_loss = F.mse_loss(pred_delta, target_delta, reduction='mean')

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
        target_flat = target_valid.reshape(num_valid_contrastive_samples, -1)  # [B, C*H*W]

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
        contrastive_loss_valid = F.cross_entropy(sim_matrix, labels, reduction='none') / math.sqrt(num_valid_contrastive_samples)

        # expand to cover cond dropout samples
        contrastive_loss_full = torch.zeros(B_full, device=model_pred.device)
        contrastive_loss_full[valid_mask] = contrastive_loss_valid

        del contrastive_loss_valid

        return contrastive_loss_full

    with (torch.autocast('cuda')):
        if args.contrastive_loss_type == 'infonce_with_text_similarity':
            contrastive_loss = text_weighted_infonce_loss(
                model_pred,
                target,
                text_embeddings=text_embeds,
                is_dropped=is_cond_dropout,
                temperature=args.contrastive_loss_temperature,
                hard_negative_weight=args.contrastive_loss_hard_negative_weight,
                use_text_similarity=True,
            )
        elif args.contrastive_loss_type == 'infonce_softrepa':
            contrastive_loss = text_weighted_infonce_loss_with_softrepa(
                model_pred,
                target,
                text_embeddings=text_embeds,
                is_dropped=is_cond_dropout,
                temperature=None,
                sigma=args.contrastive_loss_softrepa_sigma,
                hard_negative_weight=args.contrastive_loss_hard_negative_weight,
                similarity_method='softrepa'
            )
        elif args.contrastive_loss_type == 'infonce':
            contrastive_loss = contrastive_loss_infonce(model_pred, target, is_cond_dropout, timesteps)
        elif args.contrastive_loss_type == 'delta':
            contrastive_loss = contrastive_loss_delta(model_pred, target, is_cond_dropout)
        else:
            raise ValueError(f"Unrecognized contrastive loss type: {args.contrastive_loss_type}")

    contrastive_loss = contrastive_loss.view(-1, 1, 1, 1).expand_as(
        non_contrastive_loss
    )
    contrastive_loss = timestep_weight_loss(contrastive_loss, timesteps)

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


# Asks "How aligned are the directions of the prediction and target vectors?" This is a measure of cosine similarity.
def text_weighted_infonce_loss(
    model_pred,
    target,
    text_embeddings,
    is_dropped,
    temperature=0.07,
    hard_negative_weight=2.0,
    use_text_similarity=True,
):
    """
    InfoNCE with text-aware weighting:
    - Harder penalties for similar captions (confusable)
    - Can use text embeddings for semantic similarity

    Args:
        model_pred: [B, C, H, W]
        target: [B, C, H, W]
        text_embeddings: [B, T, D] pre-computed text embeddings (optional)
        is_dropped: [B] boolean mask
        temperature: scaling factor
        hard_negative_weight: extra weight for similar captions
        use_text_similarity: whether to weight by caption similarity

    Returns:
        per_sample_loss: [B] contrastive losses
    """
    B = model_pred.shape[0]
    device = model_pred.device

    # Identify valid samples
    valid_mask = ~is_dropped

    num_valid = valid_mask.sum()
    if num_valid < 2:
        return torch.zeros(B, device=device)

    # Extract valid samples
    pred_valid = model_pred[valid_mask]
    target_valid = target[valid_mask]

    # Flatten and normalize
    pred_flat = pred_valid.reshape(num_valid, -1)
    target_flat = target_valid.reshape(num_valid, -1)

    pred_norm = F.normalize(pred_flat, p=2, dim=1)
    target_norm = F.normalize(target_flat, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = (
        torch.mm(pred_norm, target_norm.t()) / temperature
    )  # [num_valid, num_valid]

    # Text-based weighting
    if use_text_similarity:
        text_sim = text_embeddings_to_similarity(text_embeddings)

        # Weight negatives by text similarity
        # Similar captions = harder negatives = higher weight
        # Create weight matrix: 1.0 for diagonal, hard_negative_weight for similar off-diagonal
        weight_matrix = torch.ones_like(sim_matrix)

        # Apply extra weight to similar captions (hard negatives)
        for i in range(num_valid):
            for j in range(num_valid):
                if i != j:
                    # Scale weight by text similarity
                    # High text_sim → hard negative → higher weight
                    weight_matrix[i, j] = (
                        1.0 + (hard_negative_weight - 1.0) * text_sim[i, j]
                    )

        # Apply weights to similarity matrix (for negatives only)
        mask = torch.eye(num_valid, device=device, dtype=torch.bool)
        sim_matrix = torch.where(mask, sim_matrix, sim_matrix * weight_matrix)

    # Standard InfoNCE, normalized by sqrt(batch_size) for stability
    labels = torch.arange(num_valid, device=device)
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction="none") / math.sqrt(num_valid)

    # Map back to full batch size with zeros for dropped samples
    full_losses = torch.zeros(B, device=device)
    full_losses[valid_mask] = loss_per_sample

    return full_losses



def text_weighted_infonce_loss_with_softrepa(
    model_pred,
    target,
    text_embeddings,
    is_dropped,
    temperature=0.07,
    sigma=1.0,
    hard_negative_weight=2.0,
    similarity_method: Literal['cosine', 'softrepa'] = 'softrepa',
):
    """
    InfoNCE with text-aware weighting, using SoftREPA's error-based similarity.

    Sigma for SoftREPA controls the sharpness of the similarity measure. A smaller sigma will make the logits more spread out (e.g., -10 vs -1000). This leads to a "sharper" softmax distribution, forcing the model to be more confident in its positive pair. A larger sigma will shrink the logits closer together, creating a "softer" distribution. You will need to tune sigma as a hyperparameter. A good starting point is 1.0.

    Args:
        model_pred: [B, C, H, W] (this is v_pred)
        target: [B, C, H, W] (this is v_target)
        text_embeddings: [B, T, D]
        is_dropped: [B]
        sigma: scaling factor for SoftREPA error-based logits
        temperature: scaling factor for normalized cosine distance
        hard_negative_weight: extra weight for similar captions
        use_text_similarity: whether to weight by caption similarity

    Returns:
        per_sample_loss: [B] contrastive losses
    """
    B = model_pred.shape[0]
    device = model_pred.device

    # Identify valid samples
    valid_mask = ~is_dropped

    num_valid = valid_mask.sum()
    if num_valid < 2:
        return torch.zeros(B, device=device)

    # Extract valid samples
    pred_valid = model_pred[valid_mask]
    target_valid = target[valid_mask]

    # Flatten (but DO NOT normalize)
    pred_flat = pred_valid.reshape(num_valid, -1)
    target_flat = target_valid.reshape(num_valid, -1)

    if similarity_method == 'softrepa':
        # Error-based logits as per SoftREPA
        # Asks "How small is the error between the prediction for image i and the target for image j?" It directly uses the model's primary objective (minimizing prediction error) as the basis for contrastive learning.
        sim_matrix = calculate_softrepa_error_based_logits(pred_flat, target_flat, sigma=sigma)
        # The output `sim_matrix` is a [num_valid, num_valid] matrix of logits.
    elif similarity_method == 'cosine':
        # cosine similarity between velocities
        # Asks "How aligned are the directions of the prediction and target vectors?"
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = (
            torch.mm(pred_norm, target_norm.t()) / temperature
        )  # [num_valid, num_valid]
    else:
        raise ValueError(f"Unknown similarity_method: {similarity_method}")

    text_sim = text_embeddings_to_similarity(text_embeddings)

    # Weight negatives by text similarity
    # Similar captions = harder negatives = higher weight
    # Create weight matrix: 1.0 for diagonal, hard_negative_weight for similar off-diagonal
    weight_matrix = torch.ones_like(sim_matrix)

    # Apply extra weight to similar captions (hard negatives)
    for i in range(num_valid):
        for j in range(num_valid):
            if i != j:
                # Scale weight by text similarity
                # High text_sim → hard negative → higher weight
                weight_matrix[i, j] = (
                    1.0 + (hard_negative_weight - 1.0) * text_sim[i, j]
                )

    # Apply weights to similarity matrix (for negatives only)
    mask = torch.eye(num_valid, device=device, dtype=torch.bool)
    sim_matrix = torch.where(mask, sim_matrix, sim_matrix * weight_matrix)

    # Standard InfoNCE (THIS PART REMAINS EXACTLY THE SAME)
    labels = torch.arange(num_valid, device=device)
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction="none") / math.sqrt(
        num_valid
    )

    # Map back to full batch size with zeros for dropped samples
    full_losses = torch.zeros(B, device=device)
    full_losses[valid_mask] = loss_per_sample

    return full_losses



def text_embeddings_to_similarity(text_embeddings):
    """
    Convert [B, T, D] text embeddings to [B, B] similarity matrix.

    Args:
        text_embeddings: [B, 77, D] CLIP penultimate hidden states

    Returns:
        sim_matrix: [B, B] pairwise similarity
    """
    B, T, D = text_embeddings.shape

    # Option 1: Mean pool over sequence dimension
    pooled = text_embeddings.mean(dim=1)  # [B, D]

    # Option 2: CLS token (first token) - often used with CLIP
    # pooled = text_embeddings[:, 0, :]  # [B, D]

    # Option 3: Max pool (take most salient features)
    # pooled = text_embeddings.max(dim=1)[0]  # [B, D]

    # Compute cosine similarity
    pooled_norm = F.normalize(pooled, p=2, dim=1)
    sim_matrix = torch.mm(pooled_norm, pooled_norm.t())

    return sim_matrix



def calculate_softrepa_error_based_logits(pred_flat, target_flat, sigma=1.0):
    """
    Calculates a similarity matrix based on negative squared error, as in SoftREPA.
    The "similarity" is actually the logit for the contrastive loss.

    Args:
        pred_flat: [N, D] tensor of flattened model predictions (v_pred).
        target_flat: [N, D] tensor of flattened targets (v_target).
        sigma: A scaling parameter, similar to temperature. Controls the sharpness.

    Returns:
        logit_matrix: [N, N] matrix where entry (i, j) is -||pred_i - target_j||^2 / sigma.
    """
    # Use broadcasting to compute the pairwise squared error efficiently
    # pred_flat:       [N, 1, D]
    # target_flat:     [1, N, D]
    # diff:            [N, N, D]
    diff = pred_flat.unsqueeze(1) - target_flat.unsqueeze(0)

    # Sum the squares over the feature dimension (D)
    # This computes ||v_pred_i - v_target_j||^2 for all i, j
    squared_error_matrix = torch.sum(diff**2, dim=2)  # Shape: [N, N]

    D = pred_flat.shape[-1]
    normalized_error = squared_error_matrix / D

    # The logit is the negative squared error, scaled by sigma
    # A small error -> large logit (good match)
    # A large error -> small (very negative) logit (bad match)
    logit_matrix = -normalized_error / sigma

    return logit_matrix


def contrastive_loss_old():

    # Generate negative samples
    # max_negative_loss = torch.tensor(args.contrastive_learning_max_negative_loss,
    #                                 dtype=positive_loss.dtype).to(positive_loss.device)
    negative_loss = torch.zeros_like(non_contrastive_loss)
    bsz = model_pred.shape[0]
    num_samples = [0] * bsz
    for i in range(bsz):
        if (caption_str[i] is None
                or len(caption_str[i].strip()) == 0
                or loss_scale[i] < 0
        ):
            continue

        for j in range(bsz):
            if (i == j  # skip self
                    or caption_str[j] is None
                    or len(caption_str[j].strip()) == 0  # skip missing or dropout
                    or caption_str[i] == caption_str[j] # skip equal captions
                    or loss_scale[j] < 0 # skip negative loss
            ):
                continue
            delta_to_wrong = model_pred[j:j + 1] - model_pred[i:i + 1]
            target_delta_to_wrong = target[j:j + 1] - target[i:i + 1]
            l_negative = compute_loss(delta_to_wrong.float(),
                                  target_delta_to_wrong.float(),
                                  timesteps=timesteps[j:j + 1],
                                  loss_scale=loss_scale[j:j + 1])
            # l_negative = F.mse_loss(delta_to_wrong.float(), target_delta_to_wrong.float(), reduction='none')
            negative_loss[i:i + 1] += l_negative
            del delta_to_wrong
            del target_delta_to_wrong
            del l_negative

            num_samples[i] += 1

    # print(' - num contrastive samples', num_samples, ', negative loss', negative_loss.mean())

    # Average over negative samples
    num_samples_safe = torch.tensor(([1] * len(num_samples)
                                     if args.contrastive_learning_no_average_negatives
                                     else [max(1, x) for x in num_samples]
                                     ), device=negative_loss.device)

    negative_loss_scale = contrastive_learning_negative_loss_scale / num_samples_safe
    negative_loss_scale = negative_loss_scale.view(-1, 1, 1, 1).to(device).expand_as(non_contrastive_loss)
    if args.contrastive_learning_delta_loss_method and args.contrastive_learning_delta_timestep_start >= 0:
        # scale negative loss with timesteps, with a minimum cutoff. ie do more delta loss as noise level increases
        max_timestep = noise_scheduler.config.num_train_timesteps
        negative_loss_timestep_start = args.contrastive_learning_delta_timestep_start
        # linear bias with offset
        early_timestep_bias = torch.maximum((timesteps - negative_loss_timestep_start)
                                            / (max_timestep - negative_loss_timestep_start),
                                            torch.tensor(0).to(device))
        early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(non_contrastive_loss)
        negative_loss_scale = negative_loss_scale * early_timestep_bias

    loss = (non_contrastive_loss + negative_loss * negative_loss_scale) * mask
    del non_contrastive_loss
    del negative_loss

    return loss

def _get_contrastive_v2_loss():
    pass

def get_model_prediction_and_target(latents, conditioning: Conditioning, noise: torch.Tensor,
                                    timesteps: torch.Tensor, model: TrainingModel,
                                    args=None, skip_contrastive: bool=False,
                                    teacher_model: TrainingModel|None=None,
                                    teacher_mask: torch.Tensor|None=None,
                                    teacher_conditioning: Conditioning|None=None,
                                    debug_fake: bool = False,
                                    log_writer=None
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #logging.info(f"get_model_prediction_and_target timesteps: {timesteps.detach().cpu().tolist()}")
    noisy_latents = _get_noisy_latents(latents, noise, model.noise_scheduler, timesteps, args.latents_perturbation)
    target = _get_target(latents, noise, model.noise_scheduler, timesteps)
    if debug_fake:
        model_pred = torch.ones_like(target).to(model.device)
    else:
        with autocast(enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16):
            if model.is_sdxl:
                model_pred = model.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=conditioning.prompt_embeds,
                        added_cond_kwargs=conditioning.added_cond_kwargs
                ).sample
            else:
                # print(f"types: {type(noisy_latents)} {type(timesteps)} {type(encoder_hidden_states)}")
                model_pred = model.unet(noisy_latents, timesteps, conditioning.prompt_embeds).sample

    if teacher_mask is not None and teacher_mask.sum() > 0:
        with (torch.no_grad()):
            if teacher_conditioning is None:
                teacher_conditioning = conditioning

            teacher_noisy_latents = _get_noisy_latents(latents, noise, teacher_model.noise_scheduler, timesteps, args.latents_perturbation)
            teacher_target = _get_teacher_target(teacher_model, teacher_conditioning, timesteps, teacher_noisy_latents, model.noise_scheduler.config.prediction_type)

            if log_writer is not None:
                loss_preview_image_rgb = torchvision.utils.make_grid(
                    log_data.loss_preview_image
                )
                log_writer.add_image(
                    tag="loss/teacher target",
                    img_tensor=loss_preview_image_rgb,
                    global_step=global_step,
                )

            target = (
                teacher_target *  teacher_mask.view(-1, 1, 1, 1).expand_as(target).to(target.device)
                +   target     * ~teacher_mask.view(-1, 1, 1, 1).expand_as(target).to(target.device)
            )

    model_pred_wrong_caption = None
    model_pred_wrong_caption_mask = None

    return model_pred, target, model_pred_wrong_caption, model_pred_wrong_caption_mask, noisy_latents




def _get_noisy_latents(latents, noise, noise_scheduler, timesteps, latents_perturbation):
    #logging.info(f"get_noisy_latents timesteps: {timesteps.detach().cpu().tolist()}")
    if not isinstance(noise_scheduler, TrainFlowMatchScheduler):
        timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    if latents_perturbation > 0:
        noisy_latents += torch.randn_like(noisy_latents) * latents_perturbation
    return noisy_latents


def _get_target(latents, noise, noise_scheduler, timesteps):
    #logging.info(f"get_target timesteps: {timesteps.detach().cpu().tolist()}")
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    elif noise_scheduler.config.prediction_type in ['flow-matching', 'flow_prediction']:
        target = noise - latents
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    return target


def _get_noisy_latents_and_target(latents, noise, noise_scheduler: (SchedulerMixin, ConfigMixin), timesteps, latents_perturbation):
    noisy_latents = _get_noisy_latents(latents, noise, noise_scheduler, timesteps, latents_perturbation)
    target = _get_target(latents, noise, noise_scheduler, timesteps)

    return noisy_latents, target


def get_timesteps(batch_size, batch_share_timesteps, device, timesteps_ranges, scheduler):
    """ if continuous_float_timesteps is True, return float timestamps (continuous), otherwise return int timestamps (discrete) """
    timesteps = torch.cat([torch.randint(a, b, size=(1,), device=device)
                           for a,b in timesteps_ranges])
    if batch_share_timesteps:
        timesteps = timesteps[:1].repeat((batch_size,))
    if isinstance(scheduler, TrainFlowMatchScheduler):
        timesteps = scheduler.get_exact_timesteps(timesteps).to(device)
    else:
        timesteps = timesteps.long()
    #logging.info(f"get_timesteps: {timesteps.detach().cpu().tolist()} from ranges: {timesteps_ranges}")
    return timesteps


def get_noise(latents_shape, device, dtype, pyramid_noise_discount, zero_frequency_noise_ratio, batch_share_noise):
    noise = torch.randn(latents_shape, dtype=dtype, device=device)
    if pyramid_noise_discount != None:
        if 0 < pyramid_noise_discount:
            noise = pyramid_noise_like(noise, discount=pyramid_noise_discount)
    if zero_frequency_noise_ratio != None:
        if zero_frequency_noise_ratio < 0:
            zero_frequency_noise_ratio = 0

        # see https://www.crosslabs.org//blog/diffusion-with-offset-noise
        zero_frequency_noise = zero_frequency_noise_ratio * torch.randn(latents_shape[0], latents_shape[1], 1,
                                                                        1, device=device)
        noise = noise + zero_frequency_noise

    if batch_share_noise:
        noise = noise[:1].repeat((noise.shape[0], 1, 1, 1))

    return noise


def pyramid_noise_like(x, discount=0.8):
  b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
  u = torch.nn.Upsample(size=(w, h), mode='bilinear')
  noise = torch.randn_like(x)
  for i in range(10):
    r = random.random()*2+2 # Rather than always going 2x,
    w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
    noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
    if w==1 or h==1: break # Lowest resolution is 1x1
  return noise/noise.std() # Scaled back to roughly unit variance

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
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
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
    SD1_5_LATENT_RGB_FACTORS = torch.tensor([
        #    R        G        B
        [0.3444, 0.1385, 0.0670],  # L1
        [0.1247, 0.4027, 0.1494],  # L2
        [-0.3192, 0.2513, 0.2103],  # L3
        [-0.1307, -0.1874, -0.7445],  # L4
    ])

    def sample_to_lowres_estimated_image(
            samples: torch.Tensor, latent_rgb_factors: torch.Tensor, smooth_matrix: Optional[torch.Tensor] = None
    ):
        if samples.dim() == 4:
            samples = samples[0]
        latent_image = samples.permute(1, 2, 0) @ latent_rgb_factors

        if smooth_matrix is not None:
            latent_image = latent_image.unsqueeze(0).permute(3, 0, 1, 2)
            latent_image = torch.nn.functional.conv2d(latent_image, smooth_matrix.reshape((1, 1, 3, 3)), padding=1)
            latent_image = latent_image.permute(1, 2, 3, 0).squeeze(0)

        latents_ubyte = (
            ((latent_image + 1) / 2).clamp(0, 1).mul(0xFF).byte()  # change scale from -1..1 to 0..1  # to 0..255
        ).cpu()

        return latents_ubyte

    return torch.stack([sample_to_lowres_estimated_image(latents[i], SD1_5_LATENT_RGB_FACTORS).permute(2, 0, 1)
                       for i in range(latents.shape[0])], dim=0)

# ref https://huggingface.co/jimmycarter/LibreFLUX
# "Beta timestep scheduling and timestep stratification"
def get_multirank_stratified_random_timesteps(batch_size, device, distribution: Literal['beta', 'mode', 'boundary-oversampling', 'lognormal'] = 'beta', alpha=2, beta=1.6, mode_scale=0.5, scheduler=None, stratify=True):
    """
    get timesteps, with stratified distribution across batches
    distribution: 'beta' or 'mode'
    alpha, beta: parameters for beta distribution
    mode_scale: parameter for mode distribution. 0 = uniform distribution, 0.5 = ts500 is about 2.2x more likely than tails
    """
    if distribution == 'boundary-oversampling':
        sigmas = _get_boundary_oversampling_sigmas(batch_size, lambda_=1e3)
    elif distribution == 'lognormal':
        std = 1
        mean = 0
        u = torch.randn(batch_size) * std + mean
        sigmas = torch.sigmoid(u)
    else:
        indices = torch.arange(0, batch_size, dtype=torch.float64)
        u = torch.rand(batch_size)
        p = ((indices + u) / batch_size) if stratify else u
        if distribution == 'beta':
            sigmas = _get_beta_sigmas(p, alpha, beta)
        elif distribution == 'mode':
            sigmas = _get_mode_sigmas(p, mode_scale)

    timesteps = (sigmas * 1000).to(device)
    # shuffle
    perm = torch.randperm(timesteps.shape[0])
    timesteps = timesteps[perm]
    timesteps = timesteps.long().clamp(min=0, max=999)
    if isinstance(scheduler, TrainFlowMatchScheduler):
        timesteps = scheduler.get_exact_timesteps(timesteps).to(device)

    #logging.info(
    #    f"get_multirank_stratified_random_timesteps: {timesteps.detach().cpu().tolist()} for batch size {batch_size} alpha {alpha} beta {beta}"
    #)
    return timesteps

def _get_beta_sigmas(p, alpha, beta):
    return torch.from_numpy(sp_beta.ppf(p.numpy(), a=alpha, b=beta))

def _get_mode_sigmas(p, mode_scale):
    return 1 - p - mode_scale * (torch.cos(math.pi * p / 2) ** 2 - 1 + p)

def _get_boundary_oversampling_sigmas(k, lambda_=1.0):
    n_bins = 1000
    t_bins = torch.linspace(0, 1, n_bins)

    # Compute unnormalized weights
    w = 1 + lambda_ * (torch.exp(-10 * t_bins**2) + torch.exp(-10 * (t_bins - 1)**2))

    # Sample indices according to weights
    indices = torch.multinomial(w, k, replacement=True)

    # Add jitter within bins for continuity
    jitter = torch.rand(k) / n_bins
    t_samples = t_bins[indices] + jitter

    return t_samples.clamp(0, 1)


def get_multirank_stratified_random_timesteps_beta(batch_size, device, alpha=2.0, beta=1.6, continuous_float_timesteps=False, offset=0):
    indices = torch.arange(0, batch_size, dtype=torch.float64)
    u = torch.rand(batch_size)
    p = (indices + u) / batch_size
    sigmas = torch.from_numpy(sp_beta.ppf(p.numpy(), a=alpha, b=beta)).to(device)
    timesteps = (sigmas * 1000)

    # shuffle
    perm = torch.randperm(timesteps.shape[0])
    timesteps = timesteps[perm]
    if not continuous_float_timesteps:
        timesteps = timesteps.long().clamp(min=0, max=999)
    #logging.info(
    #    f"get_multirank_stratified_random_timesteps: {timesteps.detach().cpu().tolist()} for batch size {batch_size} alpha {alpha} beta {beta}"
    #)
    return timesteps




