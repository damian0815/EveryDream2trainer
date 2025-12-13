import logging
import math
import random
from typing import Tuple, Optional, Literal

import torch
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.cuda.amp import autocast
import torch.nn.functional as F

from scipy.stats import beta as sp_beta

from diffusers import SchedulerMixin, ConfigMixin, UNet2DConditionModel

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



def get_latents(image, model: TrainingModel, device, args):
    with torch.no_grad():
        with autocast(enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(device, dtype=model.vae.dtype)
            latents = model.vae.encode(pixel_values, return_dict=False)
        del pixel_values
        scaling_factor = 0.13025 if model.is_sdxl else 0.18215
        latents = latents[0].sample() * scaling_factor
        return latents

def get_loss(model_pred, target, model_pred_wrong, model_pred_wrong_mask,
             caption_str, mask, timesteps, loss_scale, noise_scheduler,
             do_contrastive_learning,
             contrastive_learning_negative_loss_scale, args):

    #logging.info(f"get_loss timesteps: {timesteps.detach().cpu().tolist()}")
    device = model_pred.device

    if mask is not None:
        mask = mask.repeat(1, target.shape[1], 1, 1).to(target.device)
    else:
        mask = torch.ones_like(target)

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
            loss = loss * snr_weight.to(loss.device)

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
            loss = loss * mode_weight.to(loss.device)

        return loss

    non_contrastive_loss = compute_loss(model_pred, target, timesteps, loss_scale)
    if not do_contrastive_learning:
        return non_contrastive_loss * mask

    B = model_pred.shape[0]
    temperature = 0.07

    # Contrastive loss
    # Flatten spatial dimensions for similarity computation
    pred_flat = model_pred.reshape(B, -1)  # [B, C*H*W]
    target_flat = target.reshape(B, -1)  # [B, C*H*W]

    # Compute similarity matrix: how similar is each prediction to each target
    # Using negative L2 distance as similarity (could also use cosine)
    # sim[i, j] = similarity between prediction_i and target_j
    sim_matrix = -torch.cdist(pred_flat, target_flat, p=2)  # [B, B]

    # Alternative: cosine similarity
    # pred_norm = F.normalize(pred_flat, p=2, dim=1)
    # target_norm = F.normalize(target_flat, p=2, dim=1)
    # sim_matrix = torch.mm(pred_norm, target_norm.t())  # [B, B]

    # Scale by temperature
    sim_matrix = sim_matrix / temperature

    # InfoNCE: diagonal should be high, off-diagonal should be low
    # Loss for each sample: -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
    labels = torch.arange(B, device=model_pred.device)
    contrastive_loss = F.cross_entropy(sim_matrix, labels)

    # Combined loss
    total_loss = (non_contrastive_loss + args.contrastive_learning_negative_loss_scale * contrastive_loss) * mask  # tune the weight

    return total_loss


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
                                    teacher_unet: UNet2DConditionModel|None=None,
                                    teacher_mask: torch.Tensor|None=None,
                                    teacher_conditioning: Conditioning|None=None,
                                    debug_fake: bool = False
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #logging.info(f"get_model_prediction_and_target timesteps: {timesteps.detach().cpu().tolist()}")
    noisy_latents, target = _get_noisy_latents_and_target(latents, noise, model.noise_scheduler, timesteps, args.latents_perturbation)
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

    with torch.no_grad():
        if teacher_unet is not None:
            if teacher_conditioning is None:
                teacher_conditioning = conditioning
            teacher_target = teacher_unet(noisy_latents.half(), timesteps, teacher_conditioning.prompt_embeds.half()).sample.float()
            target = (
                teacher_target *  teacher_mask.view(-1, 1, 1, 1).expand_as(target).to(target.device)
                +   target     * ~teacher_mask.view(-1, 1, 1, 1).expand_as(target).to(target.device)
            )

    model_pred_wrong_caption = None
    model_pred_wrong_caption_mask = None

    if not skip_contrastive and args.contrastive_learning_info_nce_sample_count > 0:
        def rotate_dim0(x, c):
            return torch.cat([x[-c:], x[:-c]])
        model_pred_wrong_caption = []
        model_pred_wrong_caption_mask = []
        for i in range(min(args.contrastive_learning_info_nce_sample_count, conditioning.prompt_embeds.shape[0]-1)):
            # v_pred_positive = unet(z_t, t, c_positive).
            # v_pred_negative_1 = unet(z_t, t, c_negative_1)
            # v_pred_negative_2 = unet(z_t, t, c_negative_2)
            with autocast(
                enabled=args.amp,
                dtype=torch.bfloat16 if model.is_sdxl else torch.float16,
            ):
                wrong_caption_i = rotate_dim0(conditioning.prompt_embeds, i + 1)
                model_pred_wrong_caption_i = model.unet(noisy_latents, timesteps, wrong_caption_i).sample
                model_pred_wrong_caption.append(model_pred_wrong_caption_i)
                del wrong_caption_i, model_pred_wrong_caption_i
        if len(model_pred_wrong_caption) > 0:
            model_pred_wrong_caption = torch.stack(model_pred_wrong_caption)
            model_pred_wrong_caption_mask = torch.tensor([True] * conditioning.prompt_embeds.shape[0],
                                                              dtype=torch.bool, device=model_pred.device)
        else:
            model_pred_wrong_caption = torch.zeros_like(model_pred).unsqueeze(0)
            model_pred_wrong_caption_mask = torch.tensor([False] * conditioning.prompt_embeds.shape[0],
                                                              dtype=torch.bool, device=model_pred.device)

    return model_pred, target, model_pred_wrong_caption, model_pred_wrong_caption_mask, noisy_latents


def _get_noisy_latents(latents, noise, noise_scheduler, timesteps, latents_perturbation):
    #logging.info(f"get_noisy_latents timesteps: {timesteps.detach().cpu().tolist()}")
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
