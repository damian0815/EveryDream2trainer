import logging
import math
import random

import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

#from train import pyramid_noise_like, compute_snr


def get_encoder_hidden_states(text_encoder, cuda_caption, clip_skip, embedding_perturbation):
    encoder_output = text_encoder(cuda_caption, output_hidden_states=True)

    if clip_skip > 0:
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(
            encoder_output.hidden_states[-clip_skip])
    else:
        encoder_hidden_states = encoder_output.last_hidden_state

    # https://arxiv.org/pdf/2405.20494
    perturbation_deviation = embedding_perturbation / math.sqrt(encoder_hidden_states.shape[2])
    perturbation_delta = torch.randn_like(encoder_hidden_states) * (perturbation_deviation)
    encoder_hidden_states = encoder_hidden_states + perturbation_delta
    return encoder_hidden_states, encoder_output.pooler_output

def get_latents(image, vae, device, args):
    with torch.no_grad():
        with autocast(enabled=args.amp):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(device)
            latents = vae.encode(pixel_values, return_dict=False)
        del pixel_values
        latents = latents[0].sample() * 0.18215
        return latents


def _get_loss(model_pred, target, caption_str, mask, timesteps, loss_scale, noise_scheduler, do_contrastive_learning, args):

    device = model_pred.device

    if args.min_snr_gamma is not None and args.min_snr_gamma > 0:
        snr = compute_snr(timesteps, noise_scheduler, max_sigma=22000)
        # kohya implementation
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma))
        if noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
            snr_weight = min_snr_gamma / snr + 1
        else:
            snr_weight = min_snr_gamma / snr
        loss_scale = loss_scale * snr_weight.to(loss_scale.device)

    if mask is not None:
        mask = mask.repeat(1, target.shape[1], 1, 1).to(target.device)
    else:
        mask = torch.ones_like(target)

    def compute_loss(model_pred, target, timesteps, loss_scale, reduction="none"):
        loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction=reduction)
        loss_scale = torch.ones(model_pred.shape[0], dtype=torch.float) * loss_scale
        loss_scale = loss_scale.view(-1, 1, 1, 1).expand_as(loss_mse)

        if args.loss_type == "mse_huber":
            early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
            early_timestep_bias = torch.tensor(early_timestep_bias, dtype=torch.float).to(device)
            early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_mse = loss_mse * loss_scale.to(device) * early_timestep_bias
            loss_huber = loss_huber * loss_scale.to(device) * (1.0 - early_timestep_bias)
            loss = loss_mse + loss_huber
            del loss_mse
            del loss_huber
        elif args.loss_type == "huber_mse":
            early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
            early_timestep_bias = torch.tensor(early_timestep_bias, dtype=torch.float).to(device)
            early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_mse = loss_mse * loss_scale.to(device) * (1.0 - early_timestep_bias)
            loss_huber = loss_huber * loss_scale.to(device) * early_timestep_bias
            loss = loss_huber + loss_mse
            del loss_mse
            del loss_huber
        elif args.loss_type == "huber":
            loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction=reduction, delta=1.0)
            loss_huber = loss_huber * loss_scale.to(device)
            loss = loss_huber
            del loss_huber
        else:
            loss_mse = loss_mse * loss_scale.to(device)
            loss = loss_mse
            del loss_mse
        return loss


    if not do_contrastive_learning:
        return compute_loss(model_pred, target, timesteps, loss_scale) * mask
    else:
        positive_loss = compute_loss(model_pred, target, timesteps, loss_scale)
        # Generate negative samples
        # max_negative_loss = torch.tensor(args.contrastive_learning_max_negative_loss,
        #                                 dtype=positive_loss.dtype).to(positive_loss.device)
        negative_loss = torch.zeros_like(positive_loss)
        bsz = model_pred.shape[0]
        num_samples = [0] * bsz
        for i in range(bsz):
            if (caption_str[i] is None or caption_str[i] == " "):
                continue

            for j in range(bsz):
                if (i == j  # skip self
                        or caption_str[j] is None
                        or len(caption_str[j].strip()) == 0  # skip missing or dropout
                        or caption_str[i] == caption_str[j]  # skip equal captions
                ):
                    continue
                if args.contrastive_learning_delta_loss_method:
                    delta_to_wrong = model_pred[j:j + 1] - model_pred[i:i + 1]
                    target_delta_to_wrong = target[j:j + 1] - target[i:i + 1]
                    l_negative = compute_loss(delta_to_wrong.float(),
                                          target_delta_to_wrong.float(),
                                          timesteps=timesteps[j:j + 1], loss_scale=torch.tensor(1),
                                          reduction='none')
                    # l_negative = F.mse_loss(delta_to_wrong.float(), target_delta_to_wrong.float(), reduction='none')
                    negative_loss[i:i + 1] += l_negative
                    del delta_to_wrong
                    del target_delta_to_wrong
                    del l_negative
                else:
                    # dist_negative = torch.nn.functional.pairwise_distance(target[j:j + 1], model_pred[i:i + 1])
                    delta_negative = F.l1_loss(model_pred[i:i + 1].float(), target[j:j + 1].float(), reduction='none')
                    margin = args.contrastive_learning_max_negative_loss
                    l_negative = torch.max(margin - delta_negative, torch.zeros_like(delta_negative))
                    if not args.contrastive_learning_use_l1_loss:
                        l_negative = torch.pow(l_negative, 2.0)

                    diminishing_factor = 1  # torch.exp(-alpha * dist_negative)
                    l_negative_diminished = l_negative * diminishing_factor
                    negative_loss[i:i + 1] += l_negative_diminished
                    del l_negative_diminished
                    del l_negative
                    del delta_negative

                """contrastive_loss = get_loss(model_pred[i:i + 1], target[j:j + 1], timesteps[i:i + 1], loss_scale[i:i + 1])

                dist_negative = torch.sqrt(contrastive_loss)
                diminishing_factor = torch.exp(-args.contrastive_learning_alpha * dist_negative)
                contrastive_loss_diminished = contrastive_loss * diminishing_factor
                negative_loss[i:i + 1] = contrastive_loss_diminished
                del contrastive_loss_diminished
                del contrastive_loss
                """

                # contrastive_loss_clamped_inv = torch.maximum(max_negative_loss - contrastive_loss, torch.zeros_like(max_negative_loss))
                # negative_loss[i:i+1] += contrastive_loss_clamped_inv
                # del contrastive_loss_clamped_inv
                # del contrastive_loss

                num_samples[i] += 1

        # print(' - num contrastive samples', num_samples, ', negative loss', negative_loss.mean())

        # Average over negative samples
        num_samples_safe = torch.tensor(([1] * len(num_samples)
                                         if args.contrastive_learning_no_average_negatives
                                         else [max(1, x) for x in num_samples]
                                         ), device=negative_loss.device)

        negative_loss_scale = args.contrastive_learning_negative_loss_scale / num_samples_safe
        negative_loss_scale = negative_loss_scale.view(-1, 1, 1, 1).expand_as(positive_loss).to(device)
        if args.contrastive_learning_delta_loss_method and args.contrastive_learning_delta_timestep_start >= 0:
            # scale negative loss with timesteps, with a minimum cutoff. ie do more delta loss as noise level increases
            max_timestep = noise_scheduler.config.num_train_timesteps
            negative_loss_timestep_start = args.contrastive_learning_delta_timestep_start
            # linear bias with offset
            early_timestep_bias = torch.maximum((timesteps - negative_loss_timestep_start)
                                                / (max_timestep - negative_loss_timestep_start),
                                                torch.tensor(0).to(timesteps.device))
            early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(positive_loss).to(device)
            negative_loss_scale = negative_loss_scale * early_timestep_bias

        loss = (positive_loss + negative_loss * negative_loss_scale) * mask
        del positive_loss
        del negative_loss

        return loss


def _get_model_prediction_and_target(latents, tokens, noise, timesteps, unet, text_encoder, noise_scheduler, args):
    encoder_hidden_states = _encode_caption_tokens(tokens, text_encoder,
                                                   clip_skip=args.clip_skip,
                                                   embedding_perturbation=args.embedding_perturbation)
    noisy_latents, target = _get_noisy_latents_and_target(latents, noise, noise_scheduler, timesteps,
                                                          args.latents_perturbation)
    del latents
    del noise
    with autocast(enabled=args.amp):
        # print(f"types: {type(noisy_latents)} {type(timesteps)} {type(encoder_hidden_states)}")
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    return model_pred, target


def _encode_caption_tokens(tokens, text_encoder, clip_skip, embedding_perturbation):
    cuda_caption = tokens.to(text_encoder.device)
    encoder_hidden_states, text_features = get_encoder_hidden_states(text_encoder, cuda_caption,
                                                                     clip_skip=clip_skip,
                                                                     embedding_perturbation=embedding_perturbation)
    del cuda_caption
    return encoder_hidden_states


def _get_noisy_latents_and_target(latents, noise, noise_scheduler, timesteps, latents_perturbation):
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    if latents_perturbation > 0:
        noisy_latents += torch.randn_like(noisy_latents) * latents_perturbation

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return noisy_latents, target


def _get_timesteps(batch_size, batch_share_timesteps, device, timesteps_range):
    timestep_start, timestep_end = timesteps_range
    timesteps = torch.randint(timestep_start, timestep_end, (batch_size,), device=device)
    if batch_share_timesteps:
        timesteps = timesteps[:1].repeat((batch_size,))
    timesteps = timesteps.long()
    return timesteps


def _get_noise(latents_shape, device, dtype, pyramid_noise_discount, zero_frequency_noise_ratio, batch_share_noise):
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

def compute_snr(timesteps, noise_scheduler, max_sigma):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    minimal_value = 1/max_sigma
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
