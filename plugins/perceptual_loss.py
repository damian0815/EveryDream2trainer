import copy

import torch
from torch.cuda.amp import autocast

from plugins.plugins import BasePlugin
from train import EveryDreamTrainingState

import torch.nn.functional as F


class SampleCapturedException(Exception):
    def __init__(self, sample):
        self.sample = sample

""" 
runs the actual mid block module, grabs the result, then interrupts execution by throwing it as an exception
gross? yes. works? probably.
"""
class CapturingThrowingWrappedMidBlock(torch.nn.Module):

    def __init__(self, actual_mid_block):
        super().__init__()
        self.actual_mid_block = actual_mid_block

    def __call__(self, *args, **kwargs):
        captured_sample = self.actual_mid_block(
            *args,
            **kwargs
        )
        raise SampleCapturedException(captured_sample)

    """
    sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )"""

class PerceptualLoss(BasePlugin):

    def __init__(self):
        self.unet_frozen = None


    def on_training_start(self, **kwargs):
        ed_state: EveryDreamTrainingState = kwargs['ed_state']
        self.unet_frozen = copy.deepcopy(ed_state.unet)
        actual_mid_block = self.unet_frozen.mid_block
        self.unet_frozen.mid_block = CapturingThrowingWrappedMidBlock(actual_mid_block)
        del(self.unet_frozen.up_blocks)


    def get_model_prediction_and_target(self,
                                        image_latents,
                                        noise,
                                        encoder_hidden_states,
                                        timesteps,
                                        noise_scheduler,
                         ed_state: EveryDreamTrainingState,
                         use_amp: bool):

        #if noise_scheduler.config.prediction_type not in ["v_prediction", "v-prediction"]:
        #    raise RuntimeError("Sorry, perceptual loss only works with V-prediction models")

        x_0 = image_latents
        c = encoder_hidden_states
        t = timesteps

        eps = noise
        x_t = noise_scheduler.add_noise(x_0, eps, t)

        # Pass through model to get v prediction.
        # Then convert v_pred to x_0_pred and eps_pred.
        with autocast(enabled=use_amp):
            #print(f"types: {type(noisy_latents)} {type(timesteps)} {type(encoder_hidden_states)}")
            v_pred = ed_state.unet(x_t, t, c).sample
        x_0_pred, eps_pred = self.get_alpha_stuff(x_t=x_t,
                                                  v_pred=v_pred,
                                                  noise_scheduler=noise_scheduler,
                                                  timesteps=t,
                                                  device=x_0.device,
                                                  dtype=x_0.dtype)

        # Sample new timesteps.
        # Then perform forward diffusion twice.
        # One uses ground truth x_0 and eps.
        # Another uses predicted x_0_pred and eps_pred.
        bsz = x_0.shape[0]
        tt = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=x_0.device)
        tt = tt.long()
        x_tt = noise_scheduler.add_noise(x_0, eps, tt)
        x_tt_pred = noise_scheduler.add_noise(x_0_pred, eps, tt)

        # Pass through perceptual model.
        # Get hidden feature from midblock.
        with autocast(enabled=use_amp):
            try:
                self.unet_frozen(x_tt, tt, c)
            # sorry this is gross
            except SampleCapturedException as e:
                feature_real = e.sample
            try:
                self.unet_frozen(x_tt_pred, tt, c)
            except SampleCapturedException as e:
                feature_pred = e.sample

        return feature_pred, feature_real

    def get_alpha_stuff(self, x_t: torch.FloatTensor, v_pred: torch.FloatTensor, noise_scheduler, timesteps: torch.IntTensor, device, dtype):

        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        x_0_pred = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * v_pred
        eps_pred = sqrt_alpha_prod * v_pred + sqrt_one_minus_alpha_prod * x_t

        return x_0_pred, eps_pred


def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity