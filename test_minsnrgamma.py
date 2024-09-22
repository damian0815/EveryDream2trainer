import unittest

import torch
from diffusers import DDPMScheduler

from train import compute_snr, get_training_noise_scheduler
from utils.unet_utils import enforce_zero_terminal_snr


class MyTestCase(unittest.TestCase):
    def test_something(self):

        #model_root = '/Volumes/Samsung2TB/ml/sd-training/runs/v75/v75-428k-shotdecksf-rebal-teacher1-bs32-14k-fixdata-kissing-36k-half'
        scheduler_config = {
          "_class_name": "DDIMScheduler",
          "_diffusers_version": "0.8.0",
          "beta_end": 0.012,
          "beta_schedule": "scaled_linear",
          "beta_start": 0.00085,
          "clip_sample": False,
          "num_train_timesteps": 1000,
          "prediction_type": "v_prediction",
          "set_alpha_to_one": False,
          "skip_prk_steps": True,
          "steps_offset": 1,
          "trained_betas": None
        }
        noise_scheduler = DDPMScheduler.from_config(scheduler_config)
        ztsnr_betas = enforce_zero_terminal_snr(noise_scheduler.betas).numpy().tolist()
        scheduler_config_ztsnr = dict(scheduler_config)
        scheduler_config_ztsnr.update({'trained_betas': ztsnr_betas})
        noise_scheduler_ztsnr = DDPMScheduler.from_config(scheduler_config_ztsnr)

        timesteps = torch.tensor([0, 1, 998, 999])
        snr_base = compute_snr(timesteps=timesteps, noise_scheduler=noise_scheduler)
        snr_ztsnr = compute_snr(timesteps=timesteps, noise_scheduler=noise_scheduler_ztsnr)
        print(snr_base, snr_ztsnr)

def get_minsnr_weight(timesteps, gamma):
    scheduler_config = {
        "_class_name": "DDIMScheduler",
        "_diffusers_version": "0.8.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "num_train_timesteps": 1000,
        "prediction_type": "v_prediction",
        "set_alpha_to_one": False,
        "skip_prk_steps": True,
        "steps_offset": 1,
        "trained_betas": None
    }
    noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    snr = compute_snr(timesteps=timesteps, noise_scheduler=noise_scheduler)
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        return min_snr_gamma / (snr + 1)
    else:
        return min_snr_gamma / snr

if __name__ == '__main__':
    unittest.main()
