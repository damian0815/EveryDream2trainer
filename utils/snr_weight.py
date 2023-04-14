import torch

# adapted from https://github.com/AI-Casanova/sd-scripts/blob/6732df93e2bf119b12ef57a38c995c919a784873/library/custom_train_functions.py
def apply_snr_weight(loss, timesteps, noise_scheduler, gamma):
   alphas_cumprod = noise_scheduler.alphas_cumprod
   sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
   sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
   alpha = sqrt_alphas_cumprod
   sigma = sqrt_one_minus_alphas_cumprod
   all_snr = (alpha / sigma) ** 2
   snr = torch.stack([all_snr[t] for t in timesteps])
   gamma_over_snr = torch.div(torch.ones_like(snr)*gamma,snr)
   snr_weight = torch.minimum(gamma_over_snr,torch.ones_like(gamma_over_snr)).float() #from paper
   loss = loss * snr_weight
   return loss
