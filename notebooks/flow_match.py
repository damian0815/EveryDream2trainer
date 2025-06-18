

from diffusers import UNet2DConditionModel

#class UNet2DFlowMatchConditionModel(UNet2DConditionModel):

#    def time_embed




from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('/workspace/models/sdsf-97k-1e5qrt-12k-4e6sqrt-11k-h')
pipe.to('cuda')

from diffusers import UNet2DConditionModel
import torch
encoder_hidden_states = torch.randn((1, 77, 1024), device=pipe.device)
sample = torch.randn((1, 4, 64, 64), device=pipe.device)
t = 1
torch.set_grad_enabled(False)
unet: UNet2DConditionModel = pipe.unet

unet.forward(sample, t, encoder_hidden_states)
