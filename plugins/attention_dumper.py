import json
import logging
import os
import traceback

import PIL
from torch.utils.tensorboard import SummaryWriter
import torch

from model.training_model import EveryDreamTrainingState
from plugins.plugins import BasePlugin
from utils.sample_generator import SampleGenerator, SampleRequest

from utils.sample_generator_diffusers import ImageGenerationParams, generate_images_diffusers

SEMAPHORE_FILE = 'dump_attention.semaphore'
SAMPLE_REQUEST_FILE = 'dump_attention_image_gen_params.json'

class AttentionDumperPlugin(BasePlugin):

    def __init__(self):
        super().__init__()
        self.save_index = 0

    def should_execute(self) -> bool:
        if not os.path.exists(SEMAPHORE_FILE):
            return False
        os.unlink(SEMAPHORE_FILE)
        return True

    def on_step_end(
        self,
        global_step: int,
        ed_state: EveryDreamTrainingState,
        log_folder: str,
        log_writer: SummaryWriter,
        **kwargs
    ):
        if not self.should_execute():
            return

        try:
            with open(SAMPLE_REQUEST_FILE, 'rt') as f:
                sample_requests_json = json.load(f)

            image_generation_params = [ImageGenerationParams.from_invokeai_metadata(d, original_index=i)
                                       for i, d in enumerate(sample_requests_json)]
            pipe = ed_state.model.build_inference_pipeline()
            for image_index, param in enumerate(image_generation_params):
                with torch.amp.autocast(device_type=ed_state.model.unet.device.type), collect_attention_maps(
                    ed_state.model.unet,
                    ed_state.model.text_encoder.config.hidden_size
                ) as attention_collector:

                    def save_image_cb(image, sample_index, prompt, pngmetadata):
                        _save_sample_image(image, log_folder, sample_index)

                    generate_images_diffusers(pipe=pipe,
                                              model_name=f'training-global_step{global_step}-attention_dump',
                                              model_type='unknown',
                                              all_params=[param],
                                              batch_size=1,
                                              image_save_cb=save_image_cb)
                    for prompt_index, prompt_name in [(0, 'negative'), (1, 'positive')]: # typically negative, positive
                        for key, map in attention_collector.get_stacked_maps_by_layer(
                            latents_width=param.width//8,
                            latents_height=param.height//8,
                            prompt_index=prompt_index,
                            kernel_size=ed_state.model.unet.config["conv_in_kernel"],
                            downsample_padding=ed_state.model.unet.config["downsample_padding"],
                        ):
                            # save as image
                            _save_attention_map(f's{self.save_index}_p-{prompt_name}_{key}', map, log_folder, sample_index=image_index)
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error during attention dumping: {repr(e)}")
            return
        finally:
            self.save_index += 1


def _save_sample_image(image: PIL.Image.Image, log_folder: str, sample_index: int):
    try:
        path = os.path.join(log_folder, f"attention_maps", f"image_{sample_index:03}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
    except Exception as e:
        logging.error(f"Couldn't save sample image {sample_index}: {repr(e)}")
        return

def _save_attention_map(key: str, map: torch.Tensor, log_folder: str, sample_index: int):
    map_bytes = map.mul(0xFF).byte()
    image = PIL.Image.fromarray(map_bytes.numpy(), mode="L")
    try:
        path = os.path.join(log_folder, f"attention_maps", f"image_{sample_index:03}_attn-{key}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
    except Exception as e:
        logging.error(f"Couldn't save attention map for {key}: {repr(e)}")
        return


def _hook_attention(model):
    pass


import math
from typing import Tuple
import torch
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import PIL.Image
from torchvision.transforms.functional import resize as torchvision_resize, InterpolationMode

@contextmanager
def collect_attention_maps(unet, text_encoder_hidden_size: int):
    collector = AttentionMapCollector(unet, text_encoder_hidden_size)
    original_sdp_func = torch.nn.functional.scaled_dot_product_attention
    try:
        yield collector
    finally:
        collector.remove_hooks()
        # in case something went wrong - forward post-hooks might not have been called, so make sure we un-monkeypatch SDP
        torch.nn.functional.scaled_dot_product_attention = original_sdp_func


class AttentionMapCollector:
    def __init__(self, unet, text_encoder_hidden_size: int, verbose: bool=True, every_n_steps: int=None) -> None:
        self.unet = unet
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.attention_maps = defaultdict(list)
        self.hooks = []
        self._original_sdp_func = None
        self.verbose = verbose
        self.register_hooks()

    def register_hooks(self):
        """Register forward hooks on all cross-attention modules in the UNet."""

        def _sdp_with_map_saving(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
            target: AttentionMapCollector = None,
            map_name: str = "",
            is_cross_attn=False,
            **kwargs
        ):
            attn_output, attn_weights = (
                _scaled_dot_product_attention_with_weight_return(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            )
            # merge all heads together by averaging
            # attn_weights has shape [B, num_heads, (H*W), num_tokens]
            attn_weights = torch.mean(attn_weights, dim=1)  # now [B, (H*W), N]
            # maps = torch.mean(maps, dim=1, keepdim=True)
            if self.verbose and map_name not in target.attention_maps:
                # log on first addition
                print(f"storing maps of size {attn_weights.shape} for '{map_name}'")
            target.attention_maps[map_name].append(attn_weights.detach().cpu())
            return attn_output

        def _sdp_with_all_map_saving(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, target: AttentionMapCollector=None, map_name: str= "", is_cross_attn=False):
            attn_output, attn_weights = _scaled_dot_product_attention_with_weight_return(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            # merge all heads together by averaging
            # attn_weights has shape [B, num_heads, (H*W), num_tokens]
            attn_weights = torch.mean(attn_weights, dim=1)  # now [B, (H*W), N]
            # maps = torch.mean(maps, dim=1, keepdim=True)
            if self.verbose and map_name not in target.attention_maps:
                # log on first addition
                print(f"storing maps of size {attn_weights.shape} for '{map_name}'")

            attn_output_to_store = attn_output.detach().cpu()
            if len(attn_output_to_store.shape) == 2:
                attn_output_to_store = attn_output_to_store.unsqueeze(-1)
            target.attention_maps[map_name+'_output'].append(attn_output_to_store)

            if not is_cross_attn:
                # self-attn has a massive final dim (H*W), so reduce by taking max
                attn_weights = torch.max(attn_weights, dim=-1).values
            target.attention_maps[map_name].append(attn_weights.detach().cpu())

            return attn_output

        def pre_hook_fn(module, input, name, is_cross_attn):
            # oerride sdp
            self._original_sdp_func = torch.nn.functional.scaled_dot_product_attention
            torch.nn.functional.scaled_dot_product_attention = partial(_sdp_with_map_saving, target=self, map_name=name, is_cross_attn=is_cross_attn)

        def post_hook_fn(module, input, output):
            # restore sdp
            torch.nn.functional.scaled_dot_product_attention = self._original_sdp_func

        # Find all cross-attention modules and register hooks
        # attn_modules = {n: m for n, m in pipeline.unet.named_modules() if 'attn' in n.lower() and hasattr(m, 'to_q') and hasattr(m, 'to_k') and hasattr(m, 'to_v') and m.cross_attention_dim == pipeline.text_encoder.config.hidden_size}
        for name, module in self.unet.named_modules():
            # Look for cross-attention modules
            if 'attn' in name.lower() and (
                hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v")
            ):
                is_cross_attn = module.cross_attention_dim == self.text_encoder_hidden_size
                #if not is_cross_attn:
                #    continue
                module: torch.nn.Module
                pre_hook = module.register_forward_pre_hook(
                    lambda mod, inp, n=name, is_cross=is_cross_attn: pre_hook_fn(module=mod, input=inp, name=n, is_cross_attn=is_cross)
                )
                self.hooks.append(pre_hook)
                post_hook = module.register_forward_hook(post_hook_fn)
                self.hooks.append(post_hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_maps(self):
        """Clear collected attention maps."""
        self.attention_maps = defaultdict(list)

    def get_attention_maps(self):
        """Return the collected cross-attention maps."""
        return self.attention_maps

    def get_stacked_maps_by_layer(self, latents_width: int, latents_height: int, prompt_index: int, downsample_padding: int = 1, kernel_size: int = 3):
        maps_dict = self.get_attention_maps()
        for key, maps_list in maps_dict.items():
            if type(maps_list) is list:
                maps_full = torch.stack(maps_list, dim=0)  # [steps, B, (H*W), N]
            else:
                assert isinstance(maps_list, torch.Tensor)
                maps_full = maps_list
            maps_full = torch.swapdims(maps_full, 0, 1)  # [B, steps, (H*W), N]
            if key.endswith('_output'):
                # output maps -> extra dim -> reduce by taking max
                # [B, steps, H, (H*W), 64] (num heads? typically 5)
                maps_full = torch.max(maps_full, dim=2).values  # [B, steps, (H*W), 1]
            if len(maps_full.shape) == 3:
                # add batch dim
                maps_full = maps_full.unsqueeze(-1)
            assert len(maps_full.shape) == 4  # [B, steps, (H*W), N] or

            maps_full = maps_full[prompt_index] # only one batch slot -> [steps, (H*W), N]
            # merge by tokens N (take max)
            maps = torch.max(maps_full, dim=-1).values  # [steps, (H*W)]
            # merge over steps
            map = torch.mean(maps, dim=0)  # [(H*W)]
            # normalize
            map = (map - torch.min(map)) / (torch.max(map)-torch.min(map))

            def conv_downsample_size(input_size: int, stride=2):
                return math.floor((input_size + 2 * downsample_padding - kernel_size) / stride) + 1
            this_maps_width = latents_width
            this_maps_height = latents_height
            while this_maps_height * this_maps_width - map.shape[0] > 64:
                this_maps_width = conv_downsample_size(this_maps_width)
                this_maps_height = conv_downsample_size(this_maps_height)

            map = map.reshape(this_maps_height, this_maps_width) # [H, W]
            yield key, map


    def get_stacked_maps(self, latents_width: int, latents_height: int, kernel_size: int, downsample_padding: int,
                         prompt_index: int, eos_token_index: int = None, drop_bos_eos=True,
                         merge_timesteps=False, timestep_weighting_alpha=0, timestep_weighting_beta=1,
                         contrast_boost_pow=2) -> torch.Tensor:
        """
        Scale all collected attention maps to the same size, blend them together and return as an image.
        latents_width and latents_height are the width and height of the latent space, e.g. 64x64 for 512x512 images with 8x downsampling.
        :param latents_width: The width of the latent space.
        :param latents_height: The height of the latent space.
        :param kernel_size: The kernel size of the conv layers in the UNet.
        :param downsample_padding: The padding of the downsampling layers in the UNet.
        :param prompt_index: The index of the prompt to visualize.
        :param eos_token_index: The index of the end-of-sequence token - map stacking will be truncated to this index if provided.
        :param drop_bos_eos: If True, drop the the 0th and last token maps (BOS and EOS). Only applied if eos_token_index is provided. It's up to you to determine if your tokenizer actually outputs BOS/EOS.
        :param merge_timesteps: If True, blend all timesteps together, producing a single map per token. If False, stack timesteps horizontally, producing a grid.
        :param timestep_weighting_alpha: If merge_timesteps is True, this controls the weighting of timesteps (0 means equal weighting, 1 means linear ramp such that later timesteps carry more weight that earlier).
        :param timestep_weighting_beta: If merge_timesteps is True, this controls the sharpness of the weighting curve (1 means linear, >1 means later timesteps are weighted more strongly).
        :param contrast_boost_pow: If >1, increase the contrast of the attention maps by raising them to this power.
        :return: An image containing a vertical stack of blended attention maps, one for each requested token.
        """
        maps_dict = self.get_attention_maps()
        merged = None
        for key, maps in maps_dict.items():
            maps = torch.stack(maps, dim=0)  # [steps, B, (H*W), N]
            maps = torch.swapdims(maps, 0, 1)  # [B, steps, (H*W), N]
            assert len(maps.shape) == 4  # [B, steps, (H*W), N]

            maps = maps[prompt_index:prompt_index + 1, ...] # only one batch slot

            # drop padding tokens, bos, eos
            if eos_token_index is not None:
                maps = maps[..., :eos_token_index + 1]
                if drop_bos_eos:
                    if maps.shape[-1] > 2:
                        # drop EOS and BOS
                        maps = maps[..., 1:-1]
                    elif maps.shape[-1] == 2:
                        # drop BOS, keep EOS (must output at least 1 map)
                        maps = maps[..., 1:]
                    else:
                        raise RuntimeError("maps are missing for bos and/or eos tokens")

            # maps has shape [B, steps, (H*W), N] for N tokens
            # but we want [B, steps, N, H, W] for torchvision_resize
            # first we need to figure out what size to scale to: use standard convolution downsampling formula
            def conv_downsample_size(input_size: int, stride=2):
                return math.floor((input_size + 2 * downsample_padding - kernel_size) / stride) + 1
            this_maps_width = latents_width
            this_maps_height = latents_height
            while this_maps_height * this_maps_width - maps.shape[2] > 64:
                this_maps_width = conv_downsample_size(this_maps_width)
                this_maps_height = conv_downsample_size(this_maps_height)
            if this_maps_width * this_maps_height != maps.shape[2]:
                raise RuntimeError("Unable to determine downscaled attention map size")
            # and we need to do some dimension juggling
            bsz = maps.shape[0]
            num_steps = maps.shape[1]
            num_tokens = maps.shape[-1]
            maps = torch.reshape(torch.swapdims(maps, -2, -1),
                                 [bsz, num_steps, num_tokens, this_maps_height, this_maps_width])

            # scale to output size if necessary
            if this_maps_width != latents_width:
                # torchvision resize expects [..., H, W]
                maps = maps.reshape(bsz * num_steps, num_tokens, this_maps_height, this_maps_width)
                maps = torchvision_resize(maps, [latents_height, latents_width], InterpolationMode.BICUBIC)
                maps = maps.reshape(bsz, num_steps, num_tokens, latents_height, latents_width)

            # normalize in [N, H, W] where N=tokens
            maps_min = torch.amin(maps, dim=(-3, -2, -1), keepdim=True)
            maps_range = torch.amax(maps, dim=(-3, -2, -1), keepdim=True) - maps_min
            # print(f"map {key} size {[this_maps_width, this_maps_height]} range {[maps_min, maps_min + maps_range]}")
            maps_normalized = (maps - maps_min) / maps_range

            # increase contrast
            maps_normalized = torch.pow(maps_normalized, contrast_boost_pow)

            # stack tokens vertically
            maps_stacked = torch.reshape(maps_normalized,
                                         [bsz, num_steps, num_tokens * latents_height, latents_width])
            # map_stacked is [B, steps, (H*W), N]
            if merge_timesteps:
                # blend steps together, producing a single map per token
                num_steps = maps_stacked.shape[1]
                ramp = torch.linspace(0, 1, steps=num_steps,
                                      dtype=maps_stacked.dtype, device=maps_stacked.device)
                # beta > 1 means later timesteps are weighted (even more) strongly
                ramp_curve = ramp.pow(timestep_weighting_beta)

                weights = (1 - timestep_weighting_alpha) * torch.ones(num_steps,
                                                                      dtype=maps_stacked.dtype,
                                                                      device=maps_stacked.device) / num_steps \
                          + timestep_weighting_alpha * ramp_curve / ramp_curve.sum()
                weights = weights / weights.sum()  # Ensure weights sum to 1
                maps_stacked = torch.tensordot(maps_stacked, weights, dims=([1], [0]))
            else:
                # stack steps horizontally, producing a grid
                maps_stacked = maps_stacked.permute(0, 2, 1, 3).reshape(maps_stacked.shape[0], maps_stacked.shape[2],
                                                                        -1)
            # maps_stacked is now [B, (N*H), (W*steps)] where steps==1 if merge_timesteps is True

            if merged is None:
                merged = maps_stacked
            else:
                # screen blend
                merged = 1 - (1 - maps_stacked) * (1 - merged)

        return merged.squeeze(0)

    def get_stacked_maps_images(self, latents_width: int, latents_height: int,
                                prompt_index: int, eos_token_index: int = None, merge_timesteps=False) -> PIL.Image:
        merged = self.get_stacked_maps(latents_width=latents_width, latents_height=latents_height,
                                  prompt_index=prompt_index, eos_token_index=eos_token_index,
                                  merge_timesteps=merge_timesteps)
        # [(N*H), (W*steps)]
        assert len(merged.shape) == 2
        merged_bytes = merged.mul(0xff).byte()
        return PIL.Image.fromarray(merged_bytes.numpy(), mode='L')


def _scaled_dot_product_attention_with_weight_return(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight_with_dropout = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight_with_dropout @ value, attn_weight