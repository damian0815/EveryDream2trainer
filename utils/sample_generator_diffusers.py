import json
import os
from collections import defaultdict
from dataclasses import dataclass
import re
import itertools

import torch
from diffusers import (
    DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler, PNDMScheduler, LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverSinglestepScheduler
)
from diffusers import StableDiffusionPipeline
from compel import Compel
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import hashlib
import math

from tqdm.auto import tqdm

from semaphore_files import check_semaphore_file_and_unlink, _INTERRUPT_SAMPLES_SEMAPHORE_FILE

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ImageGenerationParams:
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    cfg_rescale_multiplier: float
    sampler: str
    prompt: str
    negative_prompt: str
    original_index: int = None

    @classmethod
    def from_invokeai_metadata(cls, m, original_index=None, cfg_rescale_override=None, karras_override=False,
                               resolution_override=None, edit_prompt_fn=None):
        scheduler = 'euler_a' if m['scheduler'] == 'k_euler_a' else m['scheduler']
        if karras_override and scheduler.endswith("_k"):
            scheduler = scheduler[:-2]

        width = int(m['width'])
        height = int(m['height'])
        if resolution_override is not None:
            original_aspect = width / height
            while width < resolution_override and height < resolution_override:
                if width / height < original_aspect:
                    width += 8
                else:
                    height += 8

        prompt = m['positive_prompt'].strip()
        if edit_prompt_fn is not None:
            prompt = edit_prompt_fn(prompt)

        return cls(
            width=width,
            height=height,
            seed=int(m['seed']),
            sampler=scheduler.strip(),
            steps=int(m['steps']),
            cfg=float(m['cfg_scale']),
            cfg_rescale_multiplier=cfg_rescale_override or m.get('cfg_rescale_multiplier', 0),
            prompt=_RE_COMBINE_WHITESPACE.sub(" ", prompt),
            negative_prompt=_RE_COMBINE_WHITESPACE.sub(" ", m['negative_prompt'].strip()),
            original_index=original_index
        )





def get_invariant_metadata(m):
    inv = dict(m)
    del inv['model']
    del inv['canvas_v2_metadata']
    del inv['regions']
    del inv['app_version']
    return inv


def get_metadata_sort_key(m):
    return json.dumps(get_invariant_metadata(m))


def collate_images_by_metadata(metadatas):
    """
    input:  dict of { <path>: <metadata_dict> }
    returns: dict of { <model-invariant metadata json string>: list[<path>] }
    """
    collated = defaultdict(list)
    for i, m in metadatas.items():
        invariant_metadata = json.dumps(get_invariant_metadata(m))
        collated[invariant_metadata].append(i)


def chunk_list(l, n):
    return [l[offset:offset + n] for offset in range(0, len(l), n)]


def get_critical_params(p: ImageGenerationParams) -> str:
    return f'{p.width}x{p.height}-s{p.steps}-c{p.cfg}*{p.cfg_rescale_multiplier}-{p.sampler}'


def get_auto1111_md_for_invoke_md(invokeai_metadata, model_name_override=None, model_hash=None):
    def map_sampler_name(invokeai_scheduler_name: str) -> str:
        invokeai_scheduler_name = invokeai_scheduler_name.lower()
        if invokeai_scheduler_name == 'dpmpp_sde_k':
            return "DPM++ SDE Karras"
        elif invokeai_scheduler_name == 'dpmpp_sde':
            return "DPM++ SDE"
        elif invokeai_scheduler_name == 'dpmpp_2m_sde_k':
            return "DPM++ 2M SDE Karras"
        elif invokeai_scheduler_name == 'dpmpp_2m_sde':
            return "DPM++ 2M SDE"
        elif invokeai_scheduler_name == 'dpmpp_2m':
            return "DPM++ 2M"
        elif invokeai_scheduler_name == 'dpmpp_3m':
            return "DPM++ 3M"
        elif invokeai_scheduler_name == 'euler':
            return "Euler"
        elif invokeai_scheduler_name == 'euler_a':
            return "Euler a"
        elif invokeai_scheduler_name == 'ddim':
            return "DDIM"
        elif invokeai_scheduler_name == 'lms':
            return "LMS"
        elif invokeai_scheduler_name == 'kdpm_2_a':
            return "DPM2 a Karras"
        else:

            raise RuntimeError("Unhandled scheduler:", invokeai_scheduler_name)

    """
format:
<prompt>
Negative prompt: <negative prompt>
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: F245dee52b
    """

    positive_prompt = invokeai_metadata['positive_prompt'].replace('\n', ' ')
    negative_prompt = invokeai_metadata.get('negative_prompt', '').replace('\n', ' ')
    sampler_name = map_sampler_name(invokeai_metadata['scheduler'])
    model_name = invokeai_metadata["model"]["name"] if model_name_override is None else model_name_override

    data = {
        'positive_prompt': positive_prompt,
        'negative_prompt': negative_prompt,
        'sampler_name': sampler_name,
        'model_name': model_name,
        'model_hash': model_hash or "",
        'cfg_scale': invokeai_metadata['cfg_scale'],
        'steps': invokeai_metadata['steps'],
        'seed': invokeai_metadata['seed'],
        'width': invokeai_metadata['width'],
        'height': invokeai_metadata['height'],
    }

    output_metadata = (f"""{data['positive_prompt']}
Negative prompt: {data['negative_prompt']}
Steps: {data['steps']}, Sampler: {data['sampler_name']}, CFG scale: {data['cfg_scale']}, """
                       f"""Seed: {data['seed']}, Size: {data['width']}x{data['height']}, """
                       f"""Model hash: {data['model_hash']}, Model: {data['model_name']}, Version: v3.0.0"""
                       )
    return output_metadata

def update_metadata(md_template, model_node_data, params):
    md = dict(md_template)
    md.update({
        "cfg_scale": params.cfg,
        "cfg_rescale_multiplier": params.cfg_rescale_multiplier,
        "height": params.height,
        "width": params.width,
        "seed": params.seed,
        "positive_prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "model": model_node_data,
        "steps": params.steps,
        "scheduler": params.sampler,
    })
    return md

def create_scheduler(name, scheduler_config: dict):
    scheduler_config = dict(scheduler_config)
    scheduler = name.lower()

    if scheduler == 'ddim':
        return DDIMScheduler.from_config(scheduler_config)
    elif scheduler == 'dpmpp_2s':
        return DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=False, solver_order=2)
    elif scheduler == 'dpmpp_2s_k':
        return DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True, solver_order=2)
    elif scheduler == 'dpmpp_2m' or scheduler == 'dpmpp':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++",
                                                       use_karras_sigmas=False, solver_order=2)
    elif scheduler == 'dpmpp_2m_k':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++",
                                                       use_karras_sigmas=True, solver_order=2)
    elif scheduler == 'dpmpp_3m':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++",
                                                       use_karras_sigmas=False, solver_order=3)
    elif scheduler == 'dpmpp_3m_k':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++",
                                                       use_karras_sigmas=True, solver_order=3)
    elif scheduler == 'dpmpp_sde':
        return DPMSolverSDEScheduler.from_config(scheduler_config, use_karras_sigmas=False, noise_sampler_seed=0)
    elif scheduler == 'dpmpp_sde_k':
        return DPMSolverSDEScheduler.from_config(scheduler_config, use_karras_sigmas=True, noise_sampler_seed=0)
    elif scheduler == 'dpmpp_2m_sde':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="sde-dpmsolver++",
                                                       use_karras_sigmas=False, solver_order=2)
    elif scheduler == 'dpmpp_2m_sde_k':
        return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="sde-dpmsolver++",
                                                       use_karras_sigmas=True, solver_order=2)
    elif scheduler == 'pndm':
        return PNDMScheduler.from_config(scheduler_config)
    elif scheduler == 'ddpm':
        return DDPMScheduler.from_config(scheduler_config)
    elif scheduler == 'lms':
        return LMSDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == 'euler':
        return EulerDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == 'euler_a':
        return EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == 'kdpm_2_a':
        return KDPM2AncestralDiscreteScheduler.from_config(scheduler_config)
    else:
        raise ValueError(f"unknown scheduler '{scheduler}'")


def generate_images_diffusers(pipe: StableDiffusionPipeline, model_name: str, model_type: str,
                              all_params: list[ImageGenerationParams],
                              batch_size: int,
                              image_save_cb,
                              generator_device='cpu',
                              pbar_update_cb=None,
                              pbar_desc=None,
                              extra_cfgs=None,
                              index_offset=0
                              ):
    extra_cfgs = extra_cfgs or []
    all_params = sorted(all_params, key=lambda x: get_critical_params(x))

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    model_node_data = {'name': model_name, "base": model_type, "type": "main"}

    try:
        font = ImageFont.truetype(font="arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    with tqdm(total=len(all_params), desc=pbar_desc, leave=True) as pbar:
        def pbar_update(n, skipped=False):
            pbar.update(n)
            if pbar_update_cb:
                pbar_update_cb(n, skipped=skipped)

        base_sample_index = index_offset

        for k, params in itertools.groupby(all_params, lambda x: get_critical_params(x)):
            if check_semaphore_file_and_unlink(_INTERRUPT_SAMPLES_SEMAPHORE_FILE):
                print("sample generation interrupted")
                break

            params = list(params)
            p0: ImageGenerationParams = params[0]

            pbar.set_postfix_str(f"{k}: {len(params)} images")
            pbar.refresh()

            # reduce the batch size when images are larger
            size_factor = (p0.width * p0.height) / (768 * 768)
            scaled_batch_size = max(batch_size, math.ceil(batch_size / size_factor))

            params_batches = chunk_list(params, scaled_batch_size)

            pipe.scheduler = create_scheduler(p0.sampler, scheduler_config=pipe.scheduler.config)
            for batch in params_batches:
                prompts = [p.prompt for p in batch]
                negative_prompts = [p.negative_prompt for p in batch]
                pipe.set_progress_bar_config(leave=False, desc=f"{len(prompts)} gens")

                prompt_embeds = compel(prompts)
                negative_prompt_embeds = compel(negative_prompts)

                # pad long prompts
                prompt_embeds_padded = []
                negative_prompt_embeds_padded = []
                for i in range(prompt_embeds.shape[0]):
                    padded = compel.pad_conditioning_tensors_to_same_length(
                        [prompt_embeds[i], negative_prompt_embeds[i]])
                    if padded[0].shape[1] > 77:
                        print('padded to shape', padded[0].shape)
                    prompt_embeds_padded.append(padded[0])
                    negative_prompt_embeds_padded.append(padded[1])

                batch_images = []

                cfgs = [p0.cfg] + extra_cfgs
                for cfg in cfgs:
                    generator = [torch.Generator(device=generator_device).manual_seed(p.seed) for p in batch]
                    images = pipe(prompt_embeds=torch.cat(prompt_embeds_padded),
                                  negative_prompt_embeds=torch.cat(negative_prompt_embeds_padded),
                                  generator=generator,
                                  width=p0.width,
                                  height=p0.height,
                                  guidance_scale=cfg,
                                  guidance_rescale=p0.cfg_rescale_multiplier,
                                  num_inference_steps=p0.steps
                                  ).images

                    for image in images:
                        draw = ImageDraw.Draw(image)
                        print_msg = f"cfg:{cfg:.1f}"

                        l, t, r, b = draw.textbbox(xy=(0, 0), text=print_msg, font=font)
                        text_width = r - l
                        text_height = b - t

                        x = float(image.width - text_width - 10)
                        y = float(image.height - text_height - 10)

                        draw.rectangle((x, y, image.width, image.height), fill="white")
                        draw.text((x, y), print_msg, fill="black", font=font)

                    batch_images.append(images)
                    del images

                for prompt_idx in range(len(batch)):
                    #print(f"batch_images[:][{prompt_idx}]: {batch_images[:][prompt_idx]}")
                    result = Image.new('RGB', (p0.width * len(cfgs), p0.height))
                    x_offset = 0

                    for cfg_idx in range(len(cfgs)):
                        image = batch_images[cfg_idx][prompt_idx]
                        result.paste(image, (x_offset, 0))
                        x_offset += image.width

                    metadata = PngInfo()
                    invokeai_metadata = update_metadata(METADATA_TEMPLATE,
                                                        model_node_data,
                                                        batch[prompt_idx])
                    auto1111_metadata = get_auto1111_md_for_invoke_md(invokeai_metadata)
                    metadata.add_text("invokeai_metadata", json.dumps(invokeai_metadata))
                    metadata.add_text("parameters", auto1111_metadata)

                    image_save_cb(result,
                                  sample_index=base_sample_index+prompt_idx,
                                  prompt=batch[prompt_idx].prompt,
                                  pngmetadata=metadata)

                    del result
                del batch_images

                pbar_update(len(batch))
                base_sample_index += len(batch)



METADATA_TEMPLATE = {
          "id": "core_metadata",
          "type": "core_metadata",
          "is_intermediate": True,
          "use_cache": True,
          "generation_mode": "txt2img",
          "cfg_scale": 7.5,
          "cfg_rescale_multiplier": 0.7,
          "width": 640,
          "height": 896,
          "negative_prompt": "negative_prompt",
          "model": {
            "name": "model_name",
            "base": "sd-2",
            "type": "main"
          },
          "steps": 30,
          "rand_device": "cpu",
          "scheduler": "dpmpp_3m",
          "clip_skip": 0,
          "seamless_x": False,
          "seamless_y": False,
          "regions": [],
          "canvas_v2_metadata": {
            "referenceImages": [],
            "controlLayers": [],
            "inpaintMasks": [
              {
                "id": "inpaint_mask",
                "name": None,
                "type": "inpaint_mask",
                "isEnabled": True,
                "isLocked": False,
                "objects": [],
                "opacity": 1,
                "position": {
                  "x": 0,
                  "y": 0
                },
                "fill": {
                  "style": "diagonal",
                  "color": {
                    "r": 224,
                    "g": 117,
                    "b": 117
                  }
                }
              }
            ],
            "rasterLayers": [],
            "regionalGuidance": []
          }
}
