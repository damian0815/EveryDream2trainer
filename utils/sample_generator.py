import json
import logging
import math
import os.path
from dataclasses import dataclass
import random
from typing import Generator, Callable, Any

import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from colorama import Fore, Style
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

from torch import FloatTensor
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from compel import Compel, ReturnedEmbeddingsType
import traceback

from model.training_model import TrainingModel
from semaphore_files import check_semaphore_file_and_unlink
from .sample_generator_diffusers import generate_images_diffusers, ImageGenerationParams

_INTERRUPT_SAMPLES_SEMAPHORE_FILE = 'no_more_samples.semaphore'

def clean_filename(filename):
    """
    removes all non-alphanumeric characters from a string so it is safe to use as a filename
    """
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

@dataclass
class SampleRequest:
    prompt: str
    negative_prompt: str
    seed: int
    size: tuple[int,int]
    wants_random_caption: bool = False

    def __str__(self):
        rep = self.prompt
        if len(self.negative_prompt) > 0:
            rep += f"\n negative prompt: {self.negative_prompt}"
        rep += f"\n seed: {self.seed}"
        return rep


def chunk_list(l: list, batch_size: int,
               compatibility_test: Callable[[Any,Any], bool]=lambda x,y: True
               ) -> Generator[list, None, None]:
    buckets = []
    for item in l:
        compatible_bucket = next((b for b in buckets if compatibility_test(item, b[0])), None)
        if compatible_bucket is not None:
            compatible_bucket.append(item)
        else:
            buckets.append([item])

    for b in buckets:
        for i in range(0, len(b), batch_size):
            yield b[i:i + batch_size]


def get_best_size_for_aspect_ratio(aspect_ratio, default_resolution) -> tuple[int, int]:
    sizes = []
    target_pixel_count = default_resolution * default_resolution
    for w in range(256, 1280, 64):
        for h in range(256, 1280, 64):
            if abs((w * h) - target_pixel_count) <= 128 * 64:
                sizes.append((w, h))
    best_size = min(sizes, key=lambda s: abs(1 - (aspect_ratio / (s[0] / s[1]))))
    return best_size


class SampleGenerator:
    seed: int
    default_resolution: int
    cfgs: list[float] = [7, 4, 1.01]
    scheduler: str = 'ddim'
    num_inference_steps: int = 30
    random_captions = False

    epoch = None
    epoch_length = None
    epoch_start_global_step = None
    steps_to_generate_this_epoch = []

    sample_requests: [str]
    log_folder: str
    log_writer: SummaryWriter

    is_ztsnr: bool

    def __init__(self,
                 log_folder: str,
                 log_writer: SummaryWriter,
                 default_resolution: int,
                 config_file_path: str,
                 batch_size: int,
                 default_seed: int,
                 default_sample_steps: int|None,
                 default_sample_epochs: int|None,
                 use_xformers: bool,
                 use_penultimate_clip_layer: bool,
                 is_ztsnr: bool,
                 guidance_rescale: float = 0):
        self.log_folder = log_folder
        self.log_writer = log_writer
        self.batch_size = batch_size
        self.config_file_path = config_file_path
        self.use_xformers = use_xformers
        self.use_compel = True
        self.show_progress_bars = True
        self.generate_pretrain_samples = False
        self.use_penultimate_clip_layer = use_penultimate_clip_layer
        self.guidance_rescale = guidance_rescale
        self.is_ztsnr = is_ztsnr

        self.default_resolution = default_resolution
        self.default_seed = default_seed
        self.sample_steps = default_sample_steps
        self.sample_epochs = default_sample_epochs

        self.sample_invokeai_info_dicts_json = None
        self.sample_invokeai_info_dicts = None
        self.sample_requests = None
        self.reload_config()
        print(f" * SampleGenerator initialized with {len(self.sample_requests)} prompts, generating samples every {self.sample_steps} training steps, using scheduler '{self.scheduler}' with {self.num_inference_steps} inference steps")
        if not os.path.exists(f"{log_folder}/samples/"):
            os.makedirs(f"{log_folder}/samples/")


    def reload_config(self):
        try:
            config_file_extension = os.path.splitext(self.config_file_path)[1].lower()
            if config_file_extension == '.txt':
                self._reload_sample_prompts_txt(self.config_file_path)
            elif config_file_extension == '.json':
                self._reload_config_json(self.config_file_path)
            else:
                raise ValueError(f"Unrecognized file type '{config_file_extension}' for sample config, must be .txt or .json")
        except Exception as e:
            logging.warning(
                f" * {Fore.LIGHTYELLOW_EX}Error trying to read sample config from {self.config_file_path}: {Style.RESET_ALL}{e}")
            logging.warning(
                f"    Edit {self.config_file_path} to fix the problem. It will be automatically reloaded next time samples are due to be generated."
            )
            if self.sample_requests == None:
                logging.warning(
                    f"    Will generate samples from random training image captions until the problem is fixed.")
                self.sample_requests = self._make_random_caption_sample_requests()

    def update_random_captions(self, possible_captions: list[str]|dict[str, str]):
        possible_captions = [p for p in possible_captions
                             if p is not None and len(p.strip())>0]
        if len(possible_captions) == 0:
            possible_captions = [' ']
        random_prompt_sample_requests = [r for r in self.sample_requests if r.wants_random_caption]
        for i, r in enumerate(random_prompt_sample_requests):
            r.prompt = possible_captions[i % len(possible_captions)]

    def _reload_sample_prompts_txt(self, path):
        with open(path, 'rt') as f:
            self.sample_requests = [SampleRequest(prompt=line.strip(),
                                                  negative_prompt='',
                                                  seed=self.default_seed,
                                                  size=(self.default_resolution, self.default_resolution)
                                                  ) for line in f]
            if len(self.sample_requests) == 0:
                self.sample_requests = self._make_random_caption_sample_requests()

    def _make_random_caption_sample_requests(self):
        num_random_captions = min(4, self.batch_size)
        return [SampleRequest(prompt='',
                              negative_prompt='',
                              seed=self.default_seed,
                              size=(self.default_resolution, self.default_resolution),
                              wants_random_caption=True)
                for _ in range(num_random_captions)]

    def _reload_config_json(self, path):
        with open(path, 'rt') as f:
            config = json.load(f)
            # if keys are missing, keep current values
            self.default_resolution = config.get('resolution', self.default_resolution)
            self.cfgs = config.get('cfgs', self.cfgs)
            self.batch_size = config.get('batch_size', self.batch_size)
            self.scheduler = config.get('scheduler', self.scheduler)
            self.num_inference_steps = config.get('num_inference_steps', self.num_inference_steps)
            self.show_progress_bars = config.get('show_progress_bars', self.show_progress_bars)
            self.generate_pretrain_samples = config.get('generate_pretrain_samples', self.generate_pretrain_samples)
            self.sample_steps = config.get('generate_samples_every_n_steps', self.sample_steps)
            self.sample_epochs = config.get('generate_samples_every_n_epochs', self.sample_epochs)
            self.sample_invokeai_info_dicts_json = config.get('invokeai_info_dicts_json', None)
            self.append_invokeai_info_dicts = config.get('append_invokeai_info_dicts', False)
            sample_requests_config = config.get('samples', None)
            if sample_requests_config is None:
                self.sample_requests = self._make_random_caption_sample_requests()
            else:
                default_seed = config.get('seed', self.default_seed)
                self.sample_requests = [SampleRequest(prompt=p.get('prompt', ''),
                                                      negative_prompt=p.get('negative_prompt', ''),
                                                      seed=p.get('seed', default_seed),
                                                      size=tuple(p.get('size', None) or
                                                                 get_best_size_for_aspect_ratio(p.get('aspect_ratio', 1), self.default_resolution)),
                                                      wants_random_caption=p.get('random_caption', False)
                                                      ) for p in sample_requests_config]
            if len(self.sample_requests) == 0:
                self.sample_requests = self._make_random_caption_sample_requests()
            self._recompute_sample_steps()


    @torch.no_grad()
    def generate_samples(self, pipe: StableDiffusionPipeline, global_step: int, extra_info: str = ""):
        """
        generates samples at different cfg scales and saves them to disk
        """
        try:
            font = ImageFont.truetype(font="arial.ttf", size=20)
        except:
            font = ImageFont.load_default()

        if not self.show_progress_bars:
            print(f" * Generating samples at gs:{global_step} for {len(self.sample_requests)} prompts")

        sample_index = 0
        with autocast():
            try:
                if self.sample_invokeai_info_dicts_json is None or self.append_invokeai_info_dicts:
                    batch: list[SampleRequest]
                    def sample_compatibility_test(a: SampleRequest, b: SampleRequest) -> bool:
                        return a.size == b.size
                    batches = list(chunk_list(self.sample_requests, self.batch_size,
                                            compatibility_test=sample_compatibility_test))
                    pbar = tqdm(total=len(batches), disable=not self.show_progress_bars, position=1, leave=False,
                                      desc=f"{Fore.YELLOW}Image samples (batches of {self.batch_size}){Style.RESET_ALL}")
                    if self.use_penultimate_clip_layer:
                        print(f"{Fore.YELLOW}Warning: use_penultimate_clip_layer ignored in samples{Style.RESET_ALL}")
                    if type(pipe) is StableDiffusionXLPipeline:
                        compel = compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
                    else:
                        compel = Compel(tokenizer=pipe.tokenizer,
                                        text_encoder=pipe.text_encoder
                                        )#,
                                        #use_penultimate_clip_layer=self.use_penultimate_clip_layer)
                    for batch in batches:
                        if check_semaphore_file_and_unlink(_INTERRUPT_SAMPLES_SEMAPHORE_FILE):
                            print("sample generation interrupted")
                            return

                        prompts = [p.prompt for p in batch]
                        negative_prompts = [p.negative_prompt for p in batch]
                        seeds = [(p.seed if p.seed != -1 else random.randint(0, 2 ** 30))
                                 for p in batch]
                        # all sizes in a batch are the same
                        size = batch[0].size
                        generators = [torch.Generator(pipe.device).manual_seed(seed) for seed in seeds]

                        batch_images = []
                        for cfg in self.cfgs:
                            pipe.set_progress_bar_config(disable=not self.show_progress_bars, position=2, leave=False,
                                                         desc=f"{Fore.LIGHTYELLOW_EX}CFG scale {cfg}{Style.RESET_ALL}")
                            if type(pipe) is StableDiffusionXLPipeline:
                                prompt_embeds, pooled_prompt_embeds = compel(prompts)
                                negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompts)
                            else:
                                prompt_embeds = compel(prompts)
                                pooled_prompt_embeds = None
                                negative_pooled_prompt_embeds = compel(negative_prompts)
                                negative_pooled_prompt_embeds = None

                            images = pipe(prompt_embeds=prompt_embeds,
                                          pooled_prompt_embeds=pooled_prompt_embeds,
                                          negative_prompt_embeds=negative_prompt_embeds,
                                          negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                          num_inference_steps=self.num_inference_steps,
                                          num_images_per_prompt=1,
                                          guidance_scale=cfg,
                                          generator=generators,
                                          width=size[0],
                                          height=size[1],
                                          guidance_rescale=self.guidance_rescale
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

                        del generators
                        #print("batch_images:", batch_images)

                        width = size[0] * len(self.cfgs)
                        height = size[1]

                        for prompt_idx in range(len(batch)):
                            #print(f"batch_images[:][{prompt_idx}]: {batch_images[:][prompt_idx]}")
                            result = Image.new('RGB', (width, height))
                            x_offset = 0

                            for cfg_idx in range(len(self.cfgs)):
                                image = batch_images[cfg_idx][prompt_idx]
                                result.paste(image, (x_offset, 0))
                                x_offset += image.width

                            prompt = prompts[prompt_idx]
                            self.save_sample_image(result,
                                                   sample_index=sample_index,
                                                   global_step=global_step,
                                                   prompt=prompt,
                                                   is_random_caption=batch[prompt_idx].wants_random_caption,
                                                   extra_info=extra_info)
                            sample_index += 1

                            del result
                        del batch_images

                        pbar.update(1)

                if self.sample_invokeai_info_dicts_json is not None:
                    with open(self.sample_invokeai_info_dicts_json, 'rt') as f:
                        invokeai_info_dicts = json.load(f)
                    self.generate_invokeai_samples(invokeai_info_dicts, pipe=pipe,
                                                   global_step=global_step, extra_info=extra_info,
                                                   index_offset=len(self.sample_requests))

            except Exception as e:
                    print(traceback.format_exc())
                    print("caught exception", e, "generating samples")

    def save_sample_image(self,
                          result: PIL.Image,
                          sample_index: int,
                          global_step: int,
                          prompt: str,
                          is_random_caption: bool,
                          extra_info: str,
                          pngmetadata: dict=None):
        clean_prompt = clean_filename(prompt)

        result.save(f"{self.log_folder}/samples/gs{global_step:05}-{sample_index}-{extra_info}{clean_prompt[:100]}.jpg",
                    format="JPEG", quality=95, optimize=True, progressive=False, pngmetadata=pngmetadata)
        with open(f"{self.log_folder}/samples/gs{global_step:05}-{sample_index}-{extra_info}{clean_prompt[:100]}.txt",
                  "w", encoding='utf-8') as f:
            f.write(prompt)
        tfimage = transforms.ToTensor()(result)
        if is_random_caption:
            self.log_writer.add_image(tag=f"sample_{sample_index}{extra_info}", img_tensor=tfimage,
                                      global_step=global_step)
        else:
            self.log_writer.add_image(tag=f"sample_{sample_index}_{extra_info}{clean_prompt[:100]}", img_tensor=tfimage,
                                      global_step=global_step)
        del tfimage

    @torch.no_grad()
    def create_inference_pipe(self, model_being_trained: TrainingModel, diffusers_scheduler_config: dict=None, **kwargs) -> StableDiffusionPipeline|StableDiffusionXLPipeline:
        """
        creates a pipeline for SD inference
        """
        scheduler = self._create_scheduler(diffusers_scheduler_config)
        if model_being_trained.is_sdxl:
            pipe = StableDiffusionXLPipeline(
                vae=model_being_trained.vae,
                text_encoder=model_being_trained.text_encoder,
                text_encoder_2=model_being_trained.text_encoder_2,
                tokenizer=model_being_trained.tokenizer,
                tokenizer_2=model_being_trained.tokenizer_2,
                unet=model_being_trained.unet,
                scheduler=scheduler,
            )
        else:
            pipe = StableDiffusionPipeline(
                vae=model_being_trained.vae,
                text_encoder=model_being_trained.text_encoder,
                tokenizer=model_being_trained.tokenizer,
                unet=model_being_trained.unet,
                scheduler=scheduler,
                safety_checker=None, # save vram
                requires_safety_checker=None, # avoid nag
                feature_extractor=None, # must be None if no safety checker
            )
        if self.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        return pipe


    @torch.no_grad()
    def _create_scheduler(self, scheduler_config: dict):
        scheduler = self.scheduler
        if scheduler not in ['ddim', 'pndm', 'ddpm', 'lms', 'euler', 'euler_a', 'kdpm2', 'dpm++',
                             'dpm++_2s', 'dpm++_2m', 'dpm++_sde', 'dpm++_2m_sde',
                             'dpm++_2s_k', 'dpm++_2m_k', 'dpm++_sde_k', 'dpm++_2m_sde_k', 'flow-matching']:
            print(f"unsupported scheduler '{self.scheduler}', falling back to ddim")
            scheduler = 'ddim'

        if scheduler == 'flow-matching':
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
            # hack for StableDiffusionPipeline support
            scheduler.init_noise_sigma = 1
            scheduler.scale_model_input = lambda x, t: x
            return scheduler

        elif scheduler == 'ddim':
            return DDIMScheduler.from_config(scheduler_config)
        elif scheduler == 'dpm++_2s':
            return DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=False)
        elif scheduler == 'dpm++_2s_k':
            return DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
        elif scheduler == 'dpm++' or scheduler == 'dpm++_2m':
            return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++", use_karras_sigmas=False)
        elif scheduler == 'dpm++_2m_k':
            return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
        elif scheduler == 'dpm++_sde':
            return DPMSolverSDEScheduler.from_config(scheduler_config, use_karras_sigmas=False, noise_sampler_seed=0)
        elif scheduler == 'dpm++_sde_k':
            return DPMSolverSDEScheduler.from_config(scheduler_config, use_karras_sigmas=True, noise_sampler_seed=0)
        elif scheduler == 'dpm++_2m_sde':
            return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=False)
        elif scheduler == 'dpm++_2m_sde_k':
            return DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
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
        elif scheduler == 'kdpm2':
            return KDPM2AncestralDiscreteScheduler.from_config(scheduler_config)
        else:
            raise ValueError(f"unknown scheduler '{scheduler}'")

    def generate_invokeai_samples(self, sample_invokeai_info_dicts: list[dict], pipe, global_step, extra_info, index_offset=0):

        params = [ImageGenerationParams.from_invokeai_metadata(d)
                  for d in sample_invokeai_info_dicts]
        def save_image(image, sample_index, prompt, pngmetadata):
            return self.save_sample_image(image, sample_index, global_step=global_step, prompt=prompt,
                                          is_random_caption=False, extra_info=extra_info, pngmetadata=pngmetadata)

        generate_images_diffusers(pipe=pipe,
                                  model_name=f'training-global_step{global_step}-{extra_info}',
                                  model_type='sd-2',
                                  all_params=params,
                                  batch_size=self.batch_size,
                                  image_save_cb=save_image,
                                  extra_cfgs=self.cfgs[1:],
                                  index_offset=index_offset)


    def on_epoch_start(self, epoch: int, global_step: int, epoch_length: int):
        self.epoch = epoch
        self.epoch_length = epoch_length
        self.epoch_start_global_step = global_step
        self._recompute_sample_steps()
        print(f"\nSample Generator generating every_n_steps {self.sample_steps} / every_n_epochs {self.sample_epochs} -> steps to generate:", self.steps_to_generate_this_epoch)


    def _recompute_sample_steps(self):
        if self.sample_steps is not None and self.sample_steps < 0:
            if self.epoch_length is None:
                # can't compute sample steps yet (no epoch length)
                return
            else:
                self.sample_epochs = -self.sample_steps / self.epoch_length
                self.sample_steps = None

        if self.sample_epochs is None:
            every_n_steps = self.sample_steps
            offset = self.epoch_start_global_step % every_n_steps
            self.steps_to_generate_this_epoch = list(range(offset, self.epoch_length, every_n_steps))
        else:
            self.steps_to_generate_this_epoch = get_generate_step_indices(self.epoch, self.epoch_length, every_n_epochs=self.sample_epochs)


    def should_generate_samples(self, global_step, local_step):
        if self.sample_steps is not None and self.sample_steps > 0:
            return ((global_step + 1) % self.sample_steps) == 0
        else:
            return local_step in self.steps_to_generate_this_epoch


def get_generate_step_indices(epoch, epoch_length_steps, every_n_epochs: float, offset: int=0) -> list[int]:
    if every_n_epochs >= 1:
        if ((epoch+1) % round(every_n_epochs)) == 0:
            # last step only
            return [offset + epoch_length_steps-1]
        else:
            return []
    else:
        # subdivide the epoch evenly, by rounding self.every_n_epochs to the nearest clean division of steps
        num_divisions = max(1, min(epoch_length_steps, round(1/every_n_epochs)))
        # if an epoch has eg 100 steps and num_divisions is 2, then validation should occur after steps 49 and 99
        generate_every_n_steps = epoch_length_steps / num_divisions
        return [offset + math.ceil((i+1)*generate_every_n_steps) - 1 for i in range(num_divisions)]
