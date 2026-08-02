import json
import logging
import math
import os.path
from dataclasses import dataclass
import random
from typing import Generator, Callable, Any
from datetime import datetime
import gc
import sys

from core.log import setup_local_logger

# Ensure project root is on sys.path so local imports (core/, model/, utils/) resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from colorama import Fore, Style
from diffusers import (
    StableDiffusionPipeline,
    SanaPipeline,
    SanaVideoPipeline,
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
    StableDiffusionXLPipeline,
)

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from compel import CompelForSD
import traceback

from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler
from model.sana_training_model import SanaTrainingModel
from model.training_model import TrainingModel
from core.semaphore_files import check_semaphore_file_and_unlink
from utils.sample_generator_diffusers import generate_images_diffusers, ImageGenerationParams
from utils.train_args import parse_train_args

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
    cfgs: list[float] | None = None

    def __str__(self):
        rep = self.prompt
        if len(self.negative_prompt) > 0:
            rep += f"\n negative prompt: {self.negative_prompt}"
        rep += f"\n seed: {self.seed}"
        return rep


@dataclass
class ExternalBatchRequest:
    json_path: str
    board_name_suffix: str = ""
    offset: int = 0
    limit: int | None = None
    every_n: int = 1
    log_to_tensorboard: bool = True


def chunk_list(l: list, batch_size: int,
               compatibility_test: Callable[[Any,Any], bool]=lambda x,y: True
               ) -> Generator[list, None, None]:
    if not l:
        return
    current_batch = [l[0]]
    for item in l[1:]:
        if len(current_batch) >= batch_size or not compatibility_test(item, current_batch[0]):
            yield current_batch
            current_batch = [item]
        else:
            current_batch.append(item)
    yield current_batch


def _select_batch_items(items: list, offset: int = 0, limit: int | None = None, every_n: int = 1) -> list:
    """
    Apply offset / limit / every_n filtering to a list of items.
    """
    sliced = items[offset:]
    if every_n > 1:
        sliced = sliced[::every_n]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def get_best_size_for_aspect_ratio(aspect_ratio, default_resolution) -> tuple[int, int]:
    sizes = []
    target_pixel_count = default_resolution * default_resolution
    for w in range(256, 1280, 64):
        for h in range(256, 1280, 64):
            if abs((w * h) - target_pixel_count) <= 128 * 64:
                sizes.append((w, h))
    best_size = min(sizes, key=lambda s: abs(1 - (aspect_ratio / (s[0] / s[1]))))
    return best_size


def _pipeline_has_quantized_components(pipe) -> bool:
    """
    Returns True if any pipeline component has been loaded with a quantizer
    (e.g. bitsandbytes 8-bit / 4-bit via transformers).  Such components have
    accelerate dispatch hooks and must NOT be moved via .to().
    """
    for component in pipe.components.values():
        if component is None:
            continue
        if getattr(component, 'is_quantized', False):
            return True
        if getattr(component, 'hf_quantizer', None) is not None:
            return True
    return False


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

    is_video: bool = False
    video_frames: int = 81
    video_fps: int = 16

    def __init__(self,
                 log_folder: str,
                 log_writer: SummaryWriter,
                 default_resolution: int = 768,
                 config_file_path: str|None = None,
                 sample_requests: list[SampleRequest]|None = None,
                 batch_size: int = 2,
                 default_seed: int = 555,
                 default_sample_steps: int|None = 1000,
                 default_sample_epochs: int|None = None,
                 use_xformers: bool = False,
                 use_penultimate_clip_layer: bool = False,
                 is_ztsnr: bool = False,
                 guidance_rescale: float = 0,
                 is_video: bool = False,
                 video_frames: int = 81,
                 video_fps: int = 16):
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
        self.is_video = is_video
        self.video_frames = video_frames
        self.video_fps = video_fps

        self.default_resolution = default_resolution
        self.default_seed = default_seed
        self.sample_steps = default_sample_steps
        self.sample_epochs = default_sample_epochs

        self.external_batch_requests: list[ExternalBatchRequest] = []
        self.external_gen_enabled: bool = False
        self.external_gen_base_url: str | None = None
        self.external_gen_model_save_dir: str | None = None
        self.keep_model_after_external_gen: bool = False
        self.offload_model_for_external_gen: bool = False
        self.sample_requests = sample_requests
        self.reload_config()
        print(f" * SampleGenerator initialized with {len(self.sample_requests)} prompts, generating samples every {self.sample_steps} training steps, using scheduler '{self.scheduler}' with {self.num_inference_steps} inference steps")
        if not os.path.exists(f"{log_folder}/samples/"):
            os.makedirs(f"{log_folder}/samples/")


    def reload_config(self):
        if self.config_file_path is not None:
            try:
                config_file_extension = os.path.splitext(self.config_file_path)[1].lower()
                if config_file_extension == '.txt':
                    self._reload_sample_prompts_txt(self.config_file_path)
                elif config_file_extension == '.json':
                    self._reload_config_json(self.config_file_path)
                else:
                    raise ValueError(f"Unrecognized file type '{config_file_extension}' for sample config, must be .txt or .json")
            except Exception as e:
                traceback.print_exc()
                logging.warning(
                    f" * {Fore.LIGHTYELLOW_EX}Error trying to read sample config from {self.config_file_path}: {Style.RESET_ALL}{e}")
                logging.warning(
                    f"    Edit {self.config_file_path} to fix the problem. It will be automatically reloaded next time samples are due to be generated."
                )
                if self.sample_requests == None:
                    logging.warning(
                        f"    Will generate samples from random training image captions until the problem is fixed.")
                    self.sample_requests = self._make_random_caption_sample_requests()
        old_steps_to_generate = self.steps_to_generate_this_epoch
        self._recompute_sample_steps()
        if self.steps_to_generate_this_epoch != old_steps_to_generate:
            logging.info(f"  Will generate samples at steps {self.steps_to_generate_this_epoch}")

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
            self.external_gen_enabled = config.get('external_gen_enabled', False)
            self.external_gen_base_url = config.get('external_gen_base_url', None)
            self.external_gen_model_save_dir = config.get('external_gen_model_save_dir', None)
            self.keep_model_after_external_gen = config.get('keep_model_after_external_gen', False)
            self.offload_model_for_external_gen = config.get('offload_model_for_external_gen', False)

            if self.external_gen_enabled and not self.external_gen_base_url:
                raise ValueError(
                    "external_gen_enabled=True requires external_gen_base_url to be set "
                    "in the config file"
                )

            sample_requests_config = config.get('samples', None)
            has_external_entry = False
            if sample_requests_config is None:
                self.sample_requests = self._make_random_caption_sample_requests()
            else:
                default_seed = config.get('seed', self.default_seed)
                sample_requests = []
                self.external_batch_requests = []
                for p in sample_requests_config:
                    if 'external_info_dicts_json' in p:
                        has_external_entry = True
                        req = ExternalBatchRequest(
                            json_path=p['external_info_dicts_json'],
                            board_name_suffix=p.get('board_name_suffix', ''),
                            offset=p.get('offset', 0),
                            limit=p.get('limit', None),
                            every_n=p.get('every_n', 1),
                            log_to_tensorboard=p.get('log_to_tensorboard', True),
                        )
                        self.external_batch_requests.append(req)
                    else:
                        cfg_val = p.get('cfg', None)
                        cfgs_val = p.get('cfgs', None)
                        if cfg_val is not None:
                            per_sample_cfgs = [cfg_val]
                        elif cfgs_val is not None:
                            per_sample_cfgs = list(cfgs_val)
                        else:
                            per_sample_cfgs = None
                        sample_requests.append(SampleRequest(
                            prompt=p.get('prompt', ''),
                            negative_prompt=p.get('negative_prompt', ''),
                            seed=p.get('seed', default_seed),
                            size=tuple(p.get('size', None) or
                                       get_best_size_for_aspect_ratio(p.get('aspect_ratio', 1), self.default_resolution)),
                            wants_random_caption=p.get('random_caption', False),
                            cfgs=per_sample_cfgs,
                        ))
                self.sample_requests = sample_requests

            # Backward compat: support top-level invokeai_info_dicts_json
            # Only add if no samples entry uses external_info_dicts_json.
            legacy_json = config.get('invokeai_info_dicts_json', None)
            if legacy_json is not None:
                print(" * Deprecation: 'invokeai_info_dicts_json' at top level — "
                      "use 'external_info_dicts_json' in a 'samples' entry instead.")
                self.external_batch_requests.append(ExternalBatchRequest(json_path=legacy_json, log_to_tensorboard=True))

            if len(self.sample_requests) == 0:
                self.sample_requests = self._make_random_caption_sample_requests()


    @torch.no_grad()
    def generate_samples(self, pipe: StableDiffusionPipeline | None, global_step: int, extra_info: str = "",
                         samples_subdir: str = "samples",
                         project_name: str | None = None,
                         log_time: str | None = None,
                         external_model_key: str | None = None,
                         external_model_save_path: str | None = None):
        """
        generates samples at different cfg scales and saves them to disk.
        samples_subdir: subfolder under log_folder to save to, also used as the tensorboard tag prefix.
                        Use "samples" for normal model output and "samples-ema" for EMA model output.
        """
        # Ensure the target subdirectory exists
        subdir_path = os.path.join(self.log_folder, samples_subdir)
        os.makedirs(subdir_path, exist_ok=True)

        try:
            font = ImageFont.truetype(font="arial.ttf", size=20)
        except:
            font = ImageFont.load_default()

        if not self.show_progress_bars:
            print(f" * Generating samples at gs:{global_step} for {len(self.sample_requests)} prompts (subdir: {samples_subdir})")

        # ── Local regular samples ──
        if pipe is not None:
            sample_index = 0
            with autocast('cuda', enabled=type(pipe) not in (SanaPipeline, SanaVideoPipeline)):
                try:
                    batch: list[SampleRequest]
                    def sample_compatibility_test(a: SampleRequest, b: SampleRequest) -> bool:
                        return a.size == b.size and a.cfgs == b.cfgs
                    batches = list(chunk_list(self.sample_requests, self.batch_size,
                                            compatibility_test=sample_compatibility_test))
                    pbar = tqdm(total=len(batches), disable=not self.show_progress_bars, position=1, leave=False,
                                      desc=f"{Fore.YELLOW}Image samples (batches of {self.batch_size}){Style.RESET_ALL}")
                    if self.use_penultimate_clip_layer:
                        print(f"{Fore.YELLOW}Warning: use_penultimate_clip_layer ignored in samples{Style.RESET_ALL}")
                    if type(pipe) in (StableDiffusionXLPipeline, SanaPipeline, SanaVideoPipeline):
                        print(f"{type(pipe).__name__} -> no Compel")
                        compel = None
                    else:
                        compel = CompelForSD(pipe)
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
                        batch_cfgs = batch[0].cfgs or self.cfgs
                        generators = [torch.Generator(pipe.device).manual_seed(seed) for seed in seeds]

                        batch_images = []
                        for cfg in batch_cfgs:
                            pipe.set_progress_bar_config(disable=not self.show_progress_bars, position=2, leave=False,
                                                         desc=f"{Fore.LIGHTYELLOW_EX}CFG scale {cfg}{Style.RESET_ALL}")

                            conditioning = None if compel is None else compel(prompts, negative_prompt=negative_prompts)

                            embeds, pooled_prompt_embeds, negative_embeds, negative_pooled_embeds = (
                                (None, None, None, None) if conditioning is None else (conditioning.embeds, conditioning.pooled_embeds, conditioning.negative_embeds, conditioning.negative_pooled_embeds)
                            )
                            prompt, negative_prompt = (
                                (prompts, negative_prompts) if conditioning is None else (None, None)
                            )

                            if self.is_video:
                                video_kwargs = dict(frames=self.video_frames)
                                image_kwargs = {}
                            else:
                                video_kwargs = {}
                                image_kwargs = dict(num_images_per_prompt=1)

                            if isinstance(pipe.scheduler, SDPipelineInferenceFlowMatchEulerDiscreteScheduler) and pipe.scheduler.config.use_dynamic_shifting:
                                image_pixel_count = size[0] * size[1]
                                # for linear shift, we go from 1 to 3 over 1 megapixel
                                assert pipe.scheduler.config.time_shift_type == 'linear'
                                shift = 1.0 + 0.5 * (image_pixel_count / 1024 ** 2)
                                pipe.scheduler.set_shift(shift)

                            extra_kwargs = dict(
                                use_resolution_binning=False
                            ) if isinstance(pipe, (SanaPipeline, SanaVideoPipeline)) else dict(
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_embeds,
                                guidance_rescale=self.guidance_rescale
                            )

                            if isinstance(pipe, (SanaPipeline, SanaVideoPipeline)):
                                width = round(size[0] / 32) * 32
                                height = round(size[1] / 32) * 32
                            else:
                                width = size[0]
                                height = size[1]

                            pipe_kwargs = dict(
                                prompt=prompt,
                                prompt_embeds=embeds,
                                negative_prompt=negative_prompt,
                                negative_prompt_embeds=negative_embeds,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=cfg,
                                generator=generators,
                                width=width,
                                height=height,
                                **image_kwargs,
                                **extra_kwargs,
                                **video_kwargs,
                            )

                            if self.is_video:
                                output = pipe(**pipe_kwargs)
                                video_frames_list = output.frames[0]
                                self._save_sample_video(
                                    video_frames_list,
                                    sample_index=sample_index,
                                    global_step=global_step,
                                    prompt=prompts[0],
                                    samples_subdir=samples_subdir,
                                    suffix=f"_cfg{cfg:.1f}"
                                )
                            else:
                                images = pipe(**pipe_kwargs).images
                                for image in images:
                                    draw = ImageDraw.Draw(image)
                                    print_msg = f"cfg:{cfg:.1f}"

                                    l, t, r, b = draw.textbbox(xy=(0, 0), text=print_msg, font=font)
                                    text_width = r - l
                                    text_height = b - t

                                    x = float(image.width - text_width - 10)
                                    y = float(image.height - text_height - 10)

                                    draw.rectangle((x, y, image.width, image.height), fill="black")
                                    draw.text((x, y), print_msg, fill="white", font=font)

                                batch_images.append(images)
                                del images

                        del generators

                        if self.is_video:
                            sample_index += 1
                            pbar.update(1)
                            continue

                        width = size[0] * len(batch_cfgs)
                        height = size[1]

                        for prompt_idx in range(len(batch)):
                            result = Image.new('RGB', (width, height))
                            x_offset = 0

                            for cfg_idx in range(len(batch_cfgs)):
                                image = batch_images[cfg_idx][prompt_idx]
                                result.paste(image, (x_offset, 0))
                                x_offset += image.width

                            prompt = prompts[prompt_idx]
                            self.save_sample_image(result,
                                                   sample_index=sample_index,
                                                   global_step=global_step,
                                                   prompt=prompt,
                                                   is_random_caption=batch[prompt_idx].wants_random_caption,
                                                   extra_info=extra_info,
                                                   samples_subdir=samples_subdir)
                            sample_index += 1

                            del result
                        del batch_images

                        pbar.update(1)

                except Exception as e:
                        print(traceback.format_exc())
                        print("caught exception", e, "generating samples")

        # ── External batch requests ──
        sample_index_offset = len(self.sample_requests)
        for batch_req in self.external_batch_requests:
            if self.external_gen_enabled and self.external_gen_base_url:
                sample_index_offset = self._generate_external_batch(
                    batch_req, global_step, extra_info,
                    samples_subdir, project_name, log_time,
                    external_model_key, external_model_save_path,
                    sample_index=sample_index_offset)
            else:
                sample_index_offset = self._generate_local_external_batch(
                    batch_req, pipe, global_step, extra_info,
                    samples_subdir, index_offset=sample_index_offset)

    def save_sample_image(self,
                          result: PIL.Image,
                          sample_index: int,
                          global_step: int,
                          prompt: str,
                          is_random_caption: bool,
                          extra_info: str,
                          pngmetadata: dict=None,
                          samples_subdir: str = "samples"):
        clean_prompt = clean_filename(prompt)

        subdir_path = os.path.join(self.log_folder, samples_subdir)
        os.makedirs(subdir_path, exist_ok=True)

        result.save(f"{subdir_path}/gs{global_step:05}-{sample_index}-{extra_info}{clean_prompt[:100]}.jpg",
                    format="JPEG", quality=95, optimize=True, progressive=False, pngmetadata=pngmetadata)
        with open(f"{subdir_path}/gs{global_step:05}-{sample_index}-{extra_info}{clean_prompt[:100]}.txt",
                  "w", encoding='utf-8') as f:
            f.write(prompt)
        tfimage = transforms.ToTensor()(result)
        # Tensorboard tag uses the samples_subdir as prefix so EMA and non-EMA images appear in separate groups
        tag_prefix = samples_subdir.replace("/", "_").replace("-", "_")
        if is_random_caption:
            self.log_writer.add_image(tag=f"{tag_prefix}_{sample_index}{extra_info}", img_tensor=tfimage,
                                      global_step=global_step)
        else:
            self.log_writer.add_image(tag=f"{tag_prefix}_{sample_index}_{extra_info}{clean_prompt[:100]}", img_tensor=tfimage,
                                      global_step=global_step)
        del tfimage

    def _save_sample_video(
        self,
        all_cfg_frames: list,
        sample_index: int,
        global_step: int,
        prompt: str,
        samples_subdir: str = "samples",
        suffix: str = ''
    ):
        from diffusers.utils import export_to_video

        clean_prompt = clean_filename(prompt)
        subdir_path = os.path.join(self.log_folder, samples_subdir)
        os.makedirs(subdir_path, exist_ok=True)

        filepath = f"{subdir_path}/gs{global_step:05}-{sample_index}-{clean_prompt[:100]}{suffix}.mp4"
        export_to_video(all_cfg_frames, filepath, fps=self.video_fps)

        with open(f"{subdir_path}/gs{global_step:05}-{sample_index}-{clean_prompt[:100]}{suffix}.txt",
                  "w", encoding='utf-8') as f:
            f.write(prompt)

        import numpy as np
        # log the first, mid, and last frames to tensorboard as a quick preview of the video
        frames_np = [np.array(f) for f in all_cfg_frames]
        mid_idx = len(frames_np) // 2
        grid_np = np.vstack([frames_np[0], frames_np[mid_idx], frames_np[-1]])
        grid_pil = Image.fromarray(grid_np)
        tfimage = transforms.ToTensor()(grid_pil)
        tag_prefix = samples_subdir.replace("/", "_").replace("-", "_")
        self.log_writer.add_image(
            tag=f"{tag_prefix}_{sample_index}",
            img_tensor=tfimage,
            global_step=global_step,
        )

    @torch.no_grad()
    def create_inference_pipe(self, model_being_trained, diffusers_scheduler_config, flow_match_shift=1, flow_match_shift_dynamic=False, use_ema: bool=False):
        """
        creates a pipeline for SD inference
        """
        if type(model_being_trained) != SanaTrainingModel:
            if model_being_trained.is_flow_matching and self.scheduler != 'flow-matching':
                print(f"Warning: model is flow-matching but scheduler is '{self.scheduler}'. Overriding.")
                self.scheduler = 'flow-matching'
        scheduler = self._create_scheduler(diffusers_scheduler_config, flow_match_shift, flow_match_shift_dynamic)
        if use_ema:
            pipe = model_being_trained.build_ema_inference_pipeline(scheduler=scheduler)
        else:
            pipe = model_being_trained.build_inference_pipeline(scheduler=scheduler)
        if self.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        # Quantized components (e.g. bitsandbytes 8-bit) have accelerate dispatch hooks
        # attached and cannot be moved via .to() — they are already on the right device.
        if not _pipeline_has_quantized_components(pipe):
            pipe = pipe.to(model_being_trained.device)
        return pipe

    @torch.no_grad()
    def create_ema_inference_pipe(self, model_being_trained: TrainingModel, diffusers_scheduler_config,
                                  flow_match_shift=1, flow_match_shift_dynamic=False
                                  ) -> StableDiffusionPipeline | StableDiffusionXLPipeline | None:
        """
        Creates an inference pipeline using EMA weights where available,
        falling back to live weights for any component without an EMA counterpart.

        All pipeline components are placed on CPU.  The caller must call
        ``pipe.to(device)`` before inference and ``del pipe`` (+ empty_cache)
        afterwards to avoid leaving stale tensors on the GPU.

        Returns None when no EMA weights exist yet.
        """
        if model_being_trained.is_flow_matching and self.scheduler != 'flow-matching':
            print(f"Warning: model is flow-matching but scheduler is '{self.scheduler}'. Overriding.")
            self.scheduler = 'flow-matching'
        scheduler = self._create_scheduler(diffusers_scheduler_config, flow_match_shift, flow_match_shift_dynamic)
        pipe = model_being_trained.build_ema_inference_pipeline(scheduler=scheduler)
        if pipe is None:
            return None
        if self.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        return pipe


    @torch.no_grad()
    def _create_scheduler(self, scheduler_config: dict, flow_match_shift: int, flow_match_shift_dynamic: bool):
        scheduler = self.scheduler
        if scheduler not in ['ddim', 'pndm', 'ddpm', 'lms', 'euler', 'euler_a', 'kdpm2', 'dpm++',
                             'dpm++_2s', 'dpm++_2m', 'dpm++_sde', 'dpm++_2m_sde',
                             'dpm++_2s_k', 'dpm++_2m_k', 'dpm++_sde_k', 'dpm++_2m_sde_k', 'flow-matching']:
            print(f"unsupported scheduler '{self.scheduler}', falling back to ddim")
            scheduler = 'ddim'

        if scheduler == 'flow-matching':
            return SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
                scheduler_config, shift=flow_match_shift, use_dynamic_shifting=flow_match_shift_dynamic, time_shift_type='linear')
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

    def _generate_local_external_batch(self, batch_req: ExternalBatchRequest, pipe, global_step, extra_info,
                                       samples_subdir: str = "samples", index_offset: int = 0) -> int:
        with open(batch_req.json_path, 'rt') as f:
            all_dicts = json.load(f)
        selected = _select_batch_items(all_dicts, batch_req.offset, batch_req.limit, batch_req.every_n)
        params = [ImageGenerationParams.from_invokeai_metadata(d)
                  for d in selected]
        def save_image(image, sample_index, prompt, pngmetadata):
            return self.save_sample_image(image, sample_index, global_step=global_step, prompt=prompt,
                                          is_random_caption=False, extra_info=extra_info, pngmetadata=pngmetadata,
                                          samples_subdir=samples_subdir)

        flow_match_shift = 1
        flow_match_shift_dynamic = False
        if pipe is not None and hasattr(pipe.scheduler, 'config'):
            flow_match_shift = pipe.scheduler.config.get('shift', 1)
            flow_match_shift_dynamic = pipe.scheduler.config.get('use_dynamic_shifting', False)

        generate_images_diffusers(pipe=pipe,
                                  model_name=f'training-global_step{global_step}-{extra_info}',
                                  model_type='sd-2',
                                  all_params=params,
                                  batch_size=self.batch_size,
                                  media_save_cb=save_image,
                                  extra_cfgs=self.cfgs[1:],
                                  index_offset=index_offset,
                                  flow_match_shift=flow_match_shift,
                                  flow_match_shift_dynamic=flow_match_shift_dynamic,
                                  show_individual_image_progress_bars=self.show_progress_bars)

        return index_offset + len(params)


    def _generate_external_batch(self, batch_req: ExternalBatchRequest, global_step, extra_info,
                                 samples_subdir, project_name, log_time,
                                 external_model_key, external_model_save_path,
                                 sample_index: int = 0) -> int:
        from utils.external_image_generator import (
            build_board_name, create_board, enqueue_batch,
            poll_until_done, download_image,
        )

        # Load and select items
        with open(batch_req.json_path, 'rt') as f:
            all_dicts = json.load(f)
        selected = _select_batch_items(all_dicts, batch_req.offset, batch_req.limit, batch_req.every_n)

        if not selected:
            print(f" * External batch from {batch_req.json_path}: no items selected (offset={batch_req.offset}, limit={batch_req.limit}, every_n={batch_req.every_n})")
            return sample_index

        # Build board name
        board_name = build_board_name(
            project_name or "model",
            log_time or "unknown",
            global_step,
            batch_req.board_name_suffix,
        )

        print(f" * External samples: creating board '{board_name}' with {len(selected)} items")

        board_id = create_board(self.external_gen_base_url, board_name)

        item_ids = enqueue_batch(self.external_gen_base_url, selected, board_id, external_model_key)

        print(f" * External samples: enqueued {len(item_ids)} items, polling...")

        try:
            completed = poll_until_done(self.external_gen_base_url, item_ids)
        except TimeoutError as e:
            print(f" * {Fore.LIGHTYELLOW_EX}Warning: external sample generation timed out: {e}{Style.RESET_ALL}")
            return sample_index

        # Download completed images with sequential indexing
        subdir_path = os.path.join(self.log_folder, samples_subdir)
        for item_id, image_name in completed.items():
            try:
                local_path = download_image(self.external_gen_base_url, image_name, subdir_path)
                # Find corresponding prompt from selected items
                prompt = ""
                for item in selected:
                    prompt = item.get("positive_prompt", item.get("prompt", ""))
                    break
                # Open and re-save with proper sample index via save_sample_image
                from PIL import Image as PILImage
                img = PILImage.open(local_path)
                self.save_sample_image(
                    img, sample_index=sample_index, global_step=global_step,
                    prompt=prompt, is_random_caption=False, extra_info=extra_info,
                    samples_subdir=samples_subdir,
                )
                # Log to tensorboard if requested (before closing img)
                if batch_req.log_to_tensorboard:
                    tfimage = transforms.ToTensor()(img)
                    tag_prefix = samples_subdir.replace("/", "_").replace("-", "_")
                    self.log_writer.add_image(
                        tag=f"{tag_prefix}_{sample_index}_{extra_info}external",
                        img_tensor=tfimage,
                        global_step=global_step,
                    )
                    del tfimage
                img.close()
                # Remove the original server-named file
                local_path.unlink(missing_ok=True)
                sample_index += 1
            except Exception as e:
                print(f" * Warning: failed to download image for item {item_id}: {e}")

        print(f" * External samples: completed {len(completed)} images")
        return sample_index


    def on_epoch_start(self, epoch: int, global_step: int, epoch_length: int):
        self.epoch = epoch
        self.epoch_length = epoch_length
        self.epoch_start_global_step = global_step
        self._recompute_sample_steps()
        print(f"\nSample Generator generating every_n_steps {self.sample_steps} / every_n_epochs {self.sample_epochs} -> steps to generate:", self.steps_to_generate_this_epoch)


    def _recompute_sample_steps(self):
        if self.epoch_length is None:
            # can't recompute sample steps yet (no epoch length)
            return
        if self.sample_steps is not None and self.sample_steps < 0:
            self.sample_epochs = -self.sample_steps / self.epoch_length
            self.sample_steps = None
        elif self.sample_epochs is None:
            every_n_steps = self.sample_steps
            offset = self.epoch_start_global_step % every_n_steps
            self.steps_to_generate_this_epoch = list(range(offset, self.epoch_length, every_n_steps))
        else:
            self.steps_to_generate_this_epoch = get_generate_step_indices(self.epoch, self.epoch_length, every_n_epochs=self.sample_epochs)
        # skip step 0
        if self.epoch_start_global_step == 0 and 0 in self.steps_to_generate_this_epoch:
            self.steps_to_generate_this_epoch.remove(0)



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


def _add_standalone_args(parser):
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace hub model ID for SANA models")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="SANA transformer checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write samples and TensorBoard events")
    parser.add_argument("--te_quantization", type=str, default='none',
                        choices=['none', 'int4', 'int8'],
                        help="Quantization for the SANA text encoder")
    parser.add_argument("--is_video", action="store_true",
                        help="Enable video mode (SanaVideoPipeline)")
    parser.add_argument("--video_frames", type=int, default=81,
                        help="Number of frames for video")
    parser.add_argument("--video_fps", type=int, default=16,
                        help="Target FPS for video")
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma token budget for SANA")


def main():
    torch.autograd.set_grad_enabled(False)
    args = parse_train_args(
        description="EveryDream2 Standalone Sample Generator",
        extra_args_fn=_add_standalone_args,
        require_resume_ckpt=False,
    )
    _ = setup_local_logger(args)

    is_sana = args.model_id is not None

    if is_sana:
        args.disable_textenc_training = True
        if args.resume_from is None:
            args.resume_from = args.resume_ckpt
        if args.resume_ckpt is None:
            args.resume_ckpt = args.resume_from

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.logdir, f"{args.project_name}_{timestamp}")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    log_writer = SummaryWriter(log_dir=output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_sana:
        from model.sana_training_model import load_sana_model
        model = load_sana_model(args)
        if device.type == 'cuda':
            model.transformer.to(device=device, dtype=torch.bfloat16)
            if model.self_flow_proj_head is not None:
                model.self_flow_proj_head.to(device)
            if not args.offload_text_encoder:
                if getattr(model.text_encoder, 'hf_quantizer', None) is None:
                    model.text_encoder.to(device)
            if not args.offload_vae:
                model.vae.to(device)
    else:
        from model.training_model import load_model
        model = load_model(args)
        if device.type == 'cuda':
            model.unet.to(device)
            if not args.offload_text_encoder:
                model.text_encoder.to(device)
                if model.text_encoder_2 is not None:
                    model.text_encoder_2.to(device)
            if not args.offload_vae:
                model.vae.to(device)

    sample_generator = SampleGenerator(
        log_folder=output_dir,
        log_writer=log_writer,
        default_resolution=args.resolution[0],
        config_file_path=args.sample_prompts,
        batch_size=max(1, args.batch_size // 2),
        default_seed=args.seed,
        default_sample_steps=args.sample_steps,
        is_video=getattr(args, 'is_video', False),
        video_frames=getattr(args, 'video_frames', 81),
        video_fps=getattr(args, 'video_fps', 16),
        use_xformers=args.attn_type == "xformers",
        use_penultimate_clip_layer=(args.clip_skip >= 2),
        guidance_rescale=0.7 if getattr(args, 'enable_zero_terminal_snr', False) else 0,
        is_ztsnr=getattr(args, 'enable_zero_terminal_snr', False),
    )

    from utils.inference_context import inference_guard

    if device.type == 'cuda':
        param_bytes = sum(p.numel() * p.element_size() for p in model.transformer.parameters()) if is_sana else sum(p.numel() * p.element_size() for p in model.unet.parameters())
        total_params = sum(p.numel() for p in (model.transformer if is_sana else model.unet).parameters())
        print(f" * Model: {total_params/1e9:.1f}B params, ~{param_bytes/1024**3:.1f}GB")
        print(f" * CUDA memory before pipe: {torch.cuda.memory_allocated(device)/1024**3:.1f}GB / {torch.cuda.memory_reserved(device)/1024**3:.1f}GB (alloc/reserved)")
        print(f" * Resolution: {args.resolution[0]}px, is_video={getattr(args, 'is_video', False)}")

    pipe = None
    try:
        with inference_guard(model.transformer if is_sana else model.unet):
            pipe = sample_generator.create_inference_pipe(
                model_being_trained=model,
                diffusers_scheduler_config=model.noise_scheduler.config,
                flow_match_shift=args.flow_match_shift,
                flow_match_shift_dynamic=args.flow_match_shift_dynamic,
            )

            was_tiling = getattr(pipe.vae, 'use_tiling', None)
            if was_tiling is not None:
                pipe.vae.enable_tiling()

            if device.type == 'cuda':
                print(f" * CUDA memory after pipe: {torch.cuda.memory_allocated(device)/1024**3:.1f}GB / {torch.cuda.memory_reserved(device)/1024**3:.1f}GB")

            sample_generator.generate_samples(pipe, global_step=0, extra_info="standalone")

            if was_tiling is not None and not was_tiling:
                pipe.vae.disable_tiling()
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        log_writer.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


