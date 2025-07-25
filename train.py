"""
Copyright [2022-2023] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import contextlib
import dataclasses
import os
import pickle
import pprint
import sys
import math
import signal
import argparse
import logging
import time
import gc
import random
import traceback
import shutil

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import safetensors.torch
import torchvision

from colorama import Fore, Style
import numpy as np
import itertools
import torch
import datetime
import json

from compel import Compel
from compel.embeddings_provider import SplitLongTextMode
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler, \
    PNDMScheduler, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
#from diffusers.models import AttentionBlock
from transformers import CLIPTextModel, CLIPTokenizer
#from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import webbrowser
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import DataLoaderMultiAspect

from data.every_dream import EveryDreamBatch, build_torch_dataloader
from data.every_dream_validation import EveryDreamValidator
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID
from flow_match_model import TrainFlowMatchScheduler
from loss import get_noise, get_timesteps, get_model_prediction_and_target, \
    get_loss, get_latents, _encode_caption_tokens, get_timestep_curriculum_range, compute_train_process_01, \
    get_exponential_scaled_value, choose_effective_batch_size, nibble_batch, vae_preview, \
    get_multirank_stratified_random_timesteps
from semaphore_files import check_semaphore_file_and_unlink, _WANT_SAMPLES_SEMAPHORE_FILE, \
    _WANT_VALIDATION_SEMAPHORE_FILE
from utils.huggingface_downloader import try_download_model_from_hf
from utils.convert_diff_to_ckpt import convert as converter
from utils.isolate_rng import isolate_rng
from utils.check_git import check_git
from optimizer.optimizers import EveryDreamOptimizer
from copy import deepcopy

if torch.cuda.is_available():
    from utils.gpu import GPU
import data.aspects as aspects
import data.resolver as resolver
from utils.sample_generator import SampleGenerator

from plugins.plugins import PluginRunner

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9

def get_training_noise_scheduler(train_sampler: str, model_root_folder, trained_betas=None, rescale_betas_zero_snr=False):
    noise_scheduler = None
    if train_sampler.lower() == "pndm":
        logging.info(f" * Using PNDM noise scheduler for training: {train_sampler}")
        noise_scheduler = PNDMScheduler.from_pretrained(model_root_folder,
                                                        subfolder="scheduler",
                                                        trained_betas=trained_betas,
                                                        rescale_betas_zero_snr=rescale_betas_zero_snr)
    elif train_sampler.lower() == "ddim":
        logging.info(f" * Using DDIM noise scheduler for training: {train_sampler}")
        noise_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler",
                                                        trained_betas=trained_betas,
                                                        rescale_betas_zero_snr=rescale_betas_zero_snr)
    elif train_sampler.lower() == "flow-matching":
        logging.info(f" * Using FlowMatching noise scheduler for training: {train_sampler}")
        noise_scheduler = TrainFlowMatchScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
    else:
        logging.info(f" * Using default (DDPM) noise scheduler for training: {train_sampler}")
        noise_scheduler = DDPMScheduler.from_pretrained(model_root_folder, subfolder="scheduler",
                                                        trained_betas=trained_betas,
                                                        rescale_betas_zero_snr=rescale_betas_zero_snr)
    return noise_scheduler

def get_hf_ckpt_cache_path(ckpt_path):
    return os.path.join("ckpt_cache", os.path.basename(ckpt_path))

def convert_to_hf(ckpt_path):

    hf_cache = get_hf_ckpt_cache_path(ckpt_path)
    from utils.unet_utils import get_attn_yaml

    if os.path.isfile(ckpt_path):
        if not os.path.exists(hf_cache):
            os.makedirs(hf_cache)
            logging.info(f"Converting {ckpt_path} to Diffusers format")
            try:
                import utils.convert_original_stable_diffusion_to_diffusers as convert
                convert.convert(ckpt_path, f"ckpt_cache/{ckpt_path}")
            except:
                logging.info("Please manually convert the checkpoint to Diffusers format (one time setup), see readme.")
                exit()
        else:
            logging.info(f"Found cached checkpoint at {hf_cache}")

        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    elif os.path.isdir(hf_cache):
        is_sd1attn, yaml = get_attn_yaml(hf_cache)
        return hf_cache, is_sd1attn, yaml
    else:
        is_sd1attn, yaml = get_attn_yaml(ckpt_path)
        return ckpt_path, is_sd1attn, yaml

class EveryDreamTrainingState:
    def __init__(self,
                 optimizer: EveryDreamOptimizer,
                 train_batch: EveryDreamBatch,
                 unet: UNet2DConditionModel,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 scheduler,
                 inference_scheduler,
                 vae: AutoencoderKL,
                 unet_ema: Optional[UNet2DConditionModel],
                 text_encoder_ema: Optional[CLIPTextModel]
                 ):
        self.optimizer = optimizer
        self.train_batch = train_batch
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.inference_scheduler = inference_scheduler
        self.vae = vae
        self.unet_ema = unet_ema
        self.text_encoder_ema = text_encoder_ema

def convert_diffusers_lora_to_civitai(diffusers_folder, civitai_path):
    broken = safetensors.torch.load_file(os.path.join(diffusers_folder, 'pytorch_lora_weights.safetensors'))

    fixed = {}
    for i, (orig_k, v) in enumerate(broken.items()):
        k = orig_k
        k = k.replace('text_encoder.', 'lora_te_')
        k = k.replace('unet.', 'lora_unet_')
        if '.lora' in k:
            parts = k.split('.lora')
            assert (len(parts) == 2)
            pre = parts[0].replace('.', '_')
            post = parts[1]
            post = post.replace('_linear_layer.', '')
            post = post.replace('.down.', 'down.')
            post = post.replace('.up.', 'up.')
            k = pre + '.lora_' + post
            # if i > offset:
            #    print(parts)
            #    print(k)
        #print(f'{orig_k} -> {k}')
        fixed[k] = v

    safetensors.torch.save_file(fixed, civitai_path)


@torch.no_grad()
def save_model(save_path, ed_state: EveryDreamTrainingState, global_step: int, save_ckpt_dir, yaml_name,
               save_full_precision=False, save_optimizer_flag=False, save_ckpt=True, save_lora=False,
               plugin_runner: PluginRunner = None):
    """
    Save the model to disk
    """

    def save_ckpt_file(diffusers_model_path, sd_ckpt_path):
        nonlocal save_ckpt_dir
        nonlocal save_full_precision
        nonlocal yaml_name

        if save_ckpt_dir is not None:
            sd_ckpt_full = os.path.join(save_ckpt_dir, sd_ckpt_path)
        else:
            sd_ckpt_full = os.path.join(os.curdir, sd_ckpt_path)
            save_ckpt_dir = os.curdir

        half = not save_full_precision

        logging.info(f" * Saving SD model to {sd_ckpt_full}")
        converter(model_path=diffusers_model_path, checkpoint_path=sd_ckpt_full, half=half)

        if yaml_name and yaml_name != "v1-inference.yaml":
            yaml_save_path = f"{os.path.join(save_ckpt_dir, os.path.basename(diffusers_model_path))}.yaml"
            logging.info(f" * Saving yaml to {yaml_save_path}")
            shutil.copyfile(yaml_name, yaml_save_path)


    if global_step is None or global_step == 0:
        logging.warning("  No model to save, something likely blew up on startup, not saving")
        return

    if plugin_runner is not None:
        plugin_runner.run_on_model_save(ed_state=ed_state, save_path=save_path)

    if ed_state.unet_ema is not None or ed_state.text_encoder_ema is not None:
        pipeline_ema = StableDiffusionPipeline(
            vae=ed_state.vae,
            text_encoder=ed_state.text_encoder_ema,
            tokenizer=ed_state.tokenizer,
            unet=ed_state.unet_ema,
            scheduler=ed_state.inference_scheduler,
            safety_checker=None, # save vram
            requires_safety_checker=None, # avoid nag
            feature_extractor=None, # must be none of no safety checker
        )

        diffusers_model_path = save_path + "_ema"
        logging.info(f" * Saving diffusers EMA model to {diffusers_model_path}")
        pipeline_ema.save_pretrained(diffusers_model_path)

        if save_ckpt:
            sd_ckpt_path_ema = f"{os.path.basename(save_path)}_ema.safetensors"

            save_ckpt_file(diffusers_model_path, sd_ckpt_path_ema)

    if save_lora:

        if hasattr(ed_state.unet, 'peft_config'):
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(ed_state.unet)
            )
        else:
            unet_lora_state_dict = None
        if hasattr(ed_state.text_encoder, 'peft_config'):
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(ed_state.text_encoder)
            )
        else:
            text_encoder_lora_state_dict = None

        print("saving diffusers LoRA to", save_path)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_state_dict,
            safe_serialization=True,
        )

        civitai_path = save_path + "_civitai_format.safetensors"
        print("saving civitai format LoRA to", civitai_path)
        convert_diffusers_lora_to_civitai(save_path, civitai_path)


    else:
        pipeline = StableDiffusionPipeline(
            vae=ed_state.vae,
            text_encoder=ed_state.text_encoder,
            tokenizer=ed_state.tokenizer,
            unet=ed_state.unet,
            scheduler=ed_state.inference_scheduler,
            safety_checker=None,  # save vram
            requires_safety_checker=None,  # avoid nag
            feature_extractor=None,  # must be none of no safety checker
        )


        diffusers_model_path = save_path
        logging.info(f" * Saving diffusers model to {diffusers_model_path}")
        pipeline.save_pretrained(diffusers_model_path)

        if save_ckpt:
            sd_ckpt_path = f"{os.path.basename(save_path)}.safetensors"
            save_ckpt_file(diffusers_model_path, sd_ckpt_path)

        if save_optimizer_flag:
            logging.info(f" Saving optimizer state to {save_path}")
            ed_state.optimizer.save(save_path)


def setup_local_logger(args):
    """
    configures logger with file and console logging, logs args, and returns the datestamp
    """
    log_path = args.logdir
    os.makedirs(log_path, exist_ok=True)

    datetimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_folder = os.path.join(log_path, f"{args.project_name}-{datetimestamp}")
    os.makedirs(log_folder, exist_ok=True)

    logfilename = os.path.join(log_folder, f"{args.project_name}-{datetimestamp}.log")

    print(f" logging to {logfilename}")
    logging.basicConfig(filename=logfilename,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                       )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(lambda msg: "Palette images with Transparency expressed in bytes" not in msg.getMessage())
    logging.getLogger().addHandler(console_handler)
    import warnings
    warnings.filterwarnings("ignore", message="UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images")
    #from PIL import Image

    return datetimestamp, log_folder

# def save_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Saves the optimizer state
#     """
#     torch.save(optimizer.state_dict(), path)

# def load_optimizer(optimizer: torch.optim.Optimizer, path: str):
#     """
#     Loads the optimizer state
#     """
#     optimizer.load_state_dict(torch.load(path))

def get_gpu_memory(nvsmi):
    """
    returns the gpu memory usage
    """
    gpu_query = nvsmi.DeviceQuery('memory.used, memory.total')
    gpu_used_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['used'])
    gpu_total_mem = int(gpu_query['gpu'][0]['fb_memory_usage']['total'])
    return gpu_used_mem, gpu_total_mem

def append_epoch_log(global_step: int, epoch_pbar, gpu, log_writer, **logs):
    """
    updates the vram usage for the epoch
    """
    if gpu is not None:
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        log_writer.add_scalar("performance/vram", gpu_used_mem, global_step)
        epoch_mem_color = Style.RESET_ALL
        if gpu_used_mem > 0.93 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTRED_EX
        elif gpu_used_mem > 0.85 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTYELLOW_EX
        elif gpu_used_mem > 0.7 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTGREEN_EX
        elif gpu_used_mem < 0.5 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTBLUE_EX

        if logs is not None:
            epoch_pbar.set_postfix(**logs, vram=f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step}")

def find_last_checkpoint(logdir, is_ema=False):
    """
    Finds the last checkpoint in the logdir, recursively
    """
    last_ckpt = None
    last_date = None

    for root, dirs, files in os.walk(logdir):
        for file in files:
            if os.path.basename(file) == "model_index.json":

                if is_ema and (not root.endswith("_ema")):
                    continue
                elif (not is_ema) and root.endswith("_ema"):
                    continue

                curr_date = os.path.getmtime(os.path.join(root,file))

                if last_date is None or curr_date > last_date:
                    last_date = curr_date
                    last_ckpt = root

    assert last_ckpt, f"Could not find last checkpoint in logdir: {logdir}"
    assert "errored" not in last_ckpt, f"Found last checkpoint: {last_ckpt}, but it was errored, cancelling"

    print(f"    {Fore.LIGHTCYAN_EX}Found last checkpoint: {last_ckpt}, resuming{Style.RESET_ALL}")

    return last_ckpt

def setup_args(args):
    """
    Sets defaults for missing args (possible if missing from json config)
    Forces some args to be set based on others for compatibility reasons
    """
    if args.disable_amp:
        logging.warning(f"{Fore.LIGHTYELLOW_EX} Disabling AMP, not recommended.{Style.RESET_ALL}")
        args.amp = False
    else:
        args.amp = True

    if args.disable_unet_training and args.disable_textenc_training:
        raise ValueError("Both unet and textenc are disabled, nothing to train")

    if args.resume_ckpt == "findlast":
        logging.info(f"{Fore.LIGHTCYAN_EX} Finding last checkpoint in logdir: {args.logdir}{Style.RESET_ALL}")
        # find the last checkpoint in the logdir
        args.resume_ckpt = find_last_checkpoint(args.logdir)

    if (args.ema_resume_model != None) and (args.ema_resume_model == "findlast"):
        logging.info(f"{Fore.LIGHTCYAN_EX} Finding last EMA decay checkpoint in logdir: {args.logdir}{Style.RESET_ALL}")

        args.ema_resume_model = find_last_checkpoint(args.logdir, is_ema=True)

    if not args.shuffle_tags:
        args.shuffle_tags = False

    if not args.keep_tags:
        args.keep_tags = 0

    args.clip_skip = max(min(4, args.clip_skip), 0)

    if args.ckpt_every_n_minutes is None and args.save_every_n_epochs is None:
        logging.info(f"{Fore.LIGHTCYAN_EX} No checkpoint saving specified, defaulting to every 20 minutes.{Style.RESET_ALL}")
        args.ckpt_every_n_minutes = 20

    if args.ckpt_every_n_minutes is None or args.ckpt_every_n_minutes < 1:
        args.ckpt_every_n_minutes = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs is None or args.save_every_n_epochs < 1:
        args.save_every_n_epochs = _VERY_LARGE_NUMBER

    if args.save_every_n_epochs < _VERY_LARGE_NUMBER and args.ckpt_every_n_minutes < _VERY_LARGE_NUMBER:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** Both save_every_n_epochs and ckpt_every_n_minutes are set, this will potentially spam a lot of checkpoints{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** save_every_n_epochs: {args.save_every_n_epochs}, ckpt_every_n_minutes: {args.ckpt_every_n_minutes}{Style.RESET_ALL}")

    if args.cond_dropout > 0.26:
        logging.warning(f"{Fore.LIGHTYELLOW_EX}** cond_dropout is set fairly high: {args.cond_dropout}, make sure this was intended{Style.RESET_ALL}")

    if args.grad_accum > 1:
        logging.info(f"{Fore.CYAN} Batch size: {args.batch_size}, grad accum: {args.grad_accum}, 'effective' batch size: {args.batch_size * args.grad_accum}{Style.RESET_ALL}")


    if args.save_ckpt_dir is not None and not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    if args.rated_dataset:
        args.rated_dataset_target_dropout_percent = min(max(args.rated_dataset_target_dropout_percent, 0), 100)

        logging.info(logging.info(f"{Fore.CYAN} * Activating rated images learning with a target rate of {args.rated_dataset_target_dropout_percent}% {Style.RESET_ALL}"))

    return args


def report_image_train_item_problems(log_folder: str, items: list[ImageTrainItem], batch_size, check_load_all=False, tokenizer=None) -> None:
    undersized_items = [item for item in items if item.is_undersized]
    if len(undersized_items) > 0:
        underized_log_path = os.path.join(log_folder, "undersized_images.txt")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Some images are smaller than the target size, consider using larger images{Style.RESET_ALL}")
        logging.warning(f"{Fore.LIGHTRED_EX} ** Check {underized_log_path} for more information.{Style.RESET_ALL}")
        with open(underized_log_path, "w", encoding='utf-8') as undersized_images_file:
            undersized_images_file.write(f" The following images are smaller than the target size, consider removing or sourcing a larger copy:")
            for undersized_item in undersized_items:
                message = f" *** {undersized_item.pathname} with size: {undersized_item.image_size} is smaller than target size: {undersized_item.target_wh}\n"
                undersized_images_file.write(message)

    if check_load_all:
        data_loader = DataLoaderMultiAspect(items)
        ed_batch = EveryDreamBatch(data_loader, tokenizer=tokenizer)
        error_count = 0
        print("checking we can load all images...")
        for i in tqdm(items):
            try:
                _ = ed_batch.get_image_for_trainer(i)
            except Exception as e:
                logging.error(f" * while loading {i.pathname}, caught {e}")
                error_count += 1
        if error_count > 0:
            logging.error(f"{error_count} broken images found, these will crash training, please delete or fix")
            exit(2)

    # warn on underfilled aspect ratio buckets

    # Intuition: if there are too few images to fill a batch, duplicates will be appended.
    # this is not a problem for large image counts but can seriously distort training if there
    # are just a handful of images for a given aspect ratio.

    # at a dupe ratio of 0.5, all images in this bucket have effective multiplier 1.5,
    # at a dupe ratio 1.0, all images in this bucket have effective multiplier 2.0
    warn_bucket_dupe_ratio = 0.5

    def make_bucket_key(item):
        return (item.batch_id, int(item.target_wh[0]), int(item.target_wh[1]))

    ar_buckets = set(make_bucket_key(i) for i in items)
    for ar_bucket in ar_buckets:
        count = len([i for i in items if make_bucket_key(i) == ar_bucket])
        runt_size = batch_size - (count % batch_size)
        bucket_dupe_ratio = runt_size / count
        if bucket_dupe_ratio > warn_bucket_dupe_ratio:
            aspect_ratio_rational = aspects.get_rational_aspect_ratio((ar_bucket[1], ar_bucket[2]))
            aspect_ratio_description = f"{aspect_ratio_rational[0]}:{aspect_ratio_rational[1]}"
            batch_id_description = "" if ar_bucket[0] == DEFAULT_BATCH_ID else f" for batch id '{ar_bucket[0]}'"
            effective_multiplier = round(1 + bucket_dupe_ratio, 1)
            logging.warning(f" * {Fore.LIGHTRED_EX}Aspect ratio bucket {ar_bucket} has only {count} "
                            f"images{Style.RESET_ALL}. At batch size {batch_size} this makes for an effective multiplier "
                            f"of {effective_multiplier}, which may cause problems. Consider adding {runt_size} or "
                            f"more images with aspect ratio {aspect_ratio_description}{batch_id_description}, or reducing your batch_size.")

def resolve_image_train_items(args: argparse.Namespace, resolution, aspects) -> list[ImageTrainItem]:
    logging.info(f"* DLMA resolution {resolution}, buckets: {aspects}")
    logging.info(" Preloading images...")

    resolved_items = resolver.resolve(args.data_root, args, resolution, aspects)

    # Remove erroneous items
    for item in resolved_items:
        if item.error is not None:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {item.error}")
    resolved_items = [item for item in resolved_items if item.error is None]

    # drop undersized, if requested
    if args.skip_undersized_images:
        full_count = len(resolved_items)
        resolved_items = [i for i in resolved_items if not i.is_undersized]
        post_drop_undersize_count = len(resolved_items)
        if full_count != post_drop_undersize_count:
            logging.info(f" * From {full_count} images, dropped {full_count - post_drop_undersize_count} undersized images ({post_drop_undersize_count} remaining). Remove --skip_undersized_images to log which ones.")

    print (f" * Found {len(resolved_items)} items in '{args.data_root}'")

    return resolved_items

def write_batch_schedule(log_folder: str, train_batch: EveryDreamBatch, epoch: int):
    with open(f"{log_folder}/ep{epoch}_batch_schedule.txt", "w", encoding='utf-8') as f:
        for i in range(len(train_batch.image_train_items)):
            try:
                item: ImageTrainItem = train_batch.image_train_items[i]
                f.write(f"step:{int(i / train_batch.batch_size):05}, wh:{item.target_wh}, r:{item.runt_size}, path:{item.pathname} captions:{item.caption}\n")
            except Exception as e:
                logging.error(f" * Error writing to batch schedule for file path: {item.pathname}")


def read_sample_prompts(sample_prompts_file_path: str):
    sample_prompts = []
    with open(sample_prompts_file_path, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())
    return sample_prompts


def log_args(log_writer, args, optimizer_config, log_folder, log_time):
    arglog = "args:\n"
    for arg, value in sorted(vars(args).items()):
        arglog += f"{arg}={value}, "
    log_writer.add_text("config", arglog)

    args_as_json = json.dumps(vars(args), indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_main.json"), "w") as f:
        f.write(args_as_json)

    optimizer_config_as_json = json.dumps(optimizer_config, indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_opt.json"), "w") as f:
        f.write(optimizer_config_as_json)


def update_ema(model, ema_model, decay, default_device, ema_device):
    with torch.no_grad():
        original_model_on_proper_device = model
        need_to_delete_original = False
        if ema_device != default_device:
            original_model_on_other_device = deepcopy(model)
            original_model_on_proper_device = original_model_on_other_device.to(ema_device, dtype=model.dtype)
            del original_model_on_other_device
            need_to_delete_original = True

        params = dict(original_model_on_proper_device.named_parameters())
        ema_params = dict(ema_model.named_parameters())

        for name in ema_params:
            #ema_params[name].data.mul_(decay).add_(params[name].data, alpha=1 - decay)
            ema_params[name].data = ema_params[name] * decay + params[name].data * (1.0 - decay)

        if need_to_delete_original:
            del(original_model_on_proper_device)

def load_train_json_from_file(args, report_load = False):
    try:
        if report_load:
            print(f"Loading training config from {args.config}.")

        with open(args.config, 'rt') as f:
            read_json = json.load(f)

        args.__dict__.update(read_json)
    except Exception as config_read:
        print(f"Error on loading training config from {args.config}:", config_read)

def main(args):
    """
    Main entry point
    """
    if os.name == 'nt':
        print(" * Windows detected, disabling Triton")
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"

    log_time, log_folder = setup_local_logger(args)
    args = setup_args(args)
    print(f" Args:")
    pprint.pprint(vars(args))

    if args.seed == -1:
        args.seed = random.randint(0, 2**30)
    seed = args.seed
    logging.info(f" Seed: {seed}")
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpuid}")
        gpu = GPU(device)
        torch.backends.cudnn.benchmark = True
    else:
        logging.warning("*** Running on CPU. This is for testing loading/config parsing code only.")
        device = 'cpu'
        gpu = None

    # fix a weird issue with dataloader?
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
    torch.multiprocessing.set_sharing_strategy("file_system")

    #log_folder = os.path.join(args.logdir, f"{args.project_name}_{log_time}")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    def release_memory(model_to_delete, original_device):
        del model_to_delete
        gc.collect()

        if 'cuda' in original_device.type:
            torch.cuda.empty_cache()


    use_ema_dacay_training = (args.ema_decay_rate != None) or (args.ema_strength_target != None)
    ema_model_loaded_from_file = False

    if use_ema_dacay_training:
        ema_device = torch.device(args.ema_device)

    optimizer_state_path = None

    try:
        # check for a local file
        hf_cache_path = get_hf_ckpt_cache_path(args.resume_ckpt)
        if os.path.exists(hf_cache_path) or os.path.exists(args.resume_ckpt):
            model_root_folder, is_sd1attn, yaml = convert_to_hf(args.resume_ckpt)
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.resume_ckpt)
            if args.lora_resume:
                pipe.load_lora_weights(args.lora_resume)
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            unet = pipe.unet
            del pipe
        else:
            if args.lora_resume:
                raise "Can't do lora_resume with downloaded models"
            # try to download from HF using resume_ckpt as a repo id
            downloaded = try_download_model_from_hf(repo_id=args.resume_ckpt)
            if downloaded is None:
                raise ValueError(f"No local file/folder for {args.resume_ckpt}, and no matching huggingface.co repo could be downloaded")
            pipe, model_root_folder, is_sd1attn, yaml = downloaded
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            unet = pipe.unet
            del pipe

        if use_ema_dacay_training and args.ema_resume_model:
            print(f"Loading EMA model: {args.ema_resume_model}")
            ema_model_loaded_from_file=True
            hf_cache_path = get_hf_ckpt_cache_path(args.ema_resume_model)

            if os.path.exists(hf_cache_path) or os.path.exists(args.ema_resume_model):
                ema_model_root_folder, ema_is_sd1attn, ema_yaml = convert_to_hf(args.resume_ckpt)
                text_encoder_ema = CLIPTextModel.from_pretrained(ema_model_root_folder, subfolder="text_encoder")
                unet_ema = UNet2DConditionModel.from_pretrained(ema_model_root_folder, subfolder="unet")

            else:
                # try to download from HF using ema_resume_model as a repo id
                ema_downloaded = try_download_model_from_hf(repo_id=args.ema_resume_model)
                if ema_downloaded is None:
                    raise ValueError(
                        f"No local file/folder for ema_resume_model {args.ema_resume_model}, and no matching huggingface.co repo could be downloaded")
                ema_pipe, ema_model_root_folder, ema_is_sd1attn, ema_yaml = ema_downloaded
                text_encoder_ema = ema_pipe.text_encoder
                unet_ema = ema_pipe.unet
                del ema_pipe

            # Make sure EMA model is on proper device, and memory released if moved
            unet_ema_current_device = next(unet_ema.parameters()).device
            if ema_device != unet_ema_current_device:
                unet_ema_on_wrong_device = unet_ema
                unet_ema = unet_ema.to(ema_device)
                release_memory(unet_ema_on_wrong_device, unet_ema_current_device)

            # Make sure EMA model is on proper device, and memory released if moved
            text_encoder_ema_current_device = next(text_encoder_ema.parameters()).device
            if ema_device != text_encoder_ema_current_device:
                text_encoder_ema_on_wrong_device = text_encoder_ema
                text_encoder_ema = text_encoder_ema.to(ema_device)
                release_memory(text_encoder_ema_on_wrong_device, text_encoder_ema_current_device)

        if args.enable_zero_terminal_snr:
            if args.train_sampler == "flow-matching":
                raise ValueError("can't use ZTSNR with flow matching")
            # Use zero terminal SNR
            from utils.unet_utils import enforce_zero_terminal_snr
            temp_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
            trained_betas = enforce_zero_terminal_snr(temp_scheduler.betas).numpy().tolist()
            inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler", trained_betas=None)
            noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder,
                                                           trained_betas=trained_betas, rescale_betas_zero_snr=False#True
            )
        else:
            inference_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
            noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder)

        tokenizer = CLIPTokenizer.from_pretrained(model_root_folder, subfolder="tokenizer", use_fast=False)

        if args.teacher is not None and args.teacher_p > 0:
            logging.info(f"* Loading teacher model @ p={args.teacher_p} from {args.teacher}")
            teacher_pipeline = StableDiffusionPipeline.from_pretrained(args.teacher)
            if teacher_pipeline.scheduler.config.prediction_type != noise_scheduler.config.prediction_type:
                raise ValueError("Teacher and training model must use same prediction_type")
            teacher_unet = teacher_pipeline.unet.to(device=device, dtype=torch.float16)
            teacher_unet.eval()
            del teacher_pipeline
        else:
            teacher_unet = None

    except Exception as e:
        traceback.print_exc()
        logging.error(" * Failed to load checkpoint *")
        raise

    compel = None
    if args.use_compel:
        compel = Compel(tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        truncate_long_prompts=False,
                        split_long_text_mode = SplitLongTextMode.SENTENCES,
                        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if args.attn_type == "xformers":
        if (args.amp and is_sd1attn) or (not is_sd1attn):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers")
            except Exception as ex:
                logging.warning("failed to load xformers, using default SDP attention instead")
                pass
        elif (args.disable_amp and is_sd1attn):
            logging.info("AMP is disabled but model is SD1.X, xformers is incompatible so using default attention")
    elif args.attn_type == "slice":
        unet.set_attention_slice("auto")
    else:
        logging.info("* Using SDP attention *")

    vae = vae.to(device, dtype=torch.float16 if args.amp else torch.float32)
    unet = unet.to(device, dtype=torch.float32)
    if args.disable_textenc_training and args.amp:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    else:
        text_encoder = text_encoder.to(device, dtype=torch.float32)


    if use_ema_dacay_training:
        if not ema_model_loaded_from_file:
            logging.info(f"EMA decay enabled, creating EMA model.")

            with torch.no_grad():
                if args.ema_device == device:
                    unet_ema = deepcopy(unet)
                    text_encoder_ema = deepcopy(text_encoder)
                else:
                    unet_ema_first = deepcopy(unet)
                    text_encoder_ema_first = deepcopy(text_encoder)
                    unet_ema = unet_ema_first.to(ema_device, dtype=unet.dtype)
                    text_encoder_ema = text_encoder_ema_first.to(ema_device, dtype=text_encoder.dtype)
                    del unet_ema_first
                    del text_encoder_ema_first
        else:
            # Make sure correct types are used for models
            unet_ema = unet_ema.to(ema_device, dtype=unet.dtype)
            text_encoder_ema = text_encoder_ema.to(ema_device, dtype=text_encoder.dtype)
    else:
        unet_ema = None
        text_encoder_ema = None

    try:
        print()
        # currently broken on most systems?
        #unet = torch.compile(unet, mode="max-autotune")
        #text_encoder = torch.compile(text_encoder, mode="max-autotune")
        #vae = torch.compile(vae, mode="max-autotune")
        #logging.info("Successfully compiled models")
    except Exception as ex:
        logging.warning(f"Failed to compile model, continuing anyway, ex: {ex}")
        pass

    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
    except Exception as ex:
        logging.warning(f"Failed to set float32 matmul precision, continuing anyway, ex: {ex}")
        pass

    optimizer_config = None
    optimizer_config_path = args.optimizer_config if args.optimizer_config else "optimizer.json"
    if os.path.exists(os.path.join(os.curdir, optimizer_config_path)):
        with open(os.path.join(os.curdir, optimizer_config_path), "r") as f:
            optimizer_config = json.load(f)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir=log_folder, pytorch=False, tensorboard_x=False, save=False)
        wandb_run = wandb.init(
            project=args.project_name,
            config={"main_cfg": vars(args), "optimizer_cfg": optimizer_config},
            name=args.run_name,
            )
        try:
            if webbrowser.get():
                webbrowser.open(wandb_run.url, new=2)
        except Exception:
            pass

    log_writer = SummaryWriter(log_dir=log_folder,
                               flush_secs=20,
                               comment=args.run_name if args.run_name is not None else log_time,
                              )

    image_train_items = []
    for batch_resolution in args.resolution:
        this_aspects = aspects.get_aspect_buckets(batch_resolution)
        image_train_items.extend(resolve_image_train_items(args, batch_resolution, this_aspects))

    validator = None
    if args.validation_config is not None and args.validation_config != "None":
        validator = EveryDreamValidator(args.validation_config,
                                        default_batch_size=args.forward_slice_size or args.batch_size,
                                        resolution=args.resolution[0],
                                        log_writer=log_writer,
                                        approx_epoch_length=sum([i.multiplier for i in image_train_items])/args.batch_size
                                        )
        # the validation dataset may need to steal some items from image_train_items
        image_train_items = validator.prepare_validation_splits(image_train_items, tokenizer=tokenizer)

    report_image_train_item_problems(log_folder, image_train_items, batch_size=args.batch_size,
                                     check_load_all=args.test_images, tokenizer=tokenizer)

    from plugins.plugins import load_plugin
    if args.plugins is not None:
        plugins = [load_plugin(name) for name in args.plugins]
    else:
        logging.info("No plugins specified")
        plugins = []

    plugin_runner = PluginRunner(plugins=plugins)
    plugin_runner.run_on_model_load(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, optimizer_config=optimizer_config)

    data_loader = DataLoaderMultiAspect(
        image_train_items=image_train_items,
        seed=seed,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        chunk_shuffle_batch_size=args.batch_size,
        batch_id_dropout_p=args.batch_id_dropout_p,
        keep_same_sample_at_different_resolutions_together=args.keep_same_sample_at_different_resolutions_together
    )

    train_batch = EveryDreamBatch(
        data_loader=data_loader,
        debug_level=1,
        conditional_dropout=args.cond_dropout,
        tokenizer=tokenizer,
        seed = seed,
        shuffle_tags=args.shuffle_tags,
        keep_tags=args.keep_tags,
        plugin_runner=plugin_runner,
        rated_dataset=args.rated_dataset,
        rated_dataset_dropout_target=(1.0 - (args.rated_dataset_target_dropout_percent / 100.0)),
        contrastive_learning_batch_ids=args.contrastive_learning_batch_ids,
        contrastive_learning_dropout_p=args.contrastive_learning_dropout_p,
        cond_dropout_noise_p=args.cond_dropout_noise_p,
        use_masks=args.use_masks
    )

    torch.cuda.benchmark = False

    epoch_len = math.ceil(len(train_batch) / args.batch_size)


    if use_ema_dacay_training:
        args.ema_update_interval = args.ema_update_interval * args.grad_accum
        if args.ema_strength_target != None:
            total_number_of_steps: float = epoch_len * args.max_epochs
            total_number_of_ema_update: float = total_number_of_steps / args.ema_update_interval
            args.ema_decay_rate = args.ema_strength_target ** (1 / total_number_of_ema_update)

            logging.info(f"ema_strength_target is {args.ema_strength_target}, calculated ema_decay_rate will be: {args.ema_decay_rate}.")

        logging.info(
            f"EMA decay enabled, with ema_decay_rate {args.ema_decay_rate}, ema_update_interval: {args.ema_update_interval}, ema_device: {args.ema_device}.")

    ed_optimizer = EveryDreamOptimizer(args,
                                       optimizer_config,
                                       text_encoder,
                                       unet,
                                       epoch_len,
                                       plugin_runner,
                                       log_writer)

    log_args(log_writer, args, optimizer_config, log_folder, log_time)

    sample_generator = SampleGenerator(log_folder=log_folder, log_writer=log_writer,
                                       default_resolution=args.resolution[0], default_seed=args.seed,
                                       config_file_path=args.sample_prompts,
                                       batch_size=max(1,args.batch_size//2),
                                       default_sample_steps=args.sample_steps,
                                       default_sample_epochs=1,
                                       use_xformers=args.attn_type == "xformers",
                                       use_penultimate_clip_layer=(args.clip_skip >= 2),
                                       guidance_rescale=0.7 if args.enable_zero_terminal_snr else 0,
                                       is_ztsnr=args.enable_zero_terminal_snr
                                       )

    """
    Train the model

    """
    print(f" {Fore.LIGHTGREEN_EX}** Welcome to EveryDream trainer 2.0!**{Style.RESET_ALL}")
    print(f" (C) 2022-2023 Victor C Hall  This program is licensed under AGPL 3.0 https://www.gnu.org/licenses/agpl-3.0.en.html")
    print()
    print("** Trainer Starting **")

    global interrupted
    interrupted = False

    def sigterm_handler(signum, frame):
        """
        handles sigterm
        """
        is_main_thread = (torch.utils.data.get_worker_info() == None)
        if is_main_thread:
            global interrupted
            if not interrupted:
                interrupted=True
                global global_step
                interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step}")
                print()
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
                logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                time.sleep(2) # give opportunity to ctrl-C again to cancel save
                save_model(interrupted_checkpoint_path, global_step=global_step, ed_state=make_current_ed_state(),
                           save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
                           save_optimizer_flag=True, save_ckpt=not args.no_save_ckpt)
            exit(_SIGTERM_EXIT_CODE)
        else:
            # non-main threads (i.e. dataloader workers) should exit cleanly
            exit(0)

    signal.signal(signal.SIGINT, sigterm_handler)

    if not os.path.exists(f"{log_folder}/samples/"):
        os.makedirs(f"{log_folder}/samples/")

    if gpu is not None:
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        logging.info(f" Pretraining GPU Memory: {gpu_used_mem} / {gpu_total_mem} MB")
    logging.info(f" saving ckpts every {args.ckpt_every_n_minutes} minutes")
    logging.info(f" saving ckpts every {args.save_every_n_epochs } epochs")

    train_dataloader = build_torch_dataloader(train_batch, batch_size=args.batch_size)

    unet.train() if (args.gradient_checkpointing or not args.disable_unet_training) else unet.eval()
    text_encoder.train() if not args.disable_textenc_training else text_encoder.eval()

    logging.info(f" unet device: {unet.device}, precision: {unet.dtype}, training: {unet.training}")
    logging.info(f" text_encoder device: {text_encoder.device}, precision: {text_encoder.dtype}, training: {text_encoder.training}")
    logging.info(f" vae device: {vae.device}, precision: {vae.dtype}, training: {vae.training}")
    logging.info(f" scheduler: {noise_scheduler.__class__}")

    logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.grad_accum}{Style.RESET_ALL}"),
    logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")

    epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True, dynamic_ncols=True)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")
    epoch_times = []

    global global_step
    global_step = 0
    training_start_time = time.time()
    last_epoch_saved_time = training_start_time

    append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    loss_log_step = []
    loss_log_step_cd = []
    loss_log_step_non_cd = []
    loss_preview_image = None
    loss_per_timestep: dict[int, dict[int, tuple[float, int]]] = {resolution: {} for resolution in args.resolution}

    assert len(train_batch) > 0, "train_batch is empty, check that your data_root is correct"

    def generate_samples(global_step: int, batch):
        nonlocal unet
        nonlocal text_encoder
        nonlocal unet_ema
        nonlocal text_encoder_ema

        with isolate_rng():
            sample_generator.reload_config()
            flattened_captions_dict = [v
                                       for _, l in batch["captions"].items()
                                       for v in l]
            sample_generator.update_random_captions(flattened_captions_dict)

            models_info = []

            if (args.ema_decay_rate is None) or args.ema_sample_nonema_model:
                models_info.append({"is_ema": False, "swap_required": False})

            if (args.ema_decay_rate is not None) and args.ema_sample_ema_model:
                models_info.append({"is_ema": True, "swap_required": ema_device != device})

            for model_info in models_info:

                extra_info: str = ""

                if model_info["is_ema"]:
                    current_unet, current_text_encoder = unet_ema, text_encoder_ema
                    extra_info = "_ema"
                else:
                    current_unet, current_text_encoder = unet, text_encoder

                torch.cuda.empty_cache()


                if model_info["swap_required"]:
                    with torch.no_grad():
                        unet_unloaded = unet.to(ema_device)
                        del unet
                        text_encoder_unloaded = text_encoder.to(ema_device)
                        del text_encoder

                        current_unet = unet_ema.to(device)
                        del unet_ema
                        current_text_encoder = text_encoder_ema.to(device)
                        del text_encoder_ema
                        gc.collect()
                        torch.cuda.empty_cache()



                inference_pipe = sample_generator.create_inference_pipe(unet=current_unet,
                                                                        text_encoder=current_text_encoder,
                                                                        tokenizer=tokenizer,
                                                                        vae=vae,
                                                                        diffusers_scheduler_config=inference_scheduler.config
                                                                        ).to(device)
                sample_generator.generate_samples(inference_pipe, global_step, extra_info=extra_info)

                # Cleanup
                del inference_pipe

                if model_info["swap_required"]:
                    with torch.no_grad():
                        unet = unet_unloaded.to(device)
                        del unet_unloaded
                        text_encoder = text_encoder_unloaded.to(device)
                        del text_encoder_unloaded

                        unet_ema = current_unet.to(ema_device)
                        del current_unet
                        text_encoder_ema = current_text_encoder.to(ema_device)
                        del current_text_encoder

                gc.collect()
                torch.cuda.empty_cache()

    def make_save_path(epoch, global_step, prepend=""):
        basename = f"{prepend}{args.project_name}"
        if epoch is not None:
            basename += f"-ep{epoch:02}"
        if global_step is not None:
            basename += f"-gs{global_step:05}"
        return os.path.join(log_folder, "ckpts", basename)

    def get_model_prediction_and_target_validation_wrapper(image, tokens
                                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = image.shape[0]
        timesteps = get_timesteps(batch_size=batch_size,
                                  batch_share_timesteps=False,
                                  device=unet.device,
                                  timesteps_ranges=[(args.timestep_start, args.timestep_end)] * batch_size,
                                  continuous_float_timestamps=noise_scheduler.config.prediction_type == 'flow-matching')
        latents = get_latents(image, vae, device=unet.device, args=args)
        noise = get_noise(latents.shape, unet.device, image.dtype,
                          pyramid_noise_discount=args.pyramid_noise_discount,
                          zero_frequency_noise_ratio=args.zero_frequency_noise_ratio,
                          batch_share_noise=False)
        encoder_hidden_states = _encode_caption_tokens(
            tokens,
            text_encoder,
            clip_skip=args.clip_skip,
            embedding_perturbation=args.embedding_perturbation,
            compel=None
        )

        model_pred, target, _, _ = get_model_prediction_and_target(
            latents,
            encoder_hidden_states,
            noise,
            timesteps,
            unet,
            noise_scheduler,
            args=args,
            skip_contrastive=True
        )

        del encoder_hidden_states
        return model_pred, target

    # Pre-train validation to establish a starting point on the loss graph
    if validator and not args.no_initial_validation:
        validator.do_validation(global_step=0,
                                get_model_prediction_and_target_callable=get_model_prediction_and_target_validation_wrapper)

    # the sample generator might be configured to generate samples before step 0
    if sample_generator.generate_pretrain_samples:
        _, batch = next(enumerate(train_dataloader))
        generate_samples(global_step=0, batch=batch)

    def make_current_ed_state() -> EveryDreamTrainingState:
        return EveryDreamTrainingState(optimizer=ed_optimizer,
                                       train_batch=train_batch,
                                       unet=unet,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       scheduler=noise_scheduler,
                                       inference_scheduler=inference_scheduler,
                                       vae=vae,
                                       unet_ema=unet_ema,
                                       text_encoder_ema=text_encoder_ema)

    epoch = None

    @dataclass
    class TrainingVariables:
        last_effective_batch_size: int = 0
        effective_backward_size: int = 0
        current_accumulated_backward_images_count: int = 0
        accumulated_loss_images_count: int = 0
        accumulated_loss: torch.Tensor|None = None
        accumulated_pathnames: list[str] = field(default_factory=list)
        accumulated_captions: list[str] = field(default_factory=list)
        accumulated_timesteps: list[int] = field(default_factory=list)
        desired_effective_batch_size: int = None
        interleave_bs1_bsN: bool = False
        interleaved_bs1_count: int = None

        prev_accumulated_pathnames: list[str] = field(default_factory=list)
        prev_accumulated_captions: list[str] = field(default_factory=list)
        prev_accumulated_timesteps: list[str] = field(default_factory=list)

        def accumulate_loss(self, loss: torch.Tensor, pathnames: list[str], captions: list[str], timesteps: list[int]):

            if loss.isnan().any():
                logging.warning(f"NaN detected after processing {pathnames} @ {timesteps} ({captions} - skipping")
                logging.warning(f" - NaN detected (current accumulated {self.accumulated_pathnames} @ {self.accumulated_timesteps} ({self.accumulated_captions}) )")
                logging.warning(f" - NaN detected (prev was {self.prev_accumulated_pathnames} @ {self.prev_accumulated_timesteps} ({self.prev_accumulated_captions}) )")
                assert False
                return

            self.accumulated_loss = (
                loss
                if self.accumulated_loss is None
                else self.accumulated_loss + loss
            )
            self.accumulated_loss_images_count += len(timesteps)
            self.accumulated_pathnames.extend(pathnames)
            self.accumulated_captions.extend(captions)
            self.accumulated_timesteps.extend(timesteps)

        def clear_accumulated_loss(self):
            self.accumulated_loss = None
            self.accumulated_loss_images_count = 0
            self.prev_accumulated_captions = self.accumulated_captions
            self.prev_accumulated_pathnames = self.accumulated_pathnames
            self.prev_accumulated_timesteps = self.accumulated_timesteps
            self.accumulated_pathnames = []
            self.accumulated_captions = []
            self.accumulated_timesteps = []


    def optimizer_backward(tv: TrainingVariables):
        if tv.accumulated_loss_images_count == 0:
            print("no accumulated loss images, not doing backward")
        else:
            ed_optimizer.backward(tv.accumulated_loss)
            tv.effective_backward_size = tv.accumulated_loss_images_count
            tv.current_accumulated_backward_images_count += tv.accumulated_loss_images_count
            #print(f"\nbackward on {tv.accumulated_loss_images_count} -> "
            #      f"backward accumulated {tv.current_accumulated_backward_images_count}")
            tv.clear_accumulated_loss()
            # reset slice sizes
            tv.forward_slice_size = args.forward_slice_size or args.batch_size
            tv.max_backward_slice_size = args.max_backward_slice_size or args.batch_size

    tv = TrainingVariables()

    try:
        plugin_runner.run_on_training_start(log_folder=log_folder, project_name=args.project_name)

        needs_samples = False
        tv.desired_effective_batch_size = choose_effective_batch_size(args, 0)
        remaining_timesteps = None

        for epoch in range(args.max_epochs):
            write_batch_schedule(log_folder, train_batch, epoch) if args.write_schedule else None
            if args.load_settings_every_epoch:
                load_train_json_from_file(args)

            epoch_len = math.ceil(len(train_batch) / args.batch_size)

            def update_arg(arg: str, newValue):
                if arg == "grad_accum":
                    args.grad_accum = newValue
                    data_loader.grad_accum = newValue
                else:
                    raise("Unrecognized arg: " + arg)

            plugin_runner.run_on_epoch_start(
                epoch=epoch,
                global_step=global_step,
                epoch_length=epoch_len,
                project_name=args.project_name,
                log_folder=log_folder,
                data_root=args.data_root,
                arg_update_callback=update_arg
            )

            sample_generator.on_epoch_start(
                epoch=epoch,
                global_step=global_step,
                epoch_length=epoch_len
            )

            loss_epoch = []
            epoch_start_time = time.time()
            images_per_sec_log_step = []

            steps_pbar = tqdm(range(epoch_len), position=1, leave=False, dynamic_ncols=True)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

            validation_steps = (
                [] if validator is None
                else validator.get_validation_step_indices(epoch, len(train_dataloader))
            )

            tv.forward_slice_size = args.forward_slice_size or args.batch_size
            tv.max_backward_slice_size = args.max_backward_slice_size or args.batch_size

            for step, full_batch in enumerate(train_dataloader):

                try:

                    train_progress_01 = compute_train_process_01(epoch=epoch, step=step, steps_per_epoch=epoch_len,
                                                                 max_epochs=args.max_epochs, max_global_steps=args.max_steps)

                    step_start_time = time.time()

                    plugin_runner.run_on_step_start(epoch=epoch,
                                                    local_step=step,
                                                    global_step=global_step,
                                                    project_name=args.project_name,
                                                    log_folder=log_folder,
                                                    batch=full_batch,
                                                    ed_state=make_current_ed_state())

                    full_batch_size = full_batch["image"].shape[0]

                    #for k in full_batch['captions'].keys():
                    #    if any(c is not None and len(c.strip()) == 0 for c in full_batch['captions'][k]):
                    #        print('a caption was already empty before cond dropout. paths:', full_batch['pathnames'], 'captions:', full_batch['captions'])

                    image_size = full_batch["image"].shape[2] * full_batch["image"].shape[3]
                    enable_vae_attention_slicing = False
                    if image_size > (1000)*(1000):
                        if tv.accumulated_loss_images_count > 1:
                            print(f"emergency backward at accumulated_loss_images_count {tv.accumulated_loss_images_count}")
                            # todo: DRY
                            optimizer_backward(tv)

                        enable_vae_attention_slicing = True
                        tv.forward_slice_size = min(tv.forward_slice_size, 1)
                        tv.max_backward_slice_size = min(tv.max_backward_slice_size, 2)

                    elif image_size > (850)*(850):
                        tv.forward_slice_size = min(tv.forward_slice_size, 2)
                        tv.max_backward_slice_size = min(tv.max_backward_slice_size, 6)

                    def lerp(x, in_min, in_max, out_min, out_max):
                        if (in_max - in_min) == 0:
                            return in_min
                        pct = (x - in_min) / (in_max - in_min)
                        return out_min + pct * (out_max - out_min)

                    if args.timestep_curriculum_alpha == 0:
                        timestep_range = (args.timestep_start, args.timestep_end)
                    else:
                        t_min_initial = args.timestep_initial_start
                        t_max_initial = args.timestep_initial_end
                        t_min_final = args.timestep_start
                        t_max_final = args.timestep_end
                        timestep_range = get_timestep_curriculum_range(progress_01=train_progress_01,
                                                                       t_min_initial=t_min_initial,
                                                                       t_max_initial=t_max_initial,
                                                                       t_min_final=t_min_final,
                                                                       t_max_final=t_max_final,
                                                                       alpha=args.timestep_curriculum_alpha)
                        # print('timestep range:', timestep_range)

                    timesteps_ranges = full_batch["timesteps_range"] or ([timestep_range] * full_batch_size)

                    # randomly expand
                    num_train_timesteps = noise_scheduler.config.num_train_timesteps
                    timesteps_ranges = [(
                        # maybe make 8 a CLI arg?
                        min(num_train_timesteps,
                            max(0, round(lerp(pow(random.random(), 8), 0, 1, tsr[0], 0)))),
                        min(num_train_timesteps,
                            max(0, round(lerp(pow(random.random(), 8), 0, 1, tsr[1], num_train_timesteps)))),
                    ) for tsr in timesteps_ranges]

                    if (tv.desired_effective_batch_size > 1
                            and args.everything_contrastive_learning_p > 0
                            and not full_batch["do_contrastive_learning"]
                    ):
                        if args.everything_contrastive_learning_curriculum_alpha > 0:
                            everything_contrastive_learning_p = get_exponential_scaled_value(train_progress_01,
                                                                                             initial_value=args.everything_contrastive_learning_p,
                                                                                             final_value=0,
                                                                                             alpha=args.everything_contrastive_learning_curriculum_alpha)
                        else:
                            everything_contrastive_learning_p = args.everything_contrastive_learning_p
                        full_batch["do_contrastive_learning"] = everything_contrastive_learning_p > random.random()


                    do_contrastive_learning = full_batch["do_contrastive_learning"]
                    if args.timesteps_multirank_stratified:
                        # the point of multirank stratified is to spread timesteps evenly across the batch.
                        # so we need to do a dance here to make sure that we're actually spreading across the
                        # desired_effective_batch_size - which will be "nibbled" below in chunks
                        while remaining_timesteps is None or remaining_timesteps.shape[0] < max(full_batch_size, tv.desired_effective_batch_size):
                            next_timesteps = get_multirank_stratified_random_timesteps(tv.desired_effective_batch_size, device=unet.device, alpha=args.timesteps_multirank_stratified_alpha, beta=args.timesteps_multirank_stratified_beta)
                            remaining_timesteps = next_timesteps if remaining_timesteps is None else torch.cat([remaining_timesteps, next_timesteps])
                        timesteps = remaining_timesteps[:full_batch_size]
                        remaining_timesteps = remaining_timesteps[full_batch_size:]
                    else:
                        timesteps: torch.LongTensor = get_timesteps(
                            batch_size=full_batch_size,
                            batch_share_timesteps=(
                            do_contrastive_learning or args.batch_share_timesteps),
                            device=unet.device,
                            timesteps_ranges=timesteps_ranges,
                            continuous_float_timestamps=noise_scheduler.config.prediction_type == 'flow-matching'
                        )

                    # apply cond dropout
                    initial_batch_size = args.initial_batch_size or args.batch_size
                    final_batch_size = args.final_batch_size or args.batch_size
                    if final_batch_size == initial_batch_size:
                        cdp_01_bs = 0
                    else:
                        cdp_01_bs = min(1, max(0, (tv.desired_effective_batch_size - initial_batch_size)
                                           / (final_batch_size - initial_batch_size)))
                    for sample_index in range(timesteps.shape[0]):
                        cdp_01_ts = 1 - (timesteps[sample_index].cpu().item() / noise_scheduler.config.num_train_timesteps)
                        if args.cond_dropout_curriculum_source == 'timestep':
                            cdp_01 = cdp_01_ts
                        elif args.cond_dropout_curriculum_source == 'batch_size':
                            assert args.final_batch_size != args.initial_batch_size
                            cdp_01 = cdp_01_bs
                        elif args.cond_dropout_curriculum_source == 'global_step':
                            cdp_01 = train_progress_01
                        elif args.cond_dropout_curriculum_source == 'batch_size_and_timestep':
                            assert args.final_batch_size != args.initial_batch_size
                            cdp_01 = cdp_01_bs * cdp_01_ts
                        else:
                            raise ValueError("Unrecognized value for --cond_dropout_curriculum_source")
                        final_cond_dropout = args.cond_dropout if args.final_cond_dropout is None else args.final_cond_dropout
                        this_cond_dropout_p = get_exponential_scaled_value(cdp_01,
                                                                           initial_value=args.cond_dropout,
                                                                           final_value=final_cond_dropout,
                                                                           alpha=args.cond_dropout_curriculum_alpha)
                        if train_batch.random_instance.random() <= this_cond_dropout_p:
                            for k in full_batch['captions'].keys():
                                full_batch['tokens'][k][sample_index] = train_batch.cond_dropout_tokens
                                full_batch['captions'][k][sample_index] = train_batch.cond_dropout_caption
                            full_batch["loss_scale"][sample_index] *= args.cond_dropout_loss_scale

                    remaining_batch = full_batch
                    while remaining_batch is not None and remaining_batch['image'].shape[0] > 0:
                        def get_nibble_size() -> int:
                            if tv.desired_effective_batch_size - tv.current_accumulated_backward_images_count <= 0:
                                raise ValueError(f"get_nibble_size: no nibble left? desired effective batch size {tv.desired_effective_batch_size} - current accumulated backward images count {tv.current_accumulated_backward_images_count}")
                            if tv.interleaved_bs1_count is not None:
                                return 1
                            required_to_fill_batch = tv.desired_effective_batch_size - tv.current_accumulated_backward_images_count
                            permitted_until_backward_step = tv.max_backward_slice_size - tv.accumulated_loss_images_count
                            return max(1, min(required_to_fill_batch, permitted_until_backward_step))

                        batch, remaining_batch = nibble_batch(remaining_batch, get_nibble_size())
                        assert batch["runt_size"] == 0

                        if args.contrastive_learning_curriculum_alpha == 0:
                            contrastive_learning_negative_loss_scale = args.contrastive_learning_negative_loss_scale
                        else:
                            contrastive_learning_negative_loss_scale = get_exponential_scaled_value(train_progress_01,
                                                                       initial_value=args.contrastive_learning_negative_loss_scale,
                                                                       final_value=0,
                                                                       alpha=args.contrastive_learning_curriculum_alpha)
                            #print('contrastive learning negative loss scale:', contrastive_learning_negative_loss_scale)

                        loss_scale = batch["loss_scale"]
                        assert loss_scale.shape[0] == batch["image"].shape[0]
                        image_shape = batch["image"].shape
                        reference_image_size = 512*512
                        #loss_scale = loss_scale.float() * (reference_image_size / image_size)
                        batch_resolution = _get_best_match_resolution(args.resolution, image_size)

                        assert type(batch["captions"]) is dict
                        caption_variants = [c for c in list(sorted(batch["captions"].keys()))
                                            if args.caption_variants is None
                                            or c in args.caption_variants
                                            and c != 'default']
                        if len(caption_variants) == 0:
                            caption_variants.append("default")
                        if not args.all_caption_variants:
                            caption_variants = [random.choice(caption_variants)]

                        batch_size = image_shape[0]

                        with torch.no_grad():
                            noise_shape = (batch_size, 4, image_shape[2] // 8, image_shape[3] // 8)
                            noise = get_noise(noise_shape, device=unet.device, dtype=batch["image"].dtype,
                                              pyramid_noise_discount=args.pyramid_noise_discount,
                                              zero_frequency_noise_ratio=args.zero_frequency_noise_ratio,
                                              batch_share_noise=(do_contrastive_learning or args.batch_share_noise)
                                              )

                        runt_size = batch["runt_size"]
                        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
                            latents_slices = []
                            pixel_values = batch["image"].to(memory_format=torch.contiguous_format).to(unet.device)
                            for slice_start, slice_end in get_slices(batch_size, tv.forward_slice_size, runt_size=runt_size):
                                try:
                                    vae: AutoencoderKL
                                    if enable_vae_attention_slicing:
                                        vae.enable_slicing()
                                    try:
                                        latents_slice = vae.encode(pixel_values[slice_start:slice_end], return_dict=False)
                                    except Exception as e:
                                        logging.error(f"vae.encode failed for batch {batch['pathnames']} slice [{slice_start}:{slice_end}] @resolution {batch_resolution}. pixel_values shape is {pixel_values.shape}")
                                        raise
                                    vae.disable_slicing()
                                    latents_slice = latents_slice[0].sample() * 0.18215
                                    latents_slices.append(latents_slice)
                                    del latents_slice
                                except torch.cuda.OutOfMemoryError:
                                    print(f"OOM in vae.encode encoding slice size {tv.forward_slice_size} from pixel values "
                                          f"of shape {pixel_values.shape}, from {slice_start} to {slice_end} of {batch_size}. "
                                          f"loss images accumulated: {tv.accumulated_loss_images_count}")
                                    raise
                            del pixel_values
                            latents = torch.cat(latents_slices)
                            del latents_slices

                        for caption_variant in caption_variants:
                            caption_str = []
                            tokens = []
                            for i in range(batch_size):
                                this_caption_str = batch["captions"][caption_variant][i]
                                this_tokens = batch["tokens"][caption_variant][i]
                                if this_caption_str is None:
                                    # pick a random caption variant to replace it
                                    non_none_variant = train_batch.random_instance.choice([k for k, v in batch["captions"].items()
                                                        if v[i] is not None])
                                    this_caption_str = batch["captions"][non_none_variant][i]
                                    this_tokens = batch["tokens"][non_none_variant][i]
                                assert this_caption_str is not None
                                assert this_tokens is not None
                                caption_str.append(this_caption_str)
                                tokens.append(this_tokens)

                            tokens = torch.stack(tokens)
                            with torch.no_grad() if args.disable_textenc_training else contextlib.nullcontext():
                                encoder_hidden_states = _encode_caption_tokens(tokens, text_encoder,
                                                                           clip_skip=args.clip_skip,
                                                                           embedding_perturbation=args.embedding_perturbation,
                                                                           compel=compel,
                                                                           caption_strings=caption_str
                                                                        )

                            model_pred_all = []
                            model_pred_wrong_all = []
                            model_pred_wrong_mask_all = []
                            target_all = []
                            if teacher_unet is None:
                                teacher_mask = None
                            else:
                                teacher_mask = torch.rand((batch_size)).to(unet.device) < args.teacher_p

                            cond_dropout_mask = torch.tensor([(s is None or len(s.strip()) == 0)
                                                              for s in caption_str[0:batch_size]], device=unet.device, dtype=torch.bool)
                            if teacher_mask is not None:
                                teacher_mask *= ~cond_dropout_mask

                            slices = list(get_slices(batch_size, tv.forward_slice_size, runt_size=runt_size))
                            for slice_index, (slice_start, slice_end) in enumerate(slices):
                                #print(f'slice {slice_index} @ res {image_shape[2:4]} (base {args.resolution[0]}), sssf {slice_size_scale_factor}, bs {batch_size}, slice size {forward_slice_size}')
                                latents_slice = latents[slice_start:slice_end]
                                encoder_hidden_states_slice = encoder_hidden_states[slice_start:slice_end]
                                noise_slice = noise[slice_start:slice_end]
                                timesteps_slice = timesteps[slice_start:slice_end]
                                if teacher_mask is None:
                                    teacher_mask_slice = None
                                else:
                                    teacher_mask_slice = teacher_mask[slice_start:slice_end]

                                model_pred, target, model_pred_wrong, model_pred_wrong_mask = get_model_prediction_and_target(
                                    latents=latents_slice,
                                    encoder_hidden_states=encoder_hidden_states_slice,
                                    noise=noise_slice,
                                    timesteps=timesteps_slice,
                                    unet=unet,
                                    noise_scheduler=noise_scheduler,
                                    args=args,
                                    teacher_unet=teacher_unet,
                                    teacher_mask=teacher_mask_slice
                                )
                                model_pred_all.append(model_pred)
                                model_pred_wrong_all.append(model_pred_wrong)
                                model_pred_wrong_mask_all.append(model_pred_wrong_mask)
                                target_all.append(target)

                            nibble_size_actual = min(slice_end, batch_size - runt_size)
                            mask = None if batch["mask"] is None else batch["mask"][0:nibble_size_actual]

                            model_pred = torch.cat(model_pred_all)
                            if any(x is not None for x in model_pred_wrong_all):
                                model_pred_wrong = torch.cat(model_pred_wrong_all, dim=1)
                                model_pred_wrong_mask = torch.cat(model_pred_wrong_mask_all)
                            else:
                                model_pred_wrong = None
                                model_pred_wrong_mask = None
                            target = torch.cat(target_all)
                            del model_pred_all, model_pred_wrong_all, model_pred_wrong_mask_all, target_all

                            loss = get_loss(model_pred,
                                            target,
                                            model_pred_wrong=model_pred_wrong,
                                            model_pred_wrong_mask=model_pred_wrong_mask,
                                            caption_str=caption_str[0:nibble_size_actual],
                                            mask=mask,
                                            timesteps=timesteps[0:nibble_size_actual],
                                            loss_scale=loss_scale[0:nibble_size_actual],
                                            noise_scheduler=noise_scheduler,
                                            do_contrastive_learning=do_contrastive_learning,
                                            contrastive_learning_negative_loss_scale=contrastive_learning_negative_loss_scale,
                                            args=args,
                                            )
                            del target, model_pred, model_pred_wrong, model_pred_wrong_mask

                            loss_log_step_cd.append(loss[cond_dropout_mask].mean().detach().item())
                            loss_log_step_non_cd.append(loss[~cond_dropout_mask].mean().detach().item())
                            for i, used_timestep in enumerate(timesteps[0:nibble_size_actual]):
                                used_timestep_detached = int(used_timestep.detach().item())
                                current, count = loss_per_timestep[batch_resolution].get(used_timestep_detached, (0, 0))
                                loss_per_timestep[batch_resolution][used_timestep_detached] = (current + loss[i].mean().detach().item(), count + 1)

                            # take mean of all dimensions except B
                            loss_sum = loss.mean(dim=list(range(1, len(loss.shape)))).sum()
                            tv.accumulate_loss(loss_sum,
                                               pathnames=batch["pathnames"][0:nibble_size_actual],
                                               captions=caption_str[0:nibble_size_actual],
                                               timesteps=timesteps[0:nibble_size_actual].tolist())
                            loss_preview_image: torch.Tensor = loss.detach().clone().cpu()
                            loss_step = loss.detach().mean().item()
                            del loss, loss_sum
                            steps_pbar.set_postfix({"loss/step": loss_step}, {"gs": global_step})
                            loss_log_step.append(loss_step)
                            loss_epoch.append(loss_step)

                            should_step_optimizer = (
                                    (tv.current_accumulated_backward_images_count + tv.accumulated_loss_images_count)
                                    >= tv.desired_effective_batch_size
                            ) or tv.interleaved_bs1_count is not None
                            if ((should_step_optimizer and tv.accumulated_loss_images_count > 0) or
                                    tv.accumulated_loss_images_count >= tv.max_backward_slice_size):
                                #accumulated_loss = accumulated_loss.mean() * (accumulated_loss_images_count / desired_effective_batch_size)
                                optimizer_backward(tv)

                            if should_step_optimizer:
                                if tv.current_accumulated_backward_images_count == 0:
                                    print("Batch has 0 images, not stepping optimizer")
                                else:
                                    #print(f'\nstepping optimizer - current_accumulated_backward_images_count '
                                    #      f'{tv.current_accumulated_backward_images_count}, '
                                    #      f'accumulated_loss_images_count {tv.accumulated_loss_images_count}')
                                    tv.last_effective_batch_size = tv.current_accumulated_backward_images_count
                                    ed_optimizer.step_optimizer(global_step)
                                    tv.current_accumulated_backward_images_count = 0

                                    # if we are interleaving BS1, increment counter
                                    if tv.interleaved_bs1_count is not None:
                                        tv.interleaved_bs1_count += 1

                                    # if we are *not* interleaving BS1, or we've reached the end of interleaving BS1, then step and perhaps toggle interleave BS1
                                    if tv.interleaved_bs1_count is None or tv.interleaved_bs1_count >= max(1, tv.desired_effective_batch_size ** args.interleave_batch_size_1_alpha):
                                        tv.desired_effective_batch_size = choose_effective_batch_size(args, train_progress_01)
                                        if args.interleave_batch_size_1:
                                            if tv.interleaved_bs1_count is None:
                                                tv.interleaved_bs1_count = 0
                                            else:
                                                tv.interleaved_bs1_count = None

                    ed_optimizer.step_schedulers(global_step)

                    if sample_generator.should_generate_samples(global_step, local_step=step):
                        needs_samples = True

                    if args.ema_decay_rate != None:
                        if ((global_step + 1) % args.ema_update_interval) == 0:
                            # debug_start_time = time.time() # Measure time

                            if args.disable_unet_training != True:
                                update_ema(unet, unet_ema, args.ema_decay_rate, default_device=device, ema_device=ema_device)

                            if args.disable_textenc_training != True:
                                update_ema(text_encoder, text_encoder_ema, args.ema_decay_rate, default_device=device, ema_device=ema_device)

                            # debug_end_time = time.time() # Measure time
                            # debug_elapsed_time = debug_end_time - debug_start_time # Measure time
                            # print(f"Command update_EMA unet and TE took {debug_elapsed_time:.3f} seconds.") # Measure time

                    if needs_samples or check_semaphore_file_and_unlink(_WANT_SAMPLES_SEMAPHORE_FILE):
                        generate_samples(global_step=global_step, batch=full_batch)
                        needs_samples = False

                    steps_pbar.update(1)

                    images_per_sec = args.batch_size / (time.time() - step_start_time)
                    images_per_sec_log_step.append(images_per_sec)

                    if (global_step + 1) % args.log_step == 0:
                        lr_unet = ed_optimizer.get_unet_lr()
                        lr_textenc = ed_optimizer.get_textenc_lr()

                        log_writer.add_scalar(tag="hyperparameter/lr unet", scalar_value=lr_unet, global_step=global_step)
                        log_writer.add_scalar(tag="hyperparameter/lr text encoder", scalar_value=lr_textenc, global_step=global_step)
                        log_writer.add_scalar(tag="hyperparameter/timestep start", scalar_value=timesteps_ranges[0][0], global_step=global_step)
                        log_writer.add_scalar(tag="hyperparameter/timestep end", scalar_value=timesteps_ranges[0][1], global_step=global_step)
                        log_writer.add_scalar(tag="hyperparameter/effective batch size", scalar_value=tv.last_effective_batch_size, global_step=global_step)
                        log_writer.add_scalar(tag="hyperparameter/effective backward size", scalar_value=tv.effective_backward_size, global_step=global_step)

                        sum_img = sum(images_per_sec_log_step)
                        avg = sum_img / len(images_per_sec_log_step)
                        images_per_sec_log_step = []
                        if args.amp:
                            log_writer.add_scalar(tag="hyperparameter/grad scale", scalar_value=ed_optimizer.get_scale(), global_step=global_step)
                        log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)

                        logs = {"lr_unet": lr_unet, "lr_te": lr_textenc, "img/s": images_per_sec}
                        if len(loss_log_step) > 0:
                            loss_step = sum(loss_log_step) / len(loss_log_step)
                            log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_step, global_step=global_step)
                            logs["loss/log_step"] = loss_step


                        # log histogram of loss vs timestep
                        loss_sums_and_counts = {
                            batch_resolution: [loss_per_timestep[batch_resolution].get(timestep, (0, 1))
                                                for timestep in range(noise_scheduler.config.num_train_timesteps + 1)]
                            for batch_resolution in args.resolution
                        }
                        loss_sums_and_counts = {
                            batch_resolution: torch.tensor([
                                loss_sum_this_step / count
                                for loss_sum_this_step, count in data
                            ])
                            for batch_resolution, data in loss_sums_and_counts.items()
                        }
                        with open(os.path.join(log_folder, "loss_sums_and_counts_per_timestep.pt"), "wb") as f:
                            torch.save(loss_sums_and_counts, f)

                        #for batch_resolution in args.resolution:
                        #    #image = _create_bar_chart_image(loss_per_timestep[batch_resolution])
                        #    loss_sums_and_counts = [loss_per_timestep[batch_resolution].get(timestep, (0, 1))
                        #                            for timestep in range(noise_scheduler.config.num_train_timesteps+1)]
                        #    #log_writer.add_histogram(tag=f"loss/timesteps/{batch_resolution}", values=torch.tensor([
                        #    #    loss_sum_this_step / count
                        #    #    for loss_sum_this_step, count in loss_sums_and_counts
                        #    #]), global_step=global_step)

                        loss_log_step_cd = [l for l in loss_log_step_cd if math.isfinite(l)]
                        if len(loss_log_step_cd) > 0:
                            loss_step_cd = sum(loss_log_step_cd) / len(loss_log_step_cd)
                            log_writer.add_scalar(tag="loss/log_step CD", scalar_value=loss_step_cd, global_step=global_step)
                            logs["loss/log_step CD"] = loss_step_cd

                        loss_log_step_non_cd = [l for l in loss_log_step_non_cd if math.isfinite(l)]
                        if len(loss_log_step_non_cd) > 0:
                            loss_step_non_cd = sum(loss_log_step_non_cd) / len(loss_log_step_non_cd)
                            log_writer.add_scalar(tag="loss/log_step non-CD", scalar_value=loss_step_non_cd, global_step=global_step)
                            logs["loss/log_step non-CD"] = loss_step_non_cd

                        if loss_preview_image is not None:
                            loss_preview_image_rgb = torchvision.utils.make_grid(
                                vae_preview((loss_preview_image / args.negative_loss_margin) * 2 - 1)
                            )
                            loss_preview_image = torchvision.utils.make_grid(
                                torch.reshape(loss_preview_image, [
                                    loss_preview_image.shape[0]*loss_preview_image.shape[1], 1, loss_preview_image.shape[2], loss_preview_image.shape[3]
                                ]),
                                nrow=loss_preview_image.shape[0],
                                normalize=True,
                                value_range=(0, args.negative_loss_margin),
                                scale_each=False)
                            log_writer.add_image(tag="loss/last vis raw", img_tensor=loss_preview_image, global_step=global_step)
                            log_writer.add_image(tag="loss/last vis rgb", img_tensor=loss_preview_image_rgb, global_step=global_step)

                        if args.log_named_parameters_magnitudes:
                            def log_named_parameters(model, prefix):
                                """Log L2 norms of parameter groups to help debug NaN issues."""
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        param_mean = param.mean().item()
                                        log_writer.add_scalar(tag=f"p-mean/{prefix}-{name}", scalar_value=param_mean, global_step=global_step)
                            if not args.disable_unet_training:
                                log_named_parameters(unet, "unet")
                            if not args.disable_textenc_training:
                                log_named_parameters(text_encoder, "textenc")

                        loss_log_step = []
                        loss_log_step_cd = []
                        loss_log_step_non_cd = []

                        append_epoch_log(global_step=global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                        torch.cuda.empty_cache()

                    if validator and (
                            step in validation_steps
                            or check_semaphore_file_and_unlink(_WANT_VALIDATION_SEMAPHORE_FILE)
                    ):
                        validator.do_validation(global_step, get_model_prediction_and_target_validation_wrapper)

                    min_since_last_ckpt =  (time.time() - last_epoch_saved_time) / 60

                    needs_save = False
                    if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                        last_epoch_saved_time = time.time()
                        logging.info(f"Saving model, {args.ckpt_every_n_minutes} mins at step {global_step}")
                        needs_save = True
                    if (epoch > 0 and epoch % args.save_every_n_epochs == 0 and step == 0 and
                            epoch < args.max_epochs and epoch >= args.save_ckpts_from_n_epochs):
                        logging.info(f" Saving model, {args.save_every_n_epochs} epochs at step {global_step}")
                        needs_save = True
                    if needs_save:
                        save_path = make_save_path(epoch, global_step)
                        save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
                                   save_ckpt_dir=args.save_ckpt_dir, yaml_name=None,
                                   save_full_precision=args.save_full_precision,
                                   save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt,
                                   save_lora=args.lora,
                                   plugin_runner=plugin_runner)

                    plugin_runner.run_on_step_end(epoch=epoch,
                                          global_step=global_step,
                                          local_step=step,
                                          project_name=args.project_name,
                                          log_folder=log_folder,
                                          data_root=args.data_root,
                                          batch=batch,
                                          ed_state=make_current_ed_state())

                    if (epoch == args.max_epochs-1
                            and ed_optimizer.will_do_grad_accum_step(step, global_step)
                            and epoch_len-step < args.grad_accum
                    ):
                        print(f"only {epoch_len-step} steps remaining at grad accum {args.grad_accum} -> early stop")
                        break

                    global_step += 1

                    if args.max_steps is not None and global_step >= args.max_steps:
                        print(f"max_steps reached, stopping")
                        break

                    # end of step

                except Exception as e:
                    try:
                        current_batch = batch
                    except NameError:
                        current_batch = None
                    logging.error(f"step {global_step} failed. full_batch: {full_batch}, current batch: {current_batch}, training values: {dataclasses.asdict(tv)}")
                    raise


            steps_pbar.close()

            elapsed_epoch_time = (time.time() - epoch_start_time) / 60
            epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
            log_writer.add_scalar(
                "performance/minutes per epoch", elapsed_epoch_time, global_step
            )

            plugin_runner.run_on_epoch_end(epoch=epoch,
                                           global_step=global_step,
                                           project_name=args.project_name,
                                           log_folder=log_folder,
                                           data_root=args.data_root,
                                           arg_update_callback=update_arg)

            epoch_pbar.update(1)
            if epoch < args.max_epochs - 1:
                train_batch.shuffle(epoch_n=epoch, max_epochs = args.max_epochs)

            if len(loss_epoch) > 0:
                loss_epoch = sum(loss_epoch) / len(loss_epoch)
                log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_epoch, global_step=global_step)

            gc.collect()

            if args.max_steps is not None and global_step >= args.max_steps:
                break

            # end of epoch

        # end of training
        epoch = args.max_epochs

        plugin_runner.run_on_training_end()

        save_path = make_save_path(epoch, global_step, prepend=("" if args.no_prepend_last else "last-"))
        save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
                   save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
                   save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, save_lora=args.lora,
                   plugin_runner=plugin_runner)

        if not sample_generator.should_generate_samples(global_step=global_step-1, local_step=step):
            print("generating final samples")
            _, batch = next(enumerate(train_dataloader))
            generate_samples(global_step=global_step, batch=batch)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        logging.error(f"{Fore.LIGHTYELLOW_EX}NOT attempting to save model{Style.RESET_ALL}")
        #save_path = make_save_path(epoch, global_step, prepend="errored-")
        #save_model(save_path, global_step=global_step, ed_state=make_current_ed_state(),
        #           save_ckpt_dir=args.save_ckpt_dir, yaml_name=yaml, save_full_precision=args.save_full_precision,
        #           save_optimizer_flag=args.save_optimizer, save_ckpt=not args.no_save_ckpt, save_lora=args.lora)
        #logging.info(f"{Fore.LIGHTYELLOW_EX}Model saved, re-raising exception and exiting.  Exception was:{Style.RESET_ALL}{Fore.LIGHTRED_EX} {ex} {Style.RESET_ALL}")
        raise ex

    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")


def get_slices(batch_size, slice_size, runt_size):
    num_slices = math.ceil(batch_size / slice_size)
    for slice_index in range(num_slices):
        slice_start = slice_index * slice_size
        slice_end = min(slice_start + slice_size, batch_size-runt_size)
        if slice_end <= slice_start:
            break
        yield slice_start, slice_end


def _get_best_match_resolution(resolutions: list[int], image_size_pixels: int) -> int:
    error = [image_size_pixels / (r*r) for r in resolutions]
    best_resolution_index = min([i for i in range(len(resolutions))], key=lambda i: abs(math.log(error[i])))
    return resolutions[best_resolution_index]


def create_bar_chart_image(values, labels=None, title="Bar Chart"):
    """
    Create a bar chart and return it as a TensorFlow image tensor.
    Use this if you want to handle the TensorBoard logging yourself.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(values))

    if labels is None:
        labels = [f"Item {i + 1}" for i in range(len(values))]

    bars = ax.bar(x_pos, values, alpha=0.8, color='steelblue')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    plt.close(fig)
    buf.close()

    return image



if __name__ == "__main__":
    check_git()
    supported_resolutions = aspects.get_supported_resolutions()
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, argv = argparser.parse_known_args()

    load_train_json_from_file(args, report_load=True)

    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--amp", action="store_true",  default=True, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--init_grad_scale", type=int, default=None, help="initial value for GradScaler (default=2^17.5)")
    argparser.add_argument("--attn_type", type=str, default="sdp", help="Attention mechanismto use", choices=["xformers", "sdp", "slice"])
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--batch_size_curriculum_alpha", type=float, default=0.5, help="curriculum alpha, default=0.5 (rapid (squared) falloff from initial)")
    argparser.add_argument("--interleave_batch_size_1", action='store_true', help="If passed, toggle between batches of BS1 and batches of current_batch_size")
    argparser.add_argument("--interleave_batch_size_1_alpha", type=float, default=0, help="How many BS1 batches to run when interleaving, as a factor of the current batch size (0=1 batch, 1=same as current batch size, 0.5=sqrt of current batch size etc)")
    argparser.add_argument("--initial_batch_size", type=int, default=None, help="initial batch size for curriculum")
    argparser.add_argument("--final_batch_size", type=int, default=None, help="final batch size for curriculum")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--cond_dropout_curriculum_alpha", type=float, default=0, help="cond dropout curriculum alpha, from cond_dropout to final_cond_dropout, controlled by --cond_dropout_curriculum_source")
    argparser.add_argument("--cond_dropout_curriculum_source", choices=['timestep', 'batch_size', 'batch_size_and_timestep', 'global_step'], default='timestep',
                           help="source for cond dropout curriculum - timestep (high timestep (high noise)...low timestep), batch size (initial_batch_size...final_batch_size), or global_step")
    argparser.add_argument("--final_cond_dropout", type=float, default=None, help="if doing cond dropout curriculum, the final cond dropout (timestep=0)")
    argparser.add_argument("--cond_dropout_loss_scale", type=float, default=1, help="additional loss scaling for cond dropout samples")
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--skip_undersized_images", action='store_true', help="If passed, ignore images that are considered undersized for the training resolution")
    argparser.add_argument("--disable_amp", action="store_true", default=False, help="disables automatic mixed precision (def: False)")
    argparser.add_argument("--disable_textenc_training", action="store_true", default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_unet_training", action="store_true", default=False, help="disables training of unet (def: False) NOT RECOMMENDED")
    argparser.add_argument("--freeze_unet_balanced", action="store_true", default=False, help="If passed, apply a 'balanced' unet freeze strategy: Train time_embedding.*, *.attentions.* (all parameters within attention blocks), conv_norm_out.*, and conv_out.*. Freeze the rest (conv_in.*, *.resnets.*, *samplers.*)." )
    argparser.add_argument("--embedding_perturbation", type=float, default=0.0, help="random perturbation of text embeddings (def: 0.0)")
    argparser.add_argument("--latents_perturbation", type=float, default=0.0, help="random perturbation of latents (def: 0.0)")
    argparser.add_argument("--flip_p", type=float, default=0.0, help="probability of flipping image horizontally (def: 0.0) use 0.0 to 1.0, ex 0.5, not good for specific faces!")
    argparser.add_argument("--gpuid", type=int, default=0, help="id of gpu to use for training, (def: 0) (ex: 1 to use GPU_ID 1), use nvidia-smi to find your GPU ids")
    argparser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="enable gradient checkpointing to reduce VRAM use, may reduce performance (def: False)")
    argparser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation factor (def: 1), (ex, 2)")
    argparser.add_argument("--forward_slice_size", type=int, default=1, help="Slice forward step into chunks of <= this ")
    argparser.add_argument("--max_backward_slice_size", type=int, default=None, help="Max number of samples to accumulate graph before doing backward (NOT optimizer step)")
    argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
    argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default!")
    argparser.add_argument("--log_named_parameters_magnitudes", action='store_true', help="If passed, log the magnitudes of all named parameters")
    argparser.add_argument("--loss_type", type=str, default="mse_huber",
                           help="type of loss / weight (def: mse_huber)", choices=["huber", "mse", "mse_huber", "v-mse"])
    argparser.add_argument("--negative_loss_margin", type=float, default=0.05, help="maximum for negative loss scale repulsion")
    argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant")
    argparser.add_argument("--lr_advance_steps", type=int, default=None, help="Steps to advance the LR during training")
    argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
    argparser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to train for")
    argparser.add_argument("--auto_decay_steps_multiplier", type=float, default=1.1, help="Multiplier for calculating decay steps from epoch count")
    argparser.add_argument("--no_prepend_last", action="store_true", help="Do not prepend 'last-' to the final checkpoint filename")
    argparser.add_argument("--no_save_ckpt", action="store_true", help="Save only diffusers files, not .safetensors files (save disk space if you do not need LDM-style checkpoints)" )
    argparser.add_argument("--optimizer_config", default="optimizer.json", help="Path to a JSON configuration file for the optimizer.  Default is 'optimizer.json'")
    argparser.add_argument('--plugins', nargs='+', help='Names of plugins to use')
    argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
    argparser.add_argument("--resolution", type=int, nargs='+', default=[512], help="resolution(s) to train", choices=supported_resolutions)
    argparser.add_argument("--keep_same_sample_at_different_resolutions_together", action='store_true', help="if passed, re-order batches to put samples with the same path but different resolutions near to each other")
    argparser.add_argument("--resume_ckpt", type=str, required=not ('resume_ckpt' in args), default="sd_v1-5_vae.ckpt", help="The checkpoint to resume from, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1 ")
    argparser.add_argument("--run_name", type=str, required=False, default=None, help="Run name for wandb (child of project name), and comment for tensorboard, (def: None)")
    argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt", help="Text file with prompts to generate test samples from, or JSON file with sample generator settings (default: sample_prompts.txt)")
    argparser.add_argument("--sample_steps", type=int, default=250, help="Number of steps between samples (def: 250)")
    argparser.add_argument("--save_ckpt_dir", type=str, default=None, help="folder to save checkpoints to (def: root training folder)")
    argparser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every n epochs, def: 0 (disabled)")
    argparser.add_argument("--save_ckpts_from_n_epochs", type=int, default=0, help="Only saves checkpoints starting an N epochs, def: 0 (disabled)")
    argparser.add_argument("--save_full_precision", action="store_true", default=False, help="save ckpts at full FP32")
    argparser.add_argument("--save_optimizer", action="store_true", default=False, help="saves optimizer state with ckpt, useful for resuming training later")
    argparser.add_argument("--seed", type=int, default=555, help="seed used for samples and shuffling, use -1 for random")
    argparser.add_argument("--shuffle_tags", action="store_true", default=False, help="randomly shuffles CSV tags in captions, for booru datasets")
    argparser.add_argument("--timestep_start", type=int, default=0, help="Noising timestep minimum (def: 0)")
    argparser.add_argument("--timestep_end", type=int, default=1000, help="Noising timestep (def: 1000)")
    argparser.add_argument("--timesteps_multirank_stratified", action="store_true", default=False, help="use multirank stratified timesteps (recommended: disable min_snr_gamma")
    argparser.add_argument("--timesteps_multirank_stratified_alpha", type=float, default=1.5, help="multirank stratified timesteps PPF alpha")
    argparser.add_argument("--timesteps_multirank_stratified_beta", type=float, default=2, help="multirank stratified timesteps PPF beta")
    argparser.add_argument("--timestep_curriculum_alpha", type=float, default=0, help="if passed, shift timestep range toward fine details as training progresses")
    argparser.add_argument("--timestep_initial_start", type=int, default=800, help="If using timestep_curriculum_alpha, the initial start timestep (default 800); will transition to --timestep_start")
    argparser.add_argument("--timestep_initial_end", type=int, default=1000, help="If using timestep_curriculum_alpha, the initial end timestep (default 1000); will transition to --timestep_end")
    argparser.add_argument("--train_sampler", type=str, default="ddpm",
                           help="noise sampler used for training, (default: ddpm)", choices=["ddpm", "pndm", "ddim", "flow-matching"])
    argparser.add_argument("--keep_tags", type=int, default=0, help="Number of tags to keep when shuffle, used to randomly select subset of tags when shuffling is enabled, def: 0 (shuffle all)")
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None, help="Path to a JSON configuration file for the validator.  Default is no validation.")
    argparser.add_argument("--no_initial_validation", action="store_true", help="If passed, don't do validation before the first step")
    argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02, help="adds zero frequency noise, for improving contrast (def: 0.0) use 0.0 to 0.15")
    argparser.add_argument("--enable_zero_terminal_snr", action="store_true", default=None, help="Use zero terminal SNR noising beta schedule")
    argparser.add_argument("--mix_zero_terminal_snr", action="store_true", default=None, help="Mix zero termianl SNR with regular training")
    argparser.add_argument("--match_zero_terminal_snr", action="store_true", default=None, help="use zero terminal SNR target as regular noise scheduler input")
    argparser.add_argument("--load_settings_every_epoch", action="store_true", default=None, help="Enable reloading of 'train.json' at start of every epoch.")
    argparser.add_argument("--min_snr_gamma", type=float, default=None, help="min-SNR-gamma parameter is the loss function into individual tasks. Recommended values: 5, 1, 20. Disabled by default and enabled when used. More info: https://arxiv.org/abs/2303.09556")
    argparser.add_argument("--min_snr_alpha", type=float, default=1, help="Blending factor for min-SNR-gamma weighting. 1=use pure min SNR gamma weighting, 0.5=use a 50/50 blend of min SNR gamma and regular loss weighting (default: 1)")
    argparser.add_argument("--debug_invert_min_snr_gamma", action='store_true', help="invert the timestep/scale equation for min snr gamma")
    argparser.add_argument("--ema_decay_rate", type=float, default=None, help="EMA decay rate. EMA model will be updated with (1 - ema_rate) from training, and the ema_rate from previous EMA, every interval. Values less than 1 and not so far from 1. Using this parameter will enable the feature.")
    argparser.add_argument("--ema_strength_target", type=float, default=None, help="EMA decay target value in range (0,1). emarate will be calculated from equation: 'ema_decay_rate=ema_strength_target^(total_steps/ema_update_interval)'. Using this parameter will enable the ema feature and overide ema_decay_rate.")
    argparser.add_argument("--ema_update_interval", type=int, default=500, help="How many steps between optimizer steps that EMA decay updates. EMA model will be update on every step modulo grad_accum times ema_update_interval.")
    argparser.add_argument("--ema_device", type=str, default='cpu', help="EMA decay device values: cpu, cuda. Using 'cpu' is taking around 4 seconds per update vs fraction of a second on 'cuda'. Using 'cuda' will reserve around 3.2GB VRAM for a model, with 'cpu' the system RAM will be used.")
    argparser.add_argument("--ema_sample_nonema_model", action="store_true", default=False, help="Will show samples from non-EMA trained model, just like regular training. Can be used with: --ema_sample_ema_model")
    argparser.add_argument("--ema_sample_ema_model", action="store_true", default=False, help="Will show samples from EMA model. May be slower when using ema cpu offloading. Can be used with: --ema_sample_nonema_model")
    argparser.add_argument("--ema_resume_model", type=str, default=None, help="The EMA decay checkpoint to resume from for EMA decay, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1-ema-decay")
    argparser.add_argument("--pyramid_noise_discount", type=float, default=None, help="Enables pyramid noise and use specified discount factor for it")
    argparser.add_argument("--batch_share_noise", action="store_true", help="All samples in a batch have the same noise")
    argparser.add_argument("--batch_share_timesteps", action="store_true", help="All samples in a batch have the same timesteps")
    argparser.add_argument("--teacher", type=str, default=None, help="Teacher model")
    argparser.add_argument("--teacher_p", type=float, default=0.5, help="Probability of teacher model being used as target")
    argparser.add_argument("--contrastive_learning_batch_ids", type=str, nargs="*", default=[], help="Batch ids for which contrastive learning should be done (default=[])")
    argparser.add_argument("--contrastive_learning_negative_loss_scale", type=float, default=0.2, help="Scaling factor for contrastive learning negative loss")
    argparser.add_argument("--contrastive_learning_max_negative_loss", type=float, default=1, help="Loss clamp max for contrastive learning negative loss (default=1)")
    argparser.add_argument("--contrastive_learning_use_l1_loss", action="store_true", help="use L1 loss instead of MSE for contrastive negative term")
    argparser.add_argument("--contrastive_learning_no_average_negatives", action="store_true", help="do not average negative terms per batch")
    argparser.add_argument("--contrastive_learning_save_on_cpu", action="store_true", help="Store grads on CPU (allows larger grad_accum sizes but slower)")
    argparser.add_argument("--contrastive_learning_delta_loss_method", action="store_true", help="If passed, contrastive learning works with deltas from correct to incorrect targets / predictions")
    argparser.add_argument("--contrastive_learning_delta_timestep_start", type=int, default=150, help="Where to start scaling delta negative loss")
    argparser.add_argument("--contrastive_learning_dropout_p", type=float, default=0, help="dropout probability for contrastive learning, 0..1")
    argparser.add_argument("--contrastive_learning_curriculum_alpha", type=float, default=0, help="if passed, exponentially disable contrastive learning as training progresses")
    argparser.add_argument("--contrastive_learning_info_nce_sample_count", type=int, default=0, help="If >0, do InfoNCE contrastive loss")
    argparser.add_argument("--contrastive_learning_info_nce_temperature", type=float, default=1, help="Temperature for InfoNCE contrastive loss")
    argparser.add_argument("--everything_contrastive_learning_p", type=float, default=0, help="probability to run contrastive learning on everything, 0..1")
    argparser.add_argument("--everything_contrastive_learning_curriculum_alpha", type=float, default=0, help="if >0, attenuate everything_contrastive_learning_p to 0 using this alpha as timestep approaches 0")
    argparser.add_argument("--caption_variants", type=str, nargs="*", default=None, help="If passed, use only these caption variants from json captions")
    argparser.add_argument("--all_caption_variants", action='store_true', help='if passed, use ALL caption variants every step')
    argparser.add_argument("--use_compel", action='store_true', help='if passed, use Compel to process prompts with long-prompt support')

    argparser.add_argument("--batch_id_dropout_p", type=float, default=0, help="dropout probability for batch ids, 0..1")
    argparser.add_argument("--cond_dropout_noise_p", type=float, default=0, help="how often to use noise (torch.randn) for the image with conditional dropout - helps prevent overfitting of unconditioned prompt")

    argparser.add_argument("--jacobian_descent", action='store_true', help="Do Jacobian Descent (see torchjd). Uses more VRAM.")
    argparser.add_argument("--use_masks", action='store_true', help="If passed, look for files called eg image_name.jpg.mask.png in the data folder and use as mask for the loss")

    argparser.add_argument("--lora", action='store_true', help="If passed, do LoRA training")
    argparser.add_argument("--lora_resume", type=str, default=None, help="resume from this lora (must be a huggingface format folder)")
    argparser.add_argument("--lora_rank", type=int, default=16)
    argparser.add_argument("--lora_alpha", type=int, default=8)

    argparser.add_argument("--test_images", action="store_true", help="check all images by trying to load them")


    # load CLI args to overwrite existing config args
    args = argparser.parse_args(args=argv, namespace=args)

    #torch.autograd.set_detect_anomaly(True)

    main(args)
