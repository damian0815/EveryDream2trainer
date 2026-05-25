"""
Copyright [2022-2023] Victor C Hall
Copyright [2023-2026] Damian Stewart

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
import os
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

from typing import Optional

import torch.distributed as dist
from utils.distributed import (
    is_distributed as _check_is_distributed,
    init_distributed,
    cleanup_distributed,
    barrier as dist_barrier,
    shard_items,
    DDPWrapper,
    get_distributed_state_signal, StateSignal, ddp_no_sync_ctx, sync_ddp_gradients,
)

from colorama import Fore, Style
import numpy as np
import torch
import datetime
import json

from compel import Compel
from compel.embeddings_provider import SplitLongTextMode
from tqdm.auto import tqdm

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from accelerate.utils import set_seed

import wandb
import webbrowser
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import DataLoaderMultiAspect

from data.every_dream import EveryDreamBatch, build_torch_dataloader, collate_fn as ed_collate_fn
from data.difficulty_estimator import DifficultyEstimator, TypeAScheduler, TypeBScheduler
from data.every_dream_validation import EveryDreamValidator, ValidationStepResult
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID
from core.log import do_log_step, append_epoch_log, write_batch_schedule, log_args, LogData
from core.loss import (
    get_noise,
    get_model_prediction_and_target,
    encode_with_vae_to_scaled_latents,
    compute_timestep_intervals,
)
from core.self_flow import SelfFlowMLPProjectionHead, get_self_flow_channels, SELF_FLOW_MODES
from core.step import nibble_batch, choose_effective_batch_size, compute_train_process_01, \
    get_exponential_scaled_value, get_best_match_resolution, train_step, get_uniform_timesteps, optimizer_backward, \
    record_performance_timing
from optimizer.attention_activation_control import ActivationLogger
from model.training_model import (
    EveryDreamTrainingState,
    save_model,
    save_model_lora,
    find_last_checkpoint,
    load_model,
    get_use_ema_decay_training,
    TrainingVariables,
    TrainingModel,
    Conditioning, load_clip_model,
)
from model.teacher import load_teacher_model
from core.semaphore_files import check_semaphore_file_and_unlink, WANT_SAMPLES_SEMAPHORE_FILE, \
    WANT_VALIDATION_SEMAPHORE_FILE, SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE, SAVE_FULL_SEMAPHORE_FILE, \
    SAVE_FULL_WITH_OPTIMIZER_AND_STOP_SEMAPHORE_FILE, SAVE_FULL_AND_STOP_SEMAPHORE_FILE
from utils.isolate_rng import isolate_rng
from utils.check_git import check_git
from optimizer.optimizers import EveryDreamOptimizer
from copy import deepcopy
import safetensors.torch

if torch.cuda.is_available():
    from utils.gpu import GPU
import data.aspects as aspects
import data.resolver as resolver
from utils.sample_generator import SampleGenerator

from plugins.plugins import PluginRunner

_SIGTERM_EXIT_CODE = 130
_VERY_LARGE_NUMBER = 1e9


class _NopWriter:
    """Drop-in replacement for SummaryWriter used on non-main ranks."""
    def add_scalar(self, *a, **kw): pass
    def add_image(self, *a, **kw): pass
    def add_histogram(self, *a, **kw): pass
    def flush(self): pass
    def close(self): pass


def _epoch_step_source(
    train_dataloader,
    train_batch: "EveryDreamBatch",
    difficulty_estimator: Optional["DifficultyEstimator"],
    epoch_len: int,
    batch_size: int,
    seed: int,
):
    """
    Yields (step, full_batch) for one training epoch.

    * If no DifficultyEstimator or a TypeA scheduler is active: just enumerates
      the pre-built train_dataloader unchanged.

    * If a TypeB (spaced-repetition) scheduler is active: rebuilds the item list
      every slab_size batches by calling build_next_slab(), then wraps the
      updated EveryDreamBatch in a fresh single-threaded DataLoader for that slab.
      num_workers=0 is intentional – there is no point spinning worker processes
      up/down at every slab boundary.
    """
    type_b: Optional["TypeBScheduler"] = None
    if difficulty_estimator is not None and isinstance(difficulty_estimator.scheduler, TypeBScheduler):
        type_b = difficulty_estimator.scheduler

    if type_b is None:
        yield from enumerate(train_dataloader)
        return

    slab_size = type_b.slab_size
    n_slots = slab_size * batch_size
    rng = random.Random(seed)
    step = 0
    all_items = train_batch.data_loader.prepared_train_data

    while step < epoch_len:
        slab_items = type_b.build_next_slab(
            items=all_items,
            scores=difficulty_estimator.scores,
            obs_counts=difficulty_estimator.obs_counts,
            min_obs_count=difficulty_estimator.min_obs_count,
            n_slots=n_slots,
            rng=rng,
        )
        # Override image_train_items for this slab (EveryDreamBatch.__len__ and
        # __getitem__ both key off self.image_train_items, so this is sufficient).
        train_batch.image_train_items = slab_items
        slab_dl = torch.utils.data.DataLoader(
            train_batch,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ed_collate_fn,
            pin_memory=False,
        )
        slab_step = 0
        for full_batch in slab_dl:
            if step >= epoch_len:
                break
            yield step, full_batch
            step += 1
            slab_step += 1
            if slab_step >= slab_size:
                break

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

    if (args.ema_resume_model is not None) and (args.ema_resume_model == "findlast"):
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

    if type(args.resolution) is not list:
        args.resolution = [args.resolution]

    if len(args.resolution_multiplier) > 0 and len(args.resolution_multiplier) != len(args.resolution):
        raise ValueError(f"when using --resolution_multiplier, you must pass exactly 1 multiplier per resolution (you passed: --resolution {args.resolution} --resolution_multiplier {args.resolution_multiplier})")

    def force_to_resolution_mapped_list_and_validate(value, name):
        if type(value) is not list:
            value = [value]
        if len(value) != len(args.resolution):
            if len(value) > 1:
                raise ValueError(
                    f"when using --{name}, you must pass exactly 1 max backward slice size per resolution (you passed: --resolution {args.resolution} --{name} {' '.join([str(v) for v in value])})")
            elif len(value) == 1:
                # expand to one per resolution
                value = value * len(args.resolution)
        return value

    args.max_backward_slice_size = force_to_resolution_mapped_list_and_validate(args.max_backward_slice_size, "max_backward_slice_size")
    args.forward_slice_size = force_to_resolution_mapped_list_and_validate(args.forward_slice_size, "forward_slice_size")

    if type(args.disable_backward_memsafe_resolutions) is not list:
        args.disable_backward_memsafe_resolutions = [args.disable_backward_memsafe_resolutions]
    if args.disable_backward_memsafe_resolutions and any(r not in args.resolution for r in args.disable_backward_memsafe_resolutions):
        raise ValueError(f"when using --disable_backward_memsafe_resolutions, all resolutions passed must be in --resolution (you passed: --resolution {args.resolution} --disable_backward_memsafe_resolutions {args.disable_backward_memsafe_resolutions})")

    if args.optimizer_batch_size is not None:
        if args.initial_batch_size is None:
            args.initial_batch_size = args.optimizer_batch_size
        else:
            logging.info(f" * overriding --optimizer_batch_size {args.optimizer_batch_size} with --initial_batch_size {args.final_batch_size}")
        if args.final_batch_size is None:
            args.final_batch_size = args.optimizer_batch_size
        else:
            logging.info(f" * overriding --optimizer_batch_size {args.optimizer_batch_size} with --final_batch_size {args.final_batch_size}")

    return args


def report_image_train_item_problems(log_folder: str, items: list[ImageTrainItem], batch_size, check_load_all=False, model: TrainingModel=None) -> None:
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
        ed_batch = EveryDreamBatch(data_loader,
                                   tokenizer=model.tokenizer,
                                   tokenizer_2=model.tokenizer_2)
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

def apply_per_path_multiplier(resolved_items: list[ImageTrainItem], per_path_multiplier_json: str):
    applied = 0
    missing = 0
    first_missing = []
    with open(per_path_multiplier_json, "rt") as f:
        per_path_multipliers = json.load(f)
    for item in tqdm(resolved_items, desc=f"applying per-path multiplier {os.path.basename(per_path_multiplier_json)}"):
        realpath = os.path.realpath(item.pathname)
        try:
            item.multiplier *= per_path_multipliers[realpath]
            applied += 1
        except KeyError:
            missing += 1
            if len(first_missing) < 5:
                first_missing.append(item.pathname)
    logging.info(f" Applied {applied} multipliers ({missing} missing) from {per_path_multiplier_json}. First 5 missing: {first_missing}")

def resolve_image_train_items(args: argparse.Namespace, resolution, aspects, global_multiplier: float=1) -> list[ImageTrainItem]:
    logging.info(f"* DLMA resolution {resolution}, buckets: {aspects}")
    logging.info(" Preloading images...")

    resolved_items = resolver.resolve(args.data_root, args, resolution, aspects)

    # Remove erroneous items
    for item in resolved_items:
        if item.error is not None:
            logging.error(f"{Fore.LIGHTRED_EX} *** Error opening {Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX} to get metadata. File may be corrupt and will be skipped.{Style.RESET_ALL}")
            logging.error(f" *** exception: {item.error}")
    resolved_items = [item for item in resolved_items if item.error is None]

    for i in resolved_items:
        i.multiplier *= global_multiplier

    if args.data_multiplier_per_path:
        # expand single str to list
        paths = [args.data_multiplier_per_path] if type(args.data_multiplier_per_path) is str else args.data_multiplier_per_path
        for p in paths:
            apply_per_path_multiplier(resolved_items, p)

    # drop undersized, if requested
    if args.skip_undersized_images:
        pre_count = len(resolved_items)
        resolved_items = [i for i in resolved_items if not i.is_undersized]
        post_drop_undersize_count = len(resolved_items)
        if pre_count != post_drop_undersize_count:
            logging.info(f" * From {pre_count} images, dropped {pre_count - post_drop_undersize_count} undersized images ({post_drop_undersize_count} remaining). Remove --skip_undersized_images to log which ones.")

    # drop empty JSON captions
    drop_empty_json_captions = True
    if drop_empty_json_captions:
        non_empty_json_captions = []
        for item in resolved_items:
            caption = item.caption.get_caption()
            if caption.startswith("<<json>>"):
                caption_data = json.loads(caption.replace("<<json>>", ""))
                if caption_data is None or all(v is None or len(v.strip()) == 0 for v in caption_data.values()):
                    #print("Empty JSON caption detected, skipping image:", item.pathname)
                    continue
            non_empty_json_captions.append(item)
        pre_count = len(resolved_items)
        resolved_items = non_empty_json_captions
        post_count = len(resolved_items)
        logging.info(
            f" * From {pre_count} images, dropped {pre_count - post_count} images with empty JSON captions ({post_count} remaining)."
        )

    print (f" * Found {len(resolved_items)} items in '{args.data_root}'")
    gc.collect()

    # Stamp base_multiplier now that all user-configured multiplier changes are done.
    # DifficultyEstimator schedulers scale relative to this value, not the mutated one.
    for item in resolved_items:
        item.base_multiplier = item.multiplier

    return resolved_items


def read_sample_prompts(sample_prompts_file_path: str):
    sample_prompts = []
    with open(sample_prompts_file_path, "r") as f:
        for line in f:
            sample_prompts.append(line.strip())
    return sample_prompts


def update_ema(model, ema_model, decay, default_device, ema_device: str):
    with torch.no_grad():
        original_model_on_proper_device = model
        need_to_delete_original = False
        if torch.device(ema_device) != torch.device(default_device):
            original_model_on_other_device = deepcopy(model)
            original_model_on_proper_device = original_model_on_other_device.to(ema_device, dtype=model.dtype)
            del original_model_on_other_device
            need_to_delete_original = True

        params: dict[str, torch.nn.Parameter] = dict(original_model_on_proper_device.named_parameters())
        ema_params: dict[str, torch.nn.Parameter] = dict(ema_model.named_parameters())

        for name in ema_params:
            #ema_params[name].data.mul_(decay).add_(params[name].data, alpha=1 - decay)
            ema_params[name].data = ema_params[name] * decay + params[name].data * (1.0 - decay)

        if need_to_delete_original:
            del(original_model_on_proper_device)

def _get_default_forward_slice_size(tv: TrainingVariables):
    return tv.default_forward_slice_size[tv.batch_resolution]

def _choose_backward_slice_size(tv: TrainingVariables):
    backward_slice_size = tv.default_max_backward_slice_size[tv.batch_resolution]
    return max(
        1,
        min(
            backward_slice_size,
            tv.desired_effective_batch_size
        ),
    )

def load_train_json_from_file(args, report_load = False):
    try:
        if report_load:
            print(f"Loading training config from {args.config}.")

        with open(args.config, 'rt') as f:
            read_json = json.load(f)

        args.__dict__.update(read_json)
    except Exception as config_read:
        print(f"Error on loading training config from {args.config}:", config_read)
        raise


def main(args):
    """
    Main entry point
    """
    if os.name == 'nt':
        print(" * Windows detected, disabling Triton")
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"

    import faulthandler
    faulthandler.enable()  # by default will dump on sys.stderr, but can also print to a regular file

    # -----------------------------------------------------------------------
    # Phase 1 – Distributed bootstrap
    # -----------------------------------------------------------------------
    _is_dist = _check_is_distributed()
    if _is_dist:
        _rank, _local_rank, _world_size = init_distributed()
    else:
        _rank, _local_rank, _world_size = 0, 0, 1
    _is_main = (_rank == 0)

    if _is_main:
        log_time, log_folder = setup_local_logger(args)
    else:
        # Non-main ranks: minimal console-only logging at WARNING level
        logging.basicConfig(
            level=logging.WARNING,
            format=f"%(asctime)s [rank{_rank}] %(message)s",
        )
        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_folder = os.path.join(args.logdir, f"{args.project_name}-{log_time}")

    args = setup_args(args)
    if _is_main:
        print(f" Args:")
        pprint.pprint(vars(args))

    if args.debug_log_on_nan:
        torch.autograd.set_detect_anomaly(True)

    if args.seed == -1:
        args.seed = random.randint(0, 2**30)
    seed = args.seed
    logging.info(f" Seed: {seed}")
    set_seed(seed)
    if torch.cuda.is_available():
        if _is_dist:
            # Phase 1: per-rank device overrides --gpuid
            device = torch.device(f"cuda:{_local_rank}")
        else:
            device = torch.device(f"cuda:{args.gpuid}")
        gpu = GPU(device)
        torch.backends.cudnn.benchmark = True
    else:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            logging.warning("*** Running on CPU. This is for testing loading/config parsing code only.")
            device = 'cpu'
        gpu = None

    # fix a weird issue with dataloader?
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
    torch.multiprocessing.set_sharing_strategy("file_system")

    # log_folder = os.path.join(args.logdir, f"{args.project_name}_{log_time}")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if args.debug_no_load_model:
        logging.warning("--debug_no_load_model passed - not loading model!")
        model = TrainingModel(
            noise_scheduler=None,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=None,
            vae=None,
            compel=None,
            yaml=None,
        )
        teacher_model = None
    else:
        try:
            model: TrainingModel = load_model(args)
        except Exception as e:
            traceback.print_exc()
            logging.error(f" * Failed to load checkpoint: {repr(e)} * ")
            raise

        if args.clip_vision_model_source is not None:
            if args.disable_textenc_training:
                logging.error(f" * Ignoring --clip_vision_model_source because --disable_textenc_training was passed")
            else:
                logging.info(f" * Loading CLIP vision model from {args.clip_vision_model_source}")
                clip_model, clip_processor = load_clip_model(args.clip_vision_model_source, args.clip_vision_model_processor_source)
                clip_model.requires_grad_(False)
                clip_model.to(device, dtype=torch.float16)
                del clip_model.text_model
                model.clip_model = clip_model
                model.clip_processor = clip_processor

        teacher_model = load_teacher_model(args, device, model,
                                           flow_match_shift_dynamic=args.flow_match_shift_dynamic, flow_match_shift=args.flow_match_shift)

        model.setup_cond_dropout_tokens()
        if teacher_model is not None:
            teacher_model.setup_cond_dropout_tokens()

        compel = None
        if args.use_compel:
            compel = Compel(tokenizer=model.tokenizer,
                            text_encoder=model.text_encoder,
                            truncate_long_prompts=False,
                            split_long_text_mode = SplitLongTextMode.SENTENCES,
                            )

        if args.gradient_checkpointing:
            model.unet.enable_gradient_checkpointing()
            model.text_encoder.gradient_checkpointing_enable()
            if model.text_encoder_2:
                model.text_encoder_2.gradient_checkpointing_enable()

        if args.attn_type == "xformers":
            if (args.amp and model.is_sd1attn) or (not model.is_sd1attn):
                try:
                    model.unet.enable_xformers_memory_efficient_attention()
                    logging.info("Enabled xformers")
                except Exception as ex:
                    logging.warning("failed to load xformers, using default SDP attention instead")
                    pass
            elif (args.disable_amp and model.is_sd1attn):
                logging.info("AMP is disabled but model is SD1.X, xformers is incompatible so using default attention")
        elif args.attn_type == "slice":
            model.unet.set_attention_slice("auto")
        else:
            logging.info("* Using SDP attention *")

        if model.is_sdxl:
            train_dtype = torch.bfloat16
            vae_dtype = torch.float32
        elif device == 'mps':
            train_dtype = torch.float16
            vae_dtype = train_dtype
        else:
            train_dtype = torch.bfloat16 if args.force_bfloat16 else torch.float32
            vae_dtype = torch.bfloat16 if args.amp else train_dtype

        model.vae = model.vae.to(device, dtype=vae_dtype)
        model.unet = model.unet.to(device, dtype=train_dtype)
        if args.disable_textenc_training and args.amp:
            model.text_encoder = model.text_encoder.to(device, dtype=torch.float16)
            if model.text_encoder_2:
                model.text_encoder_2 = model.text_encoder_2.to(device, dtype=torch.float16)
        else:
            model.text_encoder = model.text_encoder.to(device, dtype=train_dtype)
            if model.text_encoder_2:
                model.text_encoder_2 = model.text_encoder_2.to(device, dtype=train_dtype)

        if get_use_ema_decay_training(args):
            if model.unet_ema is None:
                logging.info(f"EMA decay enabled, creating EMA model.")

                with torch.no_grad():
                    if args.ema_device == device:
                        unet_ema = deepcopy(model.unet)
                        text_encoder_ema = deepcopy(model.text_encoder)
                    else:
                        unet_ema_first = deepcopy(model.unet)
                        text_encoder_ema_first = deepcopy(model.text_encoder)
                        unet_ema = unet_ema_first.to(args.ema_device, dtype=model.unet.dtype)
                        text_encoder_ema = text_encoder_ema_first.to(args.ema_device, dtype=model.text_encoder.dtype)
                        del unet_ema_first
                        del text_encoder_ema_first
            else:
                # Make sure correct types are used for models
                unet_ema = model.unet_ema.to(args.ema_device, dtype=model.unet.dtype)
                text_encoder_ema = model.text_encoder_ema.to(args.ema_device, dtype=model.text_encoder.dtype)
        else:
            unet_ema = None
            text_encoder_ema = None

        # Update model with EMA models if available
        model.unet_ema = unet_ema
        model.text_encoder_ema = text_encoder_ema

        # Self-Flow EMA teacher setup (separate from the saves/sampling EMA copy)
        if args.self_flow_p > 0.0:
            if model.self_flow_teacher_unet is None:
                logging.info(
                    f"Self-Flow enabled (p={args.self_flow_p}), creating frozen EMA teacher UNet "
                    f"(decay={args.self_flow_ema_decay})."
                )
                with torch.no_grad():
                    sf_teacher = deepcopy(model.unet).to(device, dtype=model.unet.dtype)
                sf_teacher.requires_grad_(False)
                model.self_flow_teacher_unet = sf_teacher

                # Attempt to resume teacher UNet from the checkpoint sidecar
                sf_teacher_sidecar = os.path.join(args.resume_ckpt, "self_flow_teacher_unet.safetensors")
                if os.path.exists(sf_teacher_sidecar):
                    logging.info(f"  Loading Self-Flow teacher UNet from {sf_teacher_sidecar}")
                    try:
                        state_dict = safetensors.torch.load_file(sf_teacher_sidecar, device=str(device))
                        model.self_flow_teacher_unet.load_state_dict(state_dict)
                        logging.info("  Self-Flow teacher UNet loaded successfully.")
                    except Exception as e:
                        logging.error(f" * Failed to load Self-Flow teacher UNet from {sf_teacher_sidecar}: {e}. Using fresh deepcopy.")
                else:
                    logging.info(f"  No Self-Flow teacher UNet sidecar found at {sf_teacher_sidecar}, using current model snapshot as starting point.")

            if model.self_flow_proj_head is None:
                boc = model.unet.config.block_out_channels  # e.g. [320, 640, 1280, 1280]
                sf_mode = getattr(args, 'self_flow_mode', 'shallow')
                in_ch, out_ch = get_self_flow_channels(sf_mode, boc)
                model.self_flow_proj_head = SelfFlowMLPProjectionHead(
                    in_channels=in_ch,
                    out_channels=out_ch,
                ).to(device)
                logging.info(
                    f"  Self-Flow mode={sf_mode!r}, {type(model.self_flow_proj_head).__name__}: {in_ch} -> {out_ch} channels (fp32)"
                )
                # Attempt to resume projection head from the checkpoint sidecar
                sf_proj_sidecar = os.path.join(args.resume_ckpt, "self_flow_proj_head.pt")
                if os.path.exists(sf_proj_sidecar):
                    logging.info(f"  Loading Self-Flow projection head from {sf_proj_sidecar}")
                    try:
                        model.self_flow_proj_head.load_state_dict(
                            torch.load(sf_proj_sidecar, map_location=device)
                        )
                    except RuntimeError:
                        logging.error(f" * Failed to load Self-Flow projection head from {sf_proj_sidecar}, restarting from noise")
                else:
                    logging.info(f"  No Self-Flow projection head sidecar found at {sf_proj_sidecar}, starting fresh.")

        try:
            # currently broken on most systems?
            # unet = torch.compile(unet, mode="max-autotune")
            # text_encoder = torch.compile(text_encoder, mode="max-autotune")
            # vae = torch.compile(vae, mode="max-autotune")
            # logging.info("Successfully compiled models")
            pass
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

    if len(args.caption_variants) == 0:
        logging.info("* Using all caption variants")
    else:
        logging.info(f"* Using caption variants: {args.caption_variants}")

    image_train_items: list[ImageTrainItem] = []

    for resolution_index, batch_resolution in enumerate(args.resolution):
        this_aspects = aspects.get_aspect_buckets(batch_resolution,
                                                  #square_only=model.is_sdxl,
                                                  )
        resolution_multiplier = 1.0 if len(args.resolution_multiplier) == 0 else args.resolution_multiplier[resolution_index]
        if resolution_multiplier != 1.0:
            print(f"Using global multiplier {resolution_multiplier} for resolution {batch_resolution}")
        image_train_items.extend(resolve_image_train_items(args, batch_resolution, this_aspects, global_multiplier=resolution_multiplier))
    for i in image_train_items:
        if i.cond_dropout is None:
            i.cond_dropout = args.cond_dropout
    if args.cond_dropout_global is not None:
        for i in image_train_items:
            i.cond_dropout *= args.cond_dropout_global

    validator = None
    if args.validation_config is not None and args.validation_config != "None":
        validator = EveryDreamValidator(args.validation_config,
                                        default_batch_size=args.forward_slice_size[0] if args.forward_slice_size else args.batch_size,
                                        resolution=args.resolution[0],
                                        log_writer=log_writer,
                                        approx_epoch_length=sum([i.multiplier for i in image_train_items])/args.batch_size
                                        )
        # the validation dataset may need to steal some items from image_train_items
        image_train_items = validator.prepare_validation_splits(image_train_items, model=model)

    report_image_train_item_problems(log_folder, image_train_items, batch_size=args.batch_size,
                                     check_load_all=args.test_images, model=model)

    # -----------------------------------------------------------------------
    # Phase 2 – Data sharding: each rank trains on a non-overlapping subset
    # -----------------------------------------------------------------------
    if _is_dist:
        pre_shard_count = len(image_train_items)
        image_train_items = shard_items(image_train_items, _rank, _world_size)
        logging.warning(
            f"[rank {_rank}/{_world_size}] Data shard: {len(image_train_items)} / {pre_shard_count} items"
        )

    from plugins.plugins import load_plugin
    if args.plugins is not None:
        plugins = [load_plugin(name) for name in args.plugins]
    else:
        logging.info("No plugins specified")
        plugins = []

    plugin_runner = PluginRunner(plugins=plugins)
    plugin_runner.run_on_model_load(unet=model.unet, text_encoder=model.text_encoder, tokenizer=model.tokenizer, optimizer_config=optimizer_config)

    data_loader = DataLoaderMultiAspect(
        image_train_items=image_train_items,
        seed=seed,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        chunk_shuffle_batch_size=args.batch_size,
        batch_id_dropout_p=args.batch_id_dropout_p,
        keep_same_sample_at_different_resolutions_together=args.keep_same_sample_at_different_resolutions_together,
        caption_variants=args.caption_variants,
        expand_caption_variants=args.expand_caption_variants
    )

    mask_p = 0 if args.mask_p is None else args.mask_p
    train_batch = EveryDreamBatch(
        data_loader=data_loader,
        debug_level=1,
        conditional_dropout=args.cond_dropout,
        tokenizer=model.tokenizer,
        tokenizer_2=model.tokenizer_2,
        teacher_tokenizer=teacher_model.tokenizer if teacher_model is not None else None,
        teacher_tokenizer_2=teacher_model.tokenizer_2 if teacher_model is not None else None,
        seed = seed,
        shuffle_tags=args.shuffle_tags,
        keep_tags=args.keep_tags,
        normalize_image=not args.no_normalize_images,
        plugin_runner=plugin_runner,
        rated_dataset=args.rated_dataset,
        rated_dataset_dropout_target=(1.0 - (args.rated_dataset_target_dropout_percent / 100.0)),
        contrastive_loss_batch_ids=args.contrastive_loss_batch_ids,
        contrastive_learning_dropout_p=args.contrastive_learning_dropout_p,
        cond_dropout_noise_p=args.cond_dropout_noise_p,
        mask_p=mask_p,
        invert_masks=args.invert_masks,
    )

    torch.cuda.benchmark = False

    epoch_len = math.ceil(len(train_batch) / args.batch_size)

    # -- DifficultyEstimator construction --------------------------------------
    difficulty_estimator: Optional[DifficultyEstimator] = None
    if getattr(args, "difficulty_estimator", None):
        _de_db_path = args.difficulty_estimator

        # Auto-infer model type if not specified
        _de_model_type = getattr(args, "difficulty_estimator_model_type", None)
        if not _de_model_type:
            if getattr(args, "train_sampler", None) == "flow-matching":
                _de_model_type = "flowmatching-sd2"
            else:
                logging.warning(
                    "DifficultyEstimator: --difficulty_estimator_model_type not set "
                    "and could not be auto-inferred from --train_sampler. "
                    "Loss normalisation will be skipped; scores will not update."
                )
                _de_model_type = "unknown"

        _de_scheduler_type = getattr(args, "difficulty_estimator_scheduler", "type_a")
        _de_min_mult = getattr(args, "difficulty_estimator_min_multiplier", 0.5)
        _de_max_mult = getattr(args, "difficulty_estimator_max_multiplier", 2.0)
        _de_expand = getattr(args, "difficulty_estimator_expand_factor", 1.0)
        _de_alpha = getattr(args, "difficulty_estimator_ema_alpha", 0.1)
        _de_min_obs = getattr(args, "difficulty_estimator_min_observation_count", 10)

        if _de_scheduler_type == "type_b":
            _de_slab_size = getattr(args, "difficulty_estimator_slab_size", None) or 20
            _de_base_interval = getattr(args, "difficulty_estimator_base_interval", None)
            if _de_base_interval is None:
                _de_base_interval = math.ceil(
                    len(train_batch.data_loader.prepared_train_data) /
                    (_de_slab_size * args.batch_size)
                )
                logging.info(
                    "DifficultyEstimator (TypeB): auto-computed base_interval=%d "
                    "(dataset_size=%d, slab_size=%d, batch_size=%d)",
                    _de_base_interval,
                    len(train_batch.data_loader.prepared_train_data),
                    _de_slab_size,
                    args.batch_size,
                )
            scheduler = TypeBScheduler(
                base_interval=_de_base_interval,
                slab_size=_de_slab_size,
                expand_factor=_de_expand,
            )
        else:
            scheduler = TypeAScheduler(
                min_multiplier=_de_min_mult,
                max_multiplier=_de_max_mult,
                expand_factor=_de_expand,
            )

        difficulty_estimator = DifficultyEstimator.load(
            path=_de_db_path,
            model_type_identifier=_de_model_type,
            scheduler=scheduler,
            min_obs_count=_de_min_obs,
            ema_alpha=_de_alpha,
        )
        logging.info(
            "DifficultyEstimator ready: scheduler=%s, model_type=%s, db=%s",
            _de_scheduler_type, _de_model_type, _de_db_path,
        )
    # --------------------------------------------------------------------------
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
                                       model,
                                       epoch_len,
                                       plugin_runner,
                                       log_writer)
    #ed_optimizer.register_unet_nan_hooks_simple()
    #ed_optimizer.register_unet_nan_hooks_full()

    log_args(log_writer, args, optimizer_config, log_folder, log_time)

    sample_generator = SampleGenerator(log_folder=log_folder, log_writer=log_writer,
                                       default_resolution=args.resolution[0], default_seed=args.seed,
                                       config_file_path=args.sample_prompts,
                                       batch_size=max(1,args.batch_size//2),
                                       default_sample_steps=args.sample_steps,
                                       default_sample_epochs=None,
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
                global_step = tv.global_step
                interrupted_checkpoint_path = os.path.join(f"{log_folder}/ckpts/interrupted-gs{global_step:05}")
                print()
                if not _is_main:
                    # Non-main ranks: just exit cleanly without saving
                    logging.warning(f"[rank {_rank}] CTRL-C received, exiting (rank 0 will handle save)")
                    cleanup_distributed()
                    exit(_SIGTERM_EXIT_CODE)
                if args.no_save_on_error:
                   logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, but NOT saving because --no_save_on_error was passed{Style.RESET_ALL}")
                else:
                    logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                    logging.error(f"{Fore.LIGHTRED_EX} CTRL-C received, attempting to save model to {interrupted_checkpoint_path}{Style.RESET_ALL}")
                    logging.error(f"{Fore.LIGHTRED_EX} ************************************************************************{Style.RESET_ALL}")
                    time.sleep(2) # give opportunity to ctrl-C again to cancel save
                    save_model(interrupted_checkpoint_path, global_step=global_step, ed_state=make_current_ed_state(),
                               save_ckpt_dir=args.save_ckpt_dir, yaml_name=model.yaml, save_full_precision=args.save_full_precision,
                               save_optimizer_flag=True, save_ckpt=not args.no_save_ckpt)
            cleanup_distributed()
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

    num_dataloader_workers = max(1, args.num_dataloader_workers // _world_size)
    if _world_size > 1:
        logging.info(f"Using {num_dataloader_workers} dataloader workers per rank (total {num_dataloader_workers * _world_size} across {_world_size} ranks)")
    train_dataloader = build_torch_dataloader(train_batch, batch_size=args.batch_size, num_workers=num_dataloader_workers)

    model.unet.train() if (args.gradient_checkpointing or not args.disable_unet_training) else model.unet.eval()
    if args.disable_unet_training:
        model.unet.requires_grad_(False)

    model.text_encoder.train() if not args.disable_textenc_training else model.text_encoder.eval()
    if args.disable_textenc_training:
        model.text_encoder.requires_grad_(False)
    if model.is_sdxl:
        model.text_encoder_2.train() if not args.disable_textenc_training else model.text_encoder_2.eval()
        if args.disable_textenc_training:
            model.text_encoder_2.requires_grad_(False)

    logging.info(f" unet device: {model.unet.device}, precision: {model.unet.dtype}, training: {model.unet.training}")
    logging.info(f" text_encoder device: {model.text_encoder.device}, precision: {model.text_encoder.dtype}, training: {model.text_encoder.training}")
    if model.is_sdxl:
        logging.info(f" text_encoder_2 device: {model.text_encoder_2.device}, precision: {model.text_encoder_2.dtype}, training: {model.text_encoder_2.training}")
    logging.info(f" vae device: {model.vae.device}, precision: {model.vae.dtype}, training: {model.vae.training}")
    logging.info(f" scheduler: {model.noise_scheduler.__class__}")

    # -----------------------------------------------------------------------
    # Phase 3 – DDP module wrapping (trainable modules only)
    # -----------------------------------------------------------------------
    if _is_dist:
        if not args.disable_unet_training:
            model.unet = DDPWrapper(model.unet, device_ids=[_local_rank], find_unused_parameters=True, gradient_as_bucket_view=True)
            logging.warning(f"[rank {_rank}] UNet wrapped in DDP")
        if not args.disable_textenc_training:
            model.text_encoder = DDPWrapper(model.text_encoder, device_ids=[_local_rank])
            logging.warning(f"[rank {_rank}] TextEncoder wrapped in DDP")
            if model.is_sdxl and model.text_encoder_2 is not None:
                model.text_encoder_2 = DDPWrapper(model.text_encoder_2, device_ids=[_local_rank])
                logging.warning(f"[rank {_rank}] TextEncoder2 wrapped in DDP")

    logging.info(f" {Fore.GREEN}Project name: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.project_name}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}grad_accum: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.grad_accum}{Style.RESET_ALL}"),
    logging.info(f" {Fore.GREEN}batch_size: {Style.RESET_ALL}{Fore.LIGHTGREEN_EX}{args.batch_size}{Style.RESET_ALL}")
    logging.info(f" {Fore.GREEN}epoch_len: {Fore.LIGHTGREEN_EX}{epoch_len}{Style.RESET_ALL}")

    epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True, dynamic_ncols=True)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")
    epoch_times = []

    tv = TrainingVariables()
    tv.setup_default_slice_sizes(args)
    tv.global_step = 0
    training_start_time = time.time()
    last_epoch_saved_time = training_start_time

    append_epoch_log(global_step=tv.global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    log_data = LogData()
    if args.log_attention_activations:
        log_data.attention_activation_logger = ActivationLogger(model=model.unet, writer=log_writer)

    assert len(train_batch) > 0, "train_batch is empty, check that your data_root is correct"

    def generate_samples(global_step: int, batch: dict|None):

        with isolate_rng():
            sample_generator.reload_config()
            if batch is not None:
                flattened_captions_dict = [v
                                           for _, l in batch["captions"].items()
                                           for v in l]
                sample_generator.update_random_captions(flattened_captions_dict)

            extra_info: str = ""
            torch.cuda.empty_cache()

            unet_was_training = model.unet.training
            text_encoder_was_training = model.text_encoder.training
            try:
                model.unet.eval()
                model.text_encoder.eval()
                if model.text_encoder_2:
                    model.text_encoder_2.eval()
                inference_pipe = sample_generator.create_inference_pipe(
                    model_being_trained=model,
                    diffusers_scheduler_config=model.noise_scheduler.config,
                    flow_match_shift=args.flow_match_shift,
                    flow_match_shift_dynamic=args.flow_match_shift_dynamic
                ).to(device)
                sample_generator.generate_samples(inference_pipe, global_step, extra_info=extra_info)

                # Cleanup
                del inference_pipe
            finally:
                if unet_was_training:
                    model.unet.train()
                if text_encoder_was_training:
                    model.text_encoder.train()
                    if model.text_encoder_2:
                        model.text_encoder_2.train()

            gc.collect()
            torch.cuda.empty_cache()

    def make_save_path(epoch, global_step, num_trained_samples, prepend=""):
        basename = f"{prepend}{args.project_name}"
        if epoch is not None:
            basename += f"-ep{epoch:02}"
        if global_step is not None:
            basename += f"-gs{global_step:05}"
        if num_trained_samples is not None:
            basename += f"-n{num_trained_samples:06}"
        return os.path.join(log_folder, "ckpts", basename)

    def get_model_prediction_and_target_validation_wrapper(image: torch.Tensor, conditioning: Conditioning
                                                ) -> ValidationStepResult:
        batch_size = image.shape[0]
        timesteps = get_uniform_timesteps(batch_size=batch_size,
                                          batch_share_timesteps=False,
                                          device=model.unet.device,
                                          timesteps_ranges=[(args.timestep_start, args.timestep_end)] * batch_size,
                                          )
        if type(model.noise_scheduler) is TrainFlowMatchEulerDiscreteScheduler:
            timesteps = TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(timesteps, model.noise_scheduler.timesteps)
        model.load_vae_to_device(device)
        latents = encode_with_vae_to_scaled_latents(image, model, device=model.unet.device, args=args)
        if args.offload_vae:
            model.load_vae_to_device('cpu')
        noise = get_noise(latents.shape, model.unet.device, image.dtype,
                          pyramid_noise_discount=args.pyramid_noise_discount,
                          zero_frequency_noise_ratio=args.zero_frequency_noise_ratio,
                          batch_share_noise=False)

        model_pred_result = get_model_prediction_and_target(
            latents,
            conditioning,
            noise,
            timesteps,
            model=model,
            args=args,
            skip_contrastive=True
        )

        return ValidationStepResult(
            model_pred=model_pred_result.model_pred,
            target=model_pred_result.target,
            timesteps=timesteps,
            noisy_latents=model_pred_result.noisy_latents,
        )

    def make_inference_pipe():
        """Factory that builds a fresh inference pipeline for anomaly validation."""
        return sample_generator.create_inference_pipe(
            model_being_trained=model,
            diffusers_scheduler_config=model.noise_scheduler.config,
            flow_match_shift=args.flow_match_shift,
            flow_match_shift_dynamic=args.flow_match_shift_dynamic,
        ).to(device)

    # Pre-train validation to establish a starting point on the loss graph
    if validator and not args.no_initial_validation and _is_main:
        try:
            validator.do_validation(
                model=model,
                global_step=0,
                get_model_prediction_and_target_callable=get_model_prediction_and_target_validation_wrapper,
                pipe_factory=make_inference_pipe,
            )
        except Exception as ex:
            traceback.print_exc()
            logging.warning(f"Validation failed during initial validation step, continuing anyway. Exception: {ex}")

    # the sample generator might be configured to generate samples before step 0
    if _is_main and sample_generator.generate_pretrain_samples:
        _, batch = next(enumerate(train_dataloader))
        generate_samples(global_step=0, batch=batch)

    def make_current_ed_state() -> EveryDreamTrainingState:
        return EveryDreamTrainingState(optimizer=ed_optimizer,
                                       train_batch=train_batch,
                                       model=model
                                       )

    epoch = None

    try:
        plugin_runner.run_on_training_start(log_folder=log_folder, project_name=args.project_name, max_epochs=args.max_epochs, ed_state=make_current_ed_state(),)

        tv.desired_effective_batch_size = choose_effective_batch_size(args, 0)
        tv.remaining_stratified_timesteps = None
        tv.shared_timestep = None

        if args.timestep_interval_sampling:
            if args.timesteps_multirank_stratified:
                raise ValueError("--timestep_interval_sampling and --timesteps_multirank_stratified are mutually exclusive")
            if args.timestep_curriculum_alpha != 0:
                logging.warning(" * --timestep_interval_sampling is not compatible with --timestep_curriculum_alpha; intervals are pre-computed from --timestep_start/--timestep_end and will not shift during training")
            tv.timestep_intervals = compute_timestep_intervals(
                model.noise_scheduler,
                k=args.timestep_interval_n,
                t_start=args.timestep_start,
                t_end=args.timestep_end,
            )
            logging.info(f" * Timestep interval sampling: {len(tv.timestep_intervals)} SNR-based intervals: {tv.timestep_intervals}")
        step = 0
        wants_stop = False
        force_save_optimizer = False

        with ddp_no_sync_ctx(
            model.unet,
            model.text_encoder,
            model.text_encoder_2
        ):
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
                    epoch=0 if epoch is None else epoch,
                    global_step=tv.global_step,
                    epoch_length=epoch_len,
                    project_name=args.project_name,
                    log_folder=log_folder,
                    data_root=args.data_root,
                    arg_update_callback=update_arg
                )

                sample_generator.on_epoch_start(
                    epoch=epoch,
                    global_step=tv.global_step,
                    epoch_length=epoch_len
                )

                epoch_start_time = time.time()

                steps_pbar = tqdm(range(epoch_len), position=1, leave=False, dynamic_ncols=True)
                steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps{Style.RESET_ALL}")

                validation_steps = (
                    [] if validator is None
                    else validator.get_validation_step_indices(epoch, len(train_dataloader))
                )

                timesteps = None
                #logging.info("fetching batch...")

                _dataloader_pre_time = time.perf_counter()

                for step, full_batch in enumerate(train_dataloader):
                    #logging.info("... fetched.")

                    _dataloader_post_time = time.perf_counter()
                    record_performance_timing('_dataloader_fetch', _dataloader_post_time - _dataloader_pre_time, full_batch['image'].shape[0])


                    try:
                        train_progress_01 = compute_train_process_01(epoch=epoch, step=step, steps_per_epoch=epoch_len,
                                                                     max_epochs=args.max_epochs, max_global_steps=args.max_steps)

                        step_start_time = time.time()

                        plugin_runner.run_on_step_start(epoch=epoch,
                                                        local_step=step,
                                                        global_step=tv.global_step,
                                                        project_name=args.project_name,
                                                        log_folder=log_folder,
                                                        batch=full_batch,
                                                        ed_state=make_current_ed_state())

                        # for k in full_batch['captions'].keys():
                        #    if any(c is not None and len(c.strip()) == 0 for c in full_batch['captions'][k]):
                        #        print('a caption was already empty before cond dropout. paths:', full_batch['pathnames'], 'captions:', full_batch['captions'])

                        image_pixel_count = full_batch["image"].shape[2] * full_batch["image"].shape[3]
                        tv.batch_resolution = get_best_match_resolution(
                            args.resolution, image_pixel_count=image_pixel_count
                        )
                        tv.max_backward_slice_size = _choose_backward_slice_size(tv)
                        tv.forward_slice_size = _get_default_forward_slice_size(tv)
                        if tv.max_backward_slice_size <= tv.accumulated_loss_images_count:
                            optimizer_backward(ed_optimizer, tv, plugin_runner, 'truncated backward: ')
                            gc.collect()
                            torch.cuda.empty_cache()

                        safe_backward_size = _get_safe_backward_size(gpu, model.device, image_pixel_count // 64)
                        if safe_backward_size > tv.max_backward_slice_size and tv.batch_resolution not in tv._backward_size_hint_logged:
                            logging.info(f"at resolution {tv.batch_resolution} you could probably do backward={safe_backward_size} (you requested max {tv.max_backward_slice_size})")
                            tv._backward_size_hint_logged.add(tv.batch_resolution)

                        if not args.disable_backward_memsafe and tv.batch_resolution not in args.disable_backward_memsafe_resolutions:
                            if gpu is not None:
                                torch.cuda.empty_cache()
                                gc.collect()
                                max_safe_forward_size = _get_safe_forward_size(gpu, model.device, image_pixel_count, is_sdxl=model.is_sdxl)
                                if max_safe_forward_size == 0:
                                    # maybe not enough memory. if we can, do an emergency backward pass if we have any loss pending
                                    accumulated_loss_images_count_before_emergency_backward = tv.accumulated_loss_images_count
                                    if tv.accumulated_loss_images_count > 0:
                                        with torch.amp.autocast('cuda', enabled=args.amp):
                                            optimizer_backward(ed_optimizer, tv, plugin_runner, 'emergency backward: ')
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                        max_safe_forward_size = _get_safe_forward_size(gpu, model.device, image_pixel_count, is_sdxl=model.is_sdxl)
                                    if max_safe_forward_size == 0:
                                        logging.warning(f" * Unable to free enough ram even after emergency backward of {accumulated_loss_images_count_before_emergency_backward} samples worth of accumulated loss @res {tv.batch_resolution} - possible OOM follows")
                                        max_safe_forward_size = 1
                                #if max_safe_forward_size < _get_default_forward_slice_size(tv):
                                #    logging.info(f" * forward slice size clamped from {tv.forward_slice_size} to {max_safe_forward_size} for image size {full_batch['image'].shape[2]}x{full_batch['image'].shape[3]}; num accumulated images: {tv.accumulated_loss_images_count}")
                                tv.forward_slice_size = min([_get_default_forward_slice_size(tv), max_safe_forward_size])
                            else:
                                gpu_used_mem, gpu_total_mem = 0, 0

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

                        if args.flow_match_shift_dynamic and isinstance(model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
                            # gentle shift, randomly falling back to no shift
                            assert model.noise_scheduler.config.time_shift_type == 'linear'
                            shift = 1.0 + 0.5 * (image_pixel_count / 1024**2)
                            if random.random() < args.flow_match_shift_dropout_p:
                                shift = 1.0
                            #print(f'at resolution {round(image_pixel_count ** 0.5)}, shift is {shift} ({model.noise_scheduler.config.time_shift_type})')
                            model.set_noise_scheduler_shift(shift)
                            if teacher_model is not None and isinstance(teacher_model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
                                teacher_model.set_noise_scheduler_shift(shift)

                        train_step(
                            full_batch=full_batch,
                            model=model,
                            teacher_model=teacher_model,
                            tv=tv,
                            train_progress_01=train_progress_01,
                            ed_optimizer=ed_optimizer,
                            log_writer=log_writer,
                            log_data=log_data,
                            steps_pbar=steps_pbar,
                            plugin_runner=plugin_runner,
                            did_step_optimizer_cb=None,
                            vae_dtype=vae_dtype,
                            args=args,
                        )

                        ed_optimizer.step_schedulers(tv.global_step)

                        if _is_main:
                            user_wants_samples = check_semaphore_file_and_unlink(WANT_SAMPLES_SEMAPHORE_FILE)
                            if user_wants_samples or sample_generator.should_generate_samples(tv.global_step, local_step=step):
                                generate_samples(global_step=tv.global_step, batch=full_batch)

                        if args.ema_decay_rate != None:
                            if ((tv.global_step + 1) % args.ema_update_interval) == 0:
                                if _is_main:  # EMA is rank-0 only
                                    if args.disable_unet_training != True:
                                        update_ema(model.unet, unet_ema, args.ema_decay_rate, default_device=device, ema_device=args.ema_device)

                                    if args.disable_textenc_training != True:
                                        update_ema(model.text_encoder, text_encoder_ema, args.ema_decay_rate, default_device=device, ema_device=args.ema_device)

                        # Self-Flow EMA teacher update (independent interval from main EMA)
                        if model.self_flow_teacher_unet is not None:
                            sf_interval = getattr(args, 'self_flow_ema_update_interval', 1)
                            if ((tv.global_step + 1) % sf_interval) == 0:
                                update_ema(
                                    model.unet,
                                    model.self_flow_teacher_unet,
                                    args.self_flow_ema_decay,
                                    default_device=device,
                                    ema_device=device,
                                )

                        steps_pbar.update(1)

                        images_per_sec = full_batch['image'].shape[0] / (time.time() - step_start_time)
                        log_data.images_per_sec_log_step.append(images_per_sec)

                        if (tv.global_step + 1) % args.log_step == 0:
                            if _is_main:
                                logs = do_log_step(args, ed_optimizer, log_data, log_folder,
                                                   log_writer, model, tv)
                                append_epoch_log(global_step=tv.global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer,
                                                 **logs)
                            torch.cuda.empty_cache()

                        # ------------------------------------------------------------------
                        # Validation
                        # step-based condition is deterministic (same on every rank).
                        # Semaphore check is rank-0 only; decision is broadcast so the
                        # dist_barrier() is always entered or skipped by ALL ranks.
                        # ------------------------------------------------------------------
                        _do_validate = bool(validator and step in validation_steps)

                        if _do_validate:
                            dist_barrier()
                            if _is_main:
                                try:
                                    validator.do_validation(
                                        model=model,
                                        global_step=tv.global_step,
                                        get_model_prediction_and_target_callable=get_model_prediction_and_target_validation_wrapper,
                                        pipe_factory=make_inference_pipe,
                                    )
                                except Exception as e:
                                    logging.error("Validation raised an exception: " + str(e))
                                    traceback.print_exc()
                            dist_barrier()

                        # ------------------------------------------------------------------
                        # Saving
                        # Time / epoch conditions are deterministic.  Semaphore checks are
                        # rank-0 only.  All flags are broadcast before the barrier so every
                        # rank agrees on whether to enter it.
                        # ------------------------------------------------------------------
                        min_since_last_ckpt = (time.time() - last_epoch_saved_time) / 60

                        needs_save = False
                        needs_optimizer_save = args.save_optimizer
                        if args.ckpt_every_n_minutes is not None and (min_since_last_ckpt > args.ckpt_every_n_minutes):
                            last_epoch_saved_time = time.time()
                            logging.info(f"Saving model, {args.ckpt_every_n_minutes} mins at step {tv.global_step}")
                            needs_save = True

                        if _is_main:
                            if check_semaphore_file_and_unlink(SAVE_FULL_SEMAPHORE_FILE):
                                needs_save = True
                            if check_semaphore_file_and_unlink(SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE):
                                needs_save = True
                                needs_optimizer_save = True
                            if check_semaphore_file_and_unlink(SAVE_FULL_AND_STOP_SEMAPHORE_FILE):
                                wants_stop = True
                            if check_semaphore_file_and_unlink(SAVE_FULL_WITH_OPTIMIZER_AND_STOP_SEMAPHORE_FILE):
                                wants_stop = True
                                force_save_optimizer = True

                        def is_first_step_of_save_epoch(every_n_epochs, start_epoch=0):
                            return (epoch > 0 and epoch % every_n_epochs == 0 and step == 0 and
                                    epoch < args.max_epochs and epoch >= start_epoch)

                        if is_first_step_of_save_epoch(args.save_every_n_epochs, start_epoch=args.save_ckpts_from_n_epochs):
                            logging.info(f" Saving model, {args.save_every_n_epochs} epochs at step {tv.global_step}")
                            needs_save = True

                        if needs_save:
                            if _is_main:
                                save_path = make_save_path(epoch, tv.global_step, num_trained_samples=tv.total_trained_samples_count)
                                save_model(save_path,
                                           global_step=tv.global_step,
                                           ed_state=make_current_ed_state(),
                                           save_ckpt_dir=args.save_ckpt_dir, yaml_name=None,
                                           save_full_precision=args.save_full_precision,
                                           save_optimizer_flag=needs_optimizer_save,
                                           save_ckpt=not args.no_save_ckpt,
                                           plugin_runner=plugin_runner)

                        if args.lora and is_first_step_of_save_epoch(args.lora_save_every_n_epochs, start_epoch=0):
                            if _is_main:
                                logging.info(f" Saving lora")
                                save_path = make_save_path(epoch, tv.global_step, num_trained_samples=tv.total_trained_samples_count)
                                save_model_lora(model=model, save_path=save_path)

                        # -- step end

                        plugin_runner.run_on_step_end(epoch=epoch,
                                              global_step=tv.global_step,
                                              local_step=step,
                                              num_samples=tv.total_trained_samples_count,
                                              project_name=args.project_name,
                                              log_writer=log_writer,
                                              log_folder=log_folder,
                                              data_root=args.data_root,
                                              batch=full_batch,
                                              ed_state=make_current_ed_state())

                        if (epoch == args.max_epochs-1
                                and ed_optimizer.will_do_grad_accum_step(step, tv.global_step)
                                and epoch_len-step < args.grad_accum
                        ):
                            logging.info(f"* only {epoch_len-step} steps remaining at grad accum {args.grad_accum} -> early stop")
                            break

                        tv.global_step += 1

                        if wants_stop:
                            logging.info(f"* Stop requested, stopping")
                            break

                        if args.max_steps is not None and tv.global_step >= args.max_steps:
                            logging.info(f"* max_steps reached, stopping")
                            break

                        # end of step

                    except Exception:
                        try:
                            current_timesteps = timesteps
                        except NameError:
                            current_timesteps = None
                        try:
                            batch_idx0 = nibble_batch(full_batch, 1)[0]
                            image_shape = batch_idx0['image'].shape
                        except NameError:
                            batch_idx0 = {}
                            image_shape = []
                        traceback.print_exc()
                        logging.error(f"step {tv.global_step} failed. image shape {image_shape}, full batch idx0: {batch_idx0}")
                        logging.error(f"  timesteps: {current_timesteps}, training values: {pprint.pformat(tv.filtered_for_log())}")
                        raise

                    #logging.info("fetching...")

                    _dataloader_pre_time = time.perf_counter()

                steps_pbar.close()

                elapsed_epoch_time = (time.time() - epoch_start_time) / 60
                epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
                log_writer.add_scalar(
                    "performance/minutes per epoch", elapsed_epoch_time, tv.global_step
                )

                plugin_runner.run_on_epoch_end(epoch=epoch,
                                               global_step=tv.global_step,
                                               project_name=args.project_name,
                                               log_folder=log_folder,
                                               data_root=args.data_root,
                                               arg_update_callback=update_arg,
                                               ed_state=make_current_ed_state(),
                                               training_variables=tv,
                                               sample_generator_cb=generate_samples)

                epoch_pbar.update(1)
                if epoch < args.max_epochs - 1:
                    # -- DifficultyEstimator epoch-end update --------------------
                    if difficulty_estimator is not None:
                        difficulty_estimator.ingest_epoch_losses(log_data)
                        difficulty_estimator.update_item_schedule(
                            train_batch.data_loader.prepared_train_data
                        )
                        train_batch.data_loader.recompute_expected_epoch_size()
                        difficulty_estimator.save(args.difficulty_estimator)
                    # ------------------------------------------------------------
                    train_batch.shuffle(epoch_n=epoch, max_epochs=args.max_epochs)

                if len(log_data.loss_epoch) > 0:
                    loss_epoch = sum(log_data.loss_epoch) / len(log_data.loss_epoch)
                    log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_epoch, global_step=tv.global_step)

                gc.collect()

                if wants_stop:
                    break

                if args.max_steps is not None and tv.global_step >= args.max_steps:
                    break

                # end of epoch

            # end of training
            epoch = args.max_epochs

            plugin_runner.run_on_training_end(ed_state=make_current_ed_state(), log_folder=log_folder, project_name=args.project_name, global_step=tv.global_step)

            # training is done
            # block until everyone is done
            while True:
                global_state_signal = get_distributed_state_signal(StateSignal.DONE, model.device)
                if global_state_signal == StateSignal.DONE:
                    break
                else:
                    raise ValueError(f"Unrecognized global state signal: {global_state_signal}")

        # exit no-sync context

        if _is_main:
            save_path = make_save_path(
                epoch, tv.global_step, num_trained_samples=tv.total_trained_samples_count, prepend=("" if args.no_prepend_last else "last-")
            )
            if not args.lora:
                save_model(save_path, global_step=tv.global_step, ed_state=make_current_ed_state(),
                           save_ckpt_dir=args.save_ckpt_dir, yaml_name=model.yaml, save_full_precision=args.save_full_precision,
                           save_optimizer_flag=args.save_optimizer or force_save_optimizer, save_ckpt=not args.no_save_ckpt,
                           plugin_runner=plugin_runner)
            if args.lora:
                save_model_lora(model=model, save_path=save_path)

        if validator and _is_main:
            print("doing final validation pass")
            try:
                validator.do_validation(
                    model=model,
                    global_step=tv.global_step,
                    get_model_prediction_and_target_callable=get_model_prediction_and_target_validation_wrapper,
                    pipe_factory=make_inference_pipe
                )
            except Exception as e:
                logging.error("Validation threw an exception: " + str(e))
                traceback.print_exc()

        if _is_main and not sample_generator.should_generate_samples(global_step=tv.global_step-1, local_step=step):
            print("generating final samples")
            # free up memory we might need for samples
            tv.clear_accumulated_loss()
            torch.cuda.empty_cache()
            gc.collect()
            # get samples for random captions
            _, batch = next(enumerate(train_dataloader))
            generate_samples(global_step=tv.global_step, batch=batch)

        total_elapsed_time = time.time() - training_start_time
        logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
        logging.info(f"Total training time took {total_elapsed_time/60:.2f} minutes, total steps: {tv.global_step}")
        logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")

    except Exception as ex:
        logging.error(f"{Fore.LIGHTYELLOW_EX}Something went wrong, attempting to save model{Style.RESET_ALL}")
        logging.error(f"{Fore.LIGHTYELLOW_EX}NOT attempting to save model{Style.RESET_ALL}")
        raise ex
    finally:
        cleanup_distributed()

    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} **** Finished training ****{Style.RESET_ALL}")
    logging.info(f"{Fore.LIGHTWHITE_EX} ***************************{Style.RESET_ALL}")


def _get_safe_backward_size(gpu, device, num_latent_pixels: int) -> int:

    # every forward latent pixel needs 0.4 MB to store graph
    graph_bytes_per_latent_pixel = 0.4 * 1024**2
    graph_bytes_per_sample = graph_bytes_per_latent_pixel * num_latent_pixels

    _, gpu_total_mem_mb = gpu.get_gpu_memory()
    memory_allocated = torch.cuda.memory_allocated(
        device=device
    )
    gpu_free_mem_b = gpu_total_mem_mb * 1024**2 - memory_allocated

    # observed empirically, backward() needs at least 2.2GB
    # this appears to be independent of any freezing
    # let's say 2.4 to be safe
    backward_safety_margin = 2400 * 1024**2

    graph_bytes_available = gpu_free_mem_b - backward_safety_margin
    max_backward_size = graph_bytes_available // graph_bytes_per_sample
    return max_backward_size



def _get_safe_forward_size(gpu, device, num_image_pixels: int, is_sdxl: bool) -> int:
    gpu_used_mem_mb, gpu_total_mem_mb = gpu.get_gpu_memory()
    #gpu_free_mem_b = (gpu_total_mem_mb - gpu_used_mem_mb) * (1024 ** 2)

    # discovered empirically
    #if vae_dtype in [torch.float16, torch.bfloat16]:
    #    dtype_bytes = 2
    #elif vae_dtype == torch.float32:
    #    dtype_bytes = 4
    #else:
    #    raise ValueError(f"Unrecognized VAE dtype: {model.vae.dtype}")
    #
    #def get_required_vae_vram_per_image(pixel_count):
    #    return dtype_bytes * (1e8 + 500 * pixel_count) + 600 * pixel_count
    memory_allocated = torch.cuda.memory_allocated(
        device=device
    )
    gpu_free_mem_b = gpu_total_mem_mb * 1024**2 - memory_allocated
    # memory_cached = torch.cuda.memory_cached(device=model.device)

    # each forward pass without backward increases vram peak use by ~130000 (230000 for sdxl) bytes per latent pixel
    num_vae_channels = 4
    vae_scale_factor = 8
    num_latent_pixels_per_image = num_vae_channels * num_image_pixels // (vae_scale_factor ** 2)
    memory_required_per_image = (230000 if is_sdxl else 130000) * num_latent_pixels_per_image
    max_safe_forward_size = gpu_free_mem_b // memory_required_per_image
    return max_safe_forward_size

_has_checked_bad_distribution = False



if __name__ == "__main__":
    check_git()
    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--config", type=str, required=False, default=None, help="JSON config file to load options from")
    args, argv = argparser.parse_known_args()

    load_train_json_from_file(args, report_load=True)

    argparser = argparse.ArgumentParser(description="EveryDream2 Training options")
    argparser.add_argument("--amp", action=argparse.BooleanOptionalAction,  default=True, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--amp_without_grad_scaler", action=argparse.BooleanOptionalAction, default=False, help="If passed, use AMP but without a grad scaler (default: False, meaning use grad scaler when AMP is enabled)")
    argparser.add_argument("--force_bfloat16", action=argparse.BooleanOptionalAction, default=False, help="If passed, use bfloat16 for training")
    argparser.add_argument("--init_grad_scale", type=int, default=None, help="initial value for GradScaler (default=2^17.5)")
    argparser.add_argument("--attn_type", type=str, default="sdp", help="Attention mechanismto use", choices=["xformers", "sdp", "slice"])
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--batch_size_curriculum_alpha", type=float, default=0.5, help="curriculum alpha, default=0.5 (rapid (squared) falloff from initial)")
    argparser.add_argument("--interleave_batch_size_1", action='store_true', help="If passed, toggle between batches of BS1 and batches of current_batch_size")
    argparser.add_argument("--interleave_batch_size_1_alpha", type=float, default=0, help="How many BS1 batches to run when interleaving, as a factor of the current batch size (0=1 batch, 1=same as current batch size, 0.5=sqrt of current batch size etc)")
    argparser.add_argument("--optimizer_batch_size", type=int, default=None, help="If specified, step optimizer every this many samples. overriden by --initial_batch_size and --final_batch_size.")
    argparser.add_argument("--initial_batch_size", type=int, default=None, help="initial batch size for curriculum")
    argparser.add_argument("--final_batch_size", type=int, default=None, help="final batch size for curriculum")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None, help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0, help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04, help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--cond_dropout_curriculum_alpha", type=float, default=0, help="cond dropout curriculum alpha, from cond_dropout to final_cond_dropout, controlled by --cond_dropout_curriculum_source")
    argparser.add_argument("--cond_dropout_curriculum_source", choices=['timestep', 'batch_size', 'batch_size_and_timestep', 'global_step'], default='global_step',
                           help="source for cond dropout curriculum - timestep (high timestep (high noise)...low timestep), batch size (initial_batch_size...final_batch_size), or global_step")
    argparser.add_argument("--final_cond_dropout", type=float, default=None, help="if doing cond dropout curriculum, the final cond dropout (timestep=0)")
    argparser.add_argument("--loss_scale", type=float, default=1, help="additional loss scaling")
    argparser.add_argument("--cond_dropout_global", type=float, default=None, help="Global conditioning dropout probability multiplier - if passed, all cond_dropout probabilities are multiplied by this")
    argparser.add_argument("--data_root", type=str, default="input", help="folder where your training images are")
    argparser.add_argument("--data_multiplier_per_path", type=str, nargs="*", default=[], help="optional json file(s) mapping real image paths (symlinks resolved) to an additional multiplier factor. You may pass multiple files.")
    argparser.add_argument("--num_dataloader_workers", type=int, default=None, help="number of worker threads for dataloaders (affects performance). if not specified, default is based on CPU count and batch size.")
    argparser.add_argument("--skip_undersized_images", action='store_true', help="If passed, ignore images that are considered undersized for the training resolution")
    argparser.add_argument("--disable_amp", action=argparse.BooleanOptionalAction, default=False, help="disables automatic mixed precision (def: False)")
    argparser.add_argument("--disable_textenc_training", action=argparse.BooleanOptionalAction, default=False, help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_unet_training", action=argparse.BooleanOptionalAction, default=False, help="disables training of unet (def: False) NOT RECOMMENDED")
    argparser.add_argument("--freeze_unet_balanced", action="store_true", default=False, help="If passed, apply a 'balanced' unet freeze strategy: Train time_embedding.*, *.attentions.* (all parameters within attention blocks), conv_norm_out.*, and conv_out.*. Freeze the rest (conv_in.*, *.resnets.*, *samplers.*)." )
    argparser.add_argument("--embedding_perturbation", type=float, default=0.0, help="random perturbation of text embeddings (def: 0.0)")
    argparser.add_argument("--latents_perturbation", type=float, default=0.0, help="random perturbation of latents (def: 0.0)")
    argparser.add_argument("--flip_p", type=float, default=0.0, help="probability of flipping image horizontally (def: 0.0) use 0.0 to 1.0, ex 0.5, not good for specific faces!")
    argparser.add_argument("--gpuid", type=int, default=0, help="id of gpu to use for training, (def: 0) (ex: 1 to use GPU_ID 1), use nvidia-smi to find your GPU ids")
    argparser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="enable gradient checkpointing to reduce VRAM use, may reduce performance (def: False)")
    argparser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation factor (def: 1), (ex, 2)")
    argparser.add_argument("--forward_slice_size", type=int, default=[], nargs="+", help="If specified, subdivide forward pass (max --batch_size samples) into slices of <= this size to reduce VRAM usage (loss batch size is unaffected). Pass multiple values to specify per-resolution. Slices may be dynamically reduced under memory pressure - see also --disable_backward_memsafe .")
    argparser.add_argument("--max_backward_slice_size", type=int, default=[], nargs="+", help="Max number of samples to accumulate graph before doing backward (NOT optimizer step). Pass multiple values to set per-resolution.")
    argparser.add_argument("--disable_backward_memsafe", action=argparse.BooleanOptionalAction, default=False, help="If passed, disable dynamic forward slice sizing")
    argparser.add_argument("--disable_backward_memsafe_resolutions", type=int, nargs="+", default=[], help="If passed, disable dynamic forward slice sizing on these resolutions only")

    argparser.add_argument("--logdir", type=str, default="logs", help="folder to save logs to (def: logs)")
    argparser.add_argument("--log_step", type=int, default=25, help="How often to log training stats, def: 25, recommend default!")
    argparser.add_argument("--log_named_parameters_magnitudes", action='store_true', help="If passed, log the magnitudes of all named parameters")
    argparser.add_argument('--log_attention_activations', action='store_true', help='If passed, magnitudes of attention activation modules in the unet')
    argparser.add_argument("--loss_type", type=str, default="mse_huber",
                           help="type of loss / weight (def: mse_huber)", choices=["huber", "mse", "mse_huber", "huber_mse", "sd3-cosmap", "v-mse", "cosmap-2"])
    argparser.add_argument("--loss_mean_over_full_effective_batch", default=True, action=argparse.BooleanOptionalAction, help="If passed, mean the loss over the full effective batch size, rather than the minibatch size")
    argparser.add_argument("--negative_loss_margin", type=float, default=0.05, help="maximum for negative loss scale repulsion")
    argparser.add_argument("--lr", type=float, default=None, help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--lr_end", type=float, default=None, help="Final learning rate, if using scheduler is minimum LR at end of curve")
    argparser.add_argument("--lr_d0", type=float, default=1e-6, help="d0 for adaptive optimizers")
    argparser.add_argument("--lr_decay_steps", type=int, default=0, help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler, (default: constant)", choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None, help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps), non-functional for constant")
    argparser.add_argument("--lr_advance_steps", type=int, default=None, help="Steps to advance the LR during training")
    argparser.add_argument("--lr_num_restarts", type=int, default=1, help="Number of times to (re-)start the LR scheduler, default=1 (no restarts)")
    argparser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs to train for")
    argparser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to train for")
    argparser.add_argument("--auto_decay_steps_multiplier", type=float, default=1.1, help="Multiplier for calculating decay steps from epoch count")
    argparser.add_argument("--no_prepend_last", action="store_true", help="Do not prepend 'last-' to the final checkpoint filename")
    argparser.add_argument("--no_save_ckpt", action="store_true", help="Save only diffusers files, not .safetensors files (save disk space if you do not need LDM-style checkpoints)" )
    argparser.add_argument("--optimizer_config", default="optimizer.json", help="Path to a JSON configuration file for the optimizer.  Default is 'optimizer.json'")
    argparser.add_argument("--unet_freeze_regex", default=None, help='Unet freeze regex(es). Specify multiply matches by separating with `;`. Matches are applied in order. `freeze` or `unfreeze` specifies what the match does. Last match wins. eg: --unet_freeze_regex "freeze .*; unfreeze down_blocks\\.0\\..*attentions.*; freeze .*\\.norm1" -> freeze all, then unfreeze attention modules in down_block 0 except norm1 layers. Use --debug_unet_freeze_regex to apply rules and dump result to console without actually training.')
    argparser.add_argument("--debug_unet_freeze_regex", action="store_true", help="If passed, apply unet freeze regex and dump results to console without training")
    argparser.add_argument("--optimizer_param_grouping", type=str, nargs="+", default=["single"], help="Parameter grouping strategy for optimizer. one of 'single', 'transformer10x', 'zones', 'per-module <json_path>'. Default: 'single'")
    argparser.add_argument("--optimizer_progressive_unlock", action=argparse.BooleanOptionalAction, default=False, help="If passed, progressively unlock parameters")
    argparser.add_argument("--optimizer_progressive_unlock_by_qk_proximity", action=argparse.BooleanOptionalAction, default=False, help="If passed, progressively unlock parameters by proximity to qk attention parameters")

    argparser.add_argument('--plugins', nargs='+', help='Names of plugins to use')
    argparser.add_argument("--project_name", type=str, default="myproj", help="Project name for logs and checkpoints, ex. 'tedbennett', 'superduperV1'")
    argparser.add_argument("--resolution", type=int, nargs='+', default=[512], help="resolution(s) to train", choices=aspects.get_supported_resolutions())
    argparser.add_argument("--resolution_multiplier", type=float, nargs='+', default=[], help="multipliers to apply per-resolution (specify one per resolution, like --resolution_multiply <multiplier> [<multiplier>] ...)")
    argparser.add_argument("--keep_same_sample_at_different_resolutions_together", action=argparse.BooleanOptionalAction, help="if passed, re-order batches to put samples with the same path but different resolutions near to each other")
    argparser.add_argument("--no_normalize_images", action=argparse.BooleanOptionalAction, help="if passed, do not normalize image pixels")
    argparser.add_argument("--resume_ckpt", type=str, required=not ('resume_ckpt' in args), default="sd_v1-5_vae.ckpt", help="The checkpoint to resume from, either a local .ckpt file, a converted Diffusers format folder, or a Huggingface.co repo id such as stabilityai/stable-diffusion-2-1 ")
    argparser.add_argument("--resume_ckpt_variant", type=str, required=False, default=None, help="For Hugging Face repo resume_ckpts, the variant (eg fp16) - required for some models")
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
    argparser.add_argument("--timesteps_multirank_stratified", action=argparse.BooleanOptionalAction, default=False, help="use multirank stratified timesteps (recommended: disable min_snr_gamma")
    argparser.add_argument("--timesteps_multirank_stratified_distribution", type=str, choices=['uniform', 'beta', 'mode', 'boundary-oversampling', 'lognormal'], default='beta', help="multirank stratified timesteps distribution model. for 'beta', uses alpha and beta params; for 'mode', uses mode_scale param")
    argparser.add_argument("--timesteps_multirank_stratified_stratify", action=argparse.BooleanOptionalAction, default=True, help="whether to stratify timestep distribution, or just leave to chance")
    argparser.add_argument("--timesteps_multirank_stratified_alpha", type=float, default=1.5, help="multirank stratified timesteps PPF alpha")
    argparser.add_argument("--timesteps_multirank_stratified_beta", type=float, default=2, help="multirank stratified timesteps PPF beta")
    argparser.add_argument("--timesteps_multirank_stratified_mode_scale", type=float, default=0.5, help="multirank stratified timesteps mode scale")
    argparser.add_argument("--timestep_interval_sampling", action=argparse.BooleanOptionalAction, default=False,
                           help="Sample all images in an optimizer step from the same SNR-homogeneous timestep interval, rotating the interval each step. Mutually exclusive with --timesteps_multirank_stratified.")
    argparser.add_argument("--timestep_interval_n", type=int, default=10,
                           help="Number of SNR-homogeneous intervals for --timestep_interval_sampling (default: 10)")
    argparser.add_argument("--timestep_curriculum_alpha", type=float, default=0, help="if passed, shift timestep range toward fine details as training progresses")
    argparser.add_argument("--timestep_initial_start", type=int, default=800, help="If using timestep_curriculum_alpha, the initial start timestep (default 800); will transition to --timestep_start")
    argparser.add_argument("--timestep_initial_end", type=int, default=1000, help="If using timestep_curriculum_alpha, the initial end timestep (default 1000); will transition to --timestep_end")
    argparser.add_argument("--train_sampler", type=str, default="ddpm",
                           help="noise sampler used for training, (default: ddpm)", choices=["ddpm", "pndm", "ddim", "flow-matching"])
    argparser.add_argument("--flow_match_shift", type=float, default=1.0, help="For flow-matching train sampler, the noise shift parameter (def: 1.0)")
    argparser.add_argument("--flow_match_shift_dynamic", action=argparse.BooleanOptionalAction, default=False, help="If passed, set flow-matching shift dynamically based on resolution (target shift=3 @ 1024x1024)")
    argparser.add_argument("--flow_match_shift_dropout_p", type=float, default=0.3, help="Probability that a given batch will see unshifted timesteps when doing flow-matching shift")

    argparser.add_argument("--keep_tags", type=int, default=0, help="Number of tags to keep when shuffle, used to randomly select subset of tags when shuffling is enabled, def: 0 (shuffle all)")
    argparser.add_argument("--wandb", action="store_true", default=False, help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None, help="Path to a JSON configuration file for the validator.  Default is no validation.")
    argparser.add_argument("--no_initial_validation", action="store_true", help="If passed, don't do validation before the first step")
    argparser.add_argument("--write_schedule", action="store_true", default=False, help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False, help="enable rated image set training, to less often train on lower rated images through the epochs")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50, help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02, help="adds zero frequency noise, for improving contrast (def: 0.0) use 0.0 to 0.15")
    argparser.add_argument("--enable_zero_terminal_snr", action=argparse.BooleanOptionalAction, default=None, help="Use zero terminal SNR noising beta schedule")
    argparser.add_argument("--mix_zero_terminal_snr", action="store_true", default=None, help="Mix zero termianl SNR with regular training")
    argparser.add_argument("--match_zero_terminal_snr", action="store_true", default=None, help="use zero terminal SNR target as regular noise scheduler input")
    argparser.add_argument("--load_settings_every_epoch", action="store_true", default=None, help="Enable reloading of 'train.json' at start of every epoch.")
    argparser.add_argument("--loss_mode_scale", default=0, type=float, help="Mode scale for mode-curve loss scaling. default 0/disabled, recommended 0.5 for flow matching")
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
    argparser.add_argument("--teacher_p", type=float, default=1.0, help="Probability of teacher model being used as target")
    argparser.add_argument("--teacher_lambda", type=float, default=1.0, help="When teacher is used, the scale factor for teacher loss when added to regular loss")
    argparser.add_argument("--teacher_lambda_falloff", action=argparse.BooleanOptionalAction, default=False, help="When enabled, teacher_lambda falls off linearly to 0 below teacher_lambda_falloff_tmax down to teacher_lambda_falloff_tmin")
    argparser.add_argument("--teacher_timestep_max", type=int, default=None, help="Maximum timestep where the teacher model will be used (if set, p ramps from 0 to teacher_p linearly starting from here).")
    argparser.add_argument("--teacher_prediction_type", type=str, default="auto", choices=["auto", "flow_prediction", "v_prediction", "epsilon"],
                           help="Override the teacher scheduler prediction type. 'auto' uses whatever the saved config says. 'flow_prediction' forces a FlowMatch scheduler regardless of saved config.")
    argparser.add_argument("--flow_match_t_clamp_min", type=int, default=None, help="Clamp sampled FM timestep indices to at least this value (0–999). Helps bf16 stability near t=0.")
    argparser.add_argument("--flow_match_t_clamp_max", type=int, default=None, help="Clamp sampled FM timestep indices to at most this value (0–999). Helps bf16 stability near t=999.")

    argparser.add_argument("--local_contrastive_flow_loss_p", type=float, default=0, help="Probability that a given batch will have Local Contrastive Flow loss (Zeng & Yan, 'Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy', Sept 2025) applied. Suggested value=0.2 - LCF forces all timesteps to <200 for the batch in question. Timestep stratification is applied (alpha/beta are truncated to <200). The value 200 can be overriden with --local_contrastive_flow_timestep_threshold .")
    argparser.add_argument("--local_contrastive_flow_timestep_threshold", type=int, default=200, help="Timesteps smaller than this (ie low noise timesteps) will be subject to local contrastive flow (LCF) loss")
    argparser.add_argument("--local_contrastive_flow_anchor_timestep", type=int, default=500, help="Anchor timestep (medium loss) for LCF loss")
    argparser.add_argument("--local_contrastive_flow_temperature", type=int, default=0.07, help="Temperature for LCF loss InfoNCE cross entropy computation (higher=smoother)")
    argparser.add_argument("--local_contrastive_flow_lambda", type=int, default=0.1, help="Lambda scaling factor for Local Contrastive Flow loss")

    argparser.add_argument("--contrastive_flow_matching_loss_p", type=float, default=0, help="Probability that a given batch will have Contrastive Flow Matching loss (Stoica et al., June 2025) applied")
    argparser.add_argument("--contrastive_flow_matching_loss_lambda", type=float, default=0.05, help="Lambda scaling factor for Contrastive Flow Matching loss. Ensure that K * lambda < 1")

    argparser.add_argument("--saturation_penalty_scale", type=float, default=0.0, help="Scale for the saturation penalty loss that discourages all-black or all-white/washed-out outputs. Suggested starting value: 0.02. Set to 0 to disable.")
    argparser.add_argument("--saturation_penalty_t_max", type=float, default=200.0, help="Only apply saturation penalty for timesteps < t_max (low-noise regime). Default 200 out of 1000.")

    argparser.add_argument("--self_flow_p", type=float, default=0.0,
        help="Self-Flow: probability per step that the self-distillation representation loss is applied (0=disabled). "
             "Adds ~2 extra UNet forward passes when active. Suggested starting value: 0.5.")
    argparser.add_argument("--self_flow_gamma", type=float, default=0.8,
        help="Self-Flow: weighting factor γ for the representation loss (Total = L_gen + γ·L_rep). Default: 0.8.")
    argparser.add_argument("--self_flow_mask_ratio", type=float, default=0.25,
        help="Self-Flow: fraction of spatial latent tokens that use the secondary noise level s (default: 0.25).")
    argparser.add_argument("--self_flow_ema_decay", type=float, default=0.9999,
        help="Self-Flow: EMA decay rate for the teacher UNet (default: 0.9999 as per the paper).")
    argparser.add_argument("--self_flow_ema_update_interval", type=int, default=1,
        help="Self-Flow: update the teacher EMA every N optimizer steps (default: 1).")
    argparser.add_argument("--self_flow_mode", type=str, default='shallow', choices=SELF_FLOW_MODES,
        help="Self-Flow: extraction-point arrangement. "
             "'shallow' (default): student=down_blocks[1], teacher=up_blocks[0] (H/4). "
             "'deep': student=down_blocks[2].attentions[-1], teacher=up_blocks[1].attentions[-1] (H/4). "
             "'semantic': student=down_blocks[2].attentions[-1], teacher=up_blocks[0] (H/4). "
             "'detail': student=down_blocks[1].attentions[-1], teacher=up_blocks[1] (H/2).")

    argparser.add_argument("--contrastive_learning_dropout_p", type=float, default=0, help="Probability to drop (non-LCF/non-CFM) contrastive learning")
    argparser.add_argument("--contrastive_loss_batch_ids", type=str, nargs="*", default=[], help="Batch ids for which contrastive learning should be done (default=[]). Use `--contrastive_loss_batch_ids default_batch` to do contrastive learning on all batches if you have not specified batch ids.")
    argparser.add_argument("--contrastive_loss_scale", type=float, default=1, help="Scaling factor for contrastive loss")
    argparser.add_argument("--contrastive_loss_type", type=str, choices=['infonce', 'delta', 'infonce_with_text_similarity', 'infonce_softrepa'], help="Type of contrastive loss")
    argparser.add_argument("--contrastive_loss_softrepa_sigma", type=float, default=1.0, help="Sigma value for SoftREPA infoNCE contrastive loss (only used if type == infonce_softrepa)")
    argparser.add_argument("--contrastive_loss_temperature", type=float, default=0.07, help="Temperature for infonce contrastive loss (lower=more aggressive) (only used if type == infonce or infonce_with_text_similarity")
    argparser.add_argument("--contrastive_loss_hard_negative_weight", type=float, default=2, help="weight factor for more difficult negative pairs when doing infoNCE (default=2.0)")

    argparser.add_argument("--everything_contrastive_learning_p", type=float, default=0, help="probability to run contrastive learning on everything, 0..1")
    argparser.add_argument("--everything_contrastive_learning_curriculum_alpha", type=float, default=0, help="if >0, attenuate everything_contrastive_learning_p to 0 using this alpha as timestep approaches 0")
    argparser.add_argument("--caption_variants", type=str, nargs="*", default=[], help="If passed, use only these caption variants from json captions")
    argparser.add_argument("--all_caption_variants", action=argparse.BooleanOptionalAction, help='if passed, use ALL caption variants every step')
    argparser.add_argument("--expand_caption_variants", action=argparse.BooleanOptionalAction, help='if passed, expand caption variant dicts into individual items, one for each entry in caption_variants')
    argparser.add_argument("--caption_cross_concatenation_p", type=float, default=0, help="Probability of doing caption cross concatenation. When active, two caption variants are encoded separately and concatenated (cf long prompts at inference).")
    argparser.add_argument("--caption_cross_concatenation_empty_half_p", type=float, default=0.2, help="When doing caption cross concatenation, probability of one variant being an empty prompt (default 0.2)")
    argparser.add_argument("--use_compel", action='store_true', help='if passed, use Compel to process prompts with long-prompt support')

    argparser.add_argument("--batch_id_dropout_p", type=float, default=0, help="dropout probability for batch ids, 0..1")
    argparser.add_argument("--cond_dropout_noise_p", type=float, default=0, help="how often to use noise (torch.randn) for the image with conditional dropout - helps prevent overfitting of unconditioned prompt")

    argparser.add_argument("--mask_p", type=float, default=None, help="If passed, look for files called eg image_name.jpg.mask.png in the data folder and use as mask for the loss with given probability (0..1)")
    argparser.add_argument("--invert_masks", action=argparse.BooleanOptionalAction, help="If passed, invert the masks (white<->black)")
    argparser.add_argument("--use_both_mask_sides_contrastive", action='store_true', help="If passed, when using masks, do contrastive learning between mask and inverted mask with negative loss (use --negative_loss_margin to control clamping)")

    argparser.add_argument("--lora", action='store_true', help="If passed, do LoRA training")
    argparser.add_argument("--lora_save_every_n_epochs", type=int, default=1)
    argparser.add_argument("--lora_resume", type=str, default=None, help="resume from this lora (must be a huggingface format folder)")
    argparser.add_argument("--lora_rank", type=int, default=16)
    argparser.add_argument("--lora_alpha", type=int, default=8)

    argparser.add_argument("--test_images", action="store_true", help="check all images by trying to load them")

    argparser.add_argument("--offload_vae", action="store_true", help="If passed, offload VAE to CPU when not in use, saves VRAM but is slower")
    argparser.add_argument("--offload_text_encoder", action="store_true", help="If passed, offload text encoder(s) to CPU when not in use, saves VRAM but is slower")
    argparser.add_argument("--no_save_on_error", action="store_true", help="If passed, do not save model on error/ctrl-c")

    argparser.add_argument("--clip_vision_model_source", default=None, help="If specified, the vision model to use for text encoder contrastive training, eg 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'.")
    argparser.add_argument("--clip_vision_model_processor_source", default=None, help="If specified, the preprocessor to use for text encoder contrastive training, eg 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'. If not specified, will use the same source as --clip_vision_model_source")
    argparser.add_argument("--clip_vision_contrastive_loss_lambda", type=float, default=0.1, help="Lambda scaling factor for contrastive loss between text encoder and CLIP vision model features")

    argparser.add_argument("--debug_no_load_model", action="store_true", help="If passed, do not load model weights (for testing purposes only)")
    argparser.add_argument("--debug_teacher", action="store_true", default=False, help="If passed, log detailed teacher/student latent stats and preview images for the first 10 training steps into <logdir>/debug_teacher/")
    argparser.add_argument("--debug_log_on_nan", action=argparse.BooleanOptionalAction, help="If specified, use set_detect_anomaly to find NaNs in autograd. Slow.")

    # load CLI args to overwrite existing config args
    args = argparser.parse_args(args=argv, namespace=args)

    main(args)
