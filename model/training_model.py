import argparse
import copy
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional

import diffusers
import math
import os
import shutil
from argparse import Namespace
from dataclasses import dataclass, field

import safetensors.torch
import torch
from colorama import Fore, Style
from compel import Compel, ReturnedEmbeddingsType, SplitLongTextMode, Conditioning
from diffusers import (
    PNDMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    SchedulerMixin,
    ConfigMixin,
    UNet2DConditionModel,
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from peft.utils import get_peft_model_state_dict

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from optimizer.optimizers import EveryDreamOptimizer, InfOrNanException
from plugins.plugins import PluginRunner
from utils.convert_diff_to_ckpt import convert as converter
from utils.huggingface_downloader import try_download_model_from_hf
from utils.unet_utils import check_for_sd1_attn


def get_training_noise_scheduler(scheduler, train_sampler: str, trained_betas=None, rescale_betas_zero_snr=False, flow_match_shift=1, flow_match_shift_dynamic=False):
    if train_sampler.lower() == "pndm":
        logging.info(f" * Using PNDM noise scheduler for training: {train_sampler}")
        noise_scheduler = PNDMScheduler.from_config(scheduler.config,
                                                        trained_betas=trained_betas,
                                                        rescale_betas_zero_snr=rescale_betas_zero_snr)
    elif train_sampler.lower() == "ddim":
        logging.info(f" * Using DDIM noise scheduler for training: {train_sampler}")
        noise_scheduler = DDIMScheduler.from_config(scheduler.config,
                                                        trained_betas=trained_betas,
                                                        rescale_betas_zero_snr=rescale_betas_zero_snr)
    elif train_sampler.lower() == "flow-matching":
        logging.info(f" * Using FlowMatching noise scheduler for training: {train_sampler}")
        noise_scheduler = TrainFlowMatchEulerDiscreteScheduler.from_config(scheduler.config,
                                                                           use_dynamic_shifting=flow_match_shift_dynamic,
                                                                           time_shift_type='linear',
                                                                           shift=flow_match_shift)
        noise_scheduler.config.prediction_type = 'flow_prediction'
        assert noise_scheduler.config.prediction_type == 'flow_prediction', "FlowMatching scheduler prediction_type not set correctly"
        assert noise_scheduler.config.use_dynamic_shifting == flow_match_shift_dynamic, "FlowMatching scheduler dynamic shifting not set correctly"
        assert noise_scheduler.config.time_shift_type == 'linear'
        assert noise_scheduler.shift == flow_match_shift
    else:
        logging.info(f" * Using default (DDPM) noise scheduler for training: {train_sampler}")
        noise_scheduler = DDPMScheduler.from_config(scheduler.config,
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


@dataclass
class TrainingVariables:

    global_step: int = None
    batch_resolution: int = None


    max_backward_slice_size: int = None
    default_max_backward_slice_size: dict[int, int] = field(default_factory=dict)
    backward_oom_history: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))

    forward_slice_size: int = None
    default_forward_slice_size: dict[int, int] = field(default_factory=dict)
    forward_oom_history: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))

    last_effective_batch_size: int = 0
    effective_backward_size: int = 0
    backwarded_images_count: int = 0 # images that have been backward()ed but not yet optimizer.stepped
    accumulated_loss_images_count: int = 0 # images that are in the current loss (not yet backward()ed)
    accumulated_loss: torch.Tensor|None = None
    accumulated_pathnames: list[str] = field(default_factory=list)
    accumulated_captions: list[str] = field(default_factory=list)
    accumulated_timesteps: list[int] = field(default_factory=list)
    accumulated_timesteps_shifted: list[int] = field(default_factory=list)
    desired_effective_batch_size: int|None = None
    interleave_bs1_bsN: bool = False
    interleaved_bs1_count: int|None = None
    total_trained_samples_count: int = 0
    optimizer_step: int = 0

    cond_dropouts: list[float] = field(default_factory=list)
    cond_dropout_count = 0
    non_cond_dropout_count = 0

    remaining_stratified_timesteps: torch.Tensor | None = None
    shared_timestep: int|None = None
    shared_noise: torch.Tensor|None = None

    current_timestep_interval: tuple[int, int] | None = None          # per-optimizer-step latch for interval sampling
    timestep_intervals: list[tuple[int, int]] | None = None           # pre-computed SNR-based clusters

    _backward_size_hint_logged: set = field(default_factory=set)      # resolutions for which the "could do backward=N" hint has already been printed

    timesteps_ranges: tuple[tuple[int, int], tuple[int, int]] = None

    prev_accumulated_pathnames: list[str] = field(default_factory=list)
    prev_accumulated_captions: list[str] = field(default_factory=list)
    prev_accumulated_timesteps: list[int] = field(default_factory=list)

    def setup_default_slice_sizes(self, args: argparse.Namespace):
        for index, resolution in enumerate(args.resolution):
            if args.max_backward_slice_size:
                self.default_max_backward_slice_size[resolution] = args.max_backward_slice_size[index]
            else:
                self.default_max_backward_slice_size[resolution] = args.batch_size
            if args.forward_slice_size:
                self.default_forward_slice_size[resolution] = min(args.batch_size, args.forward_slice_size[index])
            else:
                self.default_forward_slice_size[resolution] = args.batch_size


    def accumulate_loss(self, loss: torch.Tensor, pathnames: list[str], captions: list[str], timesteps: list[int]):

        if loss.isnan().any():
            logging.warning(f"NaN detected after processing {pathnames} @ {timesteps} ({captions}) - skipping")
            logging.warning(f" - NaN detected (current accumulated {self.accumulated_pathnames} @ {self.accumulated_timesteps} ({self.accumulated_captions}) )")
            logging.warning(f" - NaN detected (prev was {self.prev_accumulated_pathnames} @ {self.prev_accumulated_timesteps} ({self.prev_accumulated_captions}) )")
            raise InfOrNanException(f"NaN detected in loss after processing {pathnames} @ {timesteps} ({captions}) - skipping")

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

    def reset(self):
        # Clear all loss accumulation
        self.clear_accumulated_loss()

        self.backwarded_images_count = 0
        self.prev_accumulated_pathnames = []
        self.prev_accumulated_captions = []
        self.prev_accumulated_timesteps = []

        # Reset backward tracking
        self.last_effective_batch_size = 0
        self.effective_backward_size = 0

        # Clear conditional dropout tracking
        self.cond_dropouts = []
        self.cond_dropout_count = 0
        self.non_cond_dropout_count = 0

        # Note: global_step is NOT reset - it continues incrementing across cycles

    def filtered_for_log(self) -> 'TrainingVariables':
        filtered = copy.copy(self)
        filtered.cond_dropouts = []
        return filtered

    def register_backward_oom_or_not(self, oomed: bool):
        backward_oom_history_size = 10
        self._register_oom_or_not(oomed, self.backward_oom_history, max_history_size=backward_oom_history_size)
        batch_resolution = self.batch_resolution
        oom_pct = sum(self.backward_oom_history[batch_resolution]) / len(self.backward_oom_history[batch_resolution])
        if oom_pct > 0.75:
            logging.warning(
                f"Backward OOM'd for resolution {batch_resolution} in {oom_pct * 100}% of the last {backward_oom_history_size} batches")
            current_backward_size = self.default_max_backward_slice_size[batch_resolution]
            new_backward_size = current_backward_size - 1
            if new_backward_size > 0:
                logging.warning(f" -> dropping max backward size for {batch_resolution} to {new_backward_size}")
                self.default_max_backward_slice_size[batch_resolution] = new_backward_size
            else:
                logging.error(f" !! max backward size for {batch_resolution} is already 1, cannot drop any further")

    def register_forward_oom_or_not(self, oomed: bool):
        forward_oom_history_size = 20
        self._register_oom_or_not(oomed, self.forward_oom_history, max_history_size=forward_oom_history_size)
        batch_resolution = self.batch_resolution
        oom_pct = sum(self.forward_oom_history[batch_resolution]) / len(self.forward_oom_history[batch_resolution])
        if oom_pct > 0.75:
            logging.warning(
                f"Forward OOM'd for resolution {batch_resolution} in {oom_pct * 100}% of the last {forward_oom_history_size} batches")
            current_forward_size = self.default_forward_slice_size[batch_resolution]
            new_forward_size = current_forward_size - 1
            if new_forward_size > 0:
                logging.warning(f" -> dropping max backward size for {batch_resolution} to {new_forward_size}")
                self.default_forward_slice_size[batch_resolution] = math.floor(new_forward_size)
            else:
                logging.error(f" !! max backward size for {batch_resolution} is already 1, cannot drop any further")


    def _register_oom_or_not(self, oomed, oom_history, max_history_size):
        batch_resolution = self.batch_resolution
        oom_history[batch_resolution].append(oomed)
        if len(oom_history[batch_resolution]) > max_history_size:
            oom_history[batch_resolution].pop(0)


def _module_cpu_copy_with_optional_ema_state(
    live_module: torch.nn.Module,
    ema_path: Optional[str],
) -> torch.nn.Module:
    """
    Returns a detached CPU copy of *live_module* whose parameters are replaced
    by the EMA state loaded from *ema_path* when that file exists.  Falls back
    to the live parameters (copied to CPU) when the file is absent.

    The returned module has ``requires_grad_(False)`` applied so it is safe to
    move straight into an inference pipeline.
    """
    cpu_copy = copy.deepcopy(live_module).cpu()
    if ema_path and os.path.isfile(ema_path):
        state = safetensors.torch.load_file(ema_path, device="cpu")
        missing, _ = cpu_copy.load_state_dict(state, strict=False)
        if missing:
            logging.warning(
                f"EMA state from {ema_path}: {len(missing)} missing keys "
                f"(first 5: {missing[:5]}), using live weights for those"
            )
    else:
        if ema_path:
            logging.warning(f"EMA state file not found at {ema_path}, using live weights")
    cpu_copy.requires_grad_(False)
    return cpu_copy


@dataclass
class TrainingModel:

    @property
    def is_sdxl(self) -> bool:
        if self._is_sdxl_override is not None: # in case we're running without a text encoder
            return self._is_sdxl_override
        return self.text_encoder_2 is not None

    @property
    def is_sd1attn(self) -> bool:
        return check_for_sd1_attn(self.unet.config)

    @property
    def is_flow_matching(self) -> bool:
        return self.noise_scheduler.config.prediction_type in [
            "flow-matching",
            "flow_prediction",
        ]

    @property
    def device(self):
        # assumption: we're always training the unet
        return self.unet.device

    @property
    def dtype(self):
        # assumption: we're always training the unet
        return self.unet.dtype

    noise_scheduler: SchedulerMixin|ConfigMixin|TrainFlowMatchEulerDiscreteScheduler
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModel|None
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer|None
    unet: UNet2DConditionModel
    vae: AutoencoderKL

    compel: Compel|None
    yaml: str|None

    cond_dropout_caption: str = ' '
    cond_dropout_tokens: torch.Tensor|None = None
    cond_dropout_tokens_2: torch.Tensor|None = None

    clip_model = None  # 'CLIP'|None = None
    clip_processor = None  # 'Compose'|None = None

    # Self-Flow representation learning (set after construction in train.py)
    self_flow_teacher_unet = None   # UNet2DConditionModel|None – frozen EMA copy
    self_flow_proj_head = None      # SelfFlowProjectionHead|None – trainable 1×1 conv

    # EMA weights (in-memory mode: cpu or cuda).  None when disk-offload is used or EMA is disabled.
    unet_ema: Optional['UNet2DConditionModel'] = None
    text_encoder_ema: Optional['CLIPTextModel'] = None
    text_encoder_2_ema: Optional['CLIPTextModel'] = None

    # When ema_device='disk': the directory that holds the live *_ema.safetensors working files.
    # None for in-memory EMA modes.
    ema_working_dir: Optional[str] = None

    _is_sdxl_override: Optional[bool] = None
    def set_is_sdxl_override(self, is_sdxl_override: bool):
        """ allow overriding the default "do we have a text enc 2" is_sdxl test """
        self._is_sdxl_override = is_sdxl_override

    @staticmethod
    def from_pipeline(pipe: StableDiffusionPipeline|StableDiffusionXLPipeline, compel=None, yaml=None) -> 'TrainingModel':
        return TrainingModel(
            noise_scheduler=pipe.scheduler,
            text_encoder=pipe.text_encoder,
            text_encoder_2=getattr(pipe, 'text_encoder_2', None),
            tokenizer=pipe.tokenizer,
            tokenizer_2=getattr(pipe, 'tokenizer_2', None),
            unet=pipe.unet,
            vae=pipe.vae,
            compel=compel,
            yaml=None,
        )

    def setup_cond_dropout_tokens(self):
        self.cond_dropout_tokens = torch.tensor(self.tokenizer(self.cond_dropout_caption,
                                                               truncation=True,
                                                               padding="max_length",
                                                               max_length=self.tokenizer.model_max_length,
                                                               ).input_ids)
        if self.tokenizer_2 is not None:
            self.cond_dropout_tokens_2 = torch.tensor(self.tokenizer_2(self.cond_dropout_caption,
                                                                       truncation=True,
                                                                       padding="max_length",
                                                                       max_length=self.tokenizer_2.model_max_length,
                                                                       ).input_ids)

    def set_noise_scheduler_shift(self, shift):
        assert isinstance(self.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler), "Noise scheduler is not TrainFlowMatchEulerDiscreteScheduler"
        self.noise_scheduler.set_timesteps(num_inference_steps=self.noise_scheduler.config.num_train_timesteps,
                                           mu=shift)
        #print("set noise scheduler shift to", shift, " -> timesteps:", self.noise_scheduler.timesteps)

    def load_vae_to_device(self, device):
        self.vae.to(device)

    def load_textenc_to_device(self, device):
        self.text_encoder.to(device)
        if self.text_encoder_2:
            self.text_encoder_2.to(device)

    def _build_inference_pipeline(
        self,
        unet,
        text_encoder,
        text_encoder_2,
        scheduler: SchedulerMixin | None = None,
    ) -> StableDiffusionPipeline | StableDiffusionXLPipeline:
        """
        Core pipeline-construction logic shared by build_inference_pipeline() and
        build_ema_inference_pipeline().  The caller supplies whichever unet /
        text-encoder variants (live or EMA) should be used.
        The pipeline type (SDXL vs SD1/2) is inferred from text_encoder_2.
        """
        if scheduler is None:
            scheduler = self.noise_scheduler
        if text_encoder_2 is not None:
            return StableDiffusionXLPipeline(
                vae=self.vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
        else:
            return StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=text_encoder,
                tokenizer=self.tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,           # save vram
                requires_safety_checker=None,  # avoid nag
                feature_extractor=None,        # must be None if no safety checker
            )

    def build_inference_pipeline(self, scheduler: SchedulerMixin | None = None) -> StableDiffusionPipeline | StableDiffusionXLPipeline:
        return self._build_inference_pipeline(
            unet=self.unet,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            scheduler=scheduler,
        )

    def build_ema_inference_pipeline(self, scheduler: SchedulerMixin | None = None) -> StableDiffusionPipeline | StableDiffusionXLPipeline | None:
        """
        Builds an inference pipeline using EMA weights wherever they are
        available, falling back to the live weights for any component that has
        no EMA counterpart.

        Two EMA storage modes are handled transparently:

        * **In-memory** (``unet_ema`` / ``text_encoder_ema`` attributes set):
          the EMA modules are deepcopied to CPU so that calling ``.to(device)``
          on the returned pipeline never mutates the persistent EMA state.

        * **Disk-offload** (``ema_working_dir`` set): EMA state dicts are
          loaded from the ``*_ema.safetensors`` working files into fresh CPU
          copies of each live module.

        In both cases the returned pipeline has all components on CPU.  The
        caller is responsible for calling ``pipe.to(device)`` before inference
        and ``del pipe`` (+ ``torch.cuda.empty_cache()``) afterwards.

        Returns ``None`` when no EMA weights exist at all.
        """
        has_inmemory_ema = (
            self.unet_ema is not None
            or self.text_encoder_ema is not None
            or self.text_encoder_2_ema is not None
        )
        has_disk_ema = (
            self.ema_working_dir is not None
            and os.path.isfile(os.path.join(self.ema_working_dir, "unet_ema.safetensors"))
        )

        if not has_inmemory_ema and not has_disk_ema:
            return None

        if self.ema_working_dir is not None:
            # ── Disk-offload mode: load from the working safetensors files ──
            unet_for_pipe = _module_cpu_copy_with_optional_ema_state(
                self.unet,
                os.path.join(self.ema_working_dir, "unet_ema.safetensors"),
            )
            te_for_pipe = _module_cpu_copy_with_optional_ema_state(
                self.text_encoder,
                os.path.join(self.ema_working_dir, "text_encoder_ema.safetensors"),
            )
            te2_for_pipe = None
            if self.is_sdxl and self.text_encoder_2 is not None:
                te2_for_pipe = _module_cpu_copy_with_optional_ema_state(
                    self.text_encoder_2,
                    os.path.join(self.ema_working_dir, "text_encoder_2_ema.safetensors"),
                )
        else:
            # ── In-memory mode: deepcopy EMA modules to CPU ──────────────
            # Deepcopy is used so that pipe.to(device) never moves the
            # persistent self.*_ema modules away from their ema_device.
            def _cpu_copy(ema_module, live_module):
                src = ema_module if ema_module is not None else live_module
                c = copy.deepcopy(src).cpu()
                c.requires_grad_(False)
                return c

            unet_for_pipe = _cpu_copy(self.unet_ema, self.unet)
            te_for_pipe   = _cpu_copy(self.text_encoder_ema, self.text_encoder)
            te2_for_pipe  = None
            if self.is_sdxl:
                te2_for_pipe = _cpu_copy(self.text_encoder_2_ema, self.text_encoder_2)

        return self._build_inference_pipeline(
            unet=unet_for_pipe,
            text_encoder=te_for_pipe,
            text_encoder_2=te2_for_pipe,
            scheduler=scheduler,
        )

    @contextmanager
    def ema_inplace_swap(self):
        """
        Context manager for disk-offload EMA inference.

        For each model component that has a corresponding EMA file in
        ``ema_working_dir``:

        1. The current (live) weights are saved to a temporary backup file on
           disk (in training precision, e.g. fp16/bf16).
        2. The EMA weights are loaded from disk and copied **in-place** into
           the live module on its current device and dtype — the model never
           leaves the GPU and no second module copy is held in CPU RAM.
        3. On exit (normal or exceptional) the live weights are restored from
           the backup file and the backup is deleted.

        This lets ``build_inference_pipeline()`` be used directly for EMA
        sampling — no separate EMA pipeline object or CPU offload needed.

        Peak extra CPU RAM is bounded by the float32 EMA state dict of the
        largest component (typically the UNet, ~2× its training-precision
        size), which is freed immediately after each in-place copy.
        """
        if self.ema_working_dir is None:
            raise RuntimeError(
                "ema_inplace_swap requires disk-offload EMA (ema_device='disk')"
            )

        def _unwrap(m):
            return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

        components = [
            (_unwrap(self.unet),         "unet"),
            (_unwrap(self.text_encoder), "text_encoder"),
        ]
        if self.is_sdxl and self.text_encoder_2 is not None:
            components.append((_unwrap(self.text_encoder_2), "text_encoder_2"))

        # Registered in order of successful backup so the finally block only
        # attempts to restore what was actually swapped.
        to_restore: list[tuple[torch.nn.Module, str, str]] = []  # (module, name, backup_path)

        try:
            for module, name in components:
                ema_path = os.path.join(self.ema_working_dir, f"{name}_ema.safetensors")
                if not os.path.isfile(ema_path):
                    logging.debug(f"ema_inplace_swap: no EMA file for {name}, skipping")
                    continue

                backup_path = os.path.join(
                    self.ema_working_dir, f"{name}_live_backup.safetensors"
                )

                # ── Step 1: persist live weights to a temp backup ────────
                logging.info(f"ema_inplace_swap: backing up live {name} → {backup_path}")
                live_sd = {
                    k: v.detach().cpu().contiguous()
                    for k, v in module.state_dict().items()
                }
                safetensors.torch.save_file(live_sd, backup_path)
                del live_sd

                # Register now so the finally block restores even if the
                # EMA load below fails.
                to_restore.append((module, name, backup_path))

                # ── Step 2: load EMA weights and copy in-place ───────────
                target_dtype  = next(module.parameters()).dtype
                target_device = next(module.parameters()).device
                logging.info(
                    f"ema_inplace_swap: applying EMA {name} "
                    f"({target_dtype} on {target_device})"
                )
                ema_sd = safetensors.torch.load_file(ema_path, device="cpu")
                named_params = dict(module.named_parameters())
                with torch.no_grad():
                    for k, ema_v in ema_sd.items():
                        if k in named_params:
                            named_params[k].data.copy_(
                                ema_v.to(dtype=target_dtype, device=target_device)
                            )
                del ema_sd, named_params

            yield  # ← inference runs here

        finally:
            # ── Restore live weights from backup, then clean up ──────────
            for module, name, backup_path in to_restore:
                if not os.path.isfile(backup_path):
                    logging.error(
                        f"ema_inplace_swap: backup file missing at {backup_path} — "
                        f"live weights for this component could not be restored!"
                    )
                    continue
                target_dtype  = next(module.parameters()).dtype
                target_device = next(module.parameters()).device
                logging.info(f"ema_inplace_swap: restoring live {name} from {backup_path}")
                live_sd = safetensors.torch.load_file(backup_path, device="cpu")
                named_params = dict(module.named_parameters())
                with torch.no_grad():
                    for k, p in named_params.items():
                        if k in live_sd:
                            p.data.copy_(
                                live_sd[k].to(dtype=target_dtype, device=target_device)
                            )
                del live_sd, named_params
                os.unlink(backup_path)



@dataclass
class Conditioning:
    _text_encoder_hidden_states: torch.Tensor
    _text_encoder_pooled_embeds: torch.Tensor|None

    _text_encoder_2_hidden_states: torch.Tensor|None

    _text_encoder_2_pooled_embeds: torch.Tensor|None
    _add_time_ids: torch.Tensor|None

    @property
    def prompt_embeds(self) -> torch.Tensor:
        if self._text_encoder_2_hidden_states is not None:
            # sdxl: both text encoders fused together
            return torch.cat([self._text_encoder_hidden_states, self._text_encoder_2_hidden_states], dim=-1)
        else:
            return self._text_encoder_hidden_states

    @property
    def pooled_embeds(self) -> torch.Tensor:
        return self._text_encoder_pooled_embeds

    def get_added_cond_kwargs(self, dtype) -> dict:
        return {
            "text_embeds": self._text_encoder_2_pooled_embeds.to(dtype=dtype),
            "time_ids": self._add_time_ids.to(dtype=dtype)
        }

    @staticmethod
    def sd12_conditioning(text_encoder_hidden_states: torch.Tensor, text_encoder_pooled_embeds: torch.Tensor):
        return Conditioning(_text_encoder_hidden_states=text_encoder_hidden_states,
                            _text_encoder_pooled_embeds=text_encoder_pooled_embeds,
                            _text_encoder_2_hidden_states=None,
                            _text_encoder_2_pooled_embeds=None,
                            _add_time_ids=None)

    @staticmethod
    def sdxl_conditioning(text_encoder_hidden_states: torch.Tensor,
                          text_encoder_pooled_embeds: torch.Tensor,
                          text_encoder_2_hidden_states: torch.Tensor,
                          text_encoder_2_pooled_embeds: torch.Tensor,
                          add_time_ids: torch.Tensor):

        return Conditioning(_text_encoder_hidden_states=text_encoder_hidden_states,
                            _text_encoder_pooled_embeds=text_encoder_pooled_embeds,
                            _text_encoder_2_hidden_states=text_encoder_2_hidden_states,
                            _text_encoder_2_pooled_embeds=text_encoder_2_pooled_embeds,
                            _add_time_ids=add_time_ids)

    def get_masked(self, mask: torch.Tensor) -> 'Conditioning':
        return Conditioning(_text_encoder_hidden_states=self._text_encoder_hidden_states[mask],
                            _text_encoder_pooled_embeds=self._text_encoder_pooled_embeds[mask],
                            _text_encoder_2_hidden_states=self._text_encoder_2_hidden_states[mask] if self._text_encoder_2_hidden_states is not None else None,
                            _text_encoder_2_pooled_embeds=self._text_encoder_2_pooled_embeds[mask] if self._text_encoder_2_pooled_embeds is not None else None,
                            _add_time_ids=self._add_time_ids[mask] if self._add_time_ids is not None else None)

    def slice(self, slice_start: int, slice_end: int) -> 'Conditioning':
        return _make_conditioning_slice(self._text_encoder_hidden_states,
                                        self._text_encoder_pooled_embeds,
                                        self._text_encoder_2_hidden_states,
                                        self._text_encoder_2_pooled_embeds,
                                        add_time_ids=self._add_time_ids,
                                        slice_start=slice_start,
                                        slice_end=slice_end)

    @classmethod
    def cat(cls, conditioning: list[Conditioning]) -> 'Conditioning':
        return Conditioning(_text_encoder_hidden_states=torch.cat([c._text_encoder_hidden_states for c in conditioning], dim=1),
                            _text_encoder_2_hidden_states=torch.cat([c._text_encoder_2_hidden_states for c in conditioning], dim=1) if conditioning[0]._text_encoder_2_hidden_states is not None else None,
                            _text_encoder_pooled_embeds=conditioning[0]._text_encoder_pooled_embeds if conditioning[0]._text_encoder_pooled_embeds is not None else None,
                            _text_encoder_2_pooled_embeds=conditioning[0]._text_encoder_2_pooled_embeds if conditioning[0]._text_encoder_2_pooled_embeds is not None else None,
                            _add_time_ids=conditioning[0]._add_time_ids if conditioning[0]._add_time_ids is not None else None)


def _make_conditioning_slice(
    encoder_hidden_states: torch.Tensor,
    encoder_pooled_embeds: torch.Tensor,
    encoder_2_hidden_states: torch.Tensor|None,
    encoder_2_pooled_embeds: torch.Tensor|None,
    add_time_ids: torch.Tensor|None,
    slice_start, slice_end
) -> Conditioning:
    encoder_hidden_states_slice = encoder_hidden_states[slice_start:slice_end]
    encoder_pooled_embeds_slice = encoder_pooled_embeds[slice_start:slice_end]
    if encoder_2_hidden_states is not None:
        encoder_2_hidden_states_slice = encoder_2_hidden_states[slice_start:slice_end]
        encoder_2_pooled_embeds_slice = encoder_2_pooled_embeds[slice_start:slice_end]
        add_time_ids_slice = add_time_ids[slice_start:slice_end]
        return Conditioning.sdxl_conditioning(
            text_encoder_hidden_states=encoder_hidden_states_slice,
            text_encoder_pooled_embeds=encoder_pooled_embeds_slice,
            text_encoder_2_hidden_states=encoder_2_hidden_states_slice,
            text_encoder_2_pooled_embeds=encoder_2_pooled_embeds_slice,
            add_time_ids=add_time_ids_slice,
        )
    else:
        return Conditioning.sd12_conditioning(
            text_encoder_hidden_states=encoder_hidden_states_slice,
            text_encoder_pooled_embeds=encoder_pooled_embeds_slice,
        )


def get_text_conditioning(tokens: torch.Tensor, tokens_2: torch.Tensor, caption_str: list[str], model: TrainingModel, args: Namespace|None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor|None, torch.Tensor|None]:
    # todo: move to Conditioning
    if model.compel:
        print("Compel is setup but not being used (not implemented)")
    encoder_hidden_states, encoder_pooled_embeds = _encode_caption_tokens(
        tokens,
        model.text_encoder,
        clip_skip=args.clip_skip if args else 0,
        embedding_perturbation=args.embedding_perturbation if args else False,
        compel=model.compel,
        is_sdxl=model.is_sdxl,
        caption_strings=caption_str,
        return_pooled=True
    )
    if model.is_sdxl:
        # note: to support a "style" prompt we'd need to collect/encode a "tokens_3" (style prompt tokens)
        # so we don't support that for now
        encoder_2_hidden_states, encoder_2_pooled_embeds = _encode_caption_tokens(
            tokens_2,
            model.text_encoder_2,
            clip_skip=args.clip_skip if args else 0,
            embedding_perturbation=args.embedding_perturbation if args else False,
            compel=model.compel,
            caption_strings=caption_str,
            is_sdxl=model.is_sdxl,
            return_pooled=True
        )
    else:
        encoder_2_hidden_states = None
        encoder_2_pooled_embeds = None
    # todo: -----
    # todo: move to conditioning (end)
    return encoder_hidden_states, encoder_pooled_embeds, encoder_2_hidden_states, encoder_2_pooled_embeds


def _encode_caption_tokens(tokens, text_encoder: CLIPTextModel, clip_skip: int, embedding_perturbation: bool,
                           compel: Compel=None, caption_strings: list[str]=None, is_sdxl=False, return_pooled=False):
    cuda_caption = tokens.to(text_encoder.device)
    if compel is not None:
        if return_pooled:
            raise ValueError("Compel + SDXL not implemented yet")
        encoder_hidden_states = compel(caption_strings)
    else:
        encoder_output = text_encoder(cuda_caption, output_hidden_states=True)

        # for SDXL we use the penultimate layer (+ clip skip)
        # for SD1 we use the last layer (+ clip skip)
        # for SD2 we use the "penultimate" layer (which diffusers/HF have already dropped for us so it's the last layer)
        layer_backwards_offset = 2 if is_sdxl else 1
        encoder_hidden_states = encoder_output.hidden_states[-(clip_skip + layer_backwards_offset)]
        if not is_sdxl:
            # for SD1 and SD2 we need to normalize the hidden states
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        if is_sdxl:
            pooler_output = encoder_output[0]
        else:
            pooler_output = encoder_output.pooler_output

        if return_pooled:
            return encoder_hidden_states, pooler_output
        else:
            return encoder_hidden_states

    # https://arxiv.org/pdf/2405.20494
    perturbation_deviation_max = embedding_perturbation / math.sqrt(encoder_hidden_states.shape[2])
    perturbation_deviation = torch.randn(encoder_hidden_states.shape[0]) * perturbation_deviation_max
    perturbation_delta = torch.randn_like(encoder_hidden_states) * (perturbation_deviation.unsqueeze(-1).unsqueeze(-1))
    encoder_hidden_states = encoder_hidden_states + perturbation_delta
    del cuda_caption
    return encoder_hidden_states


@dataclass
class EveryDreamTrainingState:
    model: 'TrainingModel'
    optimizer: EveryDreamOptimizer
    train_batch: 'EveryDreamBatch'


def convert_diffusers_lora_to_single_file(diffusers_format, civitai_path):
    diffusers_format = safetensors.torch.load_file(os.path.join(diffusers_format, 'pytorch_lora_weights.safetensors'))

    fixed = {}
    for i, (orig_k, v) in enumerate(diffusers_format.items()):
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
               save_full_precision=False, save_optimizer_flag=False, save_ckpt=True,
               plugin_runner: PluginRunner=None, unet_only_with_hardlinks_source: str=None):
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
        converter(model_path=diffusers_model_path, checkpoint_path=str(sd_ckpt_full), half=half)

        if yaml_name and yaml_name != "v1-inference.yaml":
            yaml_save_path = f"{os.path.join(save_ckpt_dir, os.path.basename(diffusers_model_path))}.yaml"
            logging.info(f" * Saving yaml to {yaml_save_path}")
            shutil.copyfile(yaml_name, yaml_save_path)


    if global_step is None or global_step == 0:
        logging.warning("  No model to save, something likely blew up on startup, not saving")
        return

    if plugin_runner is not None:
        plugin_runner.run_on_model_save(ed_state=ed_state, save_path=save_path)

    noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(ed_state.model.noise_scheduler.config) if isinstance(ed_state.model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler) else ed_state.model.noise_scheduler

    def unwrap_ddp(module):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            return module.module
        else:
            return module

    unet = unwrap_ddp(ed_state.model.unet)
    text_encoder = unwrap_ddp(ed_state.model.text_encoder)
    text_encoder_2 = unwrap_ddp(ed_state.model.text_encoder_2)

    if ed_state.model.is_sdxl:
        pipeline = StableDiffusionXLPipeline(
            vae=ed_state.model.vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=ed_state.model.tokenizer,
            tokenizer_2=ed_state.model.tokenizer_2,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=None,  # must be none of no safety checker
            add_watermarker=None
        )
    else:
        pipeline = StableDiffusionPipeline(
            vae=ed_state.model.vae,
            text_encoder=text_encoder,
            tokenizer=ed_state.model.tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,  # save vram
            requires_safety_checker=None,  # avoid nag
            feature_extractor=None,  # must be none of no safety checker
        )

    diffusers_model_path = save_path
    logging.info(f" * Saving diffusers model to {diffusers_model_path}")
    pipeline.save_pretrained(diffusers_model_path)
    if unet_only_with_hardlinks_source is not None:
        # replace vae, text encoder diffusion pytorch files with hardlinks
        for subfolder in ['vae', 'text_encoder', 'text_encoder_2']:
            for file in ['model.safetensors', 'diffusion_pytorch_model.safetensors']:
                relpath = subfolder + '/' + file
                saved_path = os.path.join(diffusers_model_path, relpath)
                hardlink_source_path = os.path.join(unet_only_with_hardlinks_source, relpath)
                if os.path.exists(saved_path) and os.path.exists(hardlink_source_path):
                    os.unlink(saved_path)
                    os.link(hardlink_source_path, saved_path)

    if save_ckpt:
        sd_ckpt_path = f"{os.path.basename(save_path)}.safetensors"
        save_ckpt_file(diffusers_model_path, sd_ckpt_path)

    if save_optimizer_flag:
        logging.info(f" Saving optimizer state to {save_path}")
        ed_state.optimizer.save(save_path)

    if ed_state.model.self_flow_proj_head is not None:
        proj_head_path = os.path.join(save_path, "self_flow_proj_head.pt")
        logging.info(f" * Saving Self-Flow projection head to {proj_head_path}")
        torch.save(ed_state.model.self_flow_proj_head.state_dict(), proj_head_path)

    if ed_state.model.self_flow_teacher_unet is not None:
        teacher_path = os.path.join(save_path, "self_flow_teacher_unet.safetensors")
        logging.info(f" * Saving Self-Flow teacher UNet to {teacher_path}")
        state_dict = {k: v.cpu().contiguous() for k, v in ed_state.model.self_flow_teacher_unet.state_dict().items()}
        safetensors.torch.save_file(state_dict, teacher_path)

    # ── EMA sidecars ──────────────────────────────────────────────────────────
    # In-memory EMA (cpu / cuda): serialise the live module state_dict.
    _ema_components = [
        ("unet_ema",          ed_state.model.unet_ema),
        ("text_encoder_ema",  ed_state.model.text_encoder_ema),
        ("text_encoder_2_ema", ed_state.model.text_encoder_2_ema),
    ]
    for _ema_name, _ema_module in _ema_components:
        if _ema_module is not None:
            _ema_path = os.path.join(save_path, f"{_ema_name}.safetensors")
            logging.info(f" * Saving EMA sidecar: {_ema_path}")
            _state = {k: v.cpu().contiguous() for k, v in _ema_module.state_dict().items()}
            safetensors.torch.save_file(_state, _ema_path)

    # Disk-offload EMA: copy the working safetensors files into the checkpoint dir.
    if ed_state.model.ema_working_dir is not None:
        for _ema_name in ("unet_ema", "text_encoder_ema", "text_encoder_2_ema"):
            _src = os.path.join(ed_state.model.ema_working_dir, f"{_ema_name}.safetensors")
            if os.path.isfile(_src):
                _dst = os.path.join(save_path, f"{_ema_name}.safetensors")
                shutil.copy2(_src, _dst)
                logging.info(f" * Copied EMA sidecar (disk-offload): {_ema_name}.safetensors → {save_path}")


@torch.no_grad()
def save_model_lora(model: TrainingModel, save_path: str):
    if hasattr(model.unet, "peft_config"):
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(model.unet)
        )
    else:
        unet_lora_state_dict = None
    if hasattr(model.text_encoder, "peft_config"):
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(model.text_encoder)
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

    civitai_path = save_path + ".safetensors"
    print("saving single file format LoRA to", civitai_path)
    convert_diffusers_lora_to_single_file(save_path, civitai_path)


def find_last_checkpoint(logdir, is_ema=False):
    """
    Finds the last checkpoint in the logdir, recursively
    """
    last_ckpt = None
    last_date = None

    for root, dirs, files in os.walk(logdir):
        for file in files:
            if os.path.basename(file) == "model_index.json":

                curr_date = os.path.getmtime(os.path.join(root,file))

                if last_date is None or curr_date > last_date:
                    last_date = curr_date
                    last_ckpt = root

    assert last_ckpt, f"Could not find last checkpoint in logdir: {logdir}"
    assert "errored" not in last_ckpt, f"Found last checkpoint: {last_ckpt}, but it was errored, cancelling"

    logging.info(f"    {Fore.LIGHTCYAN_EX}Found last checkpoint: {last_ckpt}, resuming{Style.RESET_ALL}")

    return last_ckpt


def _check_pipe(pipe):
    if type(pipe) is StableDiffusionXLPipeline:
        if pipe.unet.config.time_cond_proj_dim is not None:
            logging.warning("** Pipeline config specifies time_cond_proj_dim but this will be ignored")


def load_model(args) -> TrainingModel:
    use_ema_dacay_training = get_use_ema_decay_training(args)

    optimizer_state_path = None
    # check for a local file
    hf_cache_path = get_hf_ckpt_cache_path(args.resume_ckpt)
    if os.path.exists(hf_cache_path) or os.path.exists(args.resume_ckpt):
        model_root_folder, is_sd1attn, yaml = convert_to_hf(args.resume_ckpt)
        if os.path.exists(os.path.join(model_root_folder, 'text_encoder_2')):
            pipe = StableDiffusionXLPipeline.from_pretrained(model_root_folder, variant=args.resume_ckpt_variant)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_root_folder, variant=args.resume_ckpt_variant)
        _check_pipe(pipe)
        if args.lora_resume:
            pipe.load_lora_weights(args.lora_resume)
        scheduler = pipe.scheduler
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        unet = pipe.unet

        qk_norm = unet.config.get('qk_norm', "none")
        if qk_norm is not None:
            logging.info(f" * unet has qk_norm={qk_norm}")
        else:
            logging.info(f" * no qk_norm setting found")

        if hasattr(pipe, 'text_encoder_2'):
            # sdxl
            print('sdxl detected')
            text_encoder_2 = pipe.text_encoder_2
            tokenizer_2 = pipe.tokenizer_2
            # check assumptions for _get_add_time_ids
            # this is to avoid having to pass these config values manually to eg dataloader classes
            if unet.config.addition_time_embed_dim != 256:
                raise ValueError("unet addition_time_embed_dim differs from assumed hard-coded value in _get_add_time_ids")
            if text_encoder_2.config.projection_dim != 1280:
                raise ValueError("text_encoder_2 projection_dim differs from assumed hard-coded value in _get_add_time_ids")
            if unet.add_embedding.linear_1.in_features != 2816:
                raise ValueError("unet add_embedding input dim differs from assumed hard-coded value in _get_add_time_ids")
        else:
            text_encoder_2 = None
            tokenizer_2 = None
        vae = pipe.vae
        del pipe
    else:
        if args.lora_resume:
            raise "Can't do lora_resume with downloaded models"
        # try to download from HF using resume_ckpt as a repo id
        downloaded = try_download_model_from_hf(repo_id=args.resume_ckpt, variant=args.resume_ckpt_variant)
        if downloaded is None:
            raise ValueError(
                f"No local file/folder for {args.resume_ckpt}, and no matching huggingface.co repo could be downloaded")
        pipe, model_root_folder, is_sd1attn, yaml = downloaded
        _check_pipe(pipe)
        scheduler = pipe.scheduler
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        if hasattr(pipe, 'text_encoder_2'):
            text_encoder_2 = pipe.text_encoder_2
            tokenizer_2 = pipe.tokenizer_2
        else:
            text_encoder_2 = None
            tokenizer_2 = None
        vae = pipe.vae
        unet = pipe.unet
        del pipe

    if args.enable_zero_terminal_snr:
        if args.train_sampler == "flow-matching":
            raise ValueError("can't use ZTSNR with flow matching")
        # Use zero terminal SNR
        from utils.unet_utils import enforce_zero_terminal_snr
        temp_scheduler = DDIMScheduler.from_pretrained(model_root_folder, subfolder="scheduler")
        trained_betas = enforce_zero_terminal_snr(temp_scheduler.betas).numpy().tolist()
        noise_scheduler = get_training_noise_scheduler(temp_scheduler, args.train_sampler,
                                                       trained_betas=trained_betas,
                                                       rescale_betas_zero_snr=False,
                                                       flow_match_shift=args.flow_match_shift,
                                                       flow_match_shift_dynamic=args.flow_match_shift_dynamic
                                                       )
    else:
        noise_scheduler = get_training_noise_scheduler(scheduler,
                                                       args.train_sampler,
                                                       flow_match_shift=args.flow_match_shift,
                                                       flow_match_shift_dynamic=args.flow_match_shift_dynamic
                                                       )

    compel = None
    if args.use_compel:
        tokenizer_list = tokenizer if tokenizer_2 is None else [tokenizer, tokenizer_2]
        text_encoder_list = text_encoder if text_encoder_2 is None else [text_encoder, text_encoder_2]
        return_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED if text_encoder_2 else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
        compel = Compel(tokenizer=tokenizer_list,
                        text_encoder=text_encoder_list,
                        truncate_long_prompts=False,
                        requires_pooled=True,
                        returned_embeddings_type=return_type,
                        split_long_text_mode=SplitLongTextMode.SENTENCES,
                        )
    # Construct TrainingModel instance after loading model components
    model_being_trained = TrainingModel(
        noise_scheduler=noise_scheduler,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        vae=vae,
        yaml=yaml,
        compel=compel
    )
    if isinstance(noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
        model_being_trained.set_noise_scheduler_shift(args.flow_match_shift)
    return model_being_trained

def load_clip_model(model_id: str, processor_model_id: str=None) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained('apple/DFN5B-CLIP-ViT-H-14')
    processor = CLIPProcessor.from_pretrained(processor_model_id or model_id, use_fast=True)
    return model, processor


def get_use_ema_decay_training(args):
    use_ema_dacay_training = (args.ema_decay_rate is not None) or (args.ema_strength_target != None)
    return use_ema_dacay_training
