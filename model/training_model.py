import contextlib
import gc
import logging
import math
import os
import shutil
from argparse import Namespace
from dataclasses import dataclass, field
import random

import safetensors.torch
import torch
from colorama import Fore, Style
from compel import Compel, ReturnedEmbeddingsType, SplitLongTextMode
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
    AutoModel,
)
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft.utils import get_peft_model_state_dict

from flow_match_model import TrainFlowMatchScheduler
from optimizer.optimizers import EveryDreamOptimizer
from plugins.plugins import PluginRunner
from utils.convert_diff_to_ckpt import convert as converter
from utils.huggingface_downloader import try_download_model_from_hf
from utils.unet_utils import check_for_sd1_attn


def get_training_noise_scheduler(scheduler, train_sampler: str, trained_betas=None, rescale_betas_zero_snr=False):
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
        noise_scheduler = TrainFlowMatchScheduler.from_config(scheduler.config)
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

    last_effective_batch_size: int = 0
    effective_backward_size: int = 0
    current_accumulated_backward_images_count: int = 0
    accumulated_loss_images_count: int = 0
    accumulated_loss: torch.Tensor|None = None
    accumulated_pathnames: list[str] = field(default_factory=list)
    accumulated_captions: list[str] = field(default_factory=list)
    accumulated_timesteps: list[int] = field(default_factory=list)
    desired_effective_batch_size: int|None = None
    interleave_bs1_bsN: bool = False
    interleaved_bs1_count: int|None = None

    cond_dropouts: list[float] = field(default_factory=list)

    remaining_timesteps: torch.Tensor|None = None

    timesteps_ranges: tuple[tuple[int, int], tuple[int, int]] = None

    prev_accumulated_pathnames: list[str] = field(default_factory=list)
    prev_accumulated_captions: list[str] = field(default_factory=list)
    prev_accumulated_timesteps: list[int] = field(default_factory=list)

    def accumulate_loss(self, loss: torch.Tensor, pathnames: list[str], captions: list[str], timesteps: list[int]):

        if loss.isnan().any():
            logging.warning(f"NaN detected after processing {pathnames} @ {timesteps} ({captions} - skipping")
            logging.warning(f" - NaN detected (current accumulated {self.accumulated_pathnames} @ {self.accumulated_timesteps} ({self.accumulated_captions}) )")
            logging.warning(f" - NaN detected (prev was {self.prev_accumulated_pathnames} @ {self.prev_accumulated_timesteps} ({self.prev_accumulated_captions}) )")
            assert False

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


@dataclass
class TrainingModel:

    @property
    def is_sdxl(self):
        return self.text_encoder_2 is not None

    @property
    def is_sd1attn(self):
        return check_for_sd1_attn(self.unet.config)

    @property
    def device(self):
        return self.unet.device

    noise_scheduler: SchedulerMixin|ConfigMixin
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModel|None
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer|None
    unet: UNet2DConditionModel
    vae: AutoencoderKL

    compel: Compel|None
    yaml: str|None

    def load_vae_to_device(self, device):
        self.vae.to(device)

    def load_textenc_to_device(self, device):
        self.text_encoder.to(device)
        if self.text_encoder_2:
            self.text_encoder_2.to(device)


@dataclass
class Conditioning:
    _text_encoder_hidden_states: torch.Tensor
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
    def added_cond_kwargs(self) -> dict:
        return {"text_embeds": self._text_encoder_2_pooled_embeds, "time_ids": self._add_time_ids}

    @staticmethod
    def sd12_conditioning(text_encoder_hidden_states: torch.Tensor):
        return Conditioning(_text_encoder_hidden_states=text_encoder_hidden_states,
                            _text_encoder_2_hidden_states=None,
                            _text_encoder_2_pooled_embeds=None,
                            _add_time_ids=None)

    @staticmethod
    def sdxl_conditioning(text_encoder_hidden_states: torch.Tensor,
                          text_encoder_2_hidden_states: torch.Tensor,
                          text_encoder_2_pooled_embeds: torch.Tensor,
                          add_time_ids: torch.Tensor):

        return Conditioning(_text_encoder_hidden_states=text_encoder_hidden_states,
                            _text_encoder_2_hidden_states=text_encoder_2_hidden_states,
                            _text_encoder_2_pooled_embeds=text_encoder_2_pooled_embeds,
                            _add_time_ids=add_time_ids)


def get_text_conditioning(tokens: torch.Tensor, tokens_2: torch.Tensor, caption_str: list[str], model: TrainingModel, args: Namespace|None) -> tuple[torch.Tensor, torch.Tensor|None, torch.Tensor|None]:
    # todo: move to Conditioning
    # todo: -----
    if model.compel:
        print("Compel is setup but not being used (not implemented)")
    encoder_hidden_states = _encode_caption_tokens(
        tokens,
        model.text_encoder,
        clip_skip=args.clip_skip if args else 0,
        embedding_perturbation=args.embedding_perturbation if args else False,
        compel=model.compel,
        is_sdxl=model.is_sdxl,
        caption_strings=caption_str,
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
            return_pooled = True
        )
    else:
        encoder_2_hidden_states = None
        encoder_2_pooled_embeds = None
    # todo: -----
    # todo: move to conditioning (end)
    return encoder_hidden_states, encoder_2_hidden_states, encoder_2_pooled_embeds


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

        if return_pooled:
            pooler_output = encoder_output[0]
            return encoder_hidden_states, pooler_output
        else:
            return encoder_hidden_states

    # https://arxiv.org/pdf/2405.20494
    perturbation_deviation = embedding_perturbation / math.sqrt(encoder_hidden_states.shape[2])
    perturbation_delta = torch.randn_like(encoder_hidden_states) * (perturbation_deviation)
    encoder_hidden_states = encoder_hidden_states + perturbation_delta
    del cuda_caption
    return encoder_hidden_states


@dataclass
class EveryDreamTrainingState:
    model: TrainingModel
    optimizer: EveryDreamOptimizer
    train_batch: 'EveryDreamBatch'


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
               save_full_precision=False, save_optimizer_flag=False, save_ckpt=True,
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

    if ed_state.model.is_sdxl:
        pipeline = StableDiffusionXLPipeline(
            vae=ed_state.model.vae,
            text_encoder=ed_state.model.text_encoder,
            text_encoder_2=ed_state.model.text_encoder_2,
            tokenizer=ed_state.model.tokenizer,
            tokenizer_2=ed_state.model.tokenizer_2,
            unet=ed_state.model.unet,
            scheduler=ed_state.model.noise_scheduler,
            feature_extractor=None,  # must be none of no safety checker
            add_watermarker=None
        )
    else:
        pipeline = StableDiffusionPipeline(
            vae=ed_state.model.vae,
            text_encoder=ed_state.model.text_encoder,
            tokenizer=ed_state.model.tokenizer,
            unet=ed_state.model.unet,
            scheduler=ed_state.model.noise_scheduler,
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

    civitai_path = save_path + "_civitai_format.safetensors"
    print("saving civitai format LoRA to", civitai_path)
    convert_diffusers_lora_to_civitai(save_path, civitai_path)


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

    print(f"    {Fore.LIGHTCYAN_EX}Found last checkpoint: {last_ckpt}, resuming{Style.RESET_ALL}")

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
        if hasattr(pipe, 'text_encoder_2'):
            # sdxl
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
                                                       trained_betas=trained_betas, rescale_betas_zero_snr=False
                                                       # True
                                                       )
    else:
        noise_scheduler = get_training_noise_scheduler(scheduler, args.train_sampler)

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
    return model_being_trained


def get_use_ema_decay_training(args):
    use_ema_dacay_training = (args.ema_decay_rate != None) or (args.ema_strength_target != None)
    return use_ema_dacay_training
