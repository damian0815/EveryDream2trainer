import gc
import logging
import os
import shutil
from dataclasses import dataclass

import safetensors.torch
import torch
from colorama import Fore, Style
from diffusers import PNDMScheduler, DDIMScheduler, DDPMScheduler, SchedulerMixin, ConfigMixin, UNet2DConditionModel, \
    AutoencoderKL, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft.utils import get_peft_model_state_dict

from data.every_dream import EveryDreamBatch
from flow_match_model import TrainFlowMatchScheduler
from optimizer.optimizers import EveryDreamOptimizer
from plugins.plugins import PluginRunner
from utils.convert_diff_to_ckpt import convert as converter
from utils.huggingface_downloader import try_download_model_from_hf
from utils.unet_utils import check_for_sd1_attn


def get_training_noise_scheduler(train_sampler: str, model_root_folder, trained_betas=None, rescale_betas_zero_snr=False):
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


@dataclass
class TrainingModel:

    @property
    def is_sd1attn(self):
        return check_for_sd1_attn(self.unet.config)

    @property
    def device(self):
        return self.unet.device

    noise_scheduler: SchedulerMixin|ConfigMixin
    text_encoder: CLIPTextModel
    text_encoder_ema: torch.nn.Module|None
    tokenizer: CLIPTokenizer
    unet: UNet2DConditionModel
    unet_ema: UNet2DConditionModel|None
    vae: AutoencoderKL

    yaml: str|None


@dataclass
class EveryDreamTrainingState:
    model: TrainingModel
    optimizer: EveryDreamOptimizer
    train_batch: EveryDreamBatch


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

    if ed_state.model.unet_ema is not None or ed_state.model.text_encoder_ema is not None:
        pipeline_ema = StableDiffusionPipeline(
            vae=ed_state.model.vae,
            text_encoder=ed_state.model.text_encoder_ema,
            tokenizer=ed_state.model.tokenizer,
            unet=ed_state.model.unet_ema,
            scheduler=ed_state.model.noise_scheduler,
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


def load_model(args) -> TrainingModel:
    use_ema_dacay_training = get_use_ema_decay_training(args)

    optimizer_state_path = None
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
            raise ValueError(
                f"No local file/folder for {args.resume_ckpt}, and no matching huggingface.co repo could be downloaded")
        pipe, model_root_folder, is_sd1attn, yaml = downloaded
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

    text_encoder_ema = None
    unet_ema = None
    if use_ema_dacay_training and args.ema_resume_model:
        print(f"Loading EMA model: {args.ema_resume_model}")
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
        def release_memory(model_to_delete, original_device):
            del model_to_delete
            gc.collect()

            if 'cuda' in original_device.type:
                torch.cuda.empty_cache()
        unet_ema_current_device = next(unet_ema.parameters()).device
        ema_device = torch.device(args.ema_device)
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
        noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder,
                                                       trained_betas=trained_betas, rescale_betas_zero_snr=False
                                                       # True
                                                       )
    else:
        noise_scheduler = get_training_noise_scheduler(args.train_sampler, model_root_folder)
    tokenizer = CLIPTokenizer.from_pretrained(model_root_folder, subfolder="tokenizer", use_fast=False)
    # Construct TrainingModel instance after loading model components
    model_being_trained = TrainingModel(
        noise_scheduler=noise_scheduler,
        text_encoder=text_encoder,
        text_encoder_ema=text_encoder_ema,
        tokenizer=tokenizer,
        unet=unet,
        unet_ema=unet_ema,
        vae=vae,
        yaml=yaml
    )
    return model_being_trained


def get_use_ema_decay_training(args):
    use_ema_dacay_training = (args.ema_decay_rate != None) or (args.ema_strength_target != None)
    return use_ema_dacay_training
