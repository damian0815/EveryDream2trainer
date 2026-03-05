import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Generator
from argparse import Namespace

import lpips as lpips_lib
import torch
import numpy as np
from colorama import Fore, Style
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from data.data_loader import DataLoaderMultiAspect
from data import resolver
from data import aspects
from data.image_train_item import ImageTrainItem
from model.training_model import Conditioning, get_text_conditioning, TrainingModel
from plugins.plugins import PluginRunner
from utils.isolate_rng import isolate_rng

from colorama import Fore, Style


class ValidationStepResult:
    """Return type for the get_model_prediction_and_target callable used during validation."""

    def __init__(self,
                 model_pred: torch.Tensor,
                 target: torch.Tensor,
                 timesteps: torch.Tensor,
                 noisy_latents: torch.Tensor):
        self.model_pred = model_pred
        self.target = target
        self.timesteps = timesteps
        self.noisy_latents = noisy_latents


def get_random_split(items: list[ImageTrainItem], split_proportion: float, batch_size: int) \
        -> tuple[list[ImageTrainItem], list[ImageTrainItem]]:
    split_item_count = max(1, math.ceil(split_proportion * len(items)))
    # sort first, then shuffle, to ensure determinate outcome for the current random state
    items_copy = list(sorted(items, key=lambda i: i.pathname))
    random.shuffle(items_copy)
    split_items = list(items_copy[:split_item_count])
    remaining_items = list(items_copy[split_item_count:])
    return split_items, remaining_items

def disable_multiplier_and_flip(items: list[ImageTrainItem]) -> Generator[ImageTrainItem, None, None]:
    for i in items:
        yield ImageTrainItem(image=i.image, caption=i.caption, aspects=i.aspects, pathname=i.pathname, flip_p=0, multiplier=1)


def _ssim(pred: torch.Tensor, real: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM between two [-1,1] image batches [B, C, H, W]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    # use average pooling as a proxy for a Gaussian window — fast and good enough for monitoring
    mu_p = F.avg_pool2d(pred,  kernel_size=window_size, stride=1, padding=pad)
    mu_r = F.avg_pool2d(real,  kernel_size=window_size, stride=1, padding=pad)
    mu_p_sq = mu_p * mu_p
    mu_r_sq = mu_r * mu_r
    mu_pr   = mu_p * mu_r
    sigma_p_sq = F.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=pad) - mu_p_sq
    sigma_r_sq = F.avg_pool2d(real * real, kernel_size=window_size, stride=1, padding=pad) - mu_r_sq
    sigma_pr   = F.avg_pool2d(pred * real, kernel_size=window_size, stride=1, padding=pad) - mu_pr
    ssim_map = ((2 * mu_pr + C1) * (2 * sigma_pr + C2)) / \
               ((mu_p_sq + mu_r_sq + C1) * (sigma_p_sq + sigma_r_sq + C2))
    return ssim_map.mean().item()


@dataclass
class ValidationDataset:
    name: str
    dataloader: torch.utils.data.DataLoader
    loss_history: list[float] = field(default_factory=list)
    val_loss_window_size: Optional[int] = 5  # todo: arg for this?

    def track_loss_trend(self, mean_loss: float):
        if self.val_loss_window_size is None:
            return
        self.loss_history.append(mean_loss)

        if len(self.loss_history) > ((self.val_loss_window_size * 2) + 1):
            dy = np.diff(self.loss_history[-self.val_loss_window_size:])
            if np.average(dy) > 0:
                logging.warning(f"Validation loss for {self.name} shows diverging.  Check your val/{self.name} graph.")


class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 default_batch_size: int,
                 resolution: int,
                 log_writer: SummaryWriter,
                 approx_epoch_length: int=None
    ):
        self.validation_datasets = []
        self.log_writer = log_writer
        self._lpips_fn_cache = None
        self._anomaly_model_cache = None

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,
            'resolution': None,  # use incoming resolution by default or override in validation json

            'validate_training': True,
            'val_split_mode': 'automatic',
            'auto_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15,

            'use_relative_loss': False,

            'compute_perceptual_metrics': False,  # enables LPIPS + SSIM via one-step VAE reconstruction

            'anomaly_checkpoint': None,   # path to anomaly detector .pth — enables anomaly % validation when set
            'anomaly_max_images': 64,     # cap number of images run through full inference for anomaly scoring
            'anomaly_threshold': 0.5,     # binarisation threshold passed to the anomaly detector

            'extra_manual_datasets': {
                # name: path pairs
                # eg "santa suit": "/path/to/captioned_santa_suit_images", will be logged to tensorboard as "val/santa suit"
            }
        }
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                self.config.update(json.load(f))

        self.resolution = self.config.get('resolution', None) or resolution

        if 'val_data_root' in self.config:
            logging.warning(f"   * {Fore.YELLOW}using old name 'val_data_root' for 'manual_data_root' - please "
                  f"update your validation config json{Style.RESET_ALL}")
            self.config.update({'manual_data_root': self.config['val_data_root']})

        if self.config.get('val_split_mode') == 'manual':
            manual_data_root = self.config.get('manual_data_root')
            if manual_data_root is not None:
                self.config['extra_manual_datasets'].update({'val': self.config['manual_data_root']})
            else:
                if len(self.config['extra_manual_datasets']) == 0:
                    raise ValueError("Error in validation config .json: 'manual' validation requested but no "
                                     "'manual_data_root' or 'extra_manual_datasets'")

        if 'val_split_proportion' in self.config:
            logging.warning(f"   * {Fore.YELLOW}using old name 'val_split_proportion' for 'auto_split_proportion' - please "
                  f"update your validation config json{Style.RESET_ALL}")
            self.config.update({'auto_split_proportion': self.config['val_split_proportion']})

        if self.every_n_epochs < 0:
            if approx_epoch_length is None:
                raise ValueError("missing approx_epoch_len")
            every_n_steps = 2000 if self.every_n_epochs == -1 else -self.every_n_epochs
            every_n_epochs = every_n_steps / approx_epoch_length
            if every_n_epochs > 1:
                every_n_epochs = round(every_n_epochs)
            print(f'validating every {every_n_epochs} ({every_n_steps} steps at epoch length {approx_epoch_length})')
            self.config['every_n_epochs'] = every_n_epochs

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def every_n_epochs(self):
        return self.config['every_n_epochs']

    @property
    def seed(self):
        return self.config['seed']

    @property
    def use_relative_loss(self):
        return self.config['use_relative_loss']

    @property
    def compute_perceptual_metrics(self):
        return self.config.get('compute_perceptual_metrics', False)

    @property
    def anomaly_checkpoint(self) -> Optional[str]:
        return self.config.get('anomaly_checkpoint', None)

    @property
    def anomaly_max_images(self) -> int:
        return self.config.get('anomaly_max_images', 64)

    @property
    def anomaly_threshold(self) -> float:
        return self.config.get('anomaly_threshold', 0.5)

    def _get_anomaly_model(self, device: torch.device):
        """Lazy-load the anomaly SegmentationModel on first use."""
        if self._anomaly_model_cache is None:
            from data.anomaly_detector import SegmentationModel
            ckpt = torch.load(self.anomaly_checkpoint, map_location=device)
            self._anomaly_resolution = ckpt['resolution']
            self._anomaly_model_cache = SegmentationModel(device=str(device))
            self._anomaly_model_cache.decoder.load_state_dict(ckpt['model_state_dict'])
            self._anomaly_model_cache.eval()
            logging.info(f"Anomaly detector loaded from {self.anomaly_checkpoint} (resolution={self._anomaly_resolution})")
        return self._anomaly_model_cache

    def _get_lpips_fn(self, device: torch.device):
        """Lazy-initialise the LPIPS evaluator on first use (avoids loading VGG when not needed)."""
        if not hasattr(self, '_lpips_fn_cache') or self._lpips_fn_cache is None:
            self._lpips_fn_cache = lpips_lib.LPIPS(net='vgg').to(device)
            self._lpips_fn_cache.eval()
        return self._lpips_fn_cache

    def prepare_validation_splits(self, train_items: list[ImageTrainItem], model: TrainingModel) -> list[ImageTrainItem]:
        """
        Build the validation splits as requested by the config passed at init.
        This may steal some items from `train_items`.
        If this happens, the returned `list` contains the remaining items after the required items have been stolen.
        Otherwise, the returned `list` is identical to the passed-in `train_items`.
        """
        # sort so we have a stable base point between runs
        train_items = sorted(train_items, key=lambda ti: ti.pathname)

        with isolate_rng():
            random.seed(self.seed)
            auto_dataset, remaining_train_items = self._build_automatic_validation_dataset_if_required(train_items, model=model)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            train_overlapping_dataset = self._build_train_stabilizer_dataloader_if_required(
                remaining_train_items, model)

            if auto_dataset is not None:
                self.validation_datasets.append(auto_dataset)
            if train_overlapping_dataset is not None:
                self.validation_datasets.append(train_overlapping_dataset)
            manual_splits = self._build_manual_validation_datasets(model)
            self.validation_datasets.extend(manual_splits)

            return remaining_train_items

    def get_validation_step_indices(self, epoch, epoch_length_steps: int) -> list[int]:
        if self.every_n_epochs >= 1:
            if ((epoch+1) % self.every_n_epochs) == 0:
                # last step only
                return [epoch_length_steps-1]
            else:
                return []
        else:
            # subdivide the epoch evenly, by rounding self.every_n_epochs to the nearest clean division of steps
            num_divisions = max(1, min(epoch_length_steps, round(1/self.every_n_epochs)))
            # validation happens after training:
            # if an epoch has eg 100 steps and num_divisions is 2, then validation should occur after steps 49 and 99
            validate_every_n_steps = epoch_length_steps / num_divisions
            return [math.ceil((i+1)*validate_every_n_steps) - 1 for i in range(num_divisions)]

    def do_validation(self, model: TrainingModel, global_step: int,
                      get_model_prediction_and_target_callable: Callable[
                                         [torch.Tensor, Conditioning], ValidationStepResult],
                      pipe_factory: Optional[Callable[[], any]] = None):
        mean_loss_accumulator = 0
        for i, dataset in enumerate(self.validation_datasets):
            mean_loss = self._calculate_validation_loss(model, logging_tag=dataset.name,
                                                        dataloader=dataset.dataloader,
                                                        get_model_prediction_and_target=get_model_prediction_and_target_callable,
                                                        global_step=global_step)
            mean_loss_accumulator += mean_loss
            self.log_writer.add_scalar(tag=f"val/{dataset.name}",
                                       scalar_value=mean_loss,
                                       global_step=global_step)
            dataset.track_loss_trend(mean_loss)

            if self.anomaly_checkpoint is not None and pipe_factory is not None:
                self._calculate_anomaly_metrics(model=model,
                                                logging_tag=dataset.name,
                                                dataloader=dataset.dataloader,
                                                pipe_factory=pipe_factory,
                                                global_step=global_step)

        # log combine loss to val/_all_val_combined
        if len(self.validation_datasets) > 1:
            total_mean_loss = mean_loss_accumulator / len(self.validation_datasets)
            self.log_writer.add_scalar(tag=f"val/_all_val_combined",
                                       scalar_value=total_mean_loss,
                                       global_step=global_step)

    def _calculate_validation_loss(self, model, logging_tag, dataloader, get_model_prediction_and_target: Callable[
        [torch.Tensor, Conditioning], ValidationStepResult],
                                   global_step: int = 0) -> float:
        with torch.no_grad(), isolate_rng():
            # ok to override seed here because we are in a `with isolate_rng():` block
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            loss_validation_epoch = []
            # accumulators for bias/variance and per-timestep histograms
            all_residuals = []       # list of float: mean signed residual per sample
            all_timesteps = []       # list of float: timestep per sample (normalised to [0,1])
            all_lpips = []           # list of float: per-batch LPIPS (only if compute_perceptual_metrics)
            all_ssim = []            # list of float: per-batch SSIM  (only if compute_perceptual_metrics)

            if self.compute_perceptual_metrics:
                lpips_fn = self._get_lpips_fn(model.unet.device)

            steps_pbar = tqdm(range(len(dataloader)), position=1, leave=False)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Validate ({logging_tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                keys = list(batch["captions"].keys())
                for key in keys:

                    caption_str = batch["captions"][key]
                    tokens = torch.stack(batch["tokens"][key])
                    if model.is_sdxl:
                        tokens_2 = torch.stack(batch["tokens_2"][key])
                    else:
                        tokens_2 = None

                    encoder_hidden_states, encoder_pooled_embeds, encoder_2_hidden_states, encoder_2_pooled_embeds = get_text_conditioning(
                        tokens, tokens_2, caption_str, model, args=None
                    )
                    if model.is_sdxl:
                        add_time_ids = batch["add_time_ids"].to(encoder_hidden_states.device)
                        conditioning = Conditioning.sdxl_conditioning(
                            text_encoder_hidden_states=encoder_hidden_states,
                            text_encoder_pooled_embeds=encoder_pooled_embeds,
                            text_encoder_2_hidden_states=encoder_2_hidden_states,
                            text_encoder_2_pooled_embeds=encoder_2_pooled_embeds,
                            add_time_ids=add_time_ids
                        )
                    else:
                        conditioning = Conditioning.sd12_conditioning(
                            text_encoder_hidden_states=encoder_hidden_states,
                            text_encoder_pooled_embeds=encoder_pooled_embeds
                        )

                    step_result = get_model_prediction_and_target(batch["image"], conditioning)
                    model_pred = step_result.model_pred
                    target = step_result.target
                    timesteps = step_result.timesteps
                    noisy_latents = step_result.noisy_latents

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # per-sample signed residuals (mean over C,H,W) → shape [B]
                    residuals = (model_pred.float() - target.float()).mean(dim=(1, 2, 3))
                    all_residuals.extend(residuals.detach().clone().cpu().tolist())

                    # normalise timesteps to [0,1]; they may be raw scheduler integers (0–1000) or floats
                    t_norm = timesteps.float().cpu() / model.noise_scheduler.config.num_train_timesteps
                    all_timesteps.extend(t_norm.detach().clone().cpu().tolist())

                    if self.compute_perceptual_metrics:
                        # one-step x0 reconstruction: x0_pred = x_noisy - sigma_t * v_pred
                        # sigma_t = t / num_train_timesteps (linear flow matching schedule)
                        sigma_t = t_norm.to(noisy_latents.device).to(noisy_latents.dtype).view(-1, 1, 1, 1)
                        x0_pred_latents = noisy_latents - sigma_t * model_pred.to(noisy_latents.dtype)

                        # decode to pixel space [-1, 1]
                        scaling_factor = 0.13025 if model.is_sdxl else 0.18215
                        decoded = model.vae.decode(x0_pred_latents.to(dtype=model.vae.dtype) / scaling_factor, return_dict=False)[0]
                        pixel_pred = torch.clamp(decoded, -1.0, 1.0)  # already in [-1,1] from VAE

                        # real image from batch is normalised to [-1, 1] by EveryDreamBatch
                        pixel_real = batch["image"].to(pixel_pred.device, dtype=pixel_pred.dtype)

                        # spatially align if sizes differ (can happen with multi-aspect batches)
                        if pixel_real.shape[-2:] != pixel_pred.shape[-2:]:
                            pixel_real = F.interpolate(pixel_real, size=pixel_pred.shape[-2:], mode='bilinear', align_corners=False)

                        lpips_val = lpips_fn(pixel_pred, pixel_real).mean().item()
                        all_lpips.append(lpips_val)

                        ssim_val = _ssim(pixel_pred, pixel_real)
                        all_ssim.append(ssim_val)

                        del x0_pred_latents, decoded, pixel_pred, pixel_real

                    del target, model_pred, conditioning, noisy_latents
                    loss_step = loss.detach().item()
                    loss_validation_epoch.append(loss_step)

                steps_pbar.update(1)

            steps_pbar.close()

        loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)

        # --- scalar bias & variance ---
        residuals_tensor = torch.tensor(all_residuals, dtype=torch.float32)
        bias = residuals_tensor.mean().item()
        variance = residuals_tensor.var().item()
        self.log_writer.add_scalar(tag=f"val/{logging_tag}_bias", scalar_value=bias, global_step=global_step)
        self.log_writer.add_scalar(tag=f"val/{logging_tag}_variance", scalar_value=variance, global_step=global_step)
        logging.info(f"Validation ({logging_tag}) bias={bias:.4f}  variance={variance:.4f}")

        # --- mean residual by timestep bucket ---
        # Produces a 1D tensor [n_buckets] of mean residuals, x=bucket index ~ timestep, y=mean residual
        n_buckets = 20
        timesteps_tensor = torch.tensor(all_timesteps, dtype=torch.float32)
        bucket_ids = (timesteps_tensor * n_buckets).long().clamp(0, n_buckets - 1)
        mean_residual_by_t = torch.zeros(n_buckets, dtype=torch.float32)
        for b in range(n_buckets):
            mask = bucket_ids == b
            if mask.sum() > 0:
                mean_residual_by_t[b] = residuals_tensor[mask].mean()
        self.log_writer.add_histogram(tag=f"val/{logging_tag}_residuals_by_t",
                                      values=mean_residual_by_t,
                                      global_step=global_step)

        # --- perceptual metrics (LPIPS + SSIM) ---
        if self.compute_perceptual_metrics and len(all_lpips) > 0:
            mean_lpips = sum(all_lpips) / len(all_lpips)
            mean_ssim = sum(all_ssim) / len(all_ssim)
            self.log_writer.add_scalar(tag=f"val/{logging_tag}_lpips", scalar_value=mean_lpips, global_step=global_step)
            self.log_writer.add_scalar(tag=f"val/{logging_tag}_ssim", scalar_value=mean_ssim, global_step=global_step)
            logging.info(f"Validation ({logging_tag}) lpips={mean_lpips:.4f}  ssim={mean_ssim:.4f}")

        return loss_validation_local

    def _calculate_anomaly_metrics(self, model: TrainingModel, logging_tag: str,
                                   dataloader, pipe_factory: Callable[[], any],
                                   global_step: int):
        """
        Run full denoising inference on a capped subset of the validation set, then score each
        generated image with the anomaly detector. Logs:
          val/{logging_tag}_anomaly_pct_mean  — mean anomaly pixel % across images
          val/{logging_tag}_anomaly_pct        — histogram of per-image anomaly %
        """
        from data.anomaly_detector import segment_image as anomaly_segment_image

        anomaly_model = self._get_anomaly_model(model.unet.device)
        anomaly_model.to(model.unet.device)
        logging.info("Anomaly model moved to device for validation pass.")

        # ImageNet normalisation expected by DINOv2 backbone
        _to_tensor = T.ToTensor()
        _normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # build inference pipe — caller provides the factory so we get the right scheduler config
        pipe = pipe_factory()
        pipe.to(model.unet.device)

        anomaly_pcts = []
        images_processed = 0
        max_images = self.anomaly_max_images

        logging.info(f"Anomaly validation ({logging_tag}): running inference on up to {max_images} images...")

        try:
            with torch.no_grad(), isolate_rng():
                random.seed(self.seed)
                torch.manual_seed(self.seed)

                for batch in dataloader:
                    if images_processed >= max_images:
                        break

                    keys = list(batch["captions"].keys())
                    for key in keys:
                        if images_processed >= max_images:
                            break

                        caption_list = batch["captions"][key]
                        # cap within-batch to not exceed max_images
                        n_take = min(len(caption_list), max_images - images_processed)
                        prompts = caption_list[:n_take]

                        # run denoising — pipe returns PIL images
                        with torch.autocast(device_type='cuda'):
                            pipe_output = pipe(
                                prompt=prompts,
                                num_inference_steps=20,
                                guidance_scale=7.5,
                                generator=torch.Generator(device='cpu').manual_seed(self.seed),
                            )
                        pil_images = pipe_output.images  # list of PIL Images

                        for pil_img in pil_images:
                            mask_np = anomaly_segment_image(
                                model=anomaly_model,
                                image_pil=pil_img.convert('RGB'),
                                resolution=self._anomaly_resolution,
                                device=model.unet.device,
                                threshold=self.anomaly_threshold,
                            )
                            # anomaly % = fraction of pixels flagged
                            pct = float(mask_np.sum() / 255) / (mask_np.shape[0] * mask_np.shape[1])
                            anomaly_pcts.append(pct)

                        images_processed += n_take

        finally:
            del pipe
            anomaly_model.to('cpu')
            torch.cuda.empty_cache()
            logging.info("Anomaly model moved back to CPU.")

        if not anomaly_pcts:
            return

        pcts_tensor = torch.tensor(anomaly_pcts, dtype=torch.float32)
        mean_pct = pcts_tensor.mean().item()
        self.log_writer.add_scalar(tag=f"val/{logging_tag}_anomaly_pct_mean",
                                   scalar_value=mean_pct, global_step=global_step)
        self.log_writer.add_histogram(tag=f"val/{logging_tag}_anomaly_pct",
                                      values=pcts_tensor, global_step=global_step)
        logging.info(f"Anomaly validation ({logging_tag}) mean_anomaly_pct={mean_pct:.4f}  n={len(anomaly_pcts)}")

    def _build_automatic_validation_dataset_if_required(self, image_train_items: list[ImageTrainItem], model: TrainingModel) \
            -> tuple[Optional[ValidationDataset], list[ImageTrainItem]]:
        val_split_mode = self.config['val_split_mode'] if self.config['validate_training'] else None
        if val_split_mode is None or val_split_mode == 'none' or val_split_mode == 'manual':
            # manual is handled by _build_manual_validation_datasets
            return None, image_train_items
        elif val_split_mode == 'automatic':
            auto_split_proportion = self.config['auto_split_proportion']
            val_items, remaining_train_items = get_random_split(image_train_items, auto_split_proportion, batch_size=self.batch_size)
            val_items = list(disable_multiplier_and_flip(val_items))
            logging.info(f" * Removed {len(val_items)} images from the training set to use for validation")
            val_ed_batch = self._build_ed_batch(val_items, model=model, name='val')
            val_dataloader = build_torch_dataloader(val_ed_batch, batch_size=self.batch_size)
            return ValidationDataset(name='val', dataloader=val_dataloader), remaining_train_items
        else:
            raise ValueError(f"Unrecognized validation split mode '{val_split_mode}'")

    def _build_manual_validation_datasets(self, model: TrainingModel) -> list[ValidationDataset]:
        datasets = []
        for name, root in self.config.get('extra_manual_datasets', {}).items():
            items = self._load_manual_val_split(root)
            logging.info(f" * Loaded {len(items)} validation images for validation set '{name}' from {root}")
            ed_batch = self._build_ed_batch(items, model=model, name=name)
            dataloader = build_torch_dataloader(ed_batch, batch_size=self.batch_size)
            datasets.append(ValidationDataset(name=name, dataloader=dataloader))
        return datasets

    def _build_train_stabilizer_dataloader_if_required(self, image_train_items: list[ImageTrainItem], model) \
            -> Optional[ValidationDataset]:
        stabilize_training_loss = self.config['stabilize_training_loss']
        if not stabilize_training_loss:
            return None

        stabilize_split_proportion = self.config['stabilize_split_proportion']
        stabilize_items, _ = get_random_split(image_train_items, stabilize_split_proportion, batch_size=self.batch_size)
        stabilize_items = list(disable_multiplier_and_flip(stabilize_items))
        stabilize_ed_batch = self._build_ed_batch(stabilize_items, model=model, name='stabilize-train')
        stabilize_dataloader = build_torch_dataloader(stabilize_ed_batch, batch_size=self.batch_size)
        return ValidationDataset(name='stabilize-train', dataloader=stabilize_dataloader, val_loss_window_size=None)

    def _load_manual_val_split(self, val_data_root: str):
        args = Namespace(
            aspects=aspects.get_aspect_buckets(self.resolution),
            flip_p=0.0,
            seed=self.seed,
        )
        val_items = resolver.resolve_root(val_data_root, args, self.resolution, args.aspects)
        for i in val_items:
            if i.error is not None:
                logging.warning(f" * Skipping invalid validation image {i.pathname}: {repr(i.error)}")
        val_items = [i for i in val_items if i.error is None]
        val_items.sort(key=lambda i: i.pathname)
        random.shuffle(val_items)
        return val_items

    def _build_ed_batch(self, items: list[ImageTrainItem], model: TrainingModel, name='val'):
        batch_size = self.batch_size
        seed = self.seed
        data_loader = DataLoaderMultiAspect(
            items,
            batch_size=batch_size,
            seed=seed,
        )
        empty_plugin_runner = PluginRunner()

        ed_batch = EveryDreamBatch(
            data_loader=data_loader,
            debug_level=1,
            conditional_dropout=0,
            tokenizer=model.tokenizer,
            tokenizer_2=model.tokenizer_2,
            seed=seed,
            name=name,
            crop_jitter=0,
            plugin_runner=empty_plugin_runner,
            normalize_image=True,
        )
        return ed_batch
