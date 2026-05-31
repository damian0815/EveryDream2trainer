import json
import logging
import traceback

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

from core.semaphore_files import check_semaphore_file_and_unlink, INTERRUPT_VALIDATION_SEMAPHORE_FILE
from data.every_dream import build_torch_dataloader, EveryDreamBatch
from data.data_loader import DataLoaderMultiAspect
from data import resolver
from data import aspects
from data.image_train_item import ImageTrainItem
from model.training_model import Conditioning, get_text_conditioning, TrainingModel
from plugins.plugins import PluginRunner
from utils.isolate_rng import isolate_rng

from colorama import Fore, Style


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def monge_inception_distance_torch(x: torch.Tensor, y: torch.Tensor,
                                   rng_seed: int = 42,
                                   n_projections: int = 1000) -> torch.Tensor:
    """
    Compute the MIND (Monge Inception Distance) metric via Sliced Wasserstein Distance.

    Unlike FID, MIND does not fit high-dimensional Gaussians; instead it projects
    features onto random 1-D directions, sorts the projections and computes the exact
    1-D optimal-transport distance.  This makes it >100× faster and ~10× more
    memory-efficient than FID while achieving stable results with as few as 1k–5k
    samples (vs. ~50k for FID).

    Reference: "MIND: Monge Inception Distance" (Google DeepMind, 2024).

    Args:
        x: Generated feature embeddings, shape (num_samples, dim).
        y: Real / ground-truth feature embeddings, shape (num_samples, dim).
        rng_seed: Seed for reproducible random projections.
        n_projections: Number of random projection directions (100–1000 recommended).

    Returns:
        Scalar tensor containing the MIND score (lower is better).
    """
    min_samples = min(x.shape[0], y.shape[0])
    x = x[:min_samples]
    y = y[:min_samples]

    num_samples, d = x.shape
    ALPHA = 3 * d

    generator = torch.Generator(device=x.device).manual_seed(rng_seed)
    u_proj = torch.randn((n_projections, d), generator=generator,
                         dtype=x.dtype, device=x.device)
    u_proj /= torch.linalg.norm(u_proj, dim=-1, keepdim=True)

    x_proj = u_proj @ x.T  # (n_projections, num_samples)
    y_proj = u_proj @ y.T

    dists = torch.mean(
        (torch.topk(x_proj, num_samples, dim=-1).values
         - torch.topk(y_proj, num_samples, dim=-1).values) ** 2,
        dim=1,
    )
    return ALPHA * torch.mean(dists)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute Fréchet distance between two Gaussians (N(mu1,sigma1), N(mu2,sigma2))."""
    import scipy.linalg
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def _dinov2_features(model, images_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract DINOv2 [CLS] features from a batch of [0,1] RGB images [B,3,H,W].
    Uses manual normalisation matching DINOv2's training stats.
    Returns [B, D] float32 CPU tensor.
    """
    _dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _dinov2_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    imgs = F.interpolate(images_01.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    imgs = (imgs - _dinov2_mean) / _dinov2_std
    with torch.no_grad():
        out = model(pixel_values=imgs)
    return out.last_hidden_state[:, 0].float().cpu()


def _dinov3_features(model, processor, images_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract DINOv3 pooled features from a batch of [0,1] RGB images [B,3,H,W].
    Uses AutoImageProcessor for preprocessing (normalization and resize are handled
    internally by the processor, which differs from DINOv2's manual pipeline).
    Returns [B, D] float32 CPU tensor via outputs.pooler_output.
    """
    to_pil = T.ToPILImage()
    pil_images = [to_pil(img.cpu().clamp(0.0, 1.0)) for img in images_01]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.float().cpu()


def _dino_features(model, processor, images_01: torch.Tensor, device: torch.device,
                   version: int = 2) -> torch.Tensor:
    """Dispatch to the correct DINO feature extractor based on `version` (2 or 3)."""
    if version == 3:
        return _dinov3_features(model, processor, images_01, device)
    return _dinov2_features(model, images_01, device)


def _compute_clip_score_batch(pixel_pred: torch.Tensor,
                               captions: list[str],
                               clip_model,
                               clip_tokenizer,
                               device: torch.device) -> float:
    """
    Compute mean CLIP cosine similarity between x̂₀ pixel predictions and their captions.
    pixel_pred: [B,3,H,W] in [-1,1].
    Returns a scalar float.
    """
    _clip_mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=device).view(1, 3, 1, 1)
    _clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    imgs_01 = (pixel_pred.float().to(device) + 1.0) / 2.0
    imgs    = F.interpolate(imgs_01, size=(224, 224), mode='bilinear', align_corners=False)
    imgs    = (imgs - _clip_mean) / _clip_std

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=imgs)

        text_inputs = clip_tokenizer(
            captions, padding=True, truncation=True, max_length=77, return_tensors='pt'
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = clip_model.get_text_features(**text_inputs)

    image_features = F.normalize(image_features.float(), dim=-1)
    text_features  = F.normalize(text_features.float(),  dim=-1)
    scores = (image_features * text_features).sum(dim=-1)   # [B]
    return scores.mean().item()


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------

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
    val_loss_window_size: Optional[int] = 5

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
        self._clip_model_cache = None
        self._clip_tokenizer_cache = None
        self._dinov2_model_cache = None
        self._dino_processor_cache = None  # only used for DINOv3

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,
            'resolution': None,

            'validate_training': True,
            'val_split_mode': 'automatic',
            'auto_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15,

            'use_relative_loss': False,

            'compute_perceptual_metrics': False,  # enables LPIPS + SSIM via one-step VAE reconstruction

            'anomaly_checkpoint': None,
            'anomaly_max_images': 64,
            'anomaly_threshold': 0.5,

            # CLIP score: cosine similarity between one-step x̂₀ and the caption (no generation required)
            'clip_score_enabled': False,
            'clip_score_model': 'openai/clip-vit-base-patch32',

            # FDD: Fréchet DINO Distance between generated and real images (requires generation)
            'fdd_enabled': False,
            'fdd_n_images': 256,
            'fdd_model': 'facebook/dinov2-small',
            # Which DINO version to use for feature extraction: 2 (default) or 3.
            # DINOv3 uses AutoImageProcessor + pooler_output rather than manual
            # normalisation + last_hidden_state CLS token, so it is NOT a drop-in
            # replacement; set fdd_model to a dinov3 checkpoint when using version 3.
            # Example DINOv3 checkpoint: "facebook/dinov3-vits16-pretrain-lvd1689m"
            'fdd_dino_version': 2,

            'extra_manual_datasets': {},
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

    @property
    def clip_score_enabled(self) -> bool:
        return self.config.get('clip_score_enabled', False)

    @property
    def clip_score_model(self) -> str:
        return self.config.get('clip_score_model', 'openai/clip-vit-base-patch32')

    @property
    def fdd_enabled(self) -> bool:
        return self.config.get('fdd_enabled', False)

    @property
    def fdd_n_images(self) -> int:
        return self.config.get('fdd_n_images', 256)

    @property
    def fdd_model(self) -> str:
        return self.config.get('fdd_model', 'facebook/dinov2-small')

    @property
    def fdd_dino_version(self) -> int:
        return int(self.config.get('fdd_dino_version', 2))

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

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

    def _get_clip_model(self, device: torch.device):
        """Lazy-load CLIP model and tokenizer; returns (model, tokenizer) on `device`."""
        if self._clip_model_cache is None:
            from transformers import CLIPModel, CLIPTokenizer
            model_id = self.clip_score_model
            logging.info(f"Loading CLIP model '{model_id}' for CLIP score evaluation...")
            self._clip_model_cache = CLIPModel.from_pretrained(model_id).eval()
            self._clip_tokenizer_cache = CLIPTokenizer.from_pretrained(model_id)
            logging.info("CLIP model loaded.")
        self._clip_model_cache = self._clip_model_cache.to(device)
        return self._clip_model_cache, self._clip_tokenizer_cache

    def _get_dino_model(self, device: torch.device):
        """
        Lazy-load the configured DINO model (v2 or v3) and return ``(model, processor)``.

        For DINOv2 the processor is ``None`` — preprocessing is done manually inside
        ``_dinov2_features``.  For DINOv3 an ``AutoImageProcessor`` is returned and
        used inside ``_dinov3_features``.
        """
        if self._dinov2_model_cache is None:
            from transformers import AutoModel, AutoImageProcessor
            model_id = self.fdd_model
            version  = self.fdd_dino_version
            logging.info(f"Loading DINOv{version} model '{model_id}' for FDD/MIND evaluation...")
            self._dinov2_model_cache = AutoModel.from_pretrained(model_id).eval()
            if version == 3:
                self._dino_processor_cache = AutoImageProcessor.from_pretrained(model_id)
                logging.info("DINOv3 processor loaded.")
            logging.info(f"DINOv{version} model loaded.")
        self._dinov2_model_cache = self._dinov2_model_cache.to(device)
        return self._dinov2_model_cache, self._dino_processor_cache

    # ------------------------------------------------------------------

    def prepare_validation_splits(self, train_items: list[ImageTrainItem], model: TrainingModel) -> list[ImageTrainItem]:
        """
        Build the validation splits as requested by the config passed at init.
        This may steal some items from `train_items`.
        If this happens, the returned `list` contains the remaining items after the required items have been stolen.
        Otherwise, the returned `list` is identical to the passed-in `train_items`.
        """
        train_items = sorted(train_items, key=lambda ti: ti.pathname)

        with isolate_rng():
            random.seed(self.seed)
            auto_dataset, remaining_train_items = self._build_automatic_validation_dataset_if_required(train_items, model=model)
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

    @torch.no_grad()
    def do_validation(self, model: TrainingModel, global_step: int,
                      get_model_prediction_and_target_callable: Callable[
                                         [torch.Tensor, Conditioning], ValidationStepResult],
                      pipe_factory: Optional[Callable[[], any]] = None):

        unet_was_training = model.unet.training
        text_encoder_was_training = model.text_encoder.training
        try:
            model.unet.eval()
            model.text_encoder.eval()
            if model.is_sdxl:
                model.text_encoder_2.eval()

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

                if (self.anomaly_checkpoint is not None or self.fdd_enabled) and pipe_factory is not None:
                    self._calculate_generation_based_metrics(model=model,
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
        finally:
            if unet_was_training:
                model.unet.train()
            if text_encoder_was_training:
                model.text_encoder.train()
                if model.is_sdxl:
                    model.text_encoder_2.train()

    def _calculate_validation_loss(self, model, logging_tag, dataloader, get_model_prediction_and_target: Callable[
        [torch.Tensor, Conditioning], ValidationStepResult],
                                   global_step: int = 0) -> float:

        clip_model = None
        clip_tokenizer = None
        try:
            with torch.no_grad(), isolate_rng():
                random.seed(self.seed)
                torch.manual_seed(self.seed)

                loss_validation_epoch = []
                # accumulators for bias/variance and per-timestep histograms
                all_residuals = []       # list of float: mean signed residual per sample
                all_timesteps = []       # list of float: timestep per sample (normalised to [0,1])
                all_lpips = []           # list of float: per-batch LPIPS (only if compute_perceptual_metrics)
                all_ssim = []            # list of float: per-batch SSIM  (only if compute_perceptual_metrics)
                all_clip_scores = []  # per-batch mean CLIP score (only if clip_score_enabled)

                # VAE decode is shared between LPIPS/SSIM and CLIP score
                need_pixel_decode = self.compute_perceptual_metrics or self.clip_score_enabled

                if self.compute_perceptual_metrics:
                    lpips_fn = self._get_lpips_fn(model.unet.device)

                if self.clip_score_enabled:
                    clip_model, clip_tokenizer = self._get_clip_model(model.unet.device)

                steps_pbar = tqdm(range(len(dataloader)), position=1, leave=False)
                steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Validate ({logging_tag}){Style.RESET_ALL}")

                for step, batch in enumerate(dataloader):
                    keys = list(batch["captions"].keys())
                    for key in keys:

                        caption_str = batch["captions"][key]

                        encoder_hidden_states, encoder_pooled_embeds, encoder_2_hidden_states, encoder_2_pooled_embeds = get_text_conditioning(
                            caption_str, model, args=None
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

                        if need_pixel_decode:
                            # one-step x̂₀: x̂₀ = x_noisy - σ_t · v_pred
                            sigma_t = t_norm.to(noisy_latents.device).to(noisy_latents.dtype).view(-1, 1, 1, 1)
                            x0_pred_latents = noisy_latents - sigma_t * model_pred.to(noisy_latents.dtype)

                            scaling_factor = 0.13025 if model.is_sdxl else 0.18215
                            decoded = model.vae.decode(
                                x0_pred_latents.to(dtype=model.vae.dtype) / scaling_factor,
                                return_dict=False
                            )[0]
                            pixel_pred = torch.clamp(decoded, -1.0, 1.0)

                            pixel_real = batch["image"].to(pixel_pred.device, dtype=pixel_pred.dtype)
                            if pixel_real.shape[-2:] != pixel_pred.shape[-2:]:
                                pixel_real = F.interpolate(pixel_real, size=pixel_pred.shape[-2:],
                                                           mode='bilinear', align_corners=False)

                            if self.compute_perceptual_metrics:
                                lpips_val = lpips_fn(pixel_pred, pixel_real).mean().item()
                                all_lpips.append(lpips_val)
                                ssim_val = _ssim(pixel_pred, pixel_real)
                                all_ssim.append(ssim_val)

                            if self.clip_score_enabled:
                                clip_score = _compute_clip_score_batch(
                                    pixel_pred, caption_str,
                                    clip_model, clip_tokenizer,
                                    model.unet.device
                                )
                                all_clip_scores.append(clip_score)

                            del x0_pred_latents, decoded, pixel_pred, pixel_real

                        del target, model_pred, conditioning, noisy_latents
                        loss_step = loss.detach().item()
                        loss_validation_epoch.append(loss_step)

                    steps_pbar.update(1)

                steps_pbar.close()

        finally:
            # Move CLIP model back to CPU to free VRAM
            if clip_model is not None:
                clip_model.to('cpu')
                torch.cuda.empty_cache()

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

        # --- CLIP score (x̂₀ vs prompt cosine similarity) ---
        if self.clip_score_enabled and len(all_clip_scores) > 0:
            mean_clip_score = sum(all_clip_scores) / len(all_clip_scores)
            self.log_writer.add_scalar(tag=f"val/{logging_tag}_clip_score",
                                       scalar_value=mean_clip_score, global_step=global_step)
            logging.info(f"Validation ({logging_tag}) clip_score={mean_clip_score:.4f}")

        return loss_validation_local

    def _calculate_generation_based_metrics(self, model: TrainingModel, logging_tag: str,
                                             dataloader, pipe_factory: Callable[[], any],
                                             global_step: int):
        """
        Generate images once, then score them with any combination of:
          - Anomaly detection  (if anomaly_checkpoint is set)
          - FDD                (if fdd_enabled)

        Images are generated up to max(anomaly_max_images, fdd_n_images); each metric
        independently caps its own consumption so neither forces extra generation.
        """
        run_anomaly = self.anomaly_checkpoint is not None
        run_fdd = self.fdd_enabled
        if not run_anomaly and not run_fdd:
            return

        device = model.unet.device
        n_images_needed = max(
            self.anomaly_max_images if run_anomaly else 0,
            self.fdd_n_images       if run_fdd     else 0,
        )

        # --- load auxiliary models up front ---
        anomaly_segment_image = None
        anomaly_model = None
        if run_anomaly:
            try:
                from data.anomaly_detector import segment_image as _anomaly_seg
                anomaly_segment_image = _anomaly_seg
            except ImportError:
                traceback.print_exc()
                logging.error("Anomaly checkpoint provided but failed to import anomaly detector. "
                              "Check your config and environment.")
                run_anomaly = False
            else:
                anomaly_model = self._get_anomaly_model(device)
                anomaly_model.to(device)
                logging.info("Anomaly model moved to device for validation pass.")

        dinov2 = None
        dino_processor = None
        if run_fdd:
            dinov2, dino_processor = self._get_dino_model(device)

        if not run_anomaly and not run_fdd:
            return

        n_images_needed = max(
            self.anomaly_max_images if run_anomaly else 0,
            self.fdd_n_images       if run_fdd     else 0,
        )
        logging.info(f"Generation-based metrics ({logging_tag}): generating up to {n_images_needed} images "
                     f"(anomaly={'yes' if run_anomaly else 'no'}, "
                     f"fdd+mind={'yes (DINOv' + str(self.fdd_dino_version) + ')' if run_fdd else 'no'})...")

        # accumulators
        anomaly_pcts: list[float] = []
        real_feats_list: list[torch.Tensor] = []
        gen_feats_list:  list[torch.Tensor] = []

        _to_tensor = T.ToTensor()
        pipe = pipe_factory()
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        images_done = 0

        try:
            with torch.no_grad(), isolate_rng():
                random.seed(self.seed)
                torch.manual_seed(self.seed)

                with tqdm(total=n_images_needed) as pbar:
                    for batch in dataloader:
                        if images_done >= n_images_needed:
                            break

                        if check_semaphore_file_and_unlink(INTERRUPT_VALIDATION_SEMAPHORE_FILE):
                            print("Validation interrupted")
                            break


                        keys = list(batch["captions"].keys())
                        for key in keys:
                            if images_done >= n_images_needed:
                                break

                            caption_list = batch["captions"][key]
                            n_take = min(len(caption_list), n_images_needed - images_done)
                            prompts = caption_list[:n_take]

                            # single generation call — shared by both metrics
                            with torch.amp.autocast(device_type='cuda'):
                                pipe_output = pipe(
                                    prompt=prompts,
                                    num_inference_steps=20,
                                    guidance_scale=7.5,
                                    generator=torch.Generator(device='cpu').manual_seed(self.seed),
                                )
                            pil_images = pipe_output.images  # list[PIL.Image], len == n_take

                            # --- anomaly: consume up to anomaly_max_images total ---
                            if run_anomaly:
                                anomaly_take = min(n_take, self.anomaly_max_images - len(anomaly_pcts))
                                for pil_img in pil_images[:anomaly_take]:
                                    mask_np = anomaly_segment_image(
                                        model=anomaly_model,
                                        image_pil=pil_img.convert('RGB'),
                                        resolution=self._anomaly_resolution,
                                        device=device,
                                        threshold=self.anomaly_threshold,
                                    )
                                    pct = float(mask_np.sum() / 255) / (mask_np.shape[0] * mask_np.shape[1])
                                    anomaly_pcts.append(pct)

                            # --- FDD: consume up to fdd_n_images total ---
                            if run_fdd:
                                fdd_done = sum(f.shape[0] for f in real_feats_list)
                                fdd_take = min(n_take, self.fdd_n_images - fdd_done)
                                if fdd_take > 0:
                                    real_pixels_01 = (batch["image"][:fdd_take].float() + 1.0) / 2.0
                                    real_feats_list.append(_dino_features(dinov2, dino_processor, real_pixels_01, device, self.fdd_dino_version))

                                    gen_tensor = torch.stack(
                                        [_to_tensor(img.convert('RGB')) for img in pil_images[:fdd_take]]
                                    )
                                    gen_feats_list.append(_dino_features(dinov2, dino_processor, gen_tensor, device, self.fdd_dino_version))

                            images_done += n_take
                            pbar.update(n_take)

        finally:
            del pipe
            if anomaly_model is not None:
                anomaly_model.to('cpu')
            if dinov2 is not None:
                dinov2.to('cpu')
            torch.cuda.empty_cache()
            logging.info(f"Generation-based metrics ({logging_tag}): models moved back to CPU.")

        # --- log anomaly results ---
        if run_anomaly and anomaly_pcts:
            pcts_tensor = torch.tensor(anomaly_pcts, dtype=torch.float32)
            mean_pct = pcts_tensor.mean().item()
            self.log_writer.add_scalar(tag=f"val/{logging_tag}_anomaly_pct_mean",
                                       scalar_value=mean_pct, global_step=global_step)
            self.log_writer.add_histogram(tag=f"val/{logging_tag}_anomaly_pct",
                                          values=pcts_tensor.detach().clone(), global_step=global_step)
            logging.info(f"Anomaly ({logging_tag}) mean_anomaly_pct={mean_pct:.4f}  n={len(anomaly_pcts)}")

        # --- log FDD + MIND results ---
        if run_fdd and real_feats_list and gen_feats_list:
            real_feats_tensor = torch.cat(real_feats_list, dim=0)
            gen_feats_tensor  = torch.cat(gen_feats_list,  dim=0)
            n_real = real_feats_tensor.shape[0]
            n_gen  = gen_feats_tensor.shape[0]

            if n_real < 2 or n_gen < 2:
                logging.warning(f"FDD/MIND ({logging_tag}): too few images "
                                f"({n_real} real, {n_gen} generated) — need ≥2. Skipping.")
            else:
                # --- MIND (Monge Inception Distance) ---
                try:
                    mind_score = monge_inception_distance_torch(
                        x=gen_feats_tensor,
                        y=real_feats_tensor,
                        rng_seed=self.seed,
                        n_projections=1000,
                    )
                    self.log_writer.add_scalar(tag=f"val/{logging_tag}_mind",
                                               scalar_value=mind_score.item(), global_step=global_step)
                    logging.info(f"MIND ({logging_tag}) mind={mind_score.item():.4f}  "
                                 f"n_real={n_real}  n_gen={n_gen}")
                except Exception as exc:
                    logging.warning(f"MIND ({logging_tag}): computation failed: {exc}")

                # --- FDD (Fréchet DINOv2 Distance) ---
                real_feats = real_feats_tensor.numpy().astype(np.float64)
                gen_feats  = gen_feats_tensor.numpy().astype(np.float64)
                mu_real,  sigma_real  = real_feats.mean(axis=0), np.cov(real_feats,  rowvar=False)
                mu_gen,   sigma_gen   = gen_feats.mean(axis=0),  np.cov(gen_feats,   rowvar=False)
                try:
                    fdd = _frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
                    self.log_writer.add_scalar(tag=f"val/{logging_tag}_fdd",
                                               scalar_value=fdd, global_step=global_step)
                    logging.info(f"FDD ({logging_tag}) fdd={fdd:.4f}  "
                                 f"n_real={n_real}  n_gen={n_gen}")
                except Exception as exc:
                    logging.warning(f"FDD ({logging_tag}): Fréchet distance computation failed: {exc}")



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
        data_loader = DataLoaderMultiAspect(items, seed=seed, batch_size=batch_size)
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
