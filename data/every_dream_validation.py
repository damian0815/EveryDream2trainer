import json
import logging
import math
import random
import traceback
from typing import Callable, Any, Optional, Generator
from argparse import Namespace

import torch
from colorama import Fore, Style
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from data.data_loader import DataLoaderMultiAspect
from data import resolver
from data import aspects
from data.image_train_item import ImageTrainItem
from utils.isolate_rng import isolate_rng


def get_random_split(items: list[ImageTrainItem], split_proportion: float, batch_size: int) \
        -> tuple[list[ImageTrainItem], list[ImageTrainItem]]:
    split_item_count = math.ceil(split_proportion * len(items) // batch_size) * batch_size
    # sort first, then shuffle, to ensure determinate outcome for the current random state
    items_copy = list(sorted(items, key=lambda i: i.pathname))
    random.shuffle(items_copy)
    split_items = list(items_copy[:split_item_count])
    remaining_items = list(items_copy[split_item_count:])
    return split_items, remaining_items

def disable_multiplier_and_flip(items: list[ImageTrainItem]) -> Generator[ImageTrainItem, None, None]:
    for i in items:
        yield ImageTrainItem(image=i.image, caption=i.caption, aspects=i.aspects, pathname=i.pathname, flip_p=0, multiplier=1)

class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 default_batch_size: int,
                 resolution: int,
                 log_writer: SummaryWriter,
                 log_folder: str):
        self.val_dataloader = None
        self.train_overlapping_dataloader = None

        self.log_writer = log_writer
        self.log_folder = log_folder
        self.resolution = resolution

        self.config = {
            'batch_size': default_batch_size,
            'every_n_epochs': 1,
            'seed': 555,

            'validate_training': True,
            'val_split_mode': 'automatic',
            'val_split_proportion': 0.15,

            'stabilize_training_loss': False,
            'stabilize_split_proportion': 0.15,

            'find_outliers': False,
            'find_outliers_every_n_epochs': 5,
            'find_outliers_split_proportion': 1,
            'find_outliers_pinned_image_count': 7
        }

        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                self.config.update(json.load(f))

    @property
    def batch_size(self):
        return self.config['batch_size']

    @property
    def every_n_epochs(self):
        return self.config['every_n_epochs']

    @property
    def seed(self):
        return self.config['seed']

    def prepare_validation_splits(self, train_items: list[ImageTrainItem], tokenizer: Any) -> list[ImageTrainItem]:
        """
        Build the validation splits as requested by the config passed at init.
        This may steal some items from `train_items`.
        If this happens, the returned `list` contains the remaining items after the required items have been stolen.
        Otherwise, the returned `list` is identical to the passed-in `train_items`.
        """
        with isolate_rng():
            self.val_dataloader, remaining_train_items = self._build_val_dataloader_if_required(train_items, tokenizer)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            self.train_overlapping_dataloader = self._build_train_stabilizer_dataloader_if_required(
                remaining_train_items, tokenizer)

            self.outlier_finder = self._build_outlier_finder_if_required(remaining_train_items, tokenizer)
            return remaining_train_items

    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                self._do_validation('stabilize-train', global_step, self.train_overlapping_dataloader,
                                    get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable,
                                    log_extended_stats=True)
        if self.outlier_finder is not None and (epoch % self.config['find_outliers_every_n_epochs']) == 0:
            self.outlier_finder.do_find_outliers(validator=self, global_step=global_step,
                                                 get_model_prediction_and_target_callable=get_model_prediction_and_target_callable)


    def _do_validation(self,
                       tag:str,
                       global_step:int,
                       dataloader:torch.utils.data.DataLoader,
                       get_model_prediction_and_target: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       log_loss: bool=True,
                       log_extended_stats: bool=False,
                       log_pinned_batches: Optional[list[int]]=None) -> torch.Tensor:
        """
        Do a validation pass using the given dataloader. RNG is isolated and reset to self.seed.

        tag: Used to dilineate log outputs (see e.g. `log_loss`).
        global_step: Current step.
        dataloader: Source for validation data (images and captions).
        get_model_prediction_and_target : Callable that takes image and tokens and returns predicted and target
                                          (i.e. actual) VAE latents.
        log_loss: if True, log the mean of all losses in this batch to `loss/{tag}`.
        log_extended_stats: if True, log mean, median, min, and max losses to `{tag}/mean` etc.
        log_pinned_batches: if not None, log the loss of each of the given batch indices to `{tag}/batch-#{idx}`.
        """
        with torch.no_grad(), isolate_rng():
            losses_list = []
            steps_pbar = tqdm(range(len(dataloader)), position=1)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Validate ({tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                # ok to override seed here because we are in a `with isolate_rng():` block
                torch.manual_seed(self.seed + step)
                model_pred, target = get_model_prediction_and_target(batch["image"], batch["tokens"])

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                del target, model_pred

                loss_step = loss.detach().item()
                del loss

                losses_list.append(loss_step)
                

                steps_pbar.update(1)

            steps_pbar.close()

        losses_tensor: torch.Tensor = torch.tensor(losses_list)
        if log_loss:
            self.log_writer.add_scalar(tag=f"loss/{tag}", scalar_value=torch.mean(losses_tensor).item(), global_step=global_step)
        
        if log_extended_stats:
            self.log_writer.add_scalar(tag=f"{tag}/mean", scalar_value=torch.mean(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/median", scalar_value=torch.median(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/min", scalar_value=torch.min(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/max", scalar_value=torch.max(losses_tensor).item(), global_step=global_step)

        if log_pinned_batches is not None:
            self._log_individual_batch_losses(log_pinned_batches, losses_tensor, tag=tag, global_step=global_step)
        
        return losses_tensor

    def _build_val_dataloader_if_required(self, image_train_items: list[ImageTrainItem], tokenizer)\
            -> tuple[Optional[torch.utils.data.DataLoader], list[ImageTrainItem]]:
        val_split_mode = self.config['val_split_mode'] if self.config['validate_training'] else None
        val_split_proportion = self.config['val_split_proportion']
        remaining_train_items = image_train_items
        if val_split_mode is None or val_split_mode == 'none':
            return None, image_train_items
        elif val_split_mode == 'automatic':
            val_items, remaining_train_items = get_random_split(image_train_items, val_split_proportion, batch_size=self.batch_size)
            val_items = list(disable_multiplier_and_flip(val_items))
            logging.info(f" * Removed {len(val_items)} images from the training set to use for validation")
        elif val_split_mode == 'manual':
            args = Namespace(
                aspects=aspects.get_aspect_buckets(self.resolution),
                flip_p=0.0,
                seed=self.seed,
            )
            val_data_root = self.config['val_data_root']
            val_items = resolver.resolve_root(val_data_root, args)
            logging.info(f" * Loaded {len(val_items)} validation images from {val_data_root}")
        else:
            raise ValueError(f"Unrecognized validation split mode '{val_split_mode}'")
        val_ed_batch = self._build_ed_batch(val_items, batch_size=self.batch_size, tokenizer=tokenizer, name='val')
        val_dataloader = build_torch_dataloader(val_ed_batch, batch_size=self.batch_size)
        return val_dataloader, remaining_train_items

    def _build_train_stabilizer_dataloader_if_required(self, image_train_items: list[ImageTrainItem], tokenizer) \
            -> Optional[torch.utils.data.DataLoader]:
        stabilize_training_loss = self.config['stabilize_training_loss']
        if not stabilize_training_loss:
            return None

        stabilize_split_proportion = self.config['stabilize_split_proportion']
        stabilize_items, _ = get_random_split(image_train_items, stabilize_split_proportion, batch_size=self.batch_size)
        stabilize_items = list(disable_multiplier_and_flip(stabilize_items))
        stabilize_ed_batch = self._build_ed_batch(stabilize_items, batch_size=self.batch_size, tokenizer=tokenizer,
                                                  name='stabilize-train')
        stabilize_dataloader = build_torch_dataloader(stabilize_ed_batch, batch_size=self.batch_size)
        return stabilize_dataloader

    def _build_outlier_finder_if_required(self, image_train_items: list[ImageTrainItem], tokenizer) \
            -> Optional['OutlierFinder']:
        do_find_outliers = self.config['find_outliers']
        if not do_find_outliers:
            return None
        split_proportion = self.config['find_outliers_split_proportion']
        pinned_image_count = self.config['find_outliers_pinned_image_count']
        find_outliers_items, _ = get_random_split(image_train_items, split_proportion, batch_size=1)
        outlier_finder = OutlierFinder(validator=self,
                                       items=list(disable_multiplier_and_flip(find_outliers_items)),
                                       tokenizer=tokenizer,
                                       pinned_image_count=pinned_image_count)
        return outlier_finder

    def _build_ed_batch(self, items: list[ImageTrainItem], batch_size: int, tokenizer, name='val'):
        batch_size = self.batch_size
        seed = self.seed
        data_loader = DataLoaderMultiAspect(
            items,
            batch_size=batch_size,
            seed=seed,
        )
        ed_batch = EveryDreamBatch(
            data_loader=data_loader,
            debug_level=1,
            conditional_dropout=0,
            tokenizer=tokenizer,
            seed=seed,
            name=name,
        )
        return ed_batch

    def _log_individual_batch_losses(self, batch_indices: list[int], losses_tensor:torch.Tensor, tag:str, global_step: int, batch_labels: list[str]=None):
        batch_labels = batch_labels or [f"batch-#{batch_idx}" for batch_idx in batch_indices]
        for i,batch_idx in enumerate(batch_indices):
            self.log_writer.add_scalar(tag=f"{tag}/{batch_labels[i]}", scalar_value=losses_tensor[batch_idx].item(), global_step=global_step)

class OutlierFinder:
    def __init__(self,
                 validator: EveryDreamValidator,
                 items: list[ImageTrainItem],
                 tokenizer,
                 pinned_image_count: int):

        # we need a dataloader and dataset with batch_size=1
        self.ed_batch = validator._build_ed_batch(items, batch_size=1, tokenizer=tokenizer, name='outlier-finder')
        self.dataloader = build_torch_dataloader(self.ed_batch, batch_size=1)

        self.logging_tag = 'outliers'

        self.pinned_images_count = pinned_image_count
        self.pinned_image_indices = None
        self.pinned_image_labels = None
        self.collected_losses = None
        self.collected_global_steps = None

    def do_find_outliers(self,
                         validator: EveryDreamValidator,
                         global_step: int,
                         get_model_prediction_and_target_callable):
        losses = validator._do_validation(self.logging_tag, global_step, self.dataloader,
                                          get_model_prediction_and_target_callable,
                                          log_loss=False, log_extended_stats=True)
        # to column vector
        losses_t = losses.unsqueeze(0).t()

        if self.collected_losses is None:
            self.collected_losses = losses_t
            self.collected_global_steps = [global_step]
        else:
            if len(self.collected_global_steps) == 1:
                # prepare pinning
                min_max_pin_count = (self.pinned_images_count - 1) // 2
                if min_max_pin_count == 0:
                    logging.warning("Not enough data to pin min/max outliers")
                else:
                    # collect pinned batch ids
                    delta_losses = losses_t - self.collected_losses
                    max_indices = torch.topk(delta_losses, k=min_max_pin_count, dim=0, largest=True, sorted=True).indices.flip(0)
                    min_indices = torch.topk(delta_losses, k=min_max_pin_count, dim=0, largest=False, sorted=True).indices
                    median_index = torch.median(delta_losses, dim=0).indices.unsqueeze(0)
                    self.pinned_image_indices = torch.cat([min_indices, median_index, max_indices]).t().squeeze().tolist()
                    # make labels
                    pin_types = ['max'] * min_max_pin_count + ['median'] + ['min'] * min_max_pin_count
                    self.pinned_image_labels = [f"pin{i:02}-{pin_types[i]}-i#{image_index}"
                                                for i,image_index in enumerate(self.pinned_image_indices)]

                    # one-off log of the images being pinned
                    newline = '  \n' # tensorboard uses markdown format so needs 2 spaces
                    pinned_ids_description = newline.join(
                        [f"{label}: {self.ed_batch.image_train_items[image_index].pathname}"
                         for label, image_index in zip(self.pinned_image_labels, self.pinned_image_indices)]
                    )
                    validator.log_writer.add_text(self.logging_tag, pinned_ids_description)

                    # log the initial losses that we couldn't log before because the pinned indices weren't yet known
                    skipped_losses = self.collected_losses
                    self._log_individual_item_losses(losses_tensor=skipped_losses, validator=validator,
                                                     global_step=self.collected_global_steps[0])

            self.collected_losses = torch.cat([self.collected_losses, losses_t], dim=1)
            self.collected_global_steps.append(global_step)

        loss_path_pairs_sorted = sorted(zip([i.pathname for i in self.ed_batch.image_train_items],
                                            self.collected_losses.tolist()
                                            ),
                                 key=lambda i: i[0], reverse=True)

        filename = f"{validator.log_folder}/per_item_losses.csv"
        try:
            with open(filename, "w", encoding='utf-8') as f:
                steps_list_string = ','.join([f"step {s}" for s in self.collected_global_steps])
                f.write(f"path,{steps_list_string}\n")
                for path, losses_t in loss_path_pairs_sorted:
                    path_escaped_and_quoted = '"' + path.replace('"', '\\"') + '"'
                    losses_list_string = ','.join([str(x) for x in losses_t])
                    f.write(f"{path_escaped_and_quoted},{losses_list_string}\n")
        except Exception as e:
            traceback.print_exc()
            logging.error(f" * Error {e} writing outliers to {filename}")

        if self.pinned_image_indices is not None:
            self._log_individual_item_losses(losses_tensor=losses, validator=validator, global_step=global_step)

    def _log_individual_item_losses(self, losses_tensor: torch.Tensor, validator: EveryDreamValidator, global_step: int):
        validator._log_individual_batch_losses(losses_tensor=losses_tensor,
                                               batch_indices=self.pinned_image_indices,
                                               tag=self.logging_tag, global_step=global_step,
                                               batch_labels=self.pinned_image_labels)
