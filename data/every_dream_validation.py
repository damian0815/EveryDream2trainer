import json
import logging
import random
import traceback
from typing import Callable, Any, Optional

import torch
from colorama import Fore, Style
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from utils.isolate_rng import isolate_rng


class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 train_batch: EveryDreamBatch,
                 log_writer: SummaryWriter,
                 log_folder: str):
        self.log_writer = log_writer
        self.log_folder = log_folder

        val_config = {}
        if val_config_path is not None:
            with open(val_config_path, 'rt') as f:
                val_config = json.load(f)

        do_validation = val_config.get('validate_training', False)
        val_split_mode = val_config.get('val_split_mode', 'automatic') if do_validation else 'none'
        self.val_data_root = val_config.get('val_data_root', None)
        val_split_proportion = val_config.get('val_split_proportion', 0.15)

        stabilize_training_loss = val_config.get('stabilize_training_loss', False)
        stabilize_split_proportion = val_config.get('stabilize_split_proportion', 0.15)

        self.every_n_epochs = val_config.get('every_n_epochs', 1)
        self.seed = val_config.get('seed', 555)

        do_find_outliers = val_config.get('find_outliers', False)
        self.find_outliers_every_n_epochs = val_config.get('find_outliers_every_n_epochs', self.every_n_epochs)
        self.collected_losses: Optional[torch.Tensor] = None

        with isolate_rng():
            self.val_dataloader = self._build_validation_dataloader(val_split_mode,
                                                                    split_proportion=val_split_proportion,
                                                                    val_data_root=self.val_data_root,
                                                                    train_batch=train_batch)
            if self.val_dataloader is not None:
                # pin 3 random batches
                all_val_batch_ids = list(range(len(self.val_dataloader)))
                random.shuffle(all_val_batch_ids)
                self.val_pinned_batch_ids = all_val_batch_ids[:3]

            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            self.train_overlapping_dataloader = None if not stabilize_training_loss else \
                self._build_dataloader_from_automatic_split(train_batch,
                                                            split_proportion=stabilize_split_proportion,
                                                            name='train-stabilizer',
                                                            enforce_split=False)

            if do_find_outliers:
                split_proportion = val_config.get('find_outliers_split_proportion', 1)
                pinned_image_count = val_config.get('find_outliers_pinned_image_count', 7)
                self.outlier_finder = OutlierFinder(validator=self,
                                                    reference_train_batch=train_batch,
                                                    split_proportion=split_proportion,
                                                    pinned_image_count=pinned_image_count)
            else:
                self.outlier_finder = None

    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                self._do_validation('stabilize-train', global_step, self.train_overlapping_dataloader,
                                    get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable,
                                    log_extended_stats=True, log_pinned_batches=self.val_pinned_batch_ids)
        if self.outlier_finder is not None and (epoch % self.find_outliers_every_n_epochs) == 0:
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

                # del timesteps, encoder_hidden_states, noisy_latents
                # with autocast(enabled=args.amp):
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                loss_step = loss.detach().item()
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





    def _build_validation_dataloader(self,
                                     validation_split_mode: str,
                                     split_proportion: float,
                                     val_data_root: Optional[str],
                                     train_batch: EveryDreamBatch) -> Optional[DataLoader]:
        if validation_split_mode == 'none':
            return None
        elif validation_split_mode == 'automatic':
            return self._build_dataloader_from_automatic_split(train_batch, split_proportion, name='val', enforce_split=True)
        elif validation_split_mode == 'manual':
            if val_data_root is None:
                raise ValueError("val_data_root is required for 'manual' validation split mode")
            return self._build_dataloader_from_custom_split(self.val_data_root, reference_train_batch=train_batch)
        else:
            raise ValueError(f"unhandled validation split mode '{validation_split_mode}'")


    def _build_dataloader_from_automatic_split(self,
                                               train_batch: EveryDreamBatch,
                                               split_proportion: float,
                                               name: str,
                                               enforce_split: bool=False,
                                               override_batch_size: Optional[int]=None,
                                               also_return_batch: bool=False
                                               ) -> DataLoader | tuple[DataLoader, EveryDreamBatch]:
        """
        Build a validation dataloader by copying data from the given `train_batch`. If `enforce_split` is `True`, remove
        the copied items from train_batch so that there is no overlap between `train_batch` and the new dataloader.
        """
        with isolate_rng():
            random.seed(self.seed)
            val_items = train_batch.get_random_split(split_proportion, remove_from_dataset=enforce_split)
            if enforce_split:
                print(
                f"  * Removed {len(val_items)} items for validation split from '{train_batch.name}' - {round(len(train_batch)/train_batch.batch_size)} batches are left")
            if len(train_batch) == 0:
                raise ValueError(f"Validation split used up all of the training data. Try a lower split proportion than {split_proportion}")
            val_batch = self._make_val_batch_with_train_batch_settings(
                val_items,
                train_batch,
                name=name,
                override_batch_size=override_batch_size
            )
            dataloader = build_torch_dataloader(
                items=val_batch,
                batch_size=override_batch_size or train_batch.batch_size,
            )
            if also_return_batch:
                return dataloader, val_batch
            else:
                return dataloader


    def _build_dataloader_from_custom_split(self, data_root: str, reference_train_batch: EveryDreamBatch) -> DataLoader:
        val_batch = self._make_val_batch_with_train_batch_settings(data_root, reference_train_batch)
        return build_torch_dataloader(
            items=val_batch,
            batch_size=reference_train_batch.batch_size
        )

    def _make_val_batch_with_train_batch_settings(self, data_root, reference_train_batch, name='val',
                                                  override_batch_size: Optional[int]=None) -> EveryDreamBatch:
        return EveryDreamBatch(
            data=data_root,
            debug_level=1,
            batch_size=override_batch_size or reference_train_batch.batch_size,
            conditional_dropout=0,
            resolution=reference_train_batch.resolution,
            tokenizer=reference_train_batch.tokenizer,
            seed=reference_train_batch.seed,
            log_folder=reference_train_batch.log_folder,
            write_schedule=reference_train_batch.write_schedule,
            name=name,
        )

    def _log_individual_batch_losses(self, batch_indices: list[int], losses_tensor:torch.Tensor, tag:str, global_step: int, batch_labels: list[str]=None):
        batch_labels = batch_labels or [f"batch-#{batch_idx}" for batch_idx in batch_indices]
        for i,batch_idx in enumerate(batch_indices):
            self.log_writer.add_scalar(tag=f"{tag}/{batch_labels[i]}", scalar_value=losses_tensor[batch_idx].item(), global_step=global_step)


class OutlierFinder:
    def __init__(self,
                 validator: EveryDreamValidator,
                 reference_train_batch: EveryDreamBatch,
                 split_proportion: float,
                 pinned_image_count: int):

        # we need a dataloader and dataset with batch_size=1
        self.dataloader, self.batched_dataset = validator._build_dataloader_from_automatic_split(
            reference_train_batch, split_proportion=split_proportion, name='find-outliers',
            enforce_split=False, override_batch_size=1, also_return_batch=True)

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
                        [f"{label}: {self.batched_dataset.image_train_items[image_index].pathname}"
                         for label, image_index in zip(self.pinned_image_labels, self.pinned_image_indices)]
                    )
                    validator.log_writer.add_text(self.logging_tag, pinned_ids_description)

                    # log the initial losses that we couldn't log before because the pinned indices weren't yet known
                    skipped_losses = self.collected_losses
                    self._log_individual_item_losses(losses_tensor=skipped_losses, validator=validator,
                                                     global_step=self.collected_global_steps[0])

            self.collected_losses = torch.cat([self.collected_losses, losses_t], dim=1)
            self.collected_global_steps.append(global_step)

        loss_path_pairs_sorted = sorted(zip([i.pathname for i in self.batched_dataset.image_train_items],
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
