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

        find_outliers = val_config.get('find_outliers', False)
        self.prev_losses: Optional[torch.Tensor] = None

        with isolate_rng():
            self.val_dataloader = self._build_validation_dataloader(val_split_mode,
                                                                    split_proportion=val_split_proportion,
                                                                    val_data_root=self.val_data_root,
                                                                    train_batch=train_batch)
            # order is important - if we're removing images from train, this needs to happen before making
            # the overlapping dataloader
            self.train_overlapping_dataloader = None if not stabilize_training_loss else \
                self._build_dataloader_from_automatic_split(train_batch,
                                                            split_proportion=stabilize_split_proportion,
                                                            name='train-stabilizer',
                                                            enforce_split=False)

            if find_outliers is None:
                self.find_outliers_dataloader = None
            else:
                self.find_outliers_batch = self._make_val_batch_with_train_batch_settings(train_batch.get_all_image_train_items(),
                                                                                          train_batch,
                                                                                          'find-outliers',
                                                                                          override_batch_size=1)
                self.find_outliers_dataloader = build_torch_dataloader(self.find_outliers_batch, batch_size=1)


    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                self._do_validation('stabilize-train', global_step, self.train_overlapping_dataloader, get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable)
            if self.find_outliers_dataloader is not None:
                self._do_find_outliers(global_step, get_model_prediction_and_target_callable)



    def _do_validation(self, tag, global_step, dataloader, get_model_prediction_and_target: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]], extended_logging: bool=False):
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
        self.log_writer.add_scalar(tag=f"loss/{tag}", scalar_value=torch.mean(losses_tensor).item(), global_step=global_step)

        if extended_logging:
            self.log_writer.add_scalar(tag=f"{tag}/mean", scalar_value=torch.mean(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/median", scalar_value=torch.median(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/min", scalar_value=torch.max(losses_tensor).item(), global_step=global_step)
            self.log_writer.add_scalar(tag=f"{tag}/max", scalar_value=torch.min(losses_tensor).item(), global_step=global_step)


            mean = torch.mean(losses_tensor)
            std = torch.std(losses_tensor)
            outlier_indices = torch.nonzero((losses_tensor > mean-std*2) & (losses_tensor < mean+std*2), as_tuple=True)[0]
            losses_no_outliers = torch.index_select(losses_tensor, dim=0, index=outlier_indices)
            if losses_no_outliers.numel() > 0:
                self.log_writer.add_scalar(tag=f"{tag}/trimmed-mean", scalar_value=torch.mean(losses_no_outliers).item(), global_step=global_step)
                self.log_writer.add_scalar(tag=f"{tag}/trimmed-min", scalar_value=torch.min(losses_no_outliers).item(), global_step=global_step)
                self.log_writer.add_scalar(tag=f"{tag}/trimmed-max", scalar_value=torch.max(losses_no_outliers).item(), global_step=global_step)


    def _do_find_outliers(self, global_step, get_model_prediction_and_target_callable):
        losses = self._do_validation('find-outliers', global_step, self.find_outliers_dataloader,
                                     get_model_prediction_and_target_callable, extended_logging=True)
        if self.prev_losses is None:
            self.prev_losses = losses
            return

        loss_deltas = losses - self.prev_losses
        self.prev_losses += loss_deltas

        loss_path_pairs_sorted = sorted(zip(loss_deltas.tolist(), [i.pathname for i in self.find_outliers_batch.image_train_items]),
                                 key=lambda i: i[0], reverse=True)

        filename = f"{self.log_folder}/per_item_loss_deltas-gs{global_step:05}.csv"

        # we want to prepend new data
        try:
            with open(filename, "w", encoding='utf-8') as f:
                for loss, path in loss_path_pairs_sorted:
                    f.write(f"{loss},{path}\n")
        except Exception as e:
            traceback.print_exc()
            logging.error(f" * Error {e} writing outliers to {filename}")



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
                                               override_batch_size: Optional[int]=None
                                               ) -> DataLoader:
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
            return build_torch_dataloader(
                items=val_batch,
                batch_size=train_batch.batch_size,
            )


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



