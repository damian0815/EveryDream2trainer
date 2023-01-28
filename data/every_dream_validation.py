import itertools
import json
import math
from typing import Callable, Any, Optional

import torch
from colorama import Fore, Style
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.every_dream import build_torch_dataloader, EveryDreamBatch
from utils.isolate_rng import isolate_rng


class EveryDreamValidator:
    def __init__(self,
                 val_config_path: Optional[str],
                 train_batch: EveryDreamBatch,
                 log_writer: SummaryWriter):
        self.log_writer = log_writer

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

        with isolate_rng():
            self.train_overlapping_dataloader = self._build_dataloader_from_automatic_split(train_batch,
                                                            split_proportion=stabilize_split_proportion,
                                                            enforce_split=False) if stabilize_training_loss else None
            self.val_dataloader = self._build_validation_dataloader(val_split_mode,
                                                                    split_proportion=val_split_proportion,
                                                                    val_data_root=self.val_data_root,
                                                                    train_batch=train_batch)


    def do_validation_if_appropriate(self, epoch: int, global_step: int,
                                     get_model_prediction_and_target_callable: Callable[
                                         [Any, Any], tuple[torch.Tensor, torch.Tensor]]):
        if (epoch % self.every_n_epochs) == 0:
            if self.train_overlapping_dataloader is not None:
                self._do_validation('train-stabilized', global_step, self.train_overlapping_dataloader, get_model_prediction_and_target_callable)
            if self.val_dataloader is not None:
                self._do_validation('val', global_step, self.val_dataloader, get_model_prediction_and_target_callable)


    def _do_validation(self, tag, global_step, dataloader, get_model_prediction_and_target_callable):
        with torch.no_grad(), isolate_rng():
            loss_validation_epoch = []
            validate_epoch_len = math.ceil(len(dataloader) / dataloader.batch_size)
            steps_pbar = tqdm(range(validate_epoch_len), position=1)
            steps_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Steps ({tag}){Style.RESET_ALL}")

            for step, batch in enumerate(dataloader):
                # ok to override seed here because we are in an isolate_rng() 'with' block
                torch.manual_seed(self.seed + step)
                model_pred, target = get_model_prediction_and_target_callable(batch["image"], batch["tokens"])

                # del timesteps, encoder_hidden_states, noisy_latents
                # with autocast(enabled=args.amp):
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                del target, model_pred

                loss_step = loss.detach().item()
                steps_pbar.set_postfix({f"loss/{tag} step": loss_step}, {"gs": global_step})
                steps_pbar.update(1)

                loss_validation_epoch.append(loss_step)

            steps_pbar.close()

        loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)
        self.log_writer.add_scalar(tag=f"loss/{tag}", scalar_value=loss_validation_local, global_step=global_step)


    def _build_validation_dataloader(self,
                                     validation_split_mode: str,
                                     split_proportion: float,
                                     val_data_root: Optional[str],
                                     train_batch: EveryDreamBatch) -> Optional[DataLoader]:
        if validation_split_mode == 'none':
            return None
        elif validation_split_mode == 'automatic':
            return self._build_dataloader_from_automatic_split(train_batch, split_proportion, enforce_split=True)
        elif validation_split_mode == 'custom':
            if val_data_root is None:
                raise ValueError("val_data_root is required for 'split-custom' validation mode")
            return self._build_dataloader_from_custom_split(self.val_data_root, reference_train_batch=train_batch)
        else:
            raise ValueError(f"unhandled validation split mode '{validation_split_mode}'")


    def _build_dataloader_from_automatic_split(self,
                                               train_batch: EveryDreamBatch,
                                               split_proportion: float,
                                               enforce_split: bool=False) -> DataLoader:
        """
        Build a validation dataloader by copying data from the given `train_batch`. If `enforce_split` is `True`, remove
        the copied items from train_batch so that there is no overlap between `train_batch` and the new dataloader.
        """
        val_item_count_nonbatched = split_proportion * len(train_batch)
        val_item_count_batched = ((val_item_count_nonbatched // train_batch.batch_size) + 1) * train_batch.batch_size
        print(f"* building automatic validation split by taking {val_item_count_batched} items (of {len(train_batch)}) from train_batch")

        # train_batch has been shuffled on load - bake its order here by converting from iterator to list
        val_items = list(itertools.islice(train_batch, val_item_count_batched))
        if enforce_split:
            del train_batch[0:val_item_count_batched]
        if len(train_batch) == 0:
            raise ValueError(f"validation split used up all of the training data. try a lower split proportion than {split_proportion}")
        return build_torch_dataloader(
            items=val_items,
            batch_size=train_batch.batch_size,
        )


    def _build_dataloader_from_custom_split(self, data_root: str, reference_train_batch: EveryDreamBatch) -> DataLoader:
        val_batch = EveryDreamBatch(
            data_root=data_root,
            debug_level=1,
            batch_size=reference_train_batch.batch_size,
            conditional_dropout=0,
            resolution=reference_train_batch.dataloader.resolution,
            tokenizer=reference_train_batch.tokenizer,
            seed=reference_train_batch.seed,
            log_folder=reference_train_batch.log_folder,
            write_schedule=reference_train_batch.write_schedule,
            name='val',
        )
        return build_torch_dataloader(
            items=val_batch,
            batch_size=reference_train_batch.batch_size
        )



