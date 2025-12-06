import gc
import json
import logging
import os
import shutil

import torch
import safetensors.torch
from plugins.plugins import BasePlugin
from model.training_model import (
    save_model,
    EveryDreamTrainingState,
    TrainingModel,
    TrainingVariables,
)

class RepeatTheFirstNPlugin(BasePlugin):
    """
    Plugin that trains for N epochs, accumulates the model state, resets to initial state,
    and repeats M times. At the end, saves the average of all M models.

    Configuration is loaded from repeat_the_first.json in the current working directory.
    """

    def __init__(self):
        print("RepeatTheFirstN plugin instantiated")
        self.config_path = "repeat_the_first.json"
        self.n_epochs = None
        self.merge_every_m_cycles = None
        self.storage_dir = None
        self.accumulated_state_counter = 0
        self.merge_counter = 0
        self.initial_states_saved = False
        self.max_epochs = None
        self.previous_save_path = None
        self.save_rolling_ckpts = False
        self.generate_samples = True
        self.generate_samples_every_n_cycles = None

    @property
    def accumulated_unet_state_dict_path(self):
        return os.path.join(self.storage_dir, "accumulated_unet_state_dict.safetensors")


    def on_training_start(self, **kwargs):
        # Load configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"RepeatTheFirstNPlugin requires a configuration file at {self.config_path}. "
                f"Example: {{'n_epochs': 5}}"
            )

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        self.n_epochs = config.get('n_epochs')
        if self.n_epochs is None:
            raise ValueError(f"Configuration file {self.config_path} must contain 'n_epochs'")
        self.merge_every_m_cycles = config.get('merge_every_m_cycles', None)
        self.save_rolling_ckpts = config.get('save_rolling_ckpts', False)
        self.generate_samples = config.get("generate_samples", True)
        self.generate_samples_every_n_cycles = config.get('generate_samples_every_n_cycles', 1)

        if not isinstance(self.n_epochs, int) or self.n_epochs <= 0:
            raise ValueError(f"'n_epochs' must be a positive integer, got {self.n_epochs}")

        # Get max_epochs from kwargs (passed through from args)
        self.max_epochs = kwargs['max_epochs']

        # Validate that max_epochs is divisible by n_epochs
        if self.max_epochs % self.n_epochs != 0:
            raise ValueError(
                f"Total epochs ({self.max_epochs}) must be evenly divisible by n_epochs ({self.n_epochs}). "
                f"Current division yields {self.max_epochs / self.n_epochs} cycles."
            )

        num_cycles = self.max_epochs // self.n_epochs
        print(f"\nRepeatTheFirstN Plugin Configuration:")
        print(f"  - Training for {self.n_epochs} epochs per cycle")
        if self.merge_every_m_cycles is not None:
            print(f"  - Merging every {self.merge_every_m_cycles} cycles to create new baseline")
        print(f"  - Total epochs: {self.max_epochs}")
        print(f"  - Number of cycles: {num_cycles}")
        print(f"  - Saving intermediates: {self.save_rolling_ckpts}")
        print(f"  - Generate samples: {self.generate_samples}")
        print(f"  - Models will be accumulated and averaged at the end")

        # Set up storage directory
        log_folder = kwargs['log_folder']
        self.storage_dir = os.path.join(log_folder, "ckpts", "repeat_the_first")
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save initial model states
        self._save_baseline_state(kwargs['ed_state'])

        # Initialize counter
        self.accumulated_state_counter = 0
        self.merge_counter = 0
        self._save_counter()

        print(f"  - Storage directory: {self.storage_dir}")
        print(f"  - Initial model states saved\n")

    def on_epoch_end(self, **kwargs):
        epoch = kwargs['epoch']

        # Check if we're at the end of an N-epoch cycle
        if (epoch + 1) % self.n_epochs == 0:
            print(f"\n{'='*60}")
            print(f"RepeatTheFirstN: End of cycle at epoch {epoch + 1}")
            print(f"{'='*60}")

            ed_state = kwargs['ed_state']

            # Accumulate current model state
            self._accumulate_model_state(ed_state)

            # save ckpt
            if self.save_rolling_ckpts:
                ed_state.model.unet.load_state_dict(self._load_and_average_accumulated_states())
                project_name = kwargs['project_name']
                log_folder = kwargs['log_folder']
                global_step = kwargs['global_step']
                ckpt_name = f"rolling_{project_name}_repeat_the_first_{self.n_epochs}_epochs_{self.accumulated_state_counter}_times_merge_{self.merge_counter}_gs{global_step:05}"

                save_path = os.path.join(log_folder, "ckpts", ckpt_name)
                print(f"Saving model to: {save_path}")
                self._save(ed_state, save_path=save_path, global_step=global_step)

                self._remove_previous_save()
                self.previous_save_path = save_path

            if self.generate_samples and self.accumulated_state_counter % self.generate_samples_every_n_cycles == 0:
                ed_state.model.unet.load_state_dict(self._load_and_average_accumulated_states())
                kwargs['sample_generator_cb'](global_step=kwargs['global_step'], batch=None)

            # Only reset if this is not the final epoch
            if epoch + 1 < self.max_epochs:
                if (
                    self.merge_every_m_cycles is not None
                    and self.accumulated_state_counter % self.merge_every_m_cycles == 0
                ):
                    print(
                        f"Merging last {self.merge_every_m_cycles} cycles to create new baseline..."
                    )
                    ed_state.model.unet.load_state_dict(
                        self._load_and_average_accumulated_states()
                    )
                    # Save new baseline as initial state for next cycle
                    print(f"Saving new baseline as initial state...")
                    self._save_baseline_state(ed_state)
                    print(f"Removing accumulated states for next cycle...")
                    self._remove_accumulated_states()
                    print(f"New baseline saved.")
                    self.merge_counter += 1
                    self.accumulated_state_counter = 0

                print(f"Resetting model to baseline state for next cycle...")
                training_variables = kwargs.get('training_variables')
                self._reset_to_baseline_state(ed_state, training_variables)
                print(f"Model reset complete. Starting cycle {self.accumulated_state_counter}\n")


    def on_training_end(self, **kwargs):
        print(f"\n{'='*60}")
        print(f"RepeatTheFirstN: Training complete, computing final averaged model")
        print(f"{'='*60}")

        if self.accumulated_state_counter == 0 and self.merge_counter == 0:
            print("Warning: No model states were accumulated. Skipping final save.")
            return

        ed_state = kwargs['ed_state']

        ed_state.model.unet.load_state_dict(self._load_and_average_accumulated_states())

        try:
            print(f"Cleaning up...")
            self._remove_previous_save()
            shutil.rmtree(self.storage_dir)
        except Exception as e:
            print(f"  - Warning: Failed to clean up storage directory {self.storage_dir}: {e}")
            pass

        print(f"{'='*60}\n")


    def _save_baseline_state(self, ed_state):
        """Save the initial model state dicts to disk."""
        print("Saving initial model states...")

        # Save initial states
        unet_path = os.path.join(self.storage_dir, "initial_unet_state_dict.safetensors")
        assert_not_nan(ed_state.model.unet.state_dict(), "Initial UNet state dict contains NaN values")
        safetensors.torch.save_file(ed_state.model.unet.state_dict(), unet_path)
        optimizer_unet_path = os.path.join(self.storage_dir, "initial_optimizer_unet.pt")
        torch.save(ed_state.optimizer.optimizer_unet.state_dict(), optimizer_unet_path)

        # Save initial LR scheduler states
        lr_schedulers_path = os.path.join(self.storage_dir, "initial_lr_schedulers.pt")
        lr_scheduler_states = [scheduler.state_dict() for scheduler in ed_state.optimizer.lr_schedulers]
        torch.save(lr_scheduler_states, lr_schedulers_path)

        self.initial_states_saved = True

    def _remove_accumulated_states(self):
        try:
            os.unlink(self.accumulated_unet_state_dict_path)
        except FileNotFoundError:
            pass

    def _accumulate_model_state(self, ed_state):
        """Accumulate the current model state into running sum."""
        print(f"Accumulating model state (count: {self.accumulated_state_counter + 1})...")

        # Get current model states
        current_unet_state_dict = ed_state.model.unet.state_dict()
        assert_not_nan(current_unet_state_dict, "Current UNet state dict contains NaN values")

        # Load and add, or initialize if first time
        if not os.path.exists(self.accumulated_unet_state_dict_path):
            # First accumulation - just save current state
            accumulated_unet_state_dict = current_unet_state_dict
        else:
            # Load existing accumulated states and add current states
            accumulated_unet_state_dict = safetensors.torch.load_file(self.accumulated_unet_state_dict_path)

            # Add current to accumulated (element-wise)
            for key in accumulated_unet_state_dict.keys():
                accumulated_unet_state_dict[key] = accumulated_unet_state_dict[key] + current_unet_state_dict[key].to('cpu')

        assert_not_nan(accumulated_unet_state_dict, "Current UNet state dict contains NaN values")

        # Save updated accumulated states
        safetensors.torch.save_file(accumulated_unet_state_dict, self.accumulated_unet_state_dict_path)
        self.accumulated_state_counter += 1
        self._save_counter()

        print(f"Accumulation complete. Total accumulated: {self.accumulated_state_counter}")

    def _load_baseline_state(self):
        unet_path = os.path.join(self.storage_dir, "initial_unet_state_dict.safetensors")
        return safetensors.torch.load_file(unet_path)

    def _reset_to_baseline_state(self, ed_state, training_variables: TrainingVariables):
        """Reset model, optimizer, and training variables to initial states."""
        # Load initial model states
        initial_unet = self._load_baseline_state()
        ed_state.model.unet.load_state_dict(initial_unet)

        # Reset optimizer states
        optimizer_unet_path = os.path.join(self.storage_dir, "initial_optimizer_unet.pt")
        initial_opt_unet = torch.load(optimizer_unet_path, weights_only=True)
        ed_state.optimizer.optimizer_unet.load_state_dict(initial_opt_unet)

        # Reset LR schedulers
        lr_schedulers_path = os.path.join(self.storage_dir, "initial_lr_schedulers.pt")
        if os.path.exists(lr_schedulers_path):
            initial_lr_scheduler_states = torch.load(lr_schedulers_path, weights_only=True)
            for scheduler, initial_state in zip(ed_state.optimizer.lr_schedulers, initial_lr_scheduler_states):
                scheduler.load_state_dict(initial_state)
            print("  - LR schedulers reset")

        # Reset training variables (clears loss accumulation, etc.)
        training_variables.reset()
        # reset CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        print("  - Training variables reset (loss accumulation cleared)")


    def _load_and_average_accumulated_states(self):
        """Load accumulated states and divide by counter to get average."""
        print(f"Averaging {self.accumulated_state_counter} accumulated model state...")
        acc_unet_path = os.path.join(self.storage_dir, "accumulated_unet_state_dict.safetensors")

        # Load accumulated states
        accumulated_unet = safetensors.torch.load_file(acc_unet_path)
        assert_not_nan(accumulated_unet, "Accumulated UNet state dict contains NaN values")

        # Divide by counter to get average
        for key in accumulated_unet.keys():
            accumulated_unet[key] = accumulated_unet[key] / self.accumulated_state_counter

        return accumulated_unet


    def _save_counter(self):
        """Save the counter to disk."""
        counter_path = os.path.join(self.storage_dir, "counter.txt")
        with open(counter_path, 'w') as f:
            f.write(str(self.accumulated_state_counter))


    def _save(self,
              ed_state_accumulated: EveryDreamTrainingState,
              save_path: str,
              global_step: int):

        save_model(
            save_path,
            global_step=global_step,
            ed_state=ed_state_accumulated,
            save_ckpt_dir=None,
            yaml_name=None,
            save_ckpt=False,
            save_full_precision=True,
            save_optimizer_flag=False
        )


    def _remove_previous_save(self):
        if self.previous_save_path is not None:
            shutil.rmtree(self.previous_save_path, ignore_errors=True)
        self.previous_save_path = None

def assert_not_nan(state_dict, message):
    for key, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            raise ValueError(f"{message}: Tensor '{key}' contains NaN values.")