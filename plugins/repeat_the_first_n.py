import json
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
        self.storage_dir = None
        self.counter = 0
        self.initial_states_saved = False
        self.max_epochs = None
        self.training_text_encoder = False
        self.previous_save_path = None
        self.save_rolling_ckpts = False
        self.generate_samples = True
        self.generate_samples_every_n_cycles = None


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
        self._save_initial_states(kwargs['ed_state'])

        # Initialize counter
        self.counter = 0
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
            self._accumulate_model_state(ed_state,
                                         log_folder=kwargs['log_folder'],
                                         project_name=kwargs['project_name'],
                                         global_step=kwargs['global_step'])

            if self.generate_samples and self.counter % self.generate_samples_every_n_cycles == 0:
                self._apply_accumulated_states_inplace(ed_state)
                kwargs['sample_generator_cb'](global_step=kwargs['global_step'], batch=None)

            # Only reset if this is not the final epoch
            if epoch + 1 < self.max_epochs:
                print(f"Resetting model to initial state for next cycle...")
                training_variables = kwargs.get('training_variables')
                self._reset_to_initial_state(ed_state, training_variables)
                print(f"Model reset complete. Starting cycle {self.counter}\n")


    def on_training_end(self, **kwargs):
        print(f"\n{'='*60}")
        print(f"RepeatTheFirstN: Training complete, computing final averaged model")
        print(f"{'='*60}")

        if self.counter == 0:
            print("Warning: No model states were accumulated. Skipping final save.")
            return

        ed_state = kwargs['ed_state']

        # Load accumulated states and divide by counter to get average
        print(f"Averaging {self.counter} accumulated model states...")
        self._apply_accumulated_states_inplace(ed_state)

        try:
            print(f"Cleaning up...")
            self._remove_previous_save()
            shutil.rmtree(self.storage_dir)
        except Exception as e:
            print(f"  - Warning: Failed to clean up storage directory {self.storage_dir}: {e}")
            pass

        print(f"{'='*60}\n")


    def _save_initial_states(self, ed_state):
        """Save the initial model state dicts to disk."""
        print("Saving initial model states...")

        # Check if text encoder is being trained
        self.training_text_encoder = (hasattr(ed_state.optimizer, 'optimizer_te') and
                                      ed_state.optimizer.optimizer_te is not None)

        # Save initial states
        unet_path = os.path.join(self.storage_dir, "initial_unet_state_dict.safetensors")
        safetensors.torch.save_file(ed_state.model.unet.state_dict(), unet_path)
        optimizer_unet_path = os.path.join(self.storage_dir, "initial_optimizer_unet.pt")
        torch.save(ed_state.optimizer.optimizer_unet.state_dict(), optimizer_unet_path)

        # Save text_encoder only if training it
        if self.training_text_encoder:
            te_path = os.path.join(self.storage_dir, "initial_text_encoder_state_dict.safetensors")
            safetensors.torch.save_file(ed_state.model.text_encoder.state_dict(), te_path)
            optimizer_te_path = os.path.join(self.storage_dir, "initial_optimizer_te.pt")
            torch.save(ed_state.optimizer.optimizer_te.state_dict(), optimizer_te_path)

            # Save text_encoder_2 if it exists (SDXL)
            if ed_state.model.text_encoder_2 is not None:
                te2_path = os.path.join(self.storage_dir, "initial_text_encoder_2_state_dict.safetensors")
                safetensors.torch.save_file(ed_state.model.text_encoder_2.state_dict(), te2_path)
                optimizer_te2_path = os.path.join(self.storage_dir, "initial_optimizer_te_2.pt")
                torch.save(ed_state.optimizer.optimizer_te_2.state_dict(), optimizer_te2_path)

        # Save initial LR scheduler states
        lr_schedulers_path = os.path.join(self.storage_dir, "initial_lr_schedulers.pt")
        lr_scheduler_states = [scheduler.state_dict() for scheduler in ed_state.optimizer.lr_schedulers]
        torch.save(lr_scheduler_states, lr_schedulers_path)

        self.initial_states_saved = True
        print(f"  - Text encoder training: {'enabled' if self.training_text_encoder else 'disabled'}")

    def _accumulate_model_state(self, ed_state, log_folder, project_name, global_step):
        """Accumulate the current model state into running sum."""
        print(f"Accumulating model state (count: {self.counter + 1})...")

        # Get current model states
        current_unet_state_dict = ed_state.model.unet.state_dict()

        # Only get text encoder states if training them
        if self.training_text_encoder:
            current_te_state_dict = ed_state.model.text_encoder.state_dict()
            current_te2_state_dict = ed_state.model.text_encoder_2.state_dict() if ed_state.model.text_encoder_2 is not None else None
        else:
            current_te_state_dict = None
            current_te2_state_dict = None

        # Paths for accumulated states
        acc_unet_path = os.path.join(self.storage_dir, "accumulated_unet_state_dict.safetensors")
        acc_te_path = os.path.join(self.storage_dir, "accumulated_text_encoder_state_dict.safetensors")
        acc_te2_path = os.path.join(self.storage_dir, "accumulated_text_encoder_2_state_dict.safetensors")

        accumulated_te_state_dict = None
        accumulated_te2_state_dict = None

        # Load and add, or initialize if first time
        if self.counter == 0:
            # First accumulation - just save current state
            accumulated_unet_state_dict = current_unet_state_dict
            if self.training_text_encoder:
                accumulated_te_state_dict = current_te_state_dict
                if current_te2_state_dict is not None:
                    accumulated_te2_state_dict = current_te2_state_dict
        else:
            # Load existing accumulated states and add current states
            accumulated_unet_state_dict = safetensors.torch.load_file(acc_unet_path)

            # Add current to accumulated (element-wise)
            for key in accumulated_unet_state_dict.keys():
                accumulated_unet_state_dict[key] = accumulated_unet_state_dict[key] + current_unet_state_dict[key].to('cpu')

            # Handle text encoders only if training them
            if self.training_text_encoder:
                accumulated_te_state_dict = safetensors.torch.load_file(acc_te_path)
                for key in accumulated_te_state_dict.keys():
                    accumulated_te_state_dict[key] = accumulated_te_state_dict[key] + current_te_state_dict[key].to('cpu')

                # Handle text_encoder_2 if present
                if current_te2_state_dict is not None:
                    accumulated_te2_state_dict = safetensors.torch.load_file(acc_te2_path)
                    for key in accumulated_te2_state_dict.keys():
                        accumulated_te2_state_dict[key] = accumulated_te2_state_dict[key] + current_te2_state_dict[key].to('cpu')


        # Save updated accumulated states
        safetensors.torch.save_file(accumulated_unet_state_dict, acc_unet_path)
        if accumulated_te_state_dict:
            safetensors.torch.save_file(accumulated_te_state_dict, acc_te_path)
        if accumulated_te2_state_dict:
            safetensors.torch.save_file(accumulated_te2_state_dict, acc_te2_path)

        # also save ckpt
        if self.save_rolling_ckpts:
            ckpt_name = f"rolling_{project_name}_repeat_the_first_{self.n_epochs}_epochs_{self.counter}_times"

            save_path = os.path.join(log_folder, "ckpts", ckpt_name)
            print(f"Saving model to: {save_path}")
            self._save(ed_state, save_path=save_path, global_step=global_step)

            self._remove_previous_save()
            self.previous_save_path = save_path

        self.counter += 1
        self._save_counter()
        print(f"Accumulation complete. Total accumulated: {self.counter}")

    def _reset_to_initial_state(self, ed_state, training_variables: TrainingVariables):
        """Reset model, optimizer, and training variables to initial states."""
        # Load initial model states
        unet_path = os.path.join(self.storage_dir, "initial_unet_state_dict.safetensors")
        initial_unet = safetensors.torch.load_file(unet_path)
        ed_state.model.unet.load_state_dict(initial_unet)

        # Only reset text encoders if training them
        if self.training_text_encoder:
            te_path = os.path.join(self.storage_dir, "initial_text_encoder_state_dict.safetensors")

            initial_te = safetensors.torch.load_file(te_path)
            ed_state.model.text_encoder.load_state_dict(initial_te)

            if ed_state.model.text_encoder_2 is not None:
                te2_path = os.path.join(self.storage_dir, "initial_text_encoder_2_state_dict.safetensors")
                initial_te2 = safetensors.torch.load_file(te2_path)
                ed_state.model.text_encoder_2.load_state_dict(initial_te2)

        # Reset optimizer states
        optimizer_unet_path = os.path.join(self.storage_dir, "initial_optimizer_unet.pt")
        initial_opt_unet = torch.load(optimizer_unet_path, weights_only=True)
        ed_state.optimizer.optimizer_unet.load_state_dict(initial_opt_unet)

        if self.training_text_encoder:
            optimizer_te_path = os.path.join(self.storage_dir, "initial_text_encoder_2.pt")
            if os.path.exists(optimizer_te_path):
                initial_opt_te = torch.load(optimizer_te_path, weights_only=True)
                ed_state.optimizer.optimizer_te.load_state_dict(initial_opt_te)
            if ed_state.optimizer.optimizer_te_2 is not None:
                optimizer_te2_path = os.path.join(self.storage_dir, "initial_optimizer_te_2.pt")
                if os.path.exists(optimizer_te2_path):
                    initial_opt_te2 = torch.load(optimizer_te2_path, weights_only=True)
                    ed_state.optimizer.optimizer_te_2.load_state_dict(initial_opt_te2)


        # Reset LR schedulers
        lr_schedulers_path = os.path.join(self.storage_dir, "initial_lr_schedulers.pt")
        if os.path.exists(lr_schedulers_path):
            initial_lr_scheduler_states = torch.load(lr_schedulers_path, weights_only=True)
            for scheduler, initial_state in zip(ed_state.optimizer.lr_schedulers, initial_lr_scheduler_states):
                scheduler.load_state_dict(initial_state)
            print("  - LR schedulers reset")

        # Reset training variables (clears loss accumulation, etc.)
        training_variables.reset()
        print("  - Training variables reset (loss accumulation cleared)")
        optimizer_te_path = os.path.join(self.storage_dir, "initial_optimizer_te.pt")
        if ed_state.optimizer.optimizer_te is not None:
            initial_opt_te = torch.load(optimizer_te_path, weights_only=True)
            ed_state.optimizer.optimizer_te.load_state_dict(initial_opt_te)
        if ed_state.optimizer.optimizer_te_2 is not None:
            optimizer_te2_path = os.path.join(self.storage_dir, "initial_optimizer_te_2.pt")
            if os.path.exists(optimizer_te2_path):
                initial_opt_te2 = torch.load(optimizer_te2_path, weights_only=True)
                ed_state.optimizer.optimizer_te_2.load_state_dict(initial_opt_te2)

        # Reset training variables (clears loss accumulation, etc.)
        if training_variables is not None:
            training_variables.reset()
            print("  - Training variables reset (loss accumulation cleared)")

    def _load_and_average_accumulated_states(self):
        """Load accumulated states and divide by counter to get average."""
        acc_unet_path = os.path.join(self.storage_dir, "accumulated_unet_state_dict.safetensors")

        # Load accumulated states
        accumulated_unet = safetensors.torch.load_file(acc_unet_path)

        # Divide by counter to get average
        for key in accumulated_unet.keys():
            accumulated_unet[key] = accumulated_unet[key] / self.counter

        result = {
            'unet': accumulated_unet
        }

        # Handle text encoders only if training them
        if self.training_text_encoder:
            acc_te_path = os.path.join(self.storage_dir, "accumulated_text_encoder_state_dict.safetensors")
            acc_te2_path = os.path.join(self.storage_dir, "accumulated_text_encoder_2_state_dict.safetensors")

            accumulated_te = safetensors.torch.load_file(acc_te_path)
            for key in accumulated_te.keys():
                accumulated_te[key] = accumulated_te[key] / self.counter
            result['text_encoder'] = accumulated_te

            # Handle text_encoder_2 if present
            if os.path.exists(acc_te2_path):
                accumulated_te2 = safetensors.torch.load_file(acc_te2_path)
                for key in accumulated_te2.keys():
                    accumulated_te2[key] = accumulated_te2[key] / self.counter
                result['text_encoder_2'] = accumulated_te2

        return result

    def _apply_accumulated_states_inplace(self, ed_state):
        """Apply the averaged accumulated states to the model."""
        averaged_states = self._load_and_average_accumulated_states()

        # Load averaged states into the model
        ed_state.model.unet.load_state_dict(averaged_states['unet'])

        # Only load text encoder states if they were trained
        if self.training_text_encoder:
            if 'text_encoder' in averaged_states:
                ed_state.model.text_encoder.load_state_dict(averaged_states['text_encoder'])
            if 'text_encoder_2' in averaged_states and ed_state.model.text_encoder_2 is not None:
                ed_state.model.text_encoder_2.load_state_dict(averaged_states['text_encoder_2'])


    def _save_counter(self):
        """Save the counter to disk."""
        counter_path = os.path.join(self.storage_dir, "counter.txt")
        with open(counter_path, 'w') as f:
            f.write(str(self.counter))


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
