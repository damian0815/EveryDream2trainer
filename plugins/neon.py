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

class NEONPlugin(BasePlugin):
    """
    Implementation of Sina Alemohammad, Zhangyang Wang, Richard G. Baraniuk,
    Neon: Negative Extrapolation From Self-Training Improves Image Generation
    """

    def __init__(self):
        logging.info("NEON plugin instantiated")
        self.log_folder = None
        self.storage_dir = None


    @property
    def base_unet_state_dict_path(self):
        return os.path.join(self.storage_dir, "base_unet_state_dict.pt")


    def on_training_start(
        self,
        ed_state: EveryDreamTrainingState,
        log_folder: str,
        **kwargs
    ):
        self.storage_dir = os.path.join(log_folder, 'neon_plugin_storage')
        logging.info(f" - NEON plugin: Storage directory: {self.storage_dir}")
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save initial model states
        try:
            with open(self.base_unet_state_dict_path, 'wb') as f:
                torch.save(ed_state.model.unet.state_dict(), f)
            logging.info(f" - NEON plugin: Initial model states saved\n")
        except Exception as e:
            logging.error(f" - NEON plugin: Failed to save initial model states: {e}")
            raise

    def on_training_end(self, ed_state: EveryDreamTrainingState, **kwargs):
        alpha = 1.0
        logging.info(f" - NEON plugin: Training complete, applying NEON merge with alpha={alpha}")

        try:
            base_sd = torch.load(self.base_unet_state_dict_path)
            trained_sd = ed_state.model.unet.state_dict()
            neon_sd = {}
            for k, v in base_sd.items():
                # NEON interpets the training delta as a vector pointing toward mode collapse;
                # if we apply it in reverse, we move away from mode collapse
                train_delta = trained_sd[k] - base_sd[k]
                neon_sd[k] = base_sd[k] - train_delta * alpha

            ed_state.model.unet.load_state_dict(neon_sd)

            print(f" - NEON plugin Cleaning up {self.storage_dir}...")
            shutil.rmtree(self.storage_dir)
        except Exception as e:
            print(f"  - Warning: NEON plugin: Failed to clean up storage directory {self.storage_dir}: {e}")
            pass

        print(f"{'='*60}\n")
