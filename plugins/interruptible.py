import errno
import logging
import time

import math
import os
import shutil
from plugins.plugins import BasePlugin
from model.training_model import save_model

EVERY_N_MINUTES = 20

class InterruptiblePlugin(BasePlugin):

    def __init__(self):
        self.previous_save_path = None
        self.last_save_time = time.time()
        self.save_interval = EVERY_N_MINUTES * 60  # in seconds
        print(f"InterruptiblePlugin instantiated, saving every {self.save_interval // 60} minutes")

    def on_epoch_start(self, **kwargs):
        print(f"InterruptiblePlugin: {int((self.last_save_time + self.save_interval) - time.time()) // 60} minutes remaining until next save")

    def on_step_end(self, local_step, global_step, epoch, project_name, log_folder, **kwargs):
        if self.last_save_time + self.save_interval < time.time():
            self.last_save_time = time.time()
            ckpt_name = f"rolling-{project_name}-ep{epoch:02}-gs{global_step:05}"
            save_path = os.path.join(log_folder, "ckpts", ckpt_name)
            print(f"InterruptiblePlugin: saving model to {save_path}")
            try:
                save_optimizer = False
                if not save_optimizer:
                    print("NOT saving optimizer")
                save_model(save_path, global_step=global_step, ed_state=kwargs['ed_state'], save_ckpt_dir=None,
                           yaml_name=None, save_ckpt=False, save_full_precision=True, save_optimizer_flag=save_optimizer)
                self._remove_previous()
            except OSError as e:
                save_optimizer = False
                if e.errno == errno.ENOSPC:
                    print("out of disk space for safe save, trying unsafe")
                    shutil.rmtree(save_path, ignore_errors=True)
                    self._remove_previous()
                    save_model(save_path, global_step=global_step, ed_state=kwargs['ed_state'], save_ckpt_dir=None,
                       yaml_name=None, save_ckpt=False, save_full_precision=True, save_optimizer_flag=save_optimizer)
                else:
                    raise
            self.previous_save_path = save_path

    def on_training_end(self, **kwargs):
        print(f"InterruptiblePlugin: cleaning up")
        self._remove_previous()

    def _remove_previous(self):
        if self.previous_save_path is not None:
            shutil.rmtree(self.previous_save_path, ignore_errors=True)
        self.previous_save_path = None
