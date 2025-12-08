"""
Copyright [2022-2023] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import itertools
import os
from functools import partial
from itertools import chain
from typing import List, Tuple, Generator

import math
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig

from torch.cuda.amp import GradScaler
from diffusers.optimization import get_scheduler

from colorama import Fore, Style
import pprint

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from plugins.plugins import PluginRunner
import re


BETAS_DEFAULT = [0.9, 0.999]
EPSILON_DEFAULT = 1e-8
WEIGHT_DECAY_DEFAULT = 0.01
LR_DEFAULT = 1e-6
OPTIMIZER_TE_STATE_FILENAME = "optimizer_te.pt"
OPTIMIZER_UNET_STATE_FILENAME = "optimizer_unet.pt"
SCALER_STATE_FILENAME = 'scaler.pt'


class InfOrNanException(Exception):
    pass


class EveryDreamOptimizer:
    """
    Wrapper to manage optimizers
    resume_ckpt_path: path to resume checkpoint, will try to load state (.pt) files if they exist
    optimizer_config: config for the optimizers
    text_encoder: text encoder model parameters
    unet: unet model parameters
    """
    def __init__(self, args, optimizer_config, model: 'TrainingModel', epoch_len, plugin_runner: PluginRunner, log_writer=None):
        if optimizer_config is None:
            raise ValueError("missing optimizer_config")
        if "doc" in optimizer_config:
            del optimizer_config["doc"]
        print("\n raw optimizer_config:")
        pprint.pprint(optimizer_config)
        self.epoch_len = epoch_len
        self.max_epochs = args.max_epochs
        self.unet = model.unet # needed for weight norm logging, unet.parameters() has to be called again, Diffusers quirk
        self.text_encoder = model.text_encoder
        self.text_encoder_2 = model.text_encoder_2
        self.log_writer = log_writer
        self.te_config, self.base_config = self.get_final_optimizer_configs(args, optimizer_config)
        self.te_freeze_config = optimizer_config.get("text_encoder_freezing", {})
        self.unet_component_lr_config = optimizer_config.get("unet_component_lr", {})
        print(" Final unet optimizer config:")
        pprint.pprint(self.base_config)
        print(" Final text encoder optimizer config:")
        pprint.pprint(self.te_config)
        if self.unet_component_lr_config:
            print(" UNet component-specific learning rates:")
            pprint.pprint(self.unet_component_lr_config)

        self.grad_accum = args.grad_accum
        self.next_grad_accum_step = self.grad_accum
        self.clip_grad_norm = args.clip_grad_norm
        self.apply_grad_scaler_step_tweaks = optimizer_config.get("apply_grad_scaler_step_tweaks", True)

        if 'use_grad_scaler' in optimizer_config:
            logging.warning("* Ignoring use_grad_scaler entry in optimizer config, will be set automatically")
        if model.unet.dtype == torch.bfloat16:
            self.use_grad_scaler = False
            logging.info("* bfloat16 unet: no grad scaler")
        elif model.unet.dtype == torch.float16:
            self.use_grad_scaler = True
            logging.info("* float16 unet: using grad scaler")
        elif self.base_config['optimizer'].endswith('8bit') or self.te_config['optimizer'].endswith('8bit'):
            self.use_grad_scaler = True
            logging.info("* 8bit optimizer: using grad scaler")
        elif model.unet.dtype != torch.float32:
            self.use_grad_scaler = True
            logging.info(f"* unet dtype = {model.unet.dtype}: using grad scaler")
        else:
            self.use_grad_scaler = False
            logging.info("* no grad scaler")

        self.log_grad_norm = optimizer_config.get("log_grad_norm", True)

        if args.lora:
            self.text_encoder_params, self.unet_params = self.setup_lora_training(args, model)
        else:
            self.text_encoder_params = {'default': list(itertools.chain([
                self._apply_text_encoder_freeze(model.text_encoder),
                self._apply_text_encoder_freeze(model.text_encoder_2) if model.text_encoder_2 is not None else []
            ]))}
            self.unet_params = self._apply_unet_freeze(
                model.unet,
                unet_freeze_config=optimizer_config.get("unet_freezing", {}),
                unet_component_lr_config=self.unet_component_lr_config,
                cross_attention_dim_to_find=2048 if model.is_sdxl else model.text_encoder.config.hidden_size
            )

        if args.jacobian_descent:
            from torchjd.aggregation import UPGrad
            import torchjd
            self.jacobian_aggregator = UPGrad()
            self.jacobian_backward = torchjd.backward
        else:
            self.jacobian_aggregator = None

        self.text_encoder_params, _ = plugin_runner.run_add_parameters(self.text_encoder_params, [])
        self.text_encoder_params = list(self.text_encoder_params)

        #with torch.no_grad():
        #    log_action = lambda n, label: logging.info(f"{Fore.LIGHTBLUE_EX} {label} weight normal: {n:.1f}{Style.RESET_ALL}")
        #    self._log_weight_normal(text_encoder.text_model.encoder.layers.parameters(), "text encoder", log_action)
        #    self._log_weight_normal(unet.parameters(), "unet", log_action)

        self.optimizers = []
        self.optimizer_te_2 = None  # todo: support 2nd text encoder optimizer
        self.optimizer_te, self.optimizer_unet = self.create_optimizers(args,
                                                                        self.text_encoder_params,
                                                                        self.unet_params)
        self.optimizers.append(self.optimizer_te) if self.optimizer_te is not None else None
        self.optimizers.append(self.optimizer_unet) if self.optimizer_unet is not None else None

        self.lr_schedulers = []
        schedulers = self.create_lr_schedulers(args, optimizer_config)
        self.lr_schedulers.extend(schedulers)

        if args.amp and self.use_grad_scaler:
            self.scaler = GradScaler(
                enabled=args.amp,
                init_scale=args.init_grad_scale or 2**17.5,
                growth_factor=2,
                backoff_factor=1.0/2,
                growth_interval=25,
            )
        else:
            self.scaler = None

        self.load(args.resume_ckpt)

        logging.info(f" Grad scaler enabled: {self.scaler is not None and self.scaler.is_enabled()} (amp mode)")

    def _log_gradient_normal(self, parameters: Generator, label: str, log_action=None):
        total_norm = self._get_norm([p for p in parameters if p.grad is not None], lambda p: p.grad.data)
        log_action(total_norm, label)

    def _log_weight_normal(self, parameters: Generator, label: str, log_action=None):
        total_norm = self._get_norm(parameters, lambda p: p.data)
        log_action(total_norm, label)

    def _get_norm(self, parameters: Generator, param_type):
        total_norm = 0
        for p in parameters:
            param = param_type(p)
            total_norm += self._calculate_norm(param, p)
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _calculate_norm(self, param, p):
        if param is not None:
            return param.norm(2).item() ** 2
        else:
            return 0.0

    def _should_do_grad_accum_step(self, step, global_step):
        return (
            (global_step + 1) % self.grad_accum == 0
            or ((global_step // self.epoch_len) == self.max_epochs-1 and (step == self.epoch_len - 1))
        )

    def will_do_grad_accum_step(self, step, global_step):
        return self._should_do_grad_accum_step(step, global_step)

    def backward(self, loss: torch.Tensor | list[torch.Tensor]):
        loss_scaled_if_necessary = loss if self.scaler is None else self.scaler.scale(loss)
        if self.jacobian_aggregator is not None:
            self.jacobian_backward(
                tensors=loss_scaled_if_necessary,
                inputs=itertools.chain(self.text_encoder_params, self.unet_params),
                A=self.jacobian_aggregator,
                parallel_chunk_size=None
            )
        else:
            loss_scaled_if_necessary.backward()

    def register_unet_nan_hooks_full(self):
        def detailed_nan_hook(module, grad_input, grad_output, name: str='unknown'):
            module_name = f'{name}({module.__class__.__name__})'

            # Check what's coming FROM the next layer (flowing backward)
            grad_out_has_nan_or_inf = False
            if grad_output is not None:
                for i, grad in enumerate(grad_output):
                    if grad is not None and (
                        torch.isnan(grad).any() or torch.isinf(grad).any()
                    ):
                        grad_out_has_nan_or_inf = True
                        print(
                            f"  ⬅️  NaN/inf in grad_output[{i}] flowing INTO {module_name}"
                        )
                        print(
                            f"      shape: {grad.shape}, has NaN: {torch.isnan(grad).sum().item()}/{grad.numel()} inf: {torch.isinf(grad).sum().item()}/{grad.numel()}"
                        )

            # Check what THIS layer produces (flowing backward to previous layer)
            grad_in_has_nan_or_inf = False
            if grad_input is not None:
                for i, grad in enumerate(grad_input):
                    if grad is not None and (
                        torch.isnan(grad).any() or torch.isinf(grad).any()
                    ):
                        # This layer CREATED the NaN if input is clean but output has NaN
                        # Only log this once
                        if not grad_in_has_nan_or_inf and not grad_out_has_nan_or_inf:
                            print(
                                f"🔴 ORIGIN: {module_name} created NaN/inf during its backward pass!"
                            )
                        grad_in_has_nan_or_inf = True

                        print(
                            f"  ➡️  NaN/inf in grad_input[{i}] flowing OUT OF {module_name}"
                        )
                        print(
                            f"      shape: {grad.shape}, has NaN: {torch.isnan(grad).sum().item()}/{grad.numel()} inf: {torch.isinf(grad).sum().item()}/{grad.numel()}"
                        )


        for name, module in self.unet.named_modules():
            if type(module) is UNet2DConditionModel:
                continue
            #print(f"registering detailed NaN hook for {name} ({module.__class__.__name__})")
            # pass name to hook by making a partial
            hook_with_module_name = partial(detailed_nan_hook, name=name)
            module.register_full_backward_hook(hook_with_module_name)


    def register_unet_nan_hooks_simple(self):
        # Register hooks to catch where NaN first appears
        def check_nan_hook(module, grad_input, grad_output, name):
            for i, grad in enumerate(grad_output):
                if grad is not None and (
                    torch.isnan(grad).any() or torch.isinf(grad).any()
                ):
                    print(f"NaN/Inf detected after {name}({module.__class__.__name__}) backward (hooked)")
        for name, module in self.unet.named_modules():
            if type(module) is not UNet2DConditionModel:
                continue
            module.register_backward_hook(partial(check_nan_hook, name=name))

    def unet_grads_have_inf_or_nan(self, log_hint: str) -> bool:
        # check for inf/nan in unet gradients
        has_inf_or_nan = False
        for name, p in self.unet.named_parameters():
            if p.grad is not None:
                grad_data = p.grad.data
                if torch.isinf(grad_data).any() or torch.isnan(grad_data).any():
                    logging.error(f"** {log_hint}: inf or NaN detected in UNet gradient for parameter: {name}. min: {grad_data.min()}, max: {grad_data.max()}")
                    has_inf_or_nan = True
        return has_inf_or_nan

    def check_for_inf_or_nan(self, tv: 'TrainingVariables', log_hint: str):
        if self.unet_grads_have_inf_or_nan(log_hint):
            logging.error("Dumping training vars:")
            logging.error(f" - Timesteps: {tv.accumulated_timesteps}")
            logging.error(f" - Paths: {tv.accumulated_pathnames}")
            logging.error(f" - Captions: {tv.accumulated_captions}")
            logging.error(f" - Accumulated loss: {tv.accumulated_loss}")
            logging.error(f"Global step: {tv.global_step}")
            raise InfOrNanException("** inf or NaN detected in UNet gradients")


    def step_optimizer(self, global_step, tv: 'TrainingVariables', log_hint: str=''):
        if self.scaler is not None:
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

        if self.log_grad_norm:
            with torch.no_grad():
                self.log_writer.add_scalar("optimizer/unet_grad_norm_pre_clip", _get_grad_norm(self.unet.parameters()), global_step)
                self.log_writer.add_scalar("optimizer/te_grad_norm_pre_clip", _get_grad_norm(self.text_encoder.parameters()), global_step)
                if self.text_encoder_2 is not None:
                    self.log_writer.add_scalar("optimizer/te2_grad_norm_pre_clip", _get_grad_norm(self.text_encoder_2.parameters()),global_step,)

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(parameters=self.unet.parameters(), max_norm=self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder.parameters(), max_norm=self.clip_grad_norm)
            if self.text_encoder_2 is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.text_encoder_2.parameters(),
                    max_norm=self.clip_grad_norm,
                )
            if self.log_grad_norm:
                with torch.no_grad():
                    self.log_writer.add_scalar("optimizer/unet_grad_norm_post_clip", _get_grad_norm(self.unet.parameters()), global_step)
                    self.log_writer.add_scalar("optimizer/te_grad_norm_post_clip", _get_grad_norm(self.text_encoder.parameters()), global_step)
                    if self.text_encoder_2 is not None:
                        self.log_writer.add_scalar("optimizer/te2_grad_norm_post_clip", _get_grad_norm(self.text_encoder_2.parameters()), global_step)

        if self.log_grad_norm:
            exp_avg, exp_avg_sq = _get_moment_norms(self.optimizer_unet)
            self.log_writer.add_scalar(
                "optimizer/unet_adamw_exp_avg", exp_avg, global_step
            )
            self.log_writer.add_scalar(
                "optimizer/unet_adamw_exp_avg_sq", exp_avg_sq, global_step
            )

        if self.scaler is None:
            for optimizer in self.optimizers:
                optimizer.step()
        else:
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()

        if self.log_grad_norm and self.log_writer:
            log_action = lambda n, label: self.log_writer.add_scalar(label, n, global_step)
            with (torch.no_grad()):
                self._log_gradient_normal(self.unet.parameters(), "optimizer/unet_grad_norm", log_action)
                self._log_gradient_normal(self.text_encoder.parameters(), "optimizer/te_grad_norm",
                                          log_action)
                self._log_weight_normal(self.unet.parameters(), "optimizer/unet_weight_norm", log_action)
                self._log_weight_normal(self.text_encoder.parameters(), "optimizer/te_weight_norm", log_action)
                if self.text_encoder_2 is not None:
                    self._log_gradient_normal(
                        self.text_encoder_2.parameters(),
                        "optimizer/te2_grad_norm",
                        log_action,
                    )
                    self._log_weight_normal(self.text_encoder_2.parameters(), "optimizer/te2_weight_norm", log_action)

        self._zero_grad(set_to_none=True)


    def step_schedulers(self, global_step):
        for scheduler in self.lr_schedulers:
            scheduler.step()

        if self.apply_grad_scaler_step_tweaks:
            self._update_grad_scaler(global_step)

    def _zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_scale(self):
        if self.scaler is None:
            return 1
        return self.scaler.get_scale()
    
    def get_unet_lr(self):
        """
        Returns dict of learning rates keyed by tag name.
        Single param group: {"lr unet": <value>}
        Multiple param groups: {"lr unet <group_name>": <value>, ...}
        """
        if self.optimizer_unet is None:
            return {}

        param_groups = self.optimizer_unet.param_groups
        if len(param_groups) == 1:
            # Single param group - use simple key for backward compatibility
            return {"lr unet": param_groups[0]['lr']}
        else:
            # Multiple param groups - include group name in key
            return {
                f"lr unet {pg.get('name', f'group_{idx}')}": pg['lr']
                for idx, pg in enumerate(param_groups)
            }

    def get_textenc_lr(self):
        return self.optimizer_te.param_groups[0]['lr'] if self.optimizer_te is not None else 0
    
    def save(self, ckpt_path: str):
        """
        Saves the optimizer states to path
        """
        self._save_optimizer(self.optimizer_te, os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME), optimizer_type=self.te_config['optimizer']) if self.optimizer_te is not None else None
        self._save_optimizer(self.optimizer_unet, os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME), optimizer_type=self.base_config['optimizer']) if self.optimizer_unet is not None else None
        self._save_optimizer(self.scaler, os.path.join(ckpt_path, SCALER_STATE_FILENAME), 'scaler') if self.scaler is not None else None


    def load(self, ckpt_path: str):
        """
        Loads the optimizer states from path
        """
        te_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)
        unet_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)
        scaler_state_path = os.path.join(ckpt_path, SCALER_STATE_FILENAME)
        if os.path.exists(te_optimizer_state_path) and self.optimizer_te is not None:
            self._load_optimizer(self.optimizer_te, te_optimizer_state_path, expected_type=self.te_config['optimizer']) if self.optimizer_te is not None else None
        if os.path.exists(unet_optimizer_state_path) and self.optimizer_unet is not None:
            self._load_optimizer(self.optimizer_unet, unet_optimizer_state_path, expected_type=self.base_config['optimizer']) if self.optimizer_unet is not None else None
        if os.path.exists(scaler_state_path) and self.scaler is not None:
            self._load_optimizer(self.scaler, scaler_state_path, expected_type='scaler')

    def create_optimizers(self, args, text_encoder_params: dict[str, list], unet_params: dict[str, list]):
        """
        creates optimizers from config and args for unet and text encoder
        returns (optimizer_te, optimizer_unet)
        """

        if args.disable_textenc_training:
            optimizer_te = None
        else:
            optimizer_te = self._create_optimizer("text encoder", args, self.te_config, text_encoder_params)
        if args.disable_unet_training:
            optimizer_unet = None
        else:
            optimizer_unet = self._create_optimizer("unet", args, self.base_config, unet_params)

        return optimizer_te, optimizer_unet

    def get_final_optimizer_configs(self, args, global_optimizer_config):
        """
        defaults and overrides based on priority
        cli LR arg will override LR for both unet and text encoder for legacy reasons
        """
        base_config = global_optimizer_config.get("base")
        te_config = global_optimizer_config.get("text_encoder_overrides", {})

        if args.lr_decay_steps is None or args.lr_decay_steps < 1:
            # sets cosine so the zero crossing is past the end of training, this results in a terminal LR that is about 25% of the nominal LR
            total_steps = self.epoch_len * args.max_epochs
            if args.max_steps is not None:
                total_steps = min(args.max_steps, total_steps)
            args.lr_decay_steps = int(total_steps * args.auto_decay_steps_multiplier)
            print('total_steps:', total_steps, ' -> lr_decay_steps:', args.lr_decay_steps)

        if args.lr_warmup_steps is None:
            # set warmup to 2% of decay, if decay was autoset to 150% of max epochs then warmup will end up about 3% of max epochs
            args.lr_warmup_steps = int(args.lr_decay_steps / 50)

        if args.lr_advance_steps is None:
            args.lr_advance_steps = 0

        if args.lr is not None:
            # override for legacy support reasons
            base_config["lr"] = args.lr

        base_config["lr_end"] = base_config.get("lr_end", args.lr_end) or base_config["lr"]/100.0

        base_config["optimizer"] = base_config.get("optimizer", None) or "adamw8bit"

        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler
        base_config["lr_warmup_steps"] = base_config.get("lr_warmup_steps", None) or args.lr_warmup_steps
        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_advance_steps"] = base_config.get("lr_advance_steps", None) or args.lr_advance_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler
        base_config["lr_num_restarts"] = base_config.get("lr_num_restarts", None) or args.lr_num_restarts

        te_config["lr"] = te_config.get("lr", None) or base_config["lr"]
        te_config["optimizer"] = te_config.get("optimizer", None) or base_config["optimizer"]
        te_config["lr_scheduler"] = te_config.get("lr_scheduler", None) or base_config["lr_scheduler"]
        te_config["lr_warmup_steps"] = te_config.get("lr_warmup_steps", None) or base_config["lr_warmup_steps"]
        te_config["lr_advance_steps"] = te_config.get("lr_advance_steps", None) or base_config["lr_advance_steps"]
        te_config["lr_decay_steps"] = te_config.get("lr_decay_steps", None) or base_config["lr_decay_steps"]
        te_config["weight_decay"] = te_config.get("weight_decay", None) or base_config["weight_decay"]
        te_config["betas"] = te_config.get("betas", None) or base_config["betas"]
        te_config["epsilon"] = te_config.get("epsilon", None) or base_config["epsilon"]

        return te_config, base_config

    def create_lr_schedulers(self, args, optimizer_config):
        unet_config = optimizer_config["base"]
        te_config = optimizer_config.get("text_encoder_overrides", {})

        ret_val = []

        if self.optimizer_te is not None:
            lr_scheduler = get_scheduler(
                te_config.get("lr_scheduler", args.lr_scheduler),
                optimizer=self.optimizer_te,
                num_warmup_steps=int(te_config.get("lr_warmup_steps", None) or unet_config.get("lr_warmup_steps",0)),
                num_training_steps=int(te_config.get("lr_decay_steps", None) or unet_config.get("lr_decay_steps",1e9))
            )
            ret_val.append(lr_scheduler)

        if self.optimizer_unet is not None:
            unet_config = optimizer_config["base"]

            if unet_config["lr_scheduler"] == "polynomial":
                num_restarts = unet_config.get("lr_num_restarts", 1)
                if num_restarts != 1 and args.auto_decay_steps_multiplier != 1:
                    raise ValueError("Cannot use lr_num_restarts != 1 with --auto_decay_steps_multiplier != 1 when using polynomial LR scheduler")
                if num_restarts < 1:
                    raise ValueError("Must have >=1 (re)starts for polynomial LR scheduler")
                lr_scheduler = _get_polynomial_decay_schedule_with_warmup_adj(
                    optimizer=self.optimizer_unet,
                    lr_end=unet_config.get("lr_end", unet_config["lr"]/100.0),
                    power=unet_config.get("power", 2),
                    num_cycles=num_restarts,
                    num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                    num_training_steps=int(unet_config["lr_decay_steps"]),
                )
            elif unet_config["lr_scheduler"] == "findlr":
                lr_scheduler = _get_findlr_scheduler(
                    optimizer=self.optimizer_unet,
                    base_lr=unet_config.get("lr", unet_config["lr"] * 10.0),
                    min_lr=unet_config.get("lr_end", unet_config["lr"]/100.0),
                    num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                )
            else:
                lr_scheduler = get_scheduler(
                    unet_config["lr_scheduler"],
                    optimizer=self.optimizer_unet,
                    num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                    num_training_steps=int(unet_config["lr_decay_steps"]),
                    num_cycles=int(unet_config.get("lr_num_restarts", 3)),
                )
            for i in range(unet_config["lr_advance_steps"]):
                lr_scheduler.step()
            ret_val.append(lr_scheduler)
        return ret_val

    def _update_grad_scaler(self, global_step):
        if self.scaler is None:
            return
        if global_step == 500:
            factor = 1.8
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(100)
        if global_step == 1000:
            factor = 1.6
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(200)
        if global_step == 2000:
            factor = 1.3
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(500)
        if global_step == 4000:
            factor = 1.15
            self.scaler.set_growth_factor(factor)
            self.scaler.set_backoff_factor(1/factor)
            self.scaler.set_growth_interval(2000)

    @staticmethod
    def _save_optimizer(optimizer, path: str, optimizer_type: str):
        """
        Saves the optimizer state to specific path/filename
        """
        try:
            torch.save({
                'optimizer_type': optimizer_type,
                'state_dict': optimizer.state_dict()
            }, path)
        except RuntimeError:
            logging.warning(f"  Saving optimizer state to {path} failed, deleting")
            if os.path.exists(path):
                os.unlink(path)


    @staticmethod
    def _load_optimizer(optimizer: torch.optim.Optimizer, path: str, expected_type: str=None):
        """
        Loads the optimizer state to an Optimizer object
        optimizer: torch.optim.Optimizer
        path: .pt file
        """
        try:
            state_dict = torch.load(path)
            if 'optimizer_type' in state_dict:
                optimizer_type = state_dict['optimizer_type']
                if expected_type is not None and optimizer_type != expected_type:
                    logging.warning(f"{Fore.LIGHTYELLOW_EX}**Loaded optimizer type in {path} is {optimizer_type} but we expect {expected_type} - skipping optimizer load{Style.RESET_ALL}")
                    return
                state_dict = state_dict["state_dict"]
            optimizer.load_state_dict(state_dict)
            logging.info(f" Loaded optimizer state from {path}")
        except Exception as e:
            logging.warning(f"{Fore.LIGHTYELLOW_EX}**Failed to load optimizer state from {path}, optimizer state will not be loaded, \n * Exception: {e}{Style.RESET_ALL}")
            pass

    def _create_optimizer(self, label, args, local_optimizer_config, parameters: dict[str, list]):
        """
        parameters is always a dict:
        - Single group: {"default": [(name, param), ...]}
        - Multi-group: {"cross_attention": [...], "self_attention": [...], ...}
        """
        betas = BETAS_DEFAULT
        epsilon = EPSILON_DEFAULT
        weight_decay = WEIGHT_DECAY_DEFAULT
        import bitsandbytes as bnb
        opt_class = bnb.optim.AdamW8bit
        optimizer = None

        default_lr = 1e-6
        curr_lr = args.lr
        d0 = 1e-6 # dadapt
        decouple = True # seems bad to turn off, dadapt_adam only
        momentum = 0.0 # dadapt_sgd
        no_prox = False # ????, dadapt_adan
        use_bias_correction = True # suggest by prodigy github
        growth_rate=float("inf") # dadapt various, no idea what a sane default is
        safeguard_warmup = True # per recommendation from prodigy documentation

        optimizer_name = None
        if local_optimizer_config is not None:
            betas = local_optimizer_config.get("betas", betas)
            epsilon = local_optimizer_config.get("epsilon", epsilon)
            weight_decay = local_optimizer_config.get("weight_decay", weight_decay)
            no_prox =  local_optimizer_config.get("no_prox", False)
            optimizer_name = local_optimizer_config.get("optimizer", "adamw8bit")
            curr_lr = local_optimizer_config.get("lr", curr_lr)
            d0 = local_optimizer_config.get("d0", d0)
            decouple = local_optimizer_config.get("decouple", decouple)
            momentum = local_optimizer_config.get("momentum", momentum)
            growth_rate = local_optimizer_config.get("growth_rate", growth_rate)
            safeguard_warmup = local_optimizer_config.get("safeguard_warmup", safeguard_warmup) 
            if args.lr is not None:
                curr_lr = args.lr
                logging.info(f"Overriding LR from optimizer config with main config/cli LR setting: {curr_lr}")

        if optimizer_name is None or optimizer_name == "adamw8bit":
            if not self.use_grad_scaler:
                logging.warning(
                    f"{Fore.YELLOW}** Using AdamW8bit without grad scaler is not recommended, consider enabling amp mode.{Style.RESET_ALL}"
                )

        if curr_lr is None:
            curr_lr = default_lr
            logging.warning(f"No LR setting found, defaulting to {default_lr}")

        # Check if this is component-specific LR (multi-group) or standard (single group)
        is_component_lr = label == "unet" and len(parameters) > 1 and "cross_attention" in parameters

        param_groups = []

        if is_component_lr:
            # Component-specific LR mode for UNet
            logging.info(f"{Fore.CYAN}=== Using Component-Specific Learning Rates for UNet ==={Style.RESET_ALL}")

            component_configs = {
                "cross_attention": {
                    "lr_scale": self.unet_component_lr_config.get("cross_attention_lr_scale", 1.0),
                    "weight_decay": self.unet_component_lr_config.get("cross_attention_weight_decay", weight_decay),
                },
                "self_attention": {
                    "lr_scale": self.unet_component_lr_config.get("self_attention_lr_scale", 1.0),
                    "weight_decay": self.unet_component_lr_config.get("self_attention_weight_decay", weight_decay),
                },
                "resnet": {
                    "lr_scale": self.unet_component_lr_config.get("resnet_lr_scale", 1.0),
                    "weight_decay": self.unet_component_lr_config.get("resnet_weight_decay", weight_decay),
                },
                "other": {
                    "lr_scale": self.unet_component_lr_config.get("other_lr_scale", 1.0),
                    "weight_decay": self.unet_component_lr_config.get("other_weight_decay", weight_decay),
                }
            }

            for component_name, component_params in parameters.items():
                if component_params:  # Only add non-empty groups
                    config = component_configs[component_name]
                    component_lr = curr_lr * config["lr_scale"]
                    param_groups.append({
                        "params": [p for n, p in component_params],
                        "betas": (betas[0], betas[1]),
                        "weight_decay": config["weight_decay"],
                        "lr": component_lr,
                        "name": component_name,
                    })
                    logging.info(f"{Fore.CYAN}  {component_name}: {len(component_params)} params, LR={component_lr:.2e}, WD={config['weight_decay']}{Style.RESET_ALL}")
        else:
            # Standard mode: single group (dict with one key, usually "default")
            # Get the single group's parameters
            group_name = list(parameters.keys())[0]
            param_list = parameters[group_name]

            attention_weight_decay = local_optimizer_config.get("weight_decay_attn_qk", None)
            if attention_weight_decay is None:
                param_groups = [
                    {
                        "params": [p for n, p in param_list],
                        "betas": (betas[0], betas[1]),
                        "weight_decay": weight_decay,
                        "lr": curr_lr,
                        "name": group_name,
                    }
                ]
            else:
                regular_group, attention_group = _extract_attention_parameter_group(param_list)
                logging.info(f"Using split parameter groups: {len(regular_group)} regular parameters @ weight decay {weight_decay}  , {len(attention_group)} attention parameters @ weight decay {attention_weight_decay}")
                param_groups = [
                    {
                        "params": [p for n, p in regular_group],
                        "betas": (betas[0], betas[1]),
                        "weight_decay": weight_decay,
                        "lr": curr_lr,
                        "name": "regular",
                    },
                    {
                        "params": [p for n, p in attention_group],
                        "betas": (betas[0], betas[1]),
                        "weight_decay": attention_weight_decay,
                        "lr": curr_lr,
                        "name": "attention_high_decay",
                    },
                ]

        if optimizer_name:
            optimizer_name = optimizer_name.lower()

            if optimizer_name == "lion":
                from lion_pytorch import Lion
                opt_class = Lion
                optimizer = opt_class(param_groups)
            elif optimizer_name == "lion8bit":
                from bitsandbytes.optim import Lion8bit
                opt_class = Lion8bit
                for g in param_groups:
                    g.update({"percentile_clipping": 100,
                              "min_8bit_size": 4096})
                optimizer = opt_class(param_groups)

            elif optimizer_name == "prodigy":
                from prodigyopt import Prodigy
                opt_class = Prodigy
                for g in param_groups:
                    g.update({
                        "use_bias_correction": use_bias_correction,
                        "growth_rate": growth_rate,
                        "d0": d0,
                        "safeguard_warmup": safeguard_warmup
                    })
                optimizer = opt_class(param_groups)
            elif optimizer_name == "adamw":
                opt_class = torch.optim.AdamW
            if "dowg" in optimizer_name:
                # coordinate_dowg, scalar_dowg require no additional parameters. Epsilon is overrideable but is unnecessary in all stable diffusion training situations.
                import dowg
                if optimizer_name == "coordinate_dowg":
                    opt_class = dowg.CoordinateDoWG
                elif optimizer_name == "scalar_dowg":
                    opt_class = dowg.ScalarDoWG
                else:
                    raise ValueError(f"Unknown DoWG optimizer {optimizer_name}. Available options are 'coordinate_dowg' and 'scalar_dowg'")
            elif optimizer_name in ["dadapt_adam", "dadapt_lion", "dadapt_sgd"]:
                import dadaptation

                if curr_lr < 1e-4:
                    logging.warning(f"{Fore.YELLOW} LR, {curr_lr}, is very low for Dadaptation.  Consider reviewing Dadaptation documentation, but proceeding anyway.{Style.RESET_ALL}")
                if weight_decay < 1e-3:
                    logging.warning(f"{Fore.YELLOW} Weight decay, {weight_decay}, is very low for Dadaptation.  Consider reviewing Dadaptation documentation, but proceeding anyway.{Style.RESET_ALL}")

                if optimizer_name == "dadapt_adam":
                    opt_class = dadaptation.DAdaptAdam
                    for g in param_groups:
                        g.update({
                            "eps": epsilon, # unused for dadapt_adam
                            "d0": d0,
                            "log_every": args.log_step,
                            "growth_rate": growth_rate,
                            "decouple": decouple,
                        })
                    optimizer = opt_class(param_groups)
                elif optimizer_name == "dadapt_adan":
                    opt_class = dadaptation.DAdaptAdan
                    for g in param_groups:
                        g.update({
                            "no_prox": no_prox,
                            "eps": epsilon,
                            "d0": d0,
                            "log_every": args.log_step,
                            "growth_rate": growth_rate,
                        })
                    optimizer = opt_class(param_groups)
                elif optimizer_name == "dadapt_lion":
                    opt_class = dadaptation.DAdaptLion
                    for g in param_groups:
                        g.update({
                            "eps": epsilon,
                            "d0": d0,
                            "log_every": args.log_step,
                        })
                    optimizer = opt_class(param_groups)
                elif optimizer_name == "dadapt_sgd":
                    opt_class = dadaptation.DAdaptSGD
                    for g in param_groups:
                        g.update({
                            "momentum": momentum,
                            "d0": d0,
                            "log_every": args.log_step,
                            "growth_rate": growth_rate,
                        })
                    optimizer = opt_class(param_groups)
            elif optimizer_name == "adacoor":
                from optimizer.adacoor import AdaCoor

                opt_class = AdaCoor
                for g in param_groups:
                    g.update({
                        "eps": epsilon,
                    })
                optimizer = opt_class(param_groups)

        if not optimizer:
            for g in param_groups:
                g.update({
                    "eps": epsilon,
                    "amsgrad": False,
                })
            optimizer = opt_class(param_groups)

        log_optimizer(label, optimizer, betas, epsilon, weight_decay, curr_lr)
        return optimizer



    def _get_cross_attention_layer_names(self, unet, cross_attention_dim_to_find: int) -> Generator[str, None, None]:
        for name, module in unet.named_modules():
            # Look for cross-attention modules
            if 'attn' in name.lower() and (
                hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v")
            ) and (
                module.cross_attention_dim == cross_attention_dim_to_find
            ):
                yield name

    def _get_self_attention_layer_names(self, unet, cross_attention_dim_to_ignore: int) -> Generator[str, None, None]:
        for name, module in unet.named_modules():
            # Look for cross-attention modules
            if 'attn' in name.lower() and (
                hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v")
            ) and (
                module.cross_attention_dim != cross_attention_dim_to_ignore
            ):
                yield name

    def _apply_unet_freeze(self, unet: UNet2DConditionModel,
                           unet_freeze_config: dict[str, bool],
                           unet_component_lr_config: dict,
                           cross_attention_dim_to_find: int
                           ):
        """
        Returns either:
        - A simple chain of parameters (old behavior) if no component LR config
        - A dict with component groups if component LR config is provided
        """
        freeze_prefixes = []
        if unet_freeze_config.get("freeze_in", False):
            freeze_prefixes.extend(['time_embedding.', 'conv_in.', 'down_blocks.', 'add_embedding.'])
        if unet_freeze_config.get("freeze_mid", False):
            freeze_prefixes.extend(['mid_block'])
        if unet_freeze_config.get("freeze_out", False):
            freeze_prefixes.extend(['conv_norm_out.', 'conv_out.', 'up_blocks.'])

        if 'unfreeze_cross_attention' in unet_freeze_config or 'unfreeze_self_attention' in unet_freeze_config or 'unfreeze_spatial_resnets' in unet_freeze_config:
            raise ValueError("unfreeze_* options are no longer supported in unet freeze config, please use freeze_* options instead (may be None)")

        cross_attn_layer_names = list(self._get_cross_attention_layer_names(
            unet,
            cross_attention_dim_to_find=cross_attention_dim_to_find,
        ))
        freeze_cross_attn = unet_freeze_config.get("freeze_cross_attention", None)
        if freeze_cross_attn:
            print(f"{'' if freeze_cross_attn else 'un'}freezing cross attention layers:")
            print(cross_attn_layer_names)

        freeze_self_attn = unet_freeze_config.get("freeze_self_attention", None)
        self_attn_layer_names = list(self._get_self_attention_layer_names(
            unet,
            cross_attention_dim_to_ignore=cross_attention_dim_to_find,
        ))
        if freeze_self_attn is not None:
            print(f"{'' if freeze_self_attn else 'un'}freezing self attention layers:")
            print(self_attn_layer_names)

        freeze_spatial_resnets = unet_freeze_config.get("freeze_spatial_resnets", None)
        spatial_resnet_prefixes = ["mid_block.resnets", "down_blocks.2.resnets", "down_blocks.3.resnets", "up_blocks.0.resnets", "up_blocks.1.resnets"]
        freeze_all_resnets = unet_freeze_config.get("freeze_all_resnets", None)
        resnet_layer_prefixes = [f"down_blocks.{i}.resnets" for i in range(4)] + ["mid_blocks.resnets"] + [f"up_blocks.{i}.resnets" for i in range(4)]
        freeze_late_up_resnets = unet_freeze_config.get("freeze_late_up_resnets", None)
        late_up_resnet_layer_prefixes = ["up_blocks.2.resnets", "up_blocks.3.resnets"]

        def should_freeze(n):
            # maybe freeze
            if freeze_cross_attn is not None and any(n.startswith(prefix) for prefix in cross_attn_layer_names):
                return freeze_cross_attn
            elif freeze_self_attn is not None and any(n.startswith(prefix) for prefix in self_attn_layer_names):
                return freeze_self_attn
            elif freeze_spatial_resnets is not None and any(n.startswith(prefix) for prefix in spatial_resnet_prefixes):
                return freeze_spatial_resnets
            elif freeze_late_up_resnets is not None and any(n.startswith(prefix) for prefix in late_up_resnet_layer_prefixes):
                return freeze_late_up_resnets
            elif freeze_all_resnets is not None and any(n.startswith(prefix) for prefix in resnet_layer_prefixes):
                return freeze_all_resnets
            # fallback to general prefixes
            return any(n.startswith(prefix) for prefix in freeze_prefixes)

        def get_component_type(n):
            """Determine which component type a parameter belongs to"""
            if any(n.startswith(prefix) for prefix in cross_attn_layer_names):
                return "cross_attention"
            elif any(n.startswith(prefix) for prefix in self_attn_layer_names):
                return "self_attention"
            elif "resnet" in n.lower() or "resnets" in n.lower():
                return "resnet"
            else:
                return "other"

        for n, p in unet.named_parameters():
            p.requires_grad = not should_freeze(n)

        print("unet parameters training:")
        for n, p in unet.named_parameters():
            print(f"{' ' if p.requires_grad else '❄️'} {n}")

        # Always return dict structure
        if unet_component_lr_config and unet_component_lr_config.get("enabled", False):
            # Component-specific LR mode: return 4 separate groups
            component_groups = {
                "cross_attention": [],
                "self_attention": [],
                "resnet": [],
                "other": []
            }

            for n, p in unet.named_parameters():
                if p.requires_grad:
                    component_type = get_component_type(n)
                    component_groups[component_type].append((n, p))

            # Log component counts
            print("\n=== UNet Component-Specific LR Groups ===")
            for component, params in component_groups.items():
                if params:
                    lr_scale = unet_component_lr_config.get(f"{component}_lr_scale", 1.0)
                    print(f"  {component}: {len(params)} parameters (LR scale: {lr_scale}x)")

            return component_groups
        else:
            # Standard mode: return single group dict for consistency
            return {
                "default": list(itertools.chain([np for np in unet.named_parameters() if np[1].requires_grad]))
            }


    def _apply_text_encoder_freeze(self, text_encoder) -> chain[torch.nn.Parameter]:
        num_layers = len(text_encoder.text_model.encoder.layers)
        unfreeze_embeddings = True
        unfreeze_last_n_layers = None
        unfreeze_final_layer_norm = True
        if "freeze_front_n_layers" in self.te_freeze_config:
            logging.warning(
                ' * Found "freeze_front_n_layers" in JSON, please use "unfreeze_last_n_layers" instead')
            freeze_front_n_layers = self.te_freeze_config["freeze_front_n_layers"]
            if freeze_front_n_layers<0:
                # eg -2 = freeze all but the last 2
                unfreeze_last_n_layers = -freeze_front_n_layers
            else:
                unfreeze_last_n_layers = num_layers - freeze_front_n_layers
        if "unfreeze_last_n_layers" in self.te_freeze_config:
            unfreeze_last_n_layers = self.te_freeze_config["unfreeze_last_n_layers"]

        if unfreeze_last_n_layers is None:
            # nothing specified: default behaviour
            unfreeze_last_n_layers = num_layers
        else:
            # something specified:
            #assert(unfreeze_last_n_layers > 0)
            if unfreeze_last_n_layers < num_layers:
                # if we're unfreezing layers then by default we ought to freeze the embeddings
                unfreeze_embeddings = False

        if "freeze_embeddings" in self.te_freeze_config:
            unfreeze_embeddings = not self.te_freeze_config["freeze_embeddings"]
        if "freeze_final_layer_norm" in self.te_freeze_config:
            unfreeze_final_layer_norm = not self.te_freeze_config["freeze_final_layer_norm"]

        parameters = itertools.chain([])

        if unfreeze_embeddings:
            parameters = itertools.chain(parameters, text_encoder.text_model.embeddings.named_parameters())
        else:
            print(" ❄️ freezing embeddings")
            #for p in text_encoder.text_model.embeddings.parameters():
            #    p.requires_grad = False

        if unfreeze_last_n_layers >= num_layers:
            parameters = itertools.chain(parameters, text_encoder.text_model.encoder.layers.named_parameters())
        else:
            # freeze the specified CLIP text encoder layers
            layers = text_encoder.text_model.encoder.layers
            first_layer_to_unfreeze = num_layers - unfreeze_last_n_layers
            print(f" ❄️ freezing text encoder layers 1-{first_layer_to_unfreeze} out of {num_layers} layers total")
            parameters = itertools.chain(parameters, layers[first_layer_to_unfreeze:].named_parameters())
            for l in layers[:first_layer_to_unfreeze]:
                for p in l.parameters():
                    p.requires_grad = False

        if unfreeze_final_layer_norm:
            parameters = itertools.chain(parameters, text_encoder.text_model.final_layer_norm.named_parameters())
        else:
            print(" ❄️ freezing final layer norm")
            for p in text_encoder.text_model.final_layer_norm.parameters():
                p.requires_grad = False


        return parameters

    def setup_lora_training(self, args, model: 'TrainingModel') -> tuple[dict[str, list], dict[str, list]]:

        if args.disable_textenc_training:
            text_encoder_params = []
        else:
            if not args.lora_resume:
                if self.te_freeze_config.get("unfreeze_last_n_layers", None) is not None:
                    raise ValueError("Freezing not supported with LoRA training")
                text_lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    init_lora_weights=True,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
                )
                #print("not adding lora for te")
                model.text_encoder.add_adapter(text_lora_config)
                if model.text_encoder_2 is not None:
                    model.text_encoder_2.add_adapter(text_lora_config)
            text_encoder_params = list(filter(lambda p: p[1].requires_grad, model.text_encoder.named_parameters()))
            if model.text_encoder_2 is not None:
                text_encoder_params += list(filter(lambda p: p[1].requires_grad, model.text_encoder_2.named_parameters()))

        if args.disable_unet_training:
            unet_params = []
        else:
            if not args.lora_resume:
                unet_lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    init_lora_weights=True,
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                model.unet.add_adapter(unet_lora_config)
            unet_params = list(filter(lambda p: p[1].requires_grad, model.unet.named_parameters()))

        return {'default': text_encoder_params}, {'default': unet_params}



def log_optimizer(label: str, optimizer: torch.optim.Optimizer, betas, epsilon, weight_decay, lr):
    """
    logs the optimizer settings
    """
    all_params = sum([g['params'] for g in optimizer.param_groups], [])
    frozen_parameter_count = len([p for p in all_params if not p.requires_grad])
    total_parameter_count = len(all_params)
    if frozen_parameter_count > 0:
        param_info = f"({total_parameter_count} parameters, {frozen_parameter_count} frozen)"
    else:
        param_info = f"({total_parameter_count} parameters)"

    logging.info(f"{Fore.CYAN} * {label} optimizer: {optimizer.__class__.__name__} {param_info} *{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}    lr: {lr}, betas: {betas}, epsilon: {epsilon}, weight_decay: {weight_decay} *{Style.RESET_ALL}")


def _get_grad_norm(parameters) -> float:
    parameters = [
        p for p in parameters if p.grad is not None and p.requires_grad
    ]
    if len(parameters) == 0:
        return 0.0
    else:
        device = parameters[0].grad.device
        return torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach()).to(device)
                    for p in parameters
                ]
            ),
            2.0,
        ).item()

def _get_moment_norms(optimizer) -> tuple[float, float]:
    if optimizer is None:
        return 0, 0
    with torch.no_grad():
        exp_avg_norms = []
        exp_avg_sq_norms = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        exp_avg_norms.append(torch.norm(state['exp_avg']).item() ** 2)
                        exp_avg_sq_norms.append(torch.norm(state['exp_avg_sq']).item() ** 2)

        return sum(exp_avg_norms) ** 0.5, sum(exp_avg_sq_norms) ** 0.5

def _get_findlr_scheduler(
    optimizer: Optimizer,
    min_lr: float,
    base_lr: float,
    num_warmup_steps: int
) -> LambdaLR:
    """
    Create a schedule with a learning rate that increases exponentially from min_lr to base_lr over num_warmup_steps.
    """
    def findlr_lambda(current_step: int):
        if current_step >= num_warmup_steps:
            # just use base_lr
            return 1.0

        t = current_step / num_warmup_steps
        # LambdaLR multiplies by base_lr, so we need to return the ratio of the desired LR to base_lr
        return (min_lr * (base_lr / min_lr) ** t) / base_lr

    return LambdaLR(optimizer, findlr_lambda)



def _get_polynomial_decay_schedule_with_warmup_adj(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Adapted from diffusers get_polynomial_decay_schedule_with_warmup to remove the restrictive check on strictly decreasing LR

    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        num_cycles (`int`, *optional*, defaults to 1):
            How many times to repeat the cycle of warmup/cooldown during training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]

    num_warmup_steps_cycle = math.ceil(num_warmup_steps / num_cycles)
    num_training_steps_cycle = math.ceil(num_training_steps / num_cycles)

    def lr_lambda_cycleinternal(current_cycle_step: int):
        if current_cycle_step < num_warmup_steps_cycle:
            return float(current_cycle_step) / float(max(1, num_warmup_steps_cycle))
        elif current_cycle_step > num_training_steps_cycle:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps_cycle - num_warmup_steps_cycle
            pct_remaining = 1 - (current_cycle_step - num_warmup_steps_cycle) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    def lr_lambda(current_step: int):
        current_cycle_step = current_step % int(num_warmup_steps_cycle + num_training_steps_cycle)
        return lr_lambda_cycleinternal(current_cycle_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _extract_attention_parameter_group(parameters: list[tuple[str, torch.nn.Parameter]]) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Extracts the attention parameters from the given parameters list.
    Returns a tuple of two lists: regular parameters and attention parameters.
    """
    regular_group = []
    attention_group = []

    for name, param in parameters:
        name = name.lower()
        if ("attn" in name or "attention" in name) and any(qk in name for qk in ["to_q", "to_k", "query", "key"]):
            attention_group.append((name, param))
        else:
            regular_group.append((name, param))

    return regular_group, attention_group