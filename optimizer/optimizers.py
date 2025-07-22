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
from itertools import chain
from typing import Generator

import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig

from torch.cuda.amp import GradScaler
from diffusers.optimization import get_scheduler, get_polynomial_decay_schedule_with_warmup

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

class EveryDreamOptimizer:
    """
    Wrapper to manage optimizers
    resume_ckpt_path: path to resume checkpoint, will try to load state (.pt) files if they exist
    optimizer_config: config for the optimizers
    text_encoder: text encoder model parameters
    unet: unet model parameters
    """
    def __init__(self, args, optimizer_config, text_encoder, unet, epoch_len, plugin_runner: PluginRunner, log_writer=None):
        if optimizer_config is None:
            raise ValueError("missing optimizer_config")
        if "doc" in optimizer_config:
            del optimizer_config["doc"]
        print("\n raw optimizer_config:")
        pprint.pprint(optimizer_config)
        self.epoch_len = epoch_len
        self.max_epochs = args.max_epochs
        self.unet = unet # needed for weight norm logging, unet.parameters() has to be called again, Diffusers quirk
        self.text_encoder = text_encoder
        self.log_writer = log_writer
        self.unet_freeze = args.freeze_unet_balanced
        self.te_config, self.base_config = self.get_final_optimizer_configs(args, optimizer_config)
        self.te_freeze_config = optimizer_config.get("text_encoder_freezing", {})
        print(" Final unet optimizer config:")
        pprint.pprint(self.base_config)
        print(" Final text encoder optimizer config:")
        pprint.pprint(self.te_config)

        self.grad_accum = args.grad_accum
        self.next_grad_accum_step = self.grad_accum
        self.clip_grad_norm = args.clip_grad_norm
        self.apply_grad_scaler_step_tweaks = optimizer_config.get("apply_grad_scaler_step_tweaks", True)
        self.use_grad_scaler = optimizer_config.get("use_grad_scaler", True)
        self.log_grad_norm = optimizer_config.get("log_grad_norm", True)

        if args.lora:
            self.text_encoder_params, self.unet_params = self.setup_lora_training(args, text_encoder, unet)
        else:
            self.text_encoder_params = self._apply_text_encoder_freeze(text_encoder)
            self.unet_params = self._apply_unet_freeze(unet)

        if args.jacobian_descent:
            from torchjd.aggregation import UPGrad
            import torchjd
            self.jacobian_aggregator = UPGrad()
            self.jacobian_backward = torchjd.backward
        else:
            self.jacobian_aggregator = None

        self.text_encoder_params, self.unet_params = plugin_runner.run_add_parameters(self.text_encoder_params, self.unet_params)
        self.unet_params = list(self.unet_params)
        self.text_encoder_params = list(self.text_encoder_params)

        #with torch.no_grad():
        #    log_action = lambda n, label: logging.info(f"{Fore.LIGHTBLUE_EX} {label} weight normal: {n:.1f}{Style.RESET_ALL}")
        #    self._log_weight_normal(text_encoder.text_model.encoder.layers.parameters(), "text encoder", log_action)
        #    self._log_weight_normal(unet.parameters(), "unet", log_action)

        self.optimizers = []
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

    def step_optimizer(self, global_step):
        if self.scaler is not None:
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

        if self.log_grad_norm:
            with torch.no_grad():
                self.log_writer.add_scalar("optimizer/unet_grad_norm_pre_clip", _get_grad_norm(self.unet.parameters()), global_step)
                self.log_writer.add_scalar("optimizer/te_grad_norm_pre_clip", _get_grad_norm(self.text_encoder.parameters()), global_step)

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(parameters=self.unet.parameters(), max_norm=self.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(parameters=self.text_encoder.parameters(), max_norm=self.clip_grad_norm)
            if self.log_grad_norm:
                with torch.no_grad():
                    self.log_writer.add_scalar("optimizer/unet_grad_norm_post_clip", _get_grad_norm(self.unet.parameters()), global_step)
                    self.log_writer.add_scalar("optimizer/te_grad_norm_post_clip", _get_grad_norm(self.text_encoder.parameters()), global_step)

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

        self._zero_grad(set_to_none=True)


    def step_schedulers(self, global_step):
        for scheduler in self.lr_schedulers:
            scheduler.step()

        if self.apply_grad_scaler_step_tweaks:
            self._update_grad_scaler(global_step)

    def step(self, step: int, global_step: int, ):
        if self._should_do_grad_accum_step(step, global_step):
            self.step_optimizer(global_step)

        self.step_schedulers(global_step)

    def _zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_scale(self):
        if self.scaler is None:
            return 1
        return self.scaler.get_scale()
    
    def get_unet_lr(self):
        return self.optimizer_unet.param_groups[0]['lr'] if self.optimizer_unet is not None else 0
    
    def get_textenc_lr(self):
        return self.optimizer_te.param_groups[0]['lr'] if self.optimizer_te is not None else 0
    
    def save(self, ckpt_path: str):
        """
        Saves the optimizer states to path
        """
        self._save_optimizer(self.optimizer_te, os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)) if self.optimizer_te is not None else None
        self._save_optimizer(self.optimizer_unet, os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)) if self.optimizer_unet is not None else None
        self._save_optimizer(self.scaler, os.path.join(ckpt_path, SCALER_STATE_FILENAME)) if self.scaler is not None else None

    def load(self, ckpt_path: str):
        """
        Loads the optimizer states from path
        """
        te_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_TE_STATE_FILENAME)
        unet_optimizer_state_path = os.path.join(ckpt_path, OPTIMIZER_UNET_STATE_FILENAME)
        scaler_state_path = os.path.join(ckpt_path, SCALER_STATE_FILENAME)
        if os.path.exists(te_optimizer_state_path) and self.optimizer_te is not None:
            self._load_optimizer(self.optimizer_te, te_optimizer_state_path)
        if os.path.exists(unet_optimizer_state_path) and self.optimizer_unet is not None:
            self._load_optimizer(self.optimizer_unet, unet_optimizer_state_path)
        if os.path.exists(scaler_state_path) and self.scaler is not None:
            self._load_optimizer(self.scaler, scaler_state_path)

    def create_optimizers(self, args, text_encoder_params, unet_params):
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
        te_config = global_optimizer_config.get("text_encoder_overrides")

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

        base_config["optimizer"] = base_config.get("optimizer", None) or "adamw8bit"

        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler
        base_config["lr_warmup_steps"] = base_config.get("lr_warmup_steps", None) or args.lr_warmup_steps
        base_config["lr_decay_steps"] = base_config.get("lr_decay_steps", None) or args.lr_decay_steps
        base_config["lr_advance_steps"] = base_config.get("lr_advance_steps", None) or args.lr_advance_steps
        base_config["lr_scheduler"] = base_config.get("lr_scheduler", None) or args.lr_scheduler

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
        te_config = optimizer_config["text_encoder_overrides"]

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
                lr_scheduler = _get_polynomial_decay_schedule_with_warmup_adj(
                    optimizer=self.optimizer_unet,
                    lr_end=unet_config.get("lr_end", unet_config["lr"]/100.0),
                    power=unet_config.get("power", 2),
                    num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                    num_training_steps=int(unet_config["lr_decay_steps"]),
                )
            else:
                lr_scheduler = get_scheduler(
                    unet_config["lr_scheduler"],
                    optimizer=self.optimizer_unet,
                    num_warmup_steps=int(unet_config["lr_warmup_steps"]),
                    num_training_steps=int(unet_config["lr_decay_steps"]),
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
    def _save_optimizer(optimizer, path: str):
        """
        Saves the optimizer state to specific path/filename
        """
        torch.save(optimizer.state_dict(), path)

    @staticmethod
    def _load_optimizer(optimizer: torch.optim.Optimizer, path: str):
        """
        Loads the optimizer state to an Optimizer object
        optimizer: torch.optim.Optimizer
        path: .pt file
        """
        try:
            optimizer.load_state_dict(torch.load(path))
            logging.info(f" Loaded optimizer state from {path}")
        except Exception as e:
            logging.warning(f"{Fore.LIGHTYELLOW_EX}**Failed to load optimizer state from {path}, optimizer state will not be loaded, \n * Exception: {e}{Style.RESET_ALL}")
            pass

    def _create_optimizer(self, label, args, local_optimizer_config, parameters):
        #l = [parameters]
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

        if curr_lr is None:
            curr_lr = default_lr
            logging.warning(f"No LR setting found, defaulting to {default_lr}")

        if optimizer_name:
            optimizer_name = optimizer_name.lower()

            if optimizer_name == "lion":
                from lion_pytorch import Lion
                opt_class = Lion
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                )
            elif optimizer_name == "lion8bit":
                from bitsandbytes.optim import Lion8bit
                opt_class = Lion8bit
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                    percentile_clipping=100,
                    min_8bit_size=4096,
                )
            elif optimizer_name == "prodigy":
                from prodigyopt import Prodigy
                opt_class = Prodigy
                optimizer = opt_class(
                    itertools.chain(parameters),
                    lr=curr_lr,
                    weight_decay=weight_decay,
                    use_bias_correction=use_bias_correction,
                    growth_rate=growth_rate,
                    d0=d0,
                    safeguard_warmup=safeguard_warmup
                )
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
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        weight_decay=weight_decay,
                        eps=epsilon, #unused for lion
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=growth_rate,
                        decouple=decouple,
                    )
                elif optimizer_name == "dadapt_adan":
                    opt_class = dadaptation.DAdaptAdan
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        no_prox=no_prox,
                        weight_decay=weight_decay,
                        eps=epsilon,
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=growth_rate,
                    )
                elif optimizer_name == "dadapt_lion":
                    opt_class = dadaptation.DAdaptLion
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        betas=(betas[0], betas[1]),
                        weight_decay=weight_decay,
                        d0=d0,
                        log_every=args.log_step,
                    )
                elif optimizer_name == "dadapt_sgd":
                    opt_class = dadaptation.DAdaptSGD
                    optimizer = opt_class(
                        itertools.chain(parameters),
                        lr=curr_lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        d0=d0,
                        log_every=args.log_step,
                        growth_rate=growth_rate,
                    )
            elif optimizer_name == "adacoor":
                from optimizer.adacoor import AdaCoor

                opt_class = AdaCoor
                optimizer = opt_class(
                    itertools.chain(parameters),
                    eps=epsilon
                )

        if not optimizer:
            optimizer = opt_class(
                itertools.chain(parameters),
                lr=curr_lr,
                betas=(betas[0], betas[1]),
                eps=epsilon,
                weight_decay=weight_decay,
                amsgrad=False,
            )

        log_optimizer(label, optimizer, betas, epsilon, weight_decay, curr_lr)
        return optimizer

    def _apply_unet_freeze(self, unet: UNet2DConditionModel) -> chain[torch.nn.Parameter]:
        if not self.unet_freeze:
            return itertools.chain(unet.parameters())
        def should_train(name: str) -> bool:
            # Train Time Embeddings
            if name.startswith("time_embedding."):
                return True

            # Train All Attention Layers (within down, mid, up blocks)
            # This regex matches anything containing '.attentions.'
            if re.search(r'\.attentions\.', name):
                return True

            # Train Final Output Layers
            if name.startswith("conv_norm_out.") or name.startswith("conv_out."):
                return True

            # --- Everything else is frozen based on the Balanced Strategy ---
            return False

        print(' ❄️ applying unet "balanced" freeze')
        for n, p in unet.named_parameters():
            if not should_train(n):
                p.requires_grad = False

        return itertools.chain([p for n, p in unet.named_parameters() if should_train(n)])


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
            parameters = itertools.chain(parameters, text_encoder.text_model.embeddings.parameters())
        else:
            print(" ❄️ freezing embeddings")
            #for p in text_encoder.text_model.embeddings.parameters():
            #    p.requires_grad = False

        if unfreeze_last_n_layers >= num_layers:
            parameters = itertools.chain(parameters, text_encoder.text_model.encoder.layers.parameters())
        else:
            # freeze the specified CLIP text encoder layers
            layers = text_encoder.text_model.encoder.layers
            first_layer_to_unfreeze = num_layers - unfreeze_last_n_layers
            print(f" ❄️ freezing text encoder layers 1-{first_layer_to_unfreeze} out of {num_layers} layers total")
            parameters = itertools.chain(parameters, layers[first_layer_to_unfreeze:].parameters())
            for l in layers[:first_layer_to_unfreeze]:
                for p in l.parameters():
                    p.requires_grad = False

        if unfreeze_final_layer_norm:
            parameters = itertools.chain(parameters, text_encoder.text_model.final_layer_norm.parameters())
        else:
            print(" ❄️ freezing final layer norm")
            for p in text_encoder.text_model.final_layer_norm.parameters():
                p.requires_grad = False

        return parameters

    def setup_lora_training(self, args, text_encoder, unet):

        if args.disable_textenc_training:
            text_encoder_params = []
        else:
            if not args.lora_resume:
                suffixes = ["q_proj", "k_proj", "v_proj", "out_proj"]
                target_modules = [k for k, _ in text_encoder.named_modules() if any(suffix in k for suffix in suffixes)]
                unfreeze_last_n_layers = self.te_freeze_config.get("unfreeze_last_n_layers", None)
                if unfreeze_last_n_layers is not None:
                    layer_count = len(text_encoder.text_model.encoder.layers)
                    last_n_layers = range(layer_count-unfreeze_last_n_layers, layer_count+1)
                    target_modules = [k for k in target_modules if any(f'.layers.{l}' in k for l in last_n_layers)]
                print("lora freezing means we put lora on: ", target_modules)
                text_lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=target_modules
                )
                text_encoder.add_adapter(text_lora_config)
            text_encoder_params = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

        if args.disable_unet_training:
            unet_params = []
        else:
            if not args.lora_resume:
                unet_lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                unet.add_adapter(unet_lora_config)
            unet_params = list(filter(lambda p: p.requires_grad, unet.parameters()))

        return text_encoder_params, unet_params



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



def _get_polynomial_decay_schedule_with_warmup_adj(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
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

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
