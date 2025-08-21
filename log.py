import dataclasses
import json
import logging
import math
import os
from collections import defaultdict

import torch
import torchvision
from colorama import Style, Fore

from data.every_dream import EveryDreamBatch
from data.image_train_item import ImageTrainItem

from loss import vae_preview
from model.training_model import TrainingModel, TrainingVariables

from optimizer.attention_activation_control import ActivationLogger

@dataclasses.dataclass
class LogData:
    loss_log_step = []
    loss_log_step_cd = []
    loss_log_step_non_cd = []
    loss_preview_image: torch.Tensor | None = None
    loss_per_timestep: dict[int, dict[int, tuple[float, int]]] = dataclasses.field(
        default_factory=lambda: defaultdict(dict))

    loss_epoch = []
    images_per_sec = []
    images_per_sec_log_step = []

    attention_activation_logger: ActivationLogger = None

def do_log_step(args, ed_optimizer, log_data: LogData, log_folder, log_writer, model: TrainingModel, tv: TrainingVariables):
    global_step = tv.global_step
    lr_unet = ed_optimizer.get_unet_lr()
    lr_textenc = ed_optimizer.get_textenc_lr()
    log_writer.add_scalar(tag="hyperparameter/lr unet", scalar_value=lr_unet, global_step=global_step)
    log_writer.add_scalar(tag="hyperparameter/lr text encoder", scalar_value=lr_textenc, global_step=global_step)
    if tv.timesteps_ranges:
        log_writer.add_scalar(tag="hyperparameter/timestep start", scalar_value=tv.timesteps_ranges[0][0],
                              global_step=global_step)
        log_writer.add_scalar(tag="hyperparameter/timestep end", scalar_value=tv.timesteps_ranges[0][1],
                              global_step=global_step)
    log_writer.add_scalar(tag="hyperparameter/effective batch size", scalar_value=tv.last_effective_batch_size,
                          global_step=global_step)
    log_writer.add_scalar(tag="hyperparameter/effective backward size", scalar_value=tv.effective_backward_size,
                          global_step=global_step)
    sum_img = sum(log_data.images_per_sec_log_step)
    avg = sum_img / len(log_data.images_per_sec_log_step)
    if args.amp:
        log_writer.add_scalar(tag="hyperparameter/grad scale", scalar_value=ed_optimizer.get_scale(),
                              global_step=global_step)
    log_writer.add_scalar(tag="performance/images per second", scalar_value=avg, global_step=global_step)
    logs = {"lr_unet": lr_unet, "lr_te": lr_textenc, "img/s": log_data.images_per_sec}
    if len(log_data.loss_log_step) > 0:
        loss_step = sum(log_data.loss_log_step) / len(log_data.loss_log_step)
        log_writer.add_scalar(tag="loss/log_step", scalar_value=loss_step, global_step=global_step)
        logs["loss/log_step"] = loss_step
    # log histogram of loss vs timestep
    loss_sums_and_counts = {
        batch_resolution: [log_data.loss_per_timestep[batch_resolution].get(timestep, (0, 1))
                           for timestep in range(model.noise_scheduler.config.num_train_timesteps + 1)]
        for batch_resolution in args.resolution
    }
    loss_sums_and_counts = {
        batch_resolution: torch.tensor([
            loss_sum_this_step / count
            for loss_sum_this_step, count in data
        ])
        for batch_resolution, data in loss_sums_and_counts.items()
    }
    with open(os.path.join(log_folder, "loss_sums_and_counts_per_timestep.pt"), "wb") as f:
        torch.save(loss_sums_and_counts, f)
    # for batch_resolution in args.resolution:
    #    #image = _create_bar_chart_image(loss_per_timestep[batch_resolution])
    #    loss_sums_and_counts = [loss_per_timestep[batch_resolution].get(timestep, (0, 1))
    #                            for timestep in range(noise_scheduler.config.num_train_timesteps+1)]
    #    #log_writer.add_histogram(tag=f"loss/timesteps/{batch_resolution}", values=torch.tensor([
    #    #    loss_sum_this_step / count
    #    #    for loss_sum_this_step, count in loss_sums_and_counts
    #    #]), global_step=global_step)
    loss_log_step_cd = [l for l in log_data.loss_log_step_cd if math.isfinite(l)]
    if len(loss_log_step_cd) > 0:
        loss_step_cd = sum(loss_log_step_cd) / len(loss_log_step_cd)
        log_writer.add_scalar(tag="loss/log_step CD", scalar_value=loss_step_cd, global_step=global_step)
        logs["loss/log_step CD"] = loss_step_cd
    loss_log_step_non_cd = [l for l in log_data.loss_log_step_non_cd if math.isfinite(l)]
    if len(loss_log_step_non_cd) > 0:
        loss_step_non_cd = sum(loss_log_step_non_cd) / len(loss_log_step_non_cd)
        log_writer.add_scalar(tag="loss/log_step non-CD", scalar_value=loss_step_non_cd, global_step=global_step)
        logs["loss/log_step non-CD"] = loss_step_non_cd
    if log_data.loss_preview_image is not None:
        loss_preview_image_rgb = torchvision.utils.make_grid(
            vae_preview((log_data.loss_preview_image / args.negative_loss_margin) * 2 - 1)
        )
        log_data.loss_preview_image = torchvision.utils.make_grid(
            torch.reshape(log_data.loss_preview_image, [
                log_data.loss_preview_image.shape[0] * log_data.loss_preview_image.shape[1], 1,
                log_data.loss_preview_image.shape[2], log_data.loss_preview_image.shape[3]
            ]),
            nrow=log_data.loss_preview_image.shape[0],
            normalize=True,
            value_range=(0, args.negative_loss_margin),
            scale_each=False)
        log_writer.add_image(tag="loss/last vis raw", img_tensor=log_data.loss_preview_image, global_step=global_step)
        log_writer.add_image(tag="loss/last vis rgb", img_tensor=loss_preview_image_rgb, global_step=global_step)
    if args.log_named_parameters_magnitudes:
        def log_named_parameters(model, prefix):
            """Log L2 norms of parameter groups to help debug NaN issues."""
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_mean = param.mean().item()
                    log_writer.add_scalar(tag=f"p-mean/{prefix}-{name}", scalar_value=param_mean,
                                          global_step=global_step)

        if not args.disable_unet_training:
            log_named_parameters(model.unet, "unet")
        if not args.disable_textenc_training:
            log_named_parameters(model.text_encoder, "textenc")
    log_data.loss_log_step = []
    log_data.loss_log_step_cd = []
    log_data.loss_log_step_non_cd = []

    # log activations every 4th logging action
    if log_data.attention_activation_logger and (global_step + 1) % (args.log_step * 4) == 0:
        log_data.attention_activation_logger.log_to_tensorboard(global_step=global_step)

    return logs


def append_epoch_log(global_step: int, epoch_pbar, gpu, log_writer, **logs):
    """
    updates the vram usage for the epoch
    """
    if gpu is not None:
        gpu_used_mem, gpu_total_mem = gpu.get_gpu_memory()
        log_writer.add_scalar("performance/vram", gpu_used_mem, global_step)
        epoch_mem_color = Style.RESET_ALL
        if gpu_used_mem > 0.93 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTRED_EX
        elif gpu_used_mem > 0.85 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTYELLOW_EX
        elif gpu_used_mem > 0.7 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTGREEN_EX
        elif gpu_used_mem < 0.5 * gpu_total_mem:
            epoch_mem_color = Fore.LIGHTBLUE_EX

        if logs is not None:
            epoch_pbar.set_postfix(**logs, vram=f"{epoch_mem_color}{gpu_used_mem}/{gpu_total_mem} MB{Style.RESET_ALL} gs:{global_step}")


def write_batch_schedule(log_folder: str, train_batch: EveryDreamBatch, epoch: int):
    with open(f"{log_folder}/ep{epoch}_batch_schedule.txt", "w", encoding='utf-8') as f:
        for i in range(len(train_batch.image_train_items)):
            try:
                item: ImageTrainItem = train_batch.image_train_items[i]
                f.write(f"step:{int(i / train_batch.batch_size):05}, wh:{item.target_wh}, r:{item.runt_size}, path:{item.pathname} captions:{item.caption}\n")
            except Exception as e:
                logging.error(f" * Error writing to batch schedule for file path: {item.pathname}")


def log_args(log_writer, args, optimizer_config, log_folder, log_time):
    arglog = "args:\n"
    for arg, value in sorted(vars(args).items()):
        arglog += f"{arg}={value}, "
    log_writer.add_text("config", arglog)

    args_as_json = json.dumps(vars(args), indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_main.json"), "w") as f:
        f.write(args_as_json)

    optimizer_config_as_json = json.dumps(optimizer_config, indent=2)
    with open(os.path.join(log_folder, f"{args.project_name}-{log_time}_opt.json"), "w") as f:
        f.write(optimizer_config_as_json)
