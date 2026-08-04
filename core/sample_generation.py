"""
Shared sample generation for SD/SDXL (train.py) and SANA (train_sana.py).

Extracts the common logic that was duplicated in both training scripts into a
single function, implementing DRY / SRP.
"""
import gc
import logging
import os
import shutil
from contextlib import nullcontext

import torch

from model.training_model import get_use_ema_decay_training
from utils.inference_context import inference_guard
from utils.isolate_rng import isolate_rng
from utils.model_offload import unload_model_for_pause, reload_model_for_resume


def _unwrap(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def generate_samples(
    model,
    sample_generator,
    global_step: int,
    batch: dict | None,
    args,
    device,
    log_folder: str,
    log_time: str | None,
    train_dtype,
    vae_dtype,
    swap_ema_requested: bool = False,
) -> None:
    with isolate_rng():
        sample_generator.reload_config()
        if batch is not None:
            flattened_captions_dict = [
                v for _, l in batch["captions"].items() for v in l
            ]
            sample_generator.update_random_captions(flattened_captions_dict)

        torch.cuda.empty_cache()

        # ── External-gen setup ──────────────────────────────────────────
        needs_external = (
            sample_generator.external_batch_requests
            and sample_generator.external_gen_enabled
            and sample_generator.external_gen_base_url
        )

        is_sana = hasattr(model, 'transformer')

        external_model_key = None
        external_model_save_path = None
        if needs_external:
            if is_sana:
                inference_pipe = sample_generator.create_inference_pipe(
                    model_being_trained=model,
                    diffusers_scheduler_config=model.noise_scheduler.config,
                    use_ema=False,
                )
            else:
                inference_pipe = sample_generator.create_inference_pipe(
                    model_being_trained=model,
                    diffusers_scheduler_config=model.noise_scheduler.config,
                    flow_match_shift=args.flow_match_shift,
                    flow_match_shift_dynamic=args.flow_match_shift_dynamic,
                ).to(device)

            external_model_save_path = (
                sample_generator.external_gen_model_save_dir
                or f"{log_folder}/external_samplegen_model/gs{global_step:05}"
            )

            from utils.external_image_generator import install_model
            ext_project_name = args.project_name or getattr(model, 'model_id', 'model')
            external_model_key = install_model(
                sample_generator.external_gen_base_url,
                external_model_save_path,
                model_name=f"{ext_project_name}-gs{global_step:05}",
                pipeline=inference_pipe,
            )
            del inference_pipe
            torch.cuda.empty_cache()

        do_offload = needs_external and sample_generator.offload_model_for_external_gen
        if do_offload:
            unload_model_for_pause(model)
            gc.collect()
            torch.cuda.empty_cache()

        # ── Determine passes ────────────────────────────────────────────
        ema_active = get_use_ema_decay_training(args)
        run_nonema = args.ema_sample_nonema_model
        run_ema = args.ema_sample_ema_model
        if swap_ema_requested:
            run_nonema = not run_nonema
            run_ema = not run_ema
        run_nonema = run_nonema or not ema_active
        run_ema = run_ema and ema_active

        # ── Guard modules (duck-type) ───────────────────────────────────
        if is_sana:
            guard_modules = [model.transformer]
        else:
            guard_modules = [model.unet, model.text_encoder]
            if model.text_encoder_2:
                guard_modules.append(model.text_encoder_2)

        with inference_guard(*guard_modules):
            # ── Non-EMA (live model) pass ──
            if run_nonema and not needs_external:
                logging.info(f" * Generating samples [non-EMA model] at gs:{global_step}")
                if is_sana:
                    pipe = sample_generator.create_inference_pipe(
                        model_being_trained=model,
                        diffusers_scheduler_config=model.noise_scheduler.config,
                        use_ema=False,
                    )
                    was_tiling = getattr(pipe.vae, 'use_tiling', None)
                    if was_tiling is not None:
                        pipe.vae.enable_tiling()
                    subdir = "samples"
                    project_name = args.project_name or getattr(model, 'model_id', 'sana-model')
                else:
                    pipe = sample_generator.create_inference_pipe(
                        model_being_trained=model,
                        diffusers_scheduler_config=model.noise_scheduler.config,
                        flow_match_shift=args.flow_match_shift,
                        flow_match_shift_dynamic=args.flow_match_shift_dynamic,
                    ).to(device)
                    was_tiling = None
                    subdir = "samples"
                    project_name = args.project_name

                sample_generator.generate_samples(
                    pipe, global_step,
                    extra_info="", samples_subdir=subdir,
                    project_name=project_name,
                    log_time=log_time,
                    external_model_key=external_model_key,
                    external_model_save_path=external_model_save_path,
                )
                if is_sana and was_tiling is not None:
                    if was_tiling:
                        pipe.vae.enable_tiling()
                    else:
                        pipe.vae.disable_tiling()
                del pipe
                torch.cuda.empty_cache()

            # ── EMA pass ──
            if run_ema and not needs_external:
                logging.info(f" * Generating samples [EMA model] at gs:{global_step}")

                if is_sana or model.ema_working_dir is not None:
                    # Disk-offload OR SANA in-memory EMA
                    if model.ema_working_dir is not None:
                        context = model.ema_inplace_swap()
                    else:
                        context = nullcontext()

                    with context:
                        if is_sana:
                            pipe = sample_generator.create_inference_pipe(
                                model_being_trained=model,
                                diffusers_scheduler_config=model.noise_scheduler.config,
                                use_ema=True,
                            )
                            was_tiling = getattr(pipe.vae, 'use_tiling', None)
                            if was_tiling is not None:
                                pipe.vae.enable_tiling()
                            subdir = "samples-ema"
                            project_name = args.project_name or getattr(model, 'model_id', 'sana-model')
                        else:
                            pipe = sample_generator.create_inference_pipe(
                                model_being_trained=model,
                                diffusers_scheduler_config=model.noise_scheduler.config,
                                flow_match_shift=args.flow_match_shift,
                                flow_match_shift_dynamic=args.flow_match_shift_dynamic,
                                use_ema=True,
                            ).to(device)
                            was_tiling = None
                            subdir = "samples-ema"
                            project_name = args.project_name

                        sample_generator.generate_samples(
                            pipe, global_step,
                            extra_info="", samples_subdir=subdir,
                            project_name=project_name,
                            log_time=log_time,
                            external_model_key=external_model_key,
                            external_model_save_path=external_model_save_path,
                        )
                        if is_sana and was_tiling is not None:
                            if was_tiling:
                                pipe.vae.enable_tiling()
                            else:
                                pipe.vae.disable_tiling()
                        del pipe
                    torch.cuda.empty_cache()

                else:
                    # SD/SDXL in-memory EMA
                    _unwrap(model.unet).to("cpu")
                    _unwrap(model.text_encoder).to("cpu")
                    if model.text_encoder_2 is not None:
                        _unwrap(model.text_encoder_2).to("cpu")
                    torch.cuda.empty_cache()
                    try:
                        ema_pipe = sample_generator.create_ema_inference_pipe(
                            model_being_trained=model,
                            diffusers_scheduler_config=model.noise_scheduler.config,
                            flow_match_shift=args.flow_match_shift,
                            flow_match_shift_dynamic=args.flow_match_shift_dynamic,
                        )
                        if ema_pipe is None:
                            logging.warning(
                                " * EMA sampling requested but no EMA weights found yet, "
                                "skipping EMA sample pass"
                            )
                        else:
                            ema_pipe = ema_pipe.to(device)
                            sample_generator.generate_samples(
                                ema_pipe, global_step,
                                extra_info="", samples_subdir="samples-ema",
                                project_name=args.project_name,
                                log_time=log_time,
                                external_model_key=external_model_key,
                                external_model_save_path=external_model_save_path,
                            )
                            del ema_pipe
                    finally:
                        torch.cuda.empty_cache()
                        _unwrap(model.unet).to(device)
                        _unwrap(model.text_encoder).to(device)
                        if model.text_encoder_2 is not None:
                            _unwrap(model.text_encoder_2).to(device)

        # ── External-only pass ──
        if needs_external:
            with inference_guard(*guard_modules):
                project_name = args.project_name or getattr(model, 'model_id', 'sana-model')
                sample_generator.generate_samples(
                    pipe=None,
                    global_step=global_step,
                    extra_info="", samples_subdir="samples",
                    project_name=project_name,
                    log_time=log_time,
                    external_model_key=external_model_key,
                    external_model_save_path=external_model_save_path,
                )

        # ── Reload model if offloaded ──
        if do_offload:
            reload_model_for_resume(model, device, train_dtype=train_dtype,
                                    vae_dtype=vae_dtype, args=args)

        # ── Cleanup model from server + disk ──
        if needs_external:
            from utils.external_image_generator import unload_model, uninstall_model
            unload_model(sample_generator.external_gen_base_url)
            keep_model = sample_generator.keep_model_after_external_gen
            if not keep_model:
                uninstall_model(sample_generator.external_gen_base_url, external_model_key)
                if os.path.exists(external_model_save_path):
                    shutil.rmtree(external_model_save_path, ignore_errors=True)

        gc.collect()
        torch.cuda.empty_cache()
