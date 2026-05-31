"""
Shared CLI argument parser for all EveryDream2 training scripts.

Both train.py (SD/SDXL) and train_sana.py share these arguments so that
JSON config files and CLI flags are interchangeable between the two entry
points.  Script-specific args (e.g. --model_id for SANA) are added via the
`extra_args_fn` hook in parse_train_args().
"""

import argparse
import json
import logging


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

def load_train_json_from_file(args: argparse.Namespace, report_load: bool = False) -> None:
    """Load training options from args.config (a JSON file) into args in-place."""
    if not args.config:
        return
    try:
        if report_load:
            print(f"Loading training config from {args.config}.")
        with open(args.config, 'rt') as f:
            read_json = json.load(f)
        args.__dict__.update(read_json)
    except Exception as config_read:
        print(f"Error on loading training config from {args.config}:", config_read)
        raise


# ---------------------------------------------------------------------------
# Argparser builder
# ---------------------------------------------------------------------------

def build_argparser(
    description: str = "EveryDream2 Training options",
    *,
    config_namespace: argparse.Namespace | None = None,
    require_resume_ckpt: bool = True,
) -> argparse.ArgumentParser:
    """
    Builds the full shared EveryDream2 argument parser.

    config_namespace:
        The namespace produced by a first-pass --config-only parse.  When provided,
        --resume_ckpt is made optional if the config JSON already supplied it.
    require_resume_ckpt:
        Pass False to make --resume_ckpt always optional (e.g. for SANA which uses
        --model_id as its model source instead).
    """
    import data.aspects as aspects
    from core.self_flow import SELF_FLOW_MODES

    argparser = argparse.ArgumentParser(description=description)

    argparser.add_argument("--config", type=str, required=False, default=None,
                           help="JSON config file to load options from")
    argparser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                           default=True, help="deprecated, use --disable_amp if you wish to disable AMP")
    argparser.add_argument("--amp_without_grad_scaler", action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, use AMP but without a grad scaler (default: False, meaning use grad scaler when AMP is enabled)")
    argparser.add_argument("--force_bfloat16", action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, use bfloat16 for training")
    argparser.add_argument("--init_grad_scale", type=int, default=None,
                           help="initial value for GradScaler (default=2^17.5)")
    argparser.add_argument("--attn_type", type=str, default="sdp",
                           help="Attention mechanism to use", choices=["xformers", "sdp", "slice"])
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size (def: 2)")
    argparser.add_argument("--batch_size_curriculum_alpha", type=float, default=0.5,
                           help="curriculum alpha, default=0.5 (rapid (squared) falloff from initial)")
    argparser.add_argument("--interleave_batch_size_1", action='store_true',
                           help="If passed, toggle between batches of BS1 and batches of current_batch_size")
    argparser.add_argument("--interleave_batch_size_1_alpha", type=float, default=0,
                           help="How many BS1 batches to run when interleaving, as a factor of the current batch size")
    argparser.add_argument("--optimizer_batch_size", type=int, default=None,
                           help="If specified, step optimizer every this many samples. overriden by --initial_batch_size and --final_batch_size.")
    argparser.add_argument("--initial_batch_size", type=int, default=None,
                           help="initial batch size for curriculum")
    argparser.add_argument("--final_batch_size", type=int, default=None,
                           help="final batch size for curriculum")
    argparser.add_argument("--ckpt_every_n_minutes", type=int, default=None,
                           help="Save checkpoint every n minutes, def: 20")
    argparser.add_argument("--clip_grad_norm", type=float, default=None,
                           help="Clip gradient norm (def: disabled) (ex: 1.5), useful if loss=nan?")
    argparser.add_argument("--clip_skip", type=int, default=0,
                           help="Train using penultimate layer (def: 0) (2 is 'penultimate')", choices=[0, 1, 2, 3, 4])
    argparser.add_argument("--cond_dropout", type=float, default=0.04,
                           help="Conditional drop out as decimal 0.0-1.0, see docs for more info (def: 0.04)")
    argparser.add_argument("--cond_dropout_curriculum_alpha", type=float, default=0,
                           help="cond dropout curriculum alpha, from cond_dropout to final_cond_dropout")
    argparser.add_argument("--cond_dropout_curriculum_source",
                           choices=['timestep', 'batch_size', 'batch_size_and_timestep', 'global_step'],
                           default='global_step',
                           help="source for cond dropout curriculum")
    argparser.add_argument("--final_cond_dropout", type=float, default=None,
                           help="if doing cond dropout curriculum, the final cond dropout (timestep=0)")
    argparser.add_argument("--loss_scale", type=float, default=1, help="additional loss scaling")
    argparser.add_argument("--cond_dropout_global", type=float, default=None,
                           help="Global conditioning dropout probability multiplier")
    argparser.add_argument("--data_root", type=str, default="input",
                           help="folder where your training images are")
    argparser.add_argument("--data_multiplier_per_path", type=str, nargs="*", default=[],
                           help="optional json file(s) mapping real image paths to an additional multiplier factor")
    argparser.add_argument("--num_dataloader_workers", type=int, default=None,
                           help="number of worker threads for dataloaders")
    argparser.add_argument("--skip_undersized_images", action='store_true',
                           help="If passed, ignore images that are considered undersized for the training resolution")
    argparser.add_argument("--disable_amp", action=argparse.BooleanOptionalAction, default=False,
                           help="disables automatic mixed precision (def: False)")
    argparser.add_argument("--disable_textenc_training", action=argparse.BooleanOptionalAction, default=False,
                           help="disables training of text encoder (def: False)")
    argparser.add_argument("--disable_unet_training", action=argparse.BooleanOptionalAction, default=False,
                           help="disables training of unet (def: False) NOT RECOMMENDED")
    argparser.add_argument("--freeze_unet_balanced", action="store_true", default=False,
                           help="If passed, apply a 'balanced' unet freeze strategy")
    argparser.add_argument("--embedding_perturbation", type=float, default=0.0,
                           help="random perturbation of text embeddings (def: 0.0)")
    argparser.add_argument("--latents_perturbation", type=float, default=0.0,
                           help="random perturbation of latents (def: 0.0)")
    argparser.add_argument("--flip_p", type=float, default=0.0,
                           help="probability of flipping image horizontally (def: 0.0), not good for specific faces!")
    argparser.add_argument("--gpuid", type=int, default=0,
                           help="id of gpu to use for training, (def: 0)")
    argparser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                           help="enable gradient checkpointing to reduce VRAM use (def: False)")
    argparser.add_argument("--grad_accum", type=int, default=1,
                           help="Gradient accumulation factor (def: 1)")
    argparser.add_argument("--forward_slice_size", type=int, default=[], nargs="+",
                           help="Max --batch_size samples per forward-pass slice. Pass multiple values to specify per-resolution.")
    argparser.add_argument("--max_backward_slice_size", type=int, default=[], nargs="+",
                           help="Max number of samples to accumulate graph before doing backward. Pass multiple values to set per-resolution.")
    argparser.add_argument("--disable_backward_memsafe", action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, disable dynamic forward slice sizing")
    argparser.add_argument("--disable_backward_memsafe_resolutions", type=int, nargs="+", default=[],
                           help="If passed, disable dynamic forward slice sizing on these resolutions only")
    argparser.add_argument("--logdir", type=str, default="logs",
                           help="folder to save logs to (def: logs)")
    argparser.add_argument("--log_step", type=int, default=25,
                           help="How often to log training stats, def: 25")
    argparser.add_argument("--log_named_parameters_magnitudes", action='store_true',
                           help="If passed, log the magnitudes of all named parameters")
    argparser.add_argument('--log_attention_activations', action='store_true',
                           help='If passed, magnitudes of attention activation modules in the unet')
    argparser.add_argument("--loss_type", type=str, default="mse_huber",
                           help="type of loss / weight (def: mse_huber)",
                           choices=["huber", "mse", "mse_huber", "huber_mse", "sd3-cosmap", "v-mse", "cosmap-2"])
    argparser.add_argument("--loss_mean_over_full_effective_batch", default=True,
                           action=argparse.BooleanOptionalAction,
                           help="If passed, mean the loss over the full effective batch size")
    argparser.add_argument("--negative_loss_margin", type=float, default=0.05,
                           help="maximum for negative loss scale repulsion")
    argparser.add_argument("--lr", type=float, default=None,
                           help="Learning rate, if using scheduler is maximum LR at top of curve")
    argparser.add_argument("--lr_end", type=float, default=None,
                           help="Final learning rate, if using scheduler is minimum LR at end of curve")
    argparser.add_argument("--lr_d0", type=float, default=1e-6,
                           help="d0 for adaptive optimizers")
    argparser.add_argument("--lr_decay_steps", type=int, default=0,
                           help="Steps to reach minimum LR, default: automatically set")
    argparser.add_argument("--lr_scheduler", type=str, default="constant",
                           help="LR scheduler, (default: constant)",
                           choices=["constant", "linear", "cosine", "polynomial"])
    argparser.add_argument("--lr_warmup_steps", type=int, default=None,
                           help="Steps to reach max LR during warmup (def: 0.02 of lr_decay_steps)")
    argparser.add_argument("--lr_advance_steps", type=int, default=None,
                           help="Steps to advance the LR during training")
    argparser.add_argument("--lr_num_restarts", type=int, default=1,
                           help="Number of times to (re-)start the LR scheduler, default=1 (no restarts)")
    argparser.add_argument("--max_epochs", type=int, default=300,
                           help="Maximum number of epochs to train for")
    argparser.add_argument("--max_steps", type=int, default=None,
                           help="Maximum number of steps to train for")
    argparser.add_argument("--auto_decay_steps_multiplier", type=float, default=1.1,
                           help="Multiplier for calculating decay steps from epoch count")
    argparser.add_argument("--no_prepend_last", action="store_true",
                           help="Do not prepend 'last-' to the final checkpoint filename")
    argparser.add_argument("--no_save_ckpt", action="store_true",
                           help="Save only diffusers files, not .safetensors files")
    argparser.add_argument("--optimizer_config", default="optimizer.json",
                           help="Path to a JSON configuration file for the optimizer. Default is 'optimizer.json'")
    argparser.add_argument("--unet_freeze_regex", default=None,
                           help='Unet freeze regex(es). Use --debug_unet_freeze_regex to test without training.')
    argparser.add_argument("--debug_unet_freeze_regex", action="store_true",
                           help="If passed, apply unet freeze regex and dump results to console without training")
    argparser.add_argument("--optimizer_param_grouping", type=str, nargs="+", default=["single"],
                           help="Parameter grouping strategy for optimizer. one of 'single', 'transformer10x', 'zones', 'per-module <json_path>'. Default: 'single'")
    argparser.add_argument("--optimizer_progressive_unlock", action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, progressively unlock parameters")
    argparser.add_argument("--optimizer_progressive_unlock_by_qk_proximity",
                           action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, progressively unlock parameters by proximity to qk attention parameters")
    argparser.add_argument('--plugins', nargs='+', help='Names of plugins to use')
    argparser.add_argument("--project_name", type=str, default="myproj",
                           help="Project name for logs and checkpoints")
    argparser.add_argument("--resolution", type=int, nargs='+', default=[512],
                           help="resolution(s) to train",
                           choices=aspects.get_supported_resolutions())
    argparser.add_argument("--resolution_multiplier", type=float, nargs='+', default=[],
                           help="multipliers to apply per-resolution")
    argparser.add_argument("--keep_same_sample_at_different_resolutions_together",
                           action=argparse.BooleanOptionalAction,
                           help="if passed, re-order batches to put samples with the same path but different resolutions near to each other")
    argparser.add_argument("--no_normalize_images", action=argparse.BooleanOptionalAction,
                           help="if passed, do not normalize image pixels")

    _resume_ckpt_required = require_resume_ckpt and (
        config_namespace is None or 'resume_ckpt' not in config_namespace
    )
    argparser.add_argument("--resume_ckpt", type=str,
                           required=_resume_ckpt_required,
                           default=None if not require_resume_ckpt else "sd_v1-5_vae.ckpt",
                           help="The checkpoint to resume from (local .ckpt / diffusers folder / HF repo id)")
    argparser.add_argument("--resume_ckpt_variant", type=str, required=False, default=None,
                           help="For Hugging Face repo resume_ckpts, the variant (eg fp16)")
    argparser.add_argument("--run_name", type=str, required=False, default=None,
                           help="Run name for wandb (child of project name), and comment for tensorboard")
    argparser.add_argument("--sample_prompts", type=str, default="sample_prompts.txt",
                           help="Text file with prompts to generate test samples from, or JSON file with sample generator settings")
    argparser.add_argument("--sample_steps", type=int, default=250,
                           help="Number of steps between samples (def: 250)")
    argparser.add_argument("--save_ckpt_dir", type=str, default=None,
                           help="folder to save checkpoints to (def: root training folder)")
    argparser.add_argument("--save_every_n_epochs", type=int, default=None,
                           help="Save checkpoint every n epochs, def: 0 (disabled)")
    argparser.add_argument("--save_ckpts_from_n_epochs", type=int, default=0,
                           help="Only saves checkpoints starting at N epochs, def: 0 (disabled)")
    argparser.add_argument("--save_full_precision", action="store_true", default=False,
                           help="save ckpts at full FP32")
    argparser.add_argument("--save_optimizer", action="store_true", default=False,
                           help="saves optimizer state with ckpt, useful for resuming training later")
    argparser.add_argument("--seed", type=int, default=555,
                           help="seed used for samples and shuffling, use -1 for random")
    argparser.add_argument("--shuffle_tags", action="store_true", default=False,
                           help="randomly shuffles CSV tags in captions, for booru datasets")
    argparser.add_argument("--timestep_start", type=int, default=0,
                           help="Noising timestep minimum (def: 0)")
    argparser.add_argument("--timestep_end", type=int, default=1000,
                           help="Noising timestep (def: 1000)")
    argparser.add_argument("--timesteps_multirank_stratified", action=argparse.BooleanOptionalAction, default=False,
                           help="use multirank stratified timesteps (recommended: disable min_snr_gamma)")
    argparser.add_argument("--timesteps_multirank_stratified_distribution", type=str,
                           choices=['uniform', 'beta', 'mode', 'boundary-oversampling', 'lognormal'],
                           default='beta',
                           help="multirank stratified timesteps distribution model")
    argparser.add_argument("--timesteps_multirank_stratified_stratify",
                           action=argparse.BooleanOptionalAction, default=True,
                           help="whether to stratify timestep distribution, or just leave to chance")
    argparser.add_argument("--timesteps_multirank_stratified_alpha", type=float, default=1.5,
                           help="multirank stratified timesteps PPF alpha")
    argparser.add_argument("--timesteps_multirank_stratified_beta", type=float, default=2,
                           help="multirank stratified timesteps PPF beta")
    argparser.add_argument("--timesteps_multirank_stratified_mode_scale", type=float, default=0.5,
                           help="multirank stratified timesteps mode scale")
    argparser.add_argument("--timestep_interval_sampling", action=argparse.BooleanOptionalAction, default=False,
                           help="Sample all images in an optimizer step from the same SNR-homogeneous timestep interval")
    argparser.add_argument("--timestep_interval_n", type=int, default=10,
                           help="Number of SNR-homogeneous intervals for --timestep_interval_sampling (default: 10)")
    argparser.add_argument("--timestep_curriculum_alpha", type=float, default=0,
                           help="if passed, shift timestep range toward fine details as training progresses")
    argparser.add_argument("--timestep_initial_start", type=int, default=800,
                           help="If using timestep_curriculum_alpha, the initial start timestep (default 800)")
    argparser.add_argument("--timestep_initial_end", type=int, default=1000,
                           help="If using timestep_curriculum_alpha, the initial end timestep (default 1000)")
    argparser.add_argument("--train_sampler", type=str, default="ddpm",
                           help="noise sampler used for training, (default: ddpm)",
                           choices=["ddpm", "pndm", "ddim", "flow-matching"])
    argparser.add_argument("--flow_match_shift", type=float, default=1.0,
                           help="For flow-matching train sampler, the noise shift parameter (def: 1.0)")
    argparser.add_argument("--flow_match_shift_dynamic", action=argparse.BooleanOptionalAction, default=False,
                           help="If passed, set flow-matching shift dynamically based on resolution")
    argparser.add_argument("--flow_match_shift_dropout_p", type=float, default=0.3,
                           help="Probability that a given batch will see unshifted timesteps when doing flow-matching shift")
    argparser.add_argument("--keep_tags", type=int, default=0,
                           help="Number of tags to keep when shuffle, used to randomly select subset of tags")
    argparser.add_argument("--wandb", action="store_true", default=False,
                           help="enable wandb logging instead of tensorboard, requires env var WANDB_API_KEY")
    argparser.add_argument("--validation_config", default=None,
                           help="Path to a JSON configuration file for the validator.")
    argparser.add_argument("--no_initial_validation", action="store_true",
                           help="If passed, don't do validation before the first step")
    argparser.add_argument("--write_schedule", action="store_true", default=False,
                           help="write schedule of images and their batches to file (def: False)")
    argparser.add_argument("--rated_dataset", action="store_true", default=False,
                           help="enable rated image set training")
    argparser.add_argument("--rated_dataset_target_dropout_percent", type=int, default=50,
                           help="how many images (in percent) should be included in the last epoch (Default 50)")
    argparser.add_argument("--zero_frequency_noise_ratio", type=float, default=0.02,
                           help="adds zero frequency noise, for improving contrast (def: 0.0)")
    argparser.add_argument("--enable_zero_terminal_snr", action=argparse.BooleanOptionalAction, default=None,
                           help="Use zero terminal SNR noising beta schedule")
    argparser.add_argument("--mix_zero_terminal_snr", action="store_true", default=None,
                           help="Mix zero terminal SNR with regular training")
    argparser.add_argument("--match_zero_terminal_snr", action="store_true", default=None,
                           help="use zero terminal SNR target as regular noise scheduler input")
    argparser.add_argument("--load_settings_every_epoch", action="store_true", default=None,
                           help="Enable reloading of 'train.json' at start of every epoch.")
    argparser.add_argument("--loss_mode_scale", default=0, type=float,
                           help="Mode scale for mode-curve loss scaling. default 0/disabled")
    argparser.add_argument("--min_snr_gamma", type=float, default=None,
                           help="min-SNR-gamma parameter. Recommended values: 5, 1, 20.")
    argparser.add_argument("--min_snr_alpha", type=float, default=1,
                           help="Blending factor for min-SNR-gamma weighting.")
    argparser.add_argument("--debug_invert_min_snr_gamma", action='store_true',
                           help="invert the timestep/scale equation for min snr gamma")
    argparser.add_argument("--ema_decay_rate", type=float, default=None,
                           help="EMA decay rate.")
    argparser.add_argument("--ema_strength_target", type=float, default=None,
                           help="EMA decay target value in range (0,1).")
    argparser.add_argument("--ema_update_interval", type=int, default=500,
                           help="How many steps between optimizer steps that EMA decay updates.")
    argparser.add_argument("--ema_device", type=str, default='cpu',
                           help="EMA storage location. 'cpu' / 'cuda' / 'disk'. (default: 'cpu')")
    argparser.add_argument("--ema_sample_nonema_model", action=argparse.BooleanOptionalAction, default=False,
                           help="Generate samples from non-EMA trained model.")
    argparser.add_argument("--ema_sample_ema_model", action=argparse.BooleanOptionalAction, default=True,
                           help="Generate samples from EMA model.")
    argparser.add_argument("--ema_resume_model", type=str, default=None,
                           help="The EMA decay checkpoint to resume from")
    argparser.add_argument("--pyramid_noise_discount", type=float, default=None,
                           help="Enables pyramid noise and use specified discount factor")
    argparser.add_argument("--batch_share_noise", action="store_true",
                           help="All samples in a batch have the same noise")
    argparser.add_argument("--batch_share_timesteps", action="store_true",
                           help="All samples in a batch have the same timesteps")
    argparser.add_argument("--teacher", type=str, nargs='*', default=[],
                           help="Teacher model path(s). Pass one or more paths to enable distillation from multiple teachers.")
    argparser.add_argument("--teacher_p", type=float, default=1.0,
                           help="Probability of teacher model being used as target")
    argparser.add_argument("--teacher_lambda", type=float, default=1.0,
                           help="When teacher is used, the scale factor for teacher loss")
    argparser.add_argument("--teacher_lambda_falloff", action=argparse.BooleanOptionalAction, default=False,
                           help="When enabled, teacher_lambda falls off linearly to 0")
    argparser.add_argument("--teacher_timestep_max", type=int, default=None,
                           help="Maximum timestep where the teacher model will be used")
    argparser.add_argument("--teacher_prediction_type", type=str, default="auto",
                           choices=["auto", "flow_prediction", "v_prediction", "epsilon"],
                           help="Override the teacher scheduler prediction type.")
    argparser.add_argument("--flow_match_t_clamp_min", type=int, default=None,
                           help="Clamp sampled FM timestep indices to at least this value (0-999).")
    argparser.add_argument("--flow_match_t_clamp_max", type=int, default=None,
                           help="Clamp sampled FM timestep indices to at most this value (0-999).")
    argparser.add_argument("--local_contrastive_flow_loss_p", type=float, default=0,
                           help="Probability that a given batch will have Local Contrastive Flow loss applied.")
    argparser.add_argument("--local_contrastive_flow_timestep_threshold", type=int, default=200,
                           help="Timesteps smaller than this will be subject to local contrastive flow (LCF) loss")
    argparser.add_argument("--local_contrastive_flow_anchor_timestep", type=int, default=500,
                           help="Anchor timestep (medium loss) for LCF loss")
    argparser.add_argument("--local_contrastive_flow_temperature", type=int, default=0.07,
                           help="Temperature for LCF loss InfoNCE cross entropy computation")
    argparser.add_argument("--local_contrastive_flow_lambda", type=int, default=0.1,
                           help="Lambda scaling factor for Local Contrastive Flow loss")
    argparser.add_argument("--contrastive_flow_matching_loss_p", type=float, default=0,
                           help="Probability that a given batch will have Contrastive Flow Matching loss applied")
    argparser.add_argument("--contrastive_flow_matching_loss_lambda", type=float, default=0.05,
                           help="Lambda scaling factor for Contrastive Flow Matching loss.")
    argparser.add_argument("--saturation_penalty_scale", type=float, default=0.0,
                           help="Scale for the saturation penalty loss.")
    argparser.add_argument("--saturation_penalty_t_max", type=float, default=200.0,
                           help="Only apply saturation penalty for timesteps < t_max.")
    argparser.add_argument("--self_flow_p", type=float, default=0.0,
                           help="Self-Flow: probability per step that the self-distillation representation loss is applied.")
    argparser.add_argument("--self_flow_gamma", type=float, default=0.8,
                           help="Self-Flow: weighting factor γ for the representation loss.")
    argparser.add_argument("--self_flow_mask_ratio", type=float, default=0.25,
                           help="Self-Flow: fraction of spatial latent tokens that use the secondary noise level.")
    argparser.add_argument("--self_flow_ema_decay", type=float, default=0.9999,
                           help="Self-Flow: EMA decay rate for the teacher UNet.")
    argparser.add_argument("--self_flow_ema_update_interval", type=int, default=1,
                           help="Self-Flow: update the teacher EMA every N optimizer steps.")
    argparser.add_argument("--self_flow_mode", type=str, default='shallow', choices=SELF_FLOW_MODES,
                           help="Self-Flow: extraction-point arrangement.")
    argparser.add_argument("--contrastive_learning_dropout_p", type=float, default=0,
                           help="Probability to drop (non-LCF/non-CFM) contrastive learning")
    argparser.add_argument("--contrastive_loss_batch_ids", type=str, nargs="*", default=[],
                           help="Batch ids for which contrastive learning should be done")
    argparser.add_argument("--contrastive_loss_scale", type=float, default=1,
                           help="Scaling factor for contrastive loss")
    argparser.add_argument("--contrastive_loss_type", type=str,
                           choices=['infonce', 'delta', 'infonce_with_text_similarity', 'infonce_softrepa'],
                           help="Type of contrastive loss")
    argparser.add_argument("--contrastive_loss_softrepa_sigma", type=float, default=1.0,
                           help="Sigma value for SoftREPA infoNCE contrastive loss")
    argparser.add_argument("--contrastive_loss_temperature", type=float, default=0.07,
                           help="Temperature for infonce contrastive loss")
    argparser.add_argument("--contrastive_loss_hard_negative_weight", type=float, default=2,
                           help="weight factor for more difficult negative pairs when doing infoNCE")
    argparser.add_argument("--everything_contrastive_learning_p", type=float, default=0,
                           help="probability to run contrastive learning on everything, 0..1")
    argparser.add_argument("--everything_contrastive_learning_curriculum_alpha", type=float, default=0,
                           help="if >0, attenuate everything_contrastive_learning_p to 0 as timestep approaches 0")
    argparser.add_argument("--caption_variants", type=str, nargs="*", default=[],
                           help="If passed, use only these caption variants from json captions")
    argparser.add_argument("--all_caption_variants", action=argparse.BooleanOptionalAction,
                           help='if passed, use ALL caption variants every step')
    argparser.add_argument("--expand_caption_variants", action=argparse.BooleanOptionalAction,
                           help='if passed, expand caption variant dicts into individual items')
    argparser.add_argument("--caption_cross_concatenation_p", type=float, default=0,
                           help="Probability of doing caption cross concatenation.")
    argparser.add_argument("--caption_cross_concatenation_empty_half_p", type=float, default=0.2,
                           help="When doing caption cross concatenation, probability of one variant being an empty prompt")
    argparser.add_argument("--use_compel", action='store_true',
                           help='if passed, use Compel to process prompts with long-prompt support')
    argparser.add_argument("--batch_id_dropout_p", type=float, default=0,
                           help="dropout probability for batch ids, 0..1")
    argparser.add_argument("--cond_dropout_noise_p", type=float, default=0,
                           help="how often to use noise for the image with conditional dropout")
    argparser.add_argument("--mask_p", type=float, default=None,
                           help="If passed, look for files called eg image_name.jpg.mask.png and use as mask for the loss")
    argparser.add_argument("--invert_masks", action=argparse.BooleanOptionalAction,
                           help="If passed, invert the masks (white<->black)")
    argparser.add_argument("--use_both_mask_sides_contrastive", action='store_true',
                           help="If passed, when using masks, do contrastive learning between mask and inverted mask")
    argparser.add_argument("--lora", action='store_true',
                           help="If passed, do LoRA training")
    argparser.add_argument("--lora_save_every_n_epochs", type=int, default=1)
    argparser.add_argument("--lora_resume", type=str, default=None,
                           help="resume from this lora (must be a huggingface format folder)")
    argparser.add_argument("--lora_rank", type=int, default=16)
    argparser.add_argument("--lora_alpha", type=int, default=8)
    argparser.add_argument("--test_images", action="store_true",
                           help="check all images by trying to load them")
    argparser.add_argument("--offload_vae", action="store_true",
                           help="If passed, offload VAE to CPU when not in use")
    argparser.add_argument("--offload_text_encoder", action="store_true",
                           help="If passed, offload text encoder(s) to CPU when not in use")
    argparser.add_argument("--no_save_on_error", action="store_true",
                           help="If passed, do not save model on error/ctrl-c")
    argparser.add_argument("--clip_vision_model_source", default=None,
                           help="If specified, the vision model to use for text encoder contrastive training")
    argparser.add_argument("--clip_vision_model_processor_source", default=None,
                           help="If specified, the preprocessor to use for text encoder contrastive training")
    argparser.add_argument("--clip_vision_contrastive_loss_lambda", type=float, default=0.1,
                           help="Lambda scaling factor for contrastive loss between text encoder and CLIP vision model features")
    argparser.add_argument("--debug_no_load_model", action="store_true",
                           help="If passed, do not load model weights (for testing purposes only)")
    argparser.add_argument("--debug_teacher", action="store_true", default=False,
                           help="If passed, log detailed teacher/student latent stats")
    argparser.add_argument("--debug_log_on_nan", action=argparse.BooleanOptionalAction,
                           help="If specified, use set_detect_anomaly to find NaNs in autograd. Slow.")
    # mixed_precision is used by train_sana.py but not train.py (which uses disable_amp/force_bfloat16)
    # include it here so SANA JSON configs can set it
    argparser.add_argument("--mixed_precision", type=str, default="bf16",
                           choices=["bf16", "fp16", "no"],
                           help="Mixed precision mode for SANA training (bf16/fp16/no). "
                                "For SD/SDXL use --disable_amp / --force_bfloat16 instead.")

    return argparser


# ---------------------------------------------------------------------------
# Two-pass parser entry point
# ---------------------------------------------------------------------------

def parse_train_args(
    description: str = "EveryDream2 Training options",
    extra_args_fn=None,
    require_resume_ckpt: bool = True,
    argv: list | None = None,
) -> argparse.Namespace:
    """
    Full two-pass argument parse compatible with both train.py and train_sana.py.

    Pass 1  — parse only --config so we can load any JSON config file first.
    Pass 2  — build the full shared parser, apply extra_args_fn if provided,
              parse the remaining CLI argv against the JSON-loaded namespace.

    extra_args_fn: optional callable(argparser) to add script-specific args
                   (e.g. --model_id for SANA) before the second pass.
    require_resume_ckpt: False for SANA (which uses --model_id instead).
    """
    # Pass 1: just --config
    pre_parser = argparse.ArgumentParser(description=description, add_help=False)
    pre_parser.add_argument("--config", type=str, required=False, default=None)
    args, remaining_argv = pre_parser.parse_known_args(argv)

    if args.config:
        load_train_json_from_file(args, report_load=True)

    # Pass 2: full parse
    full_parser = build_argparser(
        description,
        config_namespace=args,
        require_resume_ckpt=require_resume_ckpt,
    )
    if extra_args_fn is not None:
        extra_args_fn(full_parser)

    args = full_parser.parse_args(args=remaining_argv, namespace=args)
    return args

