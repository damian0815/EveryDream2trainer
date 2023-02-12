
import argparse

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

from data import aspects, resolver
from data.data_loader import DataLoaderMultiAspect
from data.every_dream import EveryDreamBatch, build_torch_dataloader
from data.every_dream_validation import EveryDreamValidator, OutlierFinder


def get_model_prediction_and_target(pipeline: StableDiffusionPipeline, image, tokens, amp: bool=True) -> tuple[torch.Tensor, torch.Tensor]:

    unet = pipeline.unet
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler
    text_encoder = pipeline.text_encoder

    with torch.no_grad():
        with autocast(enabled=amp):
            pixel_values = image.to(memory_format=torch.contiguous_format).to(unet.device)
            latents = vae.encode(pixel_values, return_dict=False)
        del pixel_values
        latents = latents[0].sample() * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        cuda_caption = tokens.to(text_encoder.device)

    # with autocast(enabled=args.amp):
    encoder_hidden_states = text_encoder(cuda_caption, output_hidden_states=True)

    if args.clip_skip > 0:
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(
            encoder_hidden_states.hidden_states[-args.clip_skip])
    else:
        encoder_hidden_states = encoder_hidden_states.last_hidden_state

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    del noise, latents, cuda_caption

    with autocast(enabled=amp):
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    return model_pred, target


def find_outliers(first_ckpt_path_or_repo_id, second_ckpt_path_or_repo_id, data_root: str,
                  resolution: int, validation_config: str=None, original_batch_size: int=4):

    print(f"loading first model {first_ckpt_path_or_repo_id}...")
    first_model = StableDiffusionPipeline.from_pretrained(first_ckpt_path_or_repo_id)
    print(f"loading second model {first_ckpt_path_or_repo_id}...")
    second_model = StableDiffusionPipeline.from_pretrained(second_ckpt_path_or_repo_id)

    print("preparing data...")
    args = argparse.Namespace(
        aspects=aspects.get_aspect_buckets(resolution),
        flip_p=0.0,
        seed=555,
    )
    items = resolver.resolve_root(data_root, args)

    dlma = DataLoaderMultiAspect(items, batch_size=1)
    ed_batch = EveryDreamBatch(
        data_loader=dlma,
        debug_level=1,
        conditional_dropout=0,
        tokenizer=first_model.tokenizer,
    )
    dataloader = build_torch_dataloader(ed_batch, batch_size=1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("first_ckpt", type=str, required=True, help="First (earlier) model to compare (path to ckpt or hf repo id)")
    parser.add_argument("second_ckpt", type=str, required=True, help="Second (later) model to compare (path to ckpt or hf repo id)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--resolution", type=int, required=True, help="Resolution the model was trained for")
    parser.add_argument("--validation_config", type=str, required=False, default=None, help="(optional) Path to JSON validation config (if you used 'automatic' split mode to train, you need to pass the same validation config here to reproduce the same split)")
    parser.add_argument("--original_batch_size", type=int, required=False, help="The batch size the model was trained using (necessary to reconstruct 'automatic' val split)")

    args = parser.parse_args()

    find_outliers(args.first_ckpt, args.second_ckpt, data_root=args.data_root,
                  validation_config=args.validation_config, original_batch_size=args.original_batch_size)