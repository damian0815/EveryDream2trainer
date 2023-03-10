import argparse
import os.path
from typing import Generator, Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_xformers_available
from tqdm.auto import tqdm

# make imports from data and train.py work
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data import resolver, aspects
from data.data_loader import DataLoaderMultiAspect
from data.every_dream import EveryDreamBatch, build_torch_dataloader
from data.image_train_item import ImageTrainItem
from train import ModelPredictionAndTargetComputer


def load_image_train_items(path: str, resolution: int) -> list[ImageTrainItem]:
    args = argparse.Namespace(
        aspects=aspects.get_aspect_buckets(resolution),
        flip_p=0.0,
        seed=555,
    )
    return resolver.resolve_root(path, args)


@torch.no_grad()
def compute_training_progress(image_train_items: list[ImageTrainItem],
                              model_a: StableDiffusionPipeline,
                              model_b: Optional[StableDiffusionPipeline]=None,
                              seed: int = 555,
                              use_xformers: bool = True,
                              device: str = 'cuda',
                              timesteps: list[float] = None) -> Generator[tuple[str, float, str], None, None]:

    # assumption: model_a and model_b's tokenizers are equivalent

    if timesteps is None:
        timesteps = [0.5]
    tokenizer = model_a.tokenizer

    dlma = DataLoaderMultiAspect(image_train_items=image_train_items, seed=seed)
    ed_batch = EveryDreamBatch(dlma, conditional_dropout=0, crop_jitter=0, tokenizer=tokenizer, seed=seed)
    dataloader = build_torch_dataloader(ed_batch, batch_size=1)

    all_model_losses = []
    models = [model_a] if model_b is None else [model_a, model_b]
    for model in tqdm(models, desc="models"):
        model = model.to(device)
        if use_xformers and is_xformers_available():
            model.enable_xformers_memory_efficient_attention()

        prediction_and_target_computer = ModelPredictionAndTargetComputer(vae=model.vae,
                                                                          text_encoder=model.text_encoder,
                                                                          unet=model.unet,
                                                                          noise_scheduler=model.scheduler,
                                                                          amp=False,
                                                                          clip_skip=0)
        torch.manual_seed(seed)

        model_losses = []

        iterations_pbar = tqdm(timesteps, leave=False, position=1, desc="iterations")
        for timestep in iterations_pbar:
            iterations_pbar.set_postfix({'timestep': timestep})
            iteration_losses = []
            steps_pbar = tqdm(total=len(dataloader), leave=False, position=2)
            for step, batch in enumerate(dataloader):
                model_pred, target = prediction_and_target_computer(batch["image"], batch["tokens"],
                                                                    timestep=timestep)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                del target, model_pred
                iteration_losses.append(loss.detach().item())
                del loss

                steps_pbar.update(1)

            model_losses.append(torch.tensor(iteration_losses))

        mean_model_losses = torch.mean(torch.stack(model_losses), dim=0)
        all_model_losses.append(mean_model_losses)

        model.to("cpu")

    loss_deltas = all_model_losses[0] if model_b is None else (all_model_losses[1] - all_model_losses[0])
    for i, loss_delta in enumerate(loss_deltas.tolist()):
        image_train_item = ed_batch.image_train_items[i]
        yield image_train_item.pathname, loss_delta, image_train_item.caption.get_caption()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str, help="The trained model to evaluate.")
    parser.add_argument("--baseline", required=False, type=str, help="(Optional) Baseline to compare the trained model to. If omitted, log absolute losses of the trained model (but you should set iterations >=3)")
    parser.add_argument("--data_root", required=True, type=str, help="Path to training images and captions.")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output CSV")
    parser.add_argument("--iterations", required=False, type=int, default=1, help="(Optional, default=1) How many iterations to run. More iterations gives more accurate results.")
    parser.add_argument("--timesteps", required=False, type=float, nargs="+", help="(Optional) A list of timesteps to evaluate (0-1 where 1 is pure noise and 0 is the clean, noise-free image). Overrides --iterations.")
    parser.add_argument("--resolution", type=int, required=False, default=512, help="(Optional) Resolution for the loss computations (default=512)")
    parser.add_argument("--disable_xformers", required=False, action="store_true", help="Do not use xformers")

    args = parser.parse_args()

    image_train_items = load_image_train_items(args.data_root, resolution=args.resolution)

    print(f"loading {args.trained}...")
    model_a = StableDiffusionPipeline.from_pretrained(args.trained)
    if args.baseline is not None:
        print(f"loading {args.baseline}...")
        model_b = StableDiffusionPipeline.from_pretrained(args.baseline)
    else:
        model_b = None


    if args.timesteps is not None:
        if args.iterations != 1:
            print("overriding --iterations {args.iterations} with --timesteps {args.timesteps}")
        timesteps = args.timesteps
    else:
        # for 1 iteration, timestep is 0.5
        # for 2 iterations, timesteps are 0.333, 0.666
        # for 3 iterations, timesteps are 0.25, 0.5, 0.75
        # etc.
        timesteps = [i / (args.iterations + 1) for i in range(1, args.iterations + 1)]

    per_item_progress = list(compute_training_progress(image_train_items,
                                                       model_a,
                                                       model_b,
                                                       use_xformers=not args.disable_xformers,
                                                       timesteps=timesteps,
                                                       ))
    per_item_progress = sorted(per_item_progress, key=lambda x: x[0])
    base_folder = os.path.commonpath([p[0] for p in per_item_progress])

    with open(args.output_path, "wt") as f:
        f.write('"loss delta", "path", "caption"\n')
        for progress in per_item_progress:
            f.write(f"{progress[1]}, \"{os.path.relpath(progress[0], base_folder)}\", \"{progress[2]}\"\n")


if __name__ == '__main__':
    main()