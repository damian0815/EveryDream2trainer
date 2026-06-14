
import json
import os

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import DiffusionPipeline

from utils.sample_generator_diffusers import generate_images_diffusers, ImageGenerationParams


def do_gens(model_path, gens_json_path, output_root, is_flow_matching=False, device='cuda', batch_size=12, start_offset: int=0):
    scheduler_override = 'flow-matching' if is_flow_matching else None
    with open(gens_json_path, 'r') as f:
        gens_json = json.load(f)
        gens = [ImageGenerationParams.from_invokeai_metadata(d, index, scheduler_override=scheduler_override) for index, d in enumerate(gens_json)]
    os.makedirs(output_root, exist_ok=True)

    def save_image(image,
                  sample_index: int,
                   prompt: str,
                  pngmetadata: PngInfo):
        output_path = os.path.join(output_root, f'{sample_index}.jpg')
        image.save(output_path, quality=95, pnginfo=pngmetadata)
        with open(os.path.splitext(output_path)[0] + '.txt', 'w') as f:
            f.write(prompt)

    pipe = DiffusionPipeline.from_pretrained(model_path).to(device, torchdtype=torch.float16)
    generate_images_diffusers(pipe, model_name=os.path.basename(model_path), model_type='sd-2', all_params=gens, batch_size=batch_size, flow_match_shift_dynamic=True, image_save_cb=save_image, start_offset=start_offset)



if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Evaluate a Stable Diffusion model on a set of prompts and compare to a reference model.')
    arg_parser.add_argument('--model_path', type=str, required=True, help='Path to the model to evaluate, such as a local path or a HuggingFace repo id.')
    arg_parser.add_argument('--gens_json_path', type=str, required=True, help='Path to a JSON file containing a list of dicts controlling generation')
    arg_parser.add_argument('--output_root', type=str, required=True, help='Path to a directory to save the output images')
    arg_parser.add_argument('--is_flow_matching', action=argparse.BooleanOptionalAction, help='Whether the model being evaluated is a flow-matching model - if True, overrides the scheduler to use flow-matching scheduler, and applies the flow-match shift to the dynamic parameters.')
    arg_parser.add_argument('--device', type=str, default='cuda', help='Device to use (default "cuda")')
    arg_parser.add_argument('--batch_size', type=int, default=12, help='Batch size to use for generation (default 12)')
    arg_parser.add_argument('--start_offset', type=int, default=0, help='Starting index offset to add to the sample indices when saving (default 0)')

    args = arg_parser.parse_args()

    do_gens(args.model_path, args.gens_json_path, output_root=args.output_root,
            batch_size=args.batch_size, device=args.device,
            start_offset=args.start_offset,
            is_flow_matching=args.is_flow_matching)



