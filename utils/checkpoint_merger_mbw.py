import math

import shutil

import tempfile

import glob
import argparse
import os
from typing import Dict, List, Union

import torch

from tqdm.auto import tqdm

import safetensors.torch

from huggingface_hub import snapshot_download

from diffusers import DiffusionPipeline, __version__, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, ONNX_WEIGHTS_NAME, WEIGHTS_NAME
DIFFUSERS_CACHE="~/.cache/huggingface/hub"


NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

DIFFUSERS_KEY_PREFIX_TO_WEIGHT_INDEX = {
    "time_embedding.": 0,
    "conv_in.": 0,
    "down_blocks.0.resnets.0": 1,
    "down_blocks.0.attentions.0": 1,
    "down_blocks.0.resnets.1": 2,
    "down_blocks.0.attentions.1": 2,
    "down_blocks.0.downsamplers": 3,
    "down_blocks.1.resnets.0": 4,
    "down_blocks.1.attentions.0": 4,
    "down_blocks.1.resnets.1": 5,
    "down_blocks.1.attentions.1": 5,
    "down_blocks.1.downsamplers": 6,
    "down_blocks.2.resnets.0": 7,
    "down_blocks.2.attentions.0": 7,
    "down_blocks.2.resnets.1": 8,
    "down_blocks.2.attentions.1": 8,
    "down_blocks.2.downsamplers": 9,
    "down_blocks.3.resnets.0": 10,
    "down_blocks.3.resnets.1": 11,
    "mid_block": 12,
    "up_blocks.0.resnets.0": 13,
    "up_blocks.0.resnets.1": 14,
    "up_blocks.0.resnets.2": 15,
    "up_blocks.0.upsamplers.0": 15,
    "up_blocks.1.resnets.0": 16,
    "up_blocks.1.attentions.0": 16,
    "up_blocks.1.resnets.1": 17,
    "up_blocks.1.attentions.1": 17,
    "up_blocks.1.resnets.2": 18,
    "up_blocks.1.upsamplers.0": 18,
    "up_blocks.1.attentions.2": 18,
    "up_blocks.2.resnets.0": 19,
    "up_blocks.2.attentions.0": 19,
    "up_blocks.2.resnets.1": 20,
    "up_blocks.2.attentions.1": 20,
    "up_blocks.2.resnets.2": 21,
    "up_blocks.2.upsamplers.0": 21,
    "up_blocks.2.attentions.2": 21,
    "up_blocks.3.resnets.0": 22,
    "up_blocks.3.attentions.0": 22,
    "up_blocks.3.resnets.1": 23,
    "up_blocks.3.attentions.1": 23,
    "up_blocks.3.resnets.2": 24,
    "up_blocks.3.attentions.2": 24,
    "conv_norm_out.": 24,
    "conv_out.": 24,
}

def dprint(str, flg):
    if flg:
        print(str)

def get_weight_index(key: str) -> int:
    for k, v in DIFFUSERS_KEY_PREFIX_TO_WEIGHT_INDEX.items():
        if key.startswith(k):
            return v
    raise ValueError(f"Unknown unet key: {key}")


def get_block_alpha(block_weights: list, key: str) -> float:
    weight_index = get_weight_index(key)
    return block_weights[weight_index]


class CheckpointMergerPipeline(DiffusionPipeline):
    """
    A class that that supports merging diffusion models based on the discussion here:
    https://github.com/huggingface/diffusers/issues/877

    Example usage:-

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="checkpoint_merger.py")

    merged_pipe = pipe.merge(["CompVis/stable-diffusion-v1-4","prompthero/openjourney"], interp = 'inv_sigmoid', alpha = 0.8, force = True)

    merged_pipe.to('cuda')

    prompt = "An astronaut riding a unicycle on Mars"

    results = merged_pipe(prompt)

    ## For more details, see the docstring for the merge method.

    """

    def __init__(self):
        self.register_to_config()
        super().__init__()

    def _compare_model_configs(self, dict0, dict1):
        if dict0 == dict1:
            return True
        else:
            config0, meta_keys0 = self._remove_meta_keys(dict0)
            config1, meta_keys1 = self._remove_meta_keys(dict1)
            if config0 == config1:
                print(f"Warning !: Mismatch in keys {meta_keys0} and {meta_keys1}.")
                return True
        return False

    def _remove_meta_keys(self, config_dict: Dict):
        meta_keys = []
        temp_dict = config_dict.copy()
        for key in config_dict.keys():
            if key.startswith("_"):
                temp_dict.pop(key)
                meta_keys.append(key)
        return (temp_dict, meta_keys)

    @torch.no_grad()
    def merge(self, pretrained_model_name_or_path_list: List[Union[str, os.PathLike]], **kwargs):
        """
        Returns a new pipeline object of the class 'DiffusionPipeline' with the merged checkpoints(weights) of the models passed
        in the argument 'pretrained_model_name_or_path_list' as a list.

        Parameters:
        -----------
            pretrained_model_name_or_path_list : A list of valid pretrained model names in the HuggingFace hub or paths to locally stored models in the HuggingFace format.

            **kwargs:
                Supports all the default DiffusionPipeline.get_config_dict kwargs viz..

                cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map.
                
                preloaded_model_0 - (Optional) Preloaded StableDiffusionPipeline representing `pretrained_model_name_or_path_list[0]`. Will be modified during the merge and (therefore redundantly) returned as the merged pipeline.

                alpha - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                    would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2

                interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_diff" and None.
                    Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_diff" is supported.

                force - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

                block_weights - list of 25 floats for per-block weighting. ref https://rentry.org/Merge_Block_Weight_-china-_v1_Beta#3-basic-theory-explanation

                module_override_alphas - dict of str -> float for per-module alpha overrides eg {'unet': 0.2, 'text_encoder': 0.8}

        """
        # Default kwargs from DiffusionPipeline
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        preloaded_model_0 = kwargs.pop("preloaded_model_0", None)

        alpha = kwargs.pop("alpha", 0.5)
        interp = kwargs.pop("interp", None)
        block_weights: list[float] = kwargs.pop("block_weights", None)
        module_override_alphas: dict[str, float] = kwargs.pop("module_override_alphas", {})

        print("Received list", pretrained_model_name_or_path_list)
        print(f"Combining with alpha={alpha}, interpolation mode={interp}")
        if block_weights is not None:
            print(f"Merging unet using block weights {block_weights}")

        checkpoint_count = len(pretrained_model_name_or_path_list)
        # Ignore result from model_index_json comparision of the two checkpoints
        force = kwargs.pop("force", False)

        # If less than 2 checkpoints, nothing to merge. If more than 3, not supported for now.
        if checkpoint_count > 3 or checkpoint_count < 2:
            raise ValueError(
                "Received incorrect number of checkpoints to merge. Ensure that either 2 or 3 checkpoints are being"
                " passed."
            )

        print("Received the right number of checkpoints")
        # chkpt0, chkpt1 = pretrained_model_name_or_path_list[0:2]
        # chkpt2 = pretrained_model_name_or_path_list[2] if checkpoint_count == 3 else None

        # Validate that the checkpoints can be merged
        # Step 1: Load the model config and compare the checkpoints. We'll compare the model_index.json first while ignoring the keys starting with '_'
        config_dicts = []
        for i,pretrained_model_name_or_path in enumerate(pretrained_model_name_or_path_list):
            if i == 0 and preloaded_model_0 is not None:
                print("preloaded_model_0 - skipping compatibility check for model 0")
                config_dict = None
            else:
                print(f"loading DiffusionPipeline from {pretrained_model_name_or_path}...")
                config_dict = DiffusionPipeline.load_config(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    resume_download=resume_download,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                )
            config_dicts.append(config_dict)

        comparison_result = True
        for idx in range(1, len(config_dicts)):
            if config_dicts[idx-1] is None:
                continue
            comparison_result &= self._compare_model_configs(config_dicts[idx - 1], config_dicts[idx])
            if not force and comparison_result is False:
                print(config_dicts[0], config_dicts[1])
                raise ValueError("Incompatible checkpoints. Please check model_index.json for the models.")
        print("Compatible model_index.json files found")
        # Step 2: Basic Validation has succeeded. Let's download the models and save them into our local files.
        cached_folders = []
        for i, pretrained_model_name_or_path, config_dict in zip(range(len(config_dicts)), pretrained_model_name_or_path_list, config_dicts):
            if i == 0 and preloaded_model_0 is not None:
                print("preloaded_model_0 - skipping compatibility check for model 0")
                cached_folders.append(None)
                continue
            print(f"Loading {pretrained_model_name_or_path}...")
            folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
            allow_patterns = [os.path.join(k, "*") for k in folder_names]
            allow_patterns += [
                WEIGHTS_NAME,
                SCHEDULER_CONFIG_NAME,
                CONFIG_NAME,
                ONNX_WEIGHTS_NAME,
                DiffusionPipeline.config_name,
            ]
            requested_pipeline_class = config_dict.get("_class_name")
            user_agent = {"diffusers": __version__, "pipeline_class": requested_pipeline_class}

            cached_folder = (
                pretrained_model_name_or_path
                if os.path.isdir(pretrained_model_name_or_path)
                else snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    allow_patterns=allow_patterns,
                    user_agent=user_agent,
                )
            )
            print("Cached Folder", cached_folder)
            cached_folders.append(cached_folder)

        # Step 3:-
        # Load the first checkpoint as a diffusion pipeline and modify its module state_dict in place
        if preloaded_model_0 is not None:
            final_pipe = preloaded_model_0
        else:
            final_pipe = DiffusionPipeline.from_pretrained(
                cached_folders[0], torch_dtype=torch_dtype, device_map=device_map
            )
        final_pipe.to(self.device)

        if interp == "sigmoid":
            theta_func = CheckpointMergerPipeline.sigmoid
        elif interp == "inv_sigmoid":
            theta_func = CheckpointMergerPipeline.inv_sigmoid
        elif interp == "reset_and_perturb":
            theta_func = CheckpointMergerPipeline.reset_and_perturb
        elif interp == "add_diff":
            theta_func = CheckpointMergerPipeline.add_difference
        elif interp == "weighted_sum":
            theta_func = CheckpointMergerPipeline.weighted_sum
        else:
            raise ValueError(f"unrecognized interpolation method '{interp}'")

        # Find each module's state dict.
        for attr in final_pipe.config.keys():
            if not attr.startswith("_"):
                checkpoint_path_1 = os.path.join(cached_folders[1], attr)
                if os.path.exists(checkpoint_path_1):
                    files = list(
                        (
                            *glob.glob(os.path.join(checkpoint_path_1, "*.safetensors")),
                            *glob.glob(os.path.join(checkpoint_path_1, "*.bin")),
                        )
                    )
                    checkpoint_path_1 = files[0] if len(files) > 0 else None
                if len(cached_folders) < 3:
                    checkpoint_path_2 = None
                else:
                    checkpoint_path_2 = os.path.join(cached_folders[2], attr)
                    if os.path.exists(checkpoint_path_2):
                        files = list(
                            (
                                *glob.glob(os.path.join(checkpoint_path_2, "*.safetensors")),
                                *glob.glob(os.path.join(checkpoint_path_2, "*.bin")),
                            )
                        )
                        checkpoint_path_2 = files[0] if len(files) > 0 else None
                # For an attr if both checkpoint_path_1 and 2 are None, ignore.
                # If atleast one is present, deal with it according to interp method, of course only if the state_dict keys match.
                if checkpoint_path_1 is None and checkpoint_path_2 is None:
                    print(f"Skipping {attr}: not present in 2nd or 3rd model")
                    continue
                try:
                    module = getattr(final_pipe, attr)

                    if isinstance(module, bool):  # ignore requires_safety_checker boolean
                        continue
                    theta_0 = getattr(module, "state_dict")
                    theta_0 = theta_0()

                    update_theta_0 = getattr(module, "load_state_dict")
                    theta_1 = (
                        safetensors.torch.load_file(checkpoint_path_1)
                        if (checkpoint_path_1.endswith(".safetensors"))
                        else torch.load(checkpoint_path_1, map_location="cpu")
                    )
                    theta_2 = None
                    if checkpoint_path_2:
                        theta_2 = (
                            safetensors.torch.load_file(checkpoint_path_2)
                            if (checkpoint_path_2.endswith(".safetensors"))
                            else torch.load(checkpoint_path_2, map_location="cpu")
                        )

                    if not theta_0.keys() == theta_1.keys():
                        in_theta0_not_theta1 = [k for k in theta_0.keys() if k not in theta_1.keys()]
                        in_theta1_not_theta0 = [k for k in theta_1.keys() if k not in theta_0.keys()]
                        if (
                            any(k != 'text_model.embeddings.position_ids' for k in in_theta0_not_theta1) 
                            or 
                            any(k != 'text_model.embeddings.position_ids' for k in in_theta1_not_theta0)
                        ): 
                            print(f"Skipping {attr}: key mismatch")
                            print("in theta_0, not in theta_1:", in_theta0_not_theta1)
                            print("in theta_1, not in theta_0:", in_theta1_not_theta0)
                            continue
                        else:
                            print('key mismatch but it was just text_model.embeddings.position_ids - ignoring')
                    if theta_2 and not theta_1.keys() == theta_2.keys():
                        print(f"Skipping {attr}: key mismatch")

                except Exception as e:
                    print(f"Skipping {attr} do to an unexpected error: {str(e)}")
                    continue
                print(f"MERGING {attr}")
                module_alpha = alpha
                if block_weights is not None and type(module) is UNet2DConditionModel:
                    print(f" - using block weights {block_weights}")
                elif module_override_alphas.get(attr, None) is not None:
                    print(f" - using override alpha {module_override_alphas[attr]}")
                    module_alpha = module_override_alphas.get(attr)

                for key in tqdm(list(theta_0.keys())):
                    if key not in theta_1:
                        assert key == 'text_model.embeddings.position_ids'
                        continue
                    block_alpha = (
                        module_alpha if (type(module) is not UNet2DConditionModel or block_weights is None)
                        else get_block_alpha(block_weights, key)
                    )
                    if theta_2:
                        theta_0[key] = theta_func(theta_0[key], theta_1[key], theta_2[key], block_alpha)
                    else:
                        theta_0[key] = theta_func(theta_0[key], theta_1[key], None, block_alpha)

                del theta_1
                del theta_2
                update_theta_0(theta_0)

                del theta_0
        return final_pipe

    @staticmethod
    def weighted_sum(theta0, theta1, theta2, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def sigmoid(theta0, theta1, theta2, alpha):
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def inv_sigmoid(theta0, theta1, theta2, alpha):
        import math

        alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
        return theta0 + ((theta1 - theta0) * alpha)

    @staticmethod
    def add_difference(theta0, theta1, theta2, alpha):
        return theta0 + (theta1 - theta2) * (1.0 - alpha)

    @staticmethod 
    def reset_and_perturb(theta0, theta1, theta2, alpha):
        delta = theta1 - theta0
        delta_scaled = delta * alpha
        perturbation = delta * (1.0 - alpha) * torch.rand_like(delta)
        return theta0 + delta_scaled + perturbation


def merge(a, b, alpha, preloaded_a=None, module_override_alphas: dict = None, block_weights=None, algorithm='weighted_sum'):

    print(f"merging {a} and {b} with alpha {alpha}. preloaded_a is {preloaded_a}")
    pipe: CheckpointMergerPipeline = DiffusionPipeline.from_pretrained(b, 
                                                   #custom_pipeline="./checkpoint_merger.py")
                                                    custom_pipeline = __file__)
    merged = pipe.merge([a, b], alpha=alpha, preloaded_model_0=preloaded_a, 
                        module_override_alphas=module_override_alphas or {}, 
                        interp=algorithm,
                        block_weights=block_weights,
                        force=True)
    del pipe
    return merged


def merge_add_diff(a, b, c, alpha):
    print(f"merging {a}, {b}, {c} as add_difference with alpha {alpha}")
    pipe: CheckpointMergerPipeline = DiffusionPipeline.from_pretrained(b, custom_pipeline = __file__)
    merged = pipe.merge([a, b, c], alpha=alpha,
                        interp='add_diff',
                        force=True)
    del pipe
    return merged


def do_multi_merge(models, model_weights=None, per_module_alphas: dict = None, block_weights=None,
                   num_intermediates=1,
                   algorithm='weighted_sum'):
    if not model_weights:
        model_weights = [1]*len(models)

    if num_intermediates > 1:
        chunk_size = math.ceil(len(models)/num_intermediates)
    else:
        chunk_size = len(models)

    intermediates_folder = tempfile.mkdtemp()
    try:
        merged_intermediates = []
        merged_intermediates_weights = []
        print(f"merging {len(models)} models in chunks of {chunk_size}")
        for i in range(0, len(models), chunk_size):
            chunk = models[i:i+chunk_size]
            chunk_weights = model_weights[i:i+chunk_size]
            print(f"merging chunk {i//chunk_size} with weights {chunk_weights}")
            intermediate = _do_multi_merge_core(chunk, chunk_weights, per_module_alphas, block_weights, algorithm=algorithm)
            if num_intermediates == 1:
                return intermediate
            else:
                path = os.path.join(intermediates_folder, f"intermediate_{i//chunk_size}.ckpt")
                print(f"saving intermediate to {path}")
                save_merge(intermediate, path, half=False)
                merged_intermediates.append(path)
                merged_intermediates_weights.append(sum(chunk_weights))
                print("done merging chunk")

        print(f'merging {len(merged_intermediates)} intermediates with weights {merged_intermediates_weights}')
        merged = _do_multi_merge_core(merged_intermediates, merged_intermediates_weights, per_module_alphas, block_weights, algorithm=algorithm)
        return merged
    finally:
        print(f'deleting intermediates folder {intermediates_folder}')
        shutil.rmtree(intermediates_folder)

def _do_multi_merge_core(models, model_weights, per_module_alphas, block_weights, algorithm='weighted_sum'):

    if len(models) == 0:
        raise ValueError("Must pass at least 1 model")
    elif len(models) == 1:
        # nothing to do
        return models[0]

    merged = None
    for i in tqdm(range(1, len(models))):
        if i==1:
            a = models[0]
        else:
            a = "last-merge-result"
        b = models[i]

        weight_sum_a = sum(model_weights[:i])
        weight_b = model_weights[i]
        #alpha = 1/(i+1)
        alpha = weight_b / (weight_sum_a + weight_b)
        
        # module_override_alphas - dict of str -> float for per-module alpha overrides eg {'unet': 0.2, 'text_encoder': 0.8}

        merged = merge(a, b, alpha=alpha, preloaded_a=merged, module_override_alphas = per_module_alphas, block_weights=block_weights, algorithm=algorithm)
    return merged

def save_merge(merged, path, half=False):
    if half:
        merged = merged.to('cpu', dtype=torch.float16)
    merged.save_pretrained(path)
    print("saved merged model to", path, ("at float16 precision" if half else ""))
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_ids_or_paths",
                        nargs='+',
                        required=True,
                        help="repository ids or paths to models in diffusers or ckpt format, as strings enclosed in \"\" "
                             "and separated by spaces. eg "
                             "--repo_ids_or_paths \"stablityai/stable-diffusion-2-1\" \"../models/v1-5-pruned-emaonly.ckpt\"")
    parser.add_argument("--alphas",
                        required=True,
                        type=float,
                        nargs="+",
                        help="Alpha values for each model")
    parser.add_argument("--output_path",
                        required=True,
                        type=str,
                        help="output path")
    parser.add_argument("--sdxl", action='store_true', help="if passed, merge only SDXL UNet blocks (for SDXL models); block merge disabled")
    parser.add_argument("--num_intermediates", type=int, default=1, help="number of intermediates to use, default 1 means no intermediates. >1 uses extra disk space to save intermediates")
    parser.add_argument("--half", action='store_true', help="save fp16 rather than fp32")
    parser.add_argument("--algorithm", type=str, default='weighted_sum', help="merging algorithm: weighted_sum, sigmoid, inv_sigmoid, add_diff, reset_and_perturb")

    args = parser.parse_args()

    if args.sdxl:
        print("** SDXL merge")

    if args.algorithm == 'reset_and_perturb':
        if len(args.repo_ids_or_paths) != 2:
            raise ValueError("reset_and_perturb requires 2 repos")
        merged = merge(args.repo_ids_or_paths[0], args.repo_ids_or_paths[1], args.alphas[0], algorithm='reset_and_perturb')
    elif args.algorithm == 'add_diff':
        if len(args.repo_ids_or_paths) != 3:
            raise ValueError("add_diff requires 3 repos")
        merged = merge_add_diff(args.repo_ids_or_paths[0], args.repo_ids_or_paths[1], args.repo_ids_or_paths[2], args.alphas[0])
    else:
        if len(args.repo_ids_or_paths) != len(args.alphas):
            raise ValueError(f"must pass one alpha for each model ({len(args.repo_ids_or_paths)} models)")
        merged = do_multi_merge(args.repo_ids_or_paths, args.alphas, num_intermediates=args.num_intermediates)
    save_merge(merged, args.output_path, half=args.half)
    
