from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm


def merge_safetensors(safetensors_1_path, safetensors_2_path, alpha, output_path):
    out_sd = {}
    with safe_open(safetensors_1_path, framework='pt', device='cpu') as sd1:
        with safe_open(safetensors_2_path, framework='pt') as sd2:
            for k in tqdm(sd1.keys()):
                out_sd[k] = sd1.get_tensor(k) * (1 - alpha) + sd2.get_tensor(k) * alpha

    save_file(out_sd, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge two safetensors files with a given alpha.")
    parser.add_argument("safetensors_1", help="Path to the first safetensors file")
    parser.add_argument("safetensors_2", help="Path to the second safetensors file")
    parser.add_argument("alpha", type=float, help="Alpha blending factor (0.0 to 1.0)")
    parser.add_argument("output", help="Path to save the merged safetensors file")
    args = parser.parse_args()

    merge_safetensors(args.safetensors_1, args.safetensors_2, args.alpha, args.output)
