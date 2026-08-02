from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm


def merge_safetensors(paths, alphas, output_path):
    sts = [safe_open(path, framework='pt', device='cpu') for path in paths]
    alphas_normalized = [a / sum(alphas) for a in alphas]

    out_sd = {}
    for st, alpha in tqdm(zip(sts, alphas_normalized), total=len(sts)):
        for key in tqdm(st.keys()):
            out_sd[key] = out_sd.get(key, 0) + st.get_tensor(key) * alpha

    save_file(out_sd, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge two safetensors files with a given alpha.")
    parser.add_argument("safetensors_files", nargs='+', help="Path to the first safetensors file")
    parser.add_argument("--alphas", type=float, nargs='+', required=False, help="Alpha values for each safetensors file (space-separated). If omitted, use equal-weight merging.'")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged safetensors file")
    args = parser.parse_args()

    if args.alphas is None or len(args.alphas) == 0:
        args.alphas = [1.0] * len(args.safetensors_files)
    elif len(args.alphas) == 1:
        args.alphas = [args.alphas[0]] * len(args.safetensors_files)
    elif len(args.alphas) != len(args.safetensors_files):
        raise ValueError(f"You passed {len(args.alphas)} alphas: {args.alphas}. Pass either 0 or 1 alphas, or match the same number of alphas as the number of safetensors files ({len(args.safetensors_files)}).")

    merge_safetensors(paths=args.safetensors_files, alphas=args.alphas, output_path=args.output)
