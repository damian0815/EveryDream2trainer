from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm

ZEROS_MARKER = "ZEROS"


def _is_zeros(path):
    return path.upper() == ZEROS_MARKER


def _get_ramp_blockwise_alpha(block_index, ramp_blockwise):
    # ramp_blockwise is a list of (block_index, alpha) pairs, sorted by block_index
    if block_index <= ramp_blockwise[0][0]:
        return ramp_blockwise[0][1]
    if block_index >= ramp_blockwise[-1][0]:
        return ramp_blockwise[-1][1]
    for i in range(len(ramp_blockwise) - 1):
        if ramp_blockwise[i][0] <= block_index <= ramp_blockwise[i + 1][0]:
            # Linear interpolation between the two points
            x0, y0 = ramp_blockwise[i]
            x1, y1 = ramp_blockwise[i + 1]
            slope = (y1 - y0) / (x1 - x0)
            return y0 + slope * (block_index - x0)
    raise RuntimeError("Block index not in ramp_blockwise range, this should not happen.")


def merge_safetensors(paths, alphas, output_path, ramp_blockwise=None):
    real_paths = [path for path in paths if not _is_zeros(path)]
    if not real_paths:
        raise ValueError("At least one real (non-ZEROS) checkpoint is required.")
    alphas_normalized = [a / sum(alphas) for a in alphas]

    out_sd = {}
    with tqdm(zip(paths, alphas_normalized), total=len(paths)) as pbar:
        for path_index, (path, alpha) in enumerate(pbar):
            if _is_zeros(path):
                # there's no need to literally add zeros to the output (that happens
                # implicitly when we skip the path but keep its alpha in normalization
                # computation) so we can just skip this file and continue to the next one
                continue
            st = safe_open(path, framework='pt', device='cpu')
            max_block_index = max(int(key.split('.')[1])
                                  for key in st.keys()
                                  if 'transformer_blocks' in key)
            seen_block_indexes = set()
            for key in tqdm(st.keys()):
                def _get_block_index_from_key(key):
                    if 'transformer_blocks' in key:
                        return int(key.split('.')[1])
                    elif key == 'scale_shift_table':
                        return 0
                    elif key.startswith('caption_'):
                        return 0
                    elif key.startswith('time_embed') or key.startswith('patch_embed'):
                        return None  # these are always frozen anyway
                    elif 'proj_out' in key:
                        return max_block_index
                    else:
                        raise RuntimeError("Unrecognized key, cannot extract block index: " + key)
                block_index = _get_block_index_from_key(key)
                if block_index is not None and ramp_blockwise:
                    ramp_alpha = _get_ramp_blockwise_alpha(block_index, ramp_blockwise)
                    if path_index == 0:
                        block_alpha = 1 - ramp_alpha
                    else:
                        block_alpha = ramp_alpha
                    if block_index not in seen_block_indexes:
                        pbar.write(f"{key} -> block index {block_index} alpha {block_alpha}")
                        seen_block_indexes.add(block_index)
                else:
                    block_alpha = alpha

                out_sd[key] = out_sd.get(key, 0) + st.get_tensor(key) * block_alpha

    save_file(out_sd, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge two safetensors files with a given alpha.")
    parser.add_argument("safetensors_files", nargs='+', help="Paths to safetensors files. Use 'ZEROS' in place of a path to merge a zero tensor (dilutes the result by its alpha).")
    parser.add_argument("--alphas", type=float, nargs='+', required=False, help="Alpha values for each safetensors file (space-separated). If omitted, use equal-weight merging.'")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged safetensors file")
    parser.add_argument("--ramp_blockwise_from", type=int, required=False, help="Optional: If provided, ramp the alpha values blockwise; this is the block with alpha=0.")
    parser.add_argument("--ramp_blockwise_to", type=int, required=False, help="Optional: If provided, ramp the alpha values blockwise; this is the block with alpha=1 (may be < ramp_blockwise_from, in which case the alpha goes down as block numbers increase).")
    parser.add_argument("--ramp_blockwise_complex", type=int, nargs='+', required=False, help="Optional: If provided, ramp the alpha values blockwise - this is a list of pairs of (block_index, alpha) values. The alpha for each block is linearly interpolated between the two nearest provided points. If a block index is outside the provided range, it will use the alpha of the nearest point.")
    args = parser.parse_args()

    if args.alphas is None or len(args.alphas) == 0:
        args.alphas = [1.0] * len(args.safetensors_files)
    elif len(args.alphas) == 1:
        args.alphas = [args.alphas[0]] * len(args.safetensors_files)
    elif len(args.alphas) != len(args.safetensors_files):
        raise ValueError(f"You passed {len(args.alphas)} alphas: {args.alphas}. Pass either 0 or 1 alphas, or match the same number of alphas as the number of safetensors files ({len(args.safetensors_files)}).")

    if args.ramp_blockwise_from is not None and args.ramp_blockwise_to is not None:
        if args.ramp_blockwise_complex is not None:
            raise ValueError("Cannot use --ramp_blockwise_from/--ramp_blockwise_to together with --ramp_blockwise_complex.")
        if len(args.safetensors_files) != 2:
            raise ValueError("Blockwise ramping is only supported when merging exactly 2 safetensors files.")
        ramp_blockwise = [(args.ramp_blockwise_from, 0), (args.ramp_blockwise_to, 1)]
        if ramp_blockwise[0] == ramp_blockwise[1]:
            raise ValueError("--ramp_blockwise_from and --ramp_blockwise_to cannot be the same value.")
    elif args.ramp_blockwise_from is not None or args.ramp_blockwise_to is not None:
        raise ValueError("Both --ramp_blockwise_from and --ramp_blockwise_to must be provided together.")
    else:
        ramp_blockwise = args.ramp_blockwise_complex
        if ramp_blockwise:
            if len(ramp_blockwise) % 2 != 0:
                raise ValueError("--ramp_blockwise_complex must be provided as pairs of (block_index, alpha) values.")
            ramp_blockwise = [(ramp_blockwise[i], ramp_blockwise[i + 1]) for i in range(0, len(ramp_blockwise), 2)]
            ramp_blockwise.sort(key=lambda x: x[0])  # Sort by block index
    merge_safetensors(paths=args.safetensors_files, alphas=args.alphas, output_path=args.output, ramp_blockwise=ramp_blockwise)
