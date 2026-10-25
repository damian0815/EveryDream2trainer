import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Literal

import torch


class ParamGroupBuilder:
    def __init__(self, optimizer_param_grouping: Literal['zones', 'transformer10x', 'single']|tuple[str, str], betas, weight_decay, base_lr):
        self.param_grouping = optimizer_param_grouping
        self.betas = betas
        self.weight_decay = weight_decay
        self.base_lr = base_lr

    def build_param_groups(self, parameters: tuple[str, torch.nn.Parameter]) -> list[dict]:
        if self.param_grouping[0] == 'zones':
            zone_members = defaultdict(list)
            for name, p in parameters:
                zone = get_unet_module_zone(name)
                zone_members[zone].append(p)
            return [{
                'name': z,
                'params': zone_members[z],
                'betas': self.betas,
                'weight_decay': self.weight_decay,
                'lr': self.base_lr * get_lr_scale_for_zone(z)
            } for z in zone_members.keys()]
        elif self.param_grouping[0] == 'per-module':
            if len(self.param_grouping) != 2:
                raise ValueError("must specify path when using per-module param grouping, eg --optimizer_param_grouping per-module /path/to/grad_ratios.json of format {module_name: grad_ratio} where grad_ratio is the ratio between (mean) grad after backward() and weight magnitude for each module (parameter).")
            scales = get_raw_unet_module_lr_scales(self.param_grouping[1])
            return [{
                'name': n,
                'params': [p],
                'betas': self.betas,
                'weight_decay': self.weight_decay,
                'lr': self.base_lr * scales[n]
            } for n, p in parameters]
        elif self.param_grouping[0] == 'transformer10x':
            return [{
                'name': 'transformer_blocks',
                'params': [p for n, p in parameters if 'transformer_blocks' in n],
                'betas': self.betas,
                'weight_decay': self.weight_decay,
                'lr': self.base_lr * 10
            }, {
                'name': 'non-transformer_blocks',
                'params': [p for n, p in parameters if 'transformer_blocks' not in n],
                'betas': self.betas,
                'weight_decay': self.weight_decay,
                'lr': self.base_lr
            }]
        elif self.param_grouping[0] == 'single':
            return [{
                'params': [p for n, p in parameters],
                'betas': self.betas,
                'weight_decay': self.weight_decay,
                'lr': self.base_lr
            }]
        else:
            raise ValueError(f"Unknown param grouping {self.param_grouping[0]}")



def get_lr_scale_for_zone(zone: str) -> float:

    LR_SCALES = {
        # zone            log_ratio_approx   → multiplier (exp(ref - val), ref=+1.7)
        'edge': 3.5,  # conv_in/out, embeddings — hot, careful not to overdo
        'down_outer': 1.0,  # anchor — this is your reference point
        'down_mid': 2.0,  # ~exp(1.7 - 0.5)
        'down_inner': 4.0,  # getting cold
        'mid': 8.0,  # cold across all types except ff which needs more
        'up_inner': 2.5,
        'up_mid': 1.5,
        'up_outer': 1.2,
        'other': 1.0,  # fallback

        # qkv variants: always colder than their zone by ~1.5-2 log units
        'down_outer__qkv': 4.0,
        'down_mid__qkv': 8.0,
        'down_inner__qkv': 15.0,
        'mid__ff': 20.0,  # -6.0 log ratio, most starved module in the whole network
        'mid__qkv': 20.0,  # capped — data says ~80x but that's dangerous
        'up_inner__qkv': 10.0,
        'up_mid__qkv': 6.0,
        'up_outer__qkv': 5.0,

        # attn1 (self-attention) — depth-sensitive, anchored to down_outer
        'down_outer__attn1_qkv': 4.0,
        'down_mid__attn1_qkv': 8.0,
        'down_inner__attn1_qkv': 15.0,
        'mid__attn1_qkv': 20.0,
        'up_inner__attn1_qkv': 10.0,
        'up_mid__attn1_qkv': 6.0,
        'up_outer__attn1_qkv': 5.0,

        # attn2 (cross-attention) — depth-insensitive, uniformly starved
        'down_outer__attn2_qkv': 20.0,
        'down_mid__attn2_qkv': 20.0,
        'down_inner__attn2_qkv': 20.0,
        'mid__attn2_qkv': 20.0,
        'up_inner__attn2_qkv': 20.0,
        'up_mid__attn2_qkv': 20.0,
        'up_outer__attn2_qkv': 20.0,
    }
    return LR_SCALES[zone]

def get_raw_unet_module_lr_scales(path) -> dict[str, float]:
    with open(path, 'r') as f:
        d = json.load(f)
        return {n: 1/r for n, r in d.items()}  # invert to get lr scale multipliers
        #return {n: min(100, max(0.01, 1/r)) for n, r in d.items()}  # clamp to avoid extreme outliers


def get_unet_module_zone(name: str) -> str:
    """
    Returns a zone + optional type label for a parameter name.
    Zone captures UNet block position; type flags the qkv outlier.
    """
    # --- top-level special cases ---
    if re.match(r'^conv_(in|out)\.', name):
        return 'edge'
    if re.match(r'^(time_embedding|add_embedding)\.', name):
        return 'edge'

    # --- determine zone from block path ---
    down = re.search(r'down_blocks?\.(\d+)', name)
    up   = re.search(r'up_blocks?\.(\d+)',   name)
    mid  = 'mid_block' in name

    if mid:
        zone = 'mid'
    elif down:
        i = int(down.group(1))
        zone = ['down_outer', 'down_mid', 'down_mid', 'down_inner'][min(i, 3)]
    elif up:
        i = int(up.group(1))
        zone = ['up_inner', 'up_mid', 'up_mid', 'up_outer'][min(i, 3)]
    else:
        return 'other'  # fallback for anything unmatched

    # --- type flag ---
    is_qkv = bool(re.search(r'\.to_[qkv]\.weight$', name))

    if is_qkv:
        is_attn1 = bool(re.search(r'\.attn1\.to_[qkv]\.weight$', name))
        is_attn2 = bool(re.search(r'\.attn2\.to_[qkv]\.weight$', name))

        if is_attn1:
            return f'{zone}__attn1_qkv'
        elif is_attn2:
            return f'{zone}__attn2_qkv'
        else:
            return f'{zone}__qkv'
    if zone == 'mid':
        is_ff = bool(re.search(r'\.ff\.', name))
        if is_ff:
            return 'mid__ff'
    return zone


def parse_lr_block_ramps(values: list[float] | None) -> list[tuple[int, float]]:
    if values is None:
        return []
    if len(values) % 2 != 0:
        raise ValueError("lr_block_ramps must contain an even number of values (block_index, alpha) pairs")
    pairs = [(int(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
    indices = [block_index for block_index, _ in pairs]
    if len(set(indices)) != len(indices):
        raise ValueError("lr_block_ramps contains duplicate block indices: " + str(indices))
    return sorted(pairs)


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


def _get_block_index_from_key(key: str, max_block_index: int) -> int:
    if 'transformer_blocks' in key:
        return int(key.split('.')[1])
    elif key == 'scale_shift_table':
        return 0
    elif key.startswith('caption_'):
        return 0
    elif key.startswith('time_embed') or key.startswith('patch_embed'):
        return 0
    elif 'proj_out' in key:
        return max_block_index
    else:
        raise RuntimeError("Unrecognized key, cannot extract block index: " + key)


def apply_block_ramps(param_groups, parameters, ramp_blockwise, betas, weight_decay):
    if not ramp_blockwise:
        return param_groups

    param_to_name = {id(p): name for name, p in parameters}
    block_names = [name for name, _ in parameters if 'transformer_blocks' in name]
    if not block_names:
        return param_groups
    max_block_index = max(int(name.split('.')[1]) for name in block_names)

    new_groups = []
    for group in param_groups:
        if not group['params']:
            new_groups.append(group)
            continue

        base_lr = group['lr']
        alpha_to_params = defaultdict(list)
        for p in group['params']:
            name = param_to_name[id(p)]
            block_index = _get_block_index_from_key(name, max_block_index)
            alpha = _get_ramp_blockwise_alpha(block_index, ramp_blockwise)
            alpha_to_params[alpha].append(p)

        for alpha, params in sorted(alpha_to_params.items()):
            new_groups.append({
                'name': f"block_ramp_{alpha:.4g}",
                'params': params,
                'lr': base_lr * alpha,
                'betas': betas,
                'weight_decay': weight_decay,
            })

    return new_groups
