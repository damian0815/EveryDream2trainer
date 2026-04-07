import logging
from typing import Literal

import math
import torch
from torch import nn
import re
from collections import defaultdict
from optimizer.per_unet_layer_scales import ParamGroupBuilder



"""

# Unet layout

## global (UNet2DConditionModel):
time_proj
time_embedding
conv_in
down_blocks.0-3
mid_block.
up_blocks.0-3
conv_norm_out
conv_act
conv_out


## down_blocks (CrossAttnDownBlock2D):
[resnets,
 attentions] xN
downsampler

## mid_block (UNetMidBlock2DCrossAttn):
resnets.0
[attentions.0,
resnets.1] xN with resnets.k higher than attentions.k

## up_blocks (CrossAttnUpBlock2D):
[resnets,
 attentions] xN
upsampler


## resnets: eg up_blocks.1.resnets.0
norm1
nonlinearity
upsample/downsample
conv1
time_emb_proj
norm2
nonlinearity
dropout
conv2
conv_shortcut


## attentions (transformer2dmodel):
norm
proj_in
[transformer_blocks] xN
proj_out


## transformer_blocks:
norm1
pos_embed
attn1
norm2
pos_embed
attn2
norm3
ff


"""


class ProgressiveUnlock:
    def __init__(self, optimizer, model: nn.Module, total_steps: int, param_group_builder: ParamGroupBuilder, order_by_distance_to_qk: bool = False, initial_step_offset: int = 0):
        self.optimizer = optimizer
        self.model = model
        self.param_group_builder = param_group_builder
        self.initial_step_offset = initial_step_offset

        all_params = set(n for n, _ in model.named_parameters())
        ordered_param_groups_all = order_parameters(all_params, from_qk=order_by_distance_to_qk)
        # filter out frozen
        unfrozen_params = {n for n, p in model.named_parameters() if p.requires_grad}
        ordered_param_groups_unfrozen = [[p for p in group if p in unfrozen_params] for _, group in ordered_param_groups_all]
        self.ordered_param_groups = [group for group in ordered_param_groups_unfrozen if len(group)>0]

        self.steps_per_unlock = total_steps / sum(len(g) for g in self.ordered_param_groups)
        self.unlocked_params: set[str] = set()
        # freeze everything
        for p in model.parameters():
            p.requires_grad = False

    def step(self, global_step):
        expected_unlock_count = 1 + math.floor((global_step + self.initial_step_offset) / self.steps_per_unlock)
        param_groups = self.unlock_params_and_return_groups(expected_unlock_count)
        if param_groups:
            logging.info(f"Expected unlock count at effective step {global_step + self.initial_step_offset}: {expected_unlock_count} because steps_per_unlock is {self.steps_per_unlock}")
            for g in param_groups:
                logging.info(f"* ProgressiveUnlock adding param group '{g['name']}' with {len(g['params'])} params, wd {g['weight_decay']}, betas {g['betas']}, lr {g['lr']}")
                self.optimizer.add_param_group(g)
            logging.info(f"* Optimizer now has {len(self.optimizer.param_groups)} param groups and {len(self.unlocked_params)} unlocked params. There are {sum(len(g) for g in self.ordered_param_groups) - len(self.unlocked_params)} params left to unlock.")

    def unlock_params_and_return_groups(self, expected_unlock_count):
        to_unlock: set[str] = set()

        # unlock group-wise until at least expected_unlock_count are unlocked
        for group in self.ordered_param_groups:
             for n in group:
                 if n not in self.unlocked_params:
                     to_unlock.add(n)
             if len(to_unlock) + len(self.unlocked_params) >= expected_unlock_count:
                 break

        if not to_unlock:
            return []

        logging.info(f"* ProgressiveUnlock unlocking {to_unlock}")
        params = [(n, p) for n, p in self.model.named_parameters() if n in to_unlock]
        assert len(params) == len(to_unlock), f"Expected to find parameters for all modules to unlock, but found {len(params)} parameters for {len(to_unlock)} modules. Params found: {[n for n, _ in params]}"
        for n, p in params:
            p.requires_grad = True
            self.unlocked_params.add(n)
        param_groups = self.param_group_builder.build_param_groups(params)
        return param_groups




def order_parameters(parameters: list[str], from_qk: bool = False):

    def get_resnet_offset(name):
        components = ['norm1', 'nonlinearity', 'upsample', 'downsample', 'conv1', 'time_emb_proj', 'norm2', 'dropout', 'conv2', 'conv_shortcut']

        for i, comp in enumerate(components):
            if comp in name:
                return i
        raise RuntimeError('Unmatched resnet component in name: ' + name)

    def get_attention_offset(name):

        components = ['norm1', 'pos_embed', 'attn1', 'norm2', 'attn2', 'norm3', 'ff']

        for i, comp in enumerate(components):
            if comp in name:
                return 2 + i
        if '.norm.' in name:
            return 0
        elif '.proj_in.' in name:
            return 1
        elif '.proj_out.' in name:
            return 2 + len(components)
        else:
            raise RuntimeError('Unmatched attention component in name: ' + name)

    def get_position(name):
        if re.match(r'^conv_in\.', name):
            return 0
        elif re.match(r'^time_proj\.', name):
            return 1
        elif re.match(r'^time_embedding\.', name):
            return 2
        elif re.match(r'^conv_norm_out\.', name):
            return 100000
        elif re.match(r'^conv_act\.', name):
            return 100001
        elif re.match(r'^conv_out\.', name):
            return 100002

        if re.match(r'^down_blocks\.', name):
            block_index = int(re.search(r'down_blocks\.(\d+)', name).group(1))
            offset = 1000 + block_index * 1000
        elif re.match(r'^mid_block\.', name):
            offset = 10000
        elif re.match(r'^up_blocks\.', name):
            block_index = int(re.search(r'up_blocks\.(\d+)', name).group(1))
            offset = 20000 + block_index * 1000
        else:
            raise RuntimeError('Unmatched block parameter: ' + name)

        # resnet.0, attention.0, resnet.1, attention.1, ... within each block
        if '.resnets' in name:
            resnet_index = int(re.search(r'\.resnets\.(\d+)', name).group(1))
            return offset + resnet_index * 50 + get_resnet_offset(name)
        elif '.attentions' in name:
            attn_index = int(re.search(r'\.attentions\.(\d+)', name).group(1))
            return offset + attn_index * 50 + 25 + get_attention_offset(name)
        elif '.upsampler' in name or 'downsampler' in name:
            return offset + 999

        raise RuntimeError('Unmatched down_blocks parameter: ' + name)

    groups = defaultdict(list)
    for n in parameters:
        try:
            score = get_position(n)
        except RuntimeError as e:
            print("Error processing parameter name '" + n + "': " + str(e))
            continue
        groups[score].append(n)

    groups_sorted = list(sorted(groups.items(), key=lambda x: x[0]))
    if not from_qk:
        # unlock in backward order
        return reversed(list(groups_sorted))

    # for each group, get its distance from the nearest q/k/v parameter, and sort by that distance instead of absolute position, to unlock in qkv order rather than strict forward order
    qk_group_indices: set[int] = set()
    for i, (score, group) in enumerate(groups_sorted):
        for n in group:
            if re.search(r'\.to_[qkv]\.weight$', n):
                qk_group_indices.add(i)
                break

    def distance_from_qk(i):
        if i in qk_group_indices:
            return 0
        return min(abs(i - qk_i) for qk_i in qk_group_indices)
    group_by_qk_distance = defaultdict(list)
    for i, group in enumerate(groups_sorted):
        dist = distance_from_qk(i)
        group_by_qk_distance[dist].extend(group[1])

    return list(sorted(group_by_qk_distance.items(), key=lambda x: x[0]))



