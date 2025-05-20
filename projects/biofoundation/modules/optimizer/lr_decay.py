import json
import numpy as np
from scipy.stats import norm
from typing import Literal
import re

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, decay_type: Literal['right', 'inverse_normal', 'normal']="right"):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    # num_layers = len(model.blocks) + 1
    num_layers = model.get_num_layers() +1

    if decay_type == "right":
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    elif decay_type == "normal":
        #center_layer = num_layers // 2
        x = np.linspace(-3, 3, num_layers + 1)  # Adjust spread to match the required range
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # Normalize so the max is 1
        layer_scales = [s for s in normal_dist]  # Convert to list

    elif decay_type == "inverse_normal":
        x = np.linspace(-3, 3, num_layers + 1)  # Adjust spread to match the required range
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # Normalize so the max is 1

        inverted_dist = 1 - normal_dist  # Flip values so the middle becomes the lowest

        midpoint = len(inverted_dist) // 2
        position_counts = np.arange(1, num_layers + 1)  # Generate 1 to num_layers
        scaling_factors = layer_decay ** position_counts[::-1]  # Reverse to start from 0.75^13

        scaled_left = scaling_factors[:midpoint]  # Use appropriate portion of scaling factors
        right = inverted_dist[midpoint-1:]  # Keep the right side unchanged
        right[midpoint-1] += 0.1
        right[midpoint] += 0.1

        layer_scales = np.concatenate([scaled_left, right])

        layer_scales = layer_scales.tolist()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "names": [],
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["names"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif re.search(r'layers\.(\d+)\.', name):
        return int(re.search(r'layers\.(\d+)\.', name).group(1))
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('ppnet'):
        return num_layers
    else:
        return num_layers


