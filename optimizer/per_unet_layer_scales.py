
import re

def get_lr_scale(module_name: str) -> float:
    zone = _categorise_v2(module_name)

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
    }
    return LR_SCALES[zone]


def _categorise_v2(name: str) -> str:
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

    # --- type flag: qkv is a consistent outlier in every zone ---
    is_qkv = bool(re.search(r'\.to_[qkv]\.weight$', name))

    # --- type flag ---
    is_qkv = bool(re.search(r'\.to_[qkv]\.weight$', name))

    if is_qkv:
        return f'{zone}__qkv'
    if zone == 'mid':
        is_ff = bool(re.search(r'\.ff\.', name))
        if is_ff:
            return 'mid__ff'
    return zone
