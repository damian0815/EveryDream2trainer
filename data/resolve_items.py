"""
Image source resolution for EveryDream2trainer.

Moved from train.py and train_sana.py during refactoring.
"""
import gc
import json
import logging
import os
import argparse
from collections import defaultdict

from colorama import Fore, Style
from tqdm.auto import tqdm

import data.resolver as resolver
from data.image_train_item import ImageSourceItem


def apply_per_path_multiplier(resolved_items: list, per_path_multiplier_json: str):
    """Apply per-image multipliers loaded from a JSON file.  Works with both
    ImageTrainItem and ImageSourceItem objects (both expose .multiplier and .pathname)."""
    applied = 0
    missing = 0
    first_missing = []
    with open(per_path_multiplier_json, "rt") as f:
        per_path_multipliers = json.load(f)
    for item in tqdm(resolved_items, desc=f"applying per-path multiplier {os.path.basename(per_path_multiplier_json)}"):
        realpath = os.path.realpath(item.pathname)
        try:
            item.multiplier *= per_path_multipliers[realpath]
            applied += 1
        except KeyError:
            missing += 1
            if len(first_missing) < 5:
                first_missing.append(item.pathname)
    logging.info(f" Applied {applied} multipliers ({missing} missing) from {per_path_multiplier_json}. First 5 missing: {first_missing}")


def resolve_image_source_items(
    args: argparse.Namespace,
    aspects_per_resolution: dict,
    divisible_by: int = 8,
) -> list:
    """
    Resolve training images for all resolutions at once, returning a list of
    ImageSourceItem objects (one per image, resolution assignment deferred).

    Replaces the old per-resolution resolve_image_train_items loop.
    """
    logging.info(f"* Loading images for resolutions: {list(aspects_per_resolution.keys())}")
    logging.info("  Preloading image metadata (one file-open per image)...")

    source_items = resolver.resolve_sources(args.data_root, args, aspects_per_resolution)

    # Log and remove images that could not be opened
    for item in source_items:
        if item.error is not None:
            logging.error(
                f"{Fore.LIGHTRED_EX} *** Error opening "
                f"{Fore.LIGHTYELLOW_EX}{item.pathname}{Fore.LIGHTRED_EX}: "
                f"{item.error} — skipping.{Style.RESET_ALL}"
            )
    source_items = [s for s in source_items if s.error is None]

    if args.data_multiplier_per_path:
        paths = (
            [args.data_multiplier_per_path]
            if isinstance(args.data_multiplier_per_path, str)
            else args.data_multiplier_per_path
        )
        for p in paths:
            apply_per_path_multiplier(source_items, p)

    # --use_only_largest_resolution_per_image (shared arg, SANA-specific usage)
    if getattr(args, 'use_only_largest_resolution_per_image', False):
        resolutions = sorted(aspects_per_resolution.keys(), reverse=True)
        lower_res = defaultdict(int)
        for s in source_items:
            fallback_resolution = None if args.skip_undersized_images else resolutions[-1]
            largest_feasible_resolution = next((k for k in resolutions
                                                if s.resolution_options[k].is_feasible), fallback_resolution)
            if largest_feasible_resolution is None:
                logging.warning(
                    f" * Image {s.pathname} is undersized for all resolutions → dropping"
                )
                s.resolution_options = {}
            else:
                if largest_feasible_resolution != resolutions[0]:
                    logging.debug(f" * Image {s.pathname} is undersized for {resolutions[0]} → using {largest_feasible_resolution}")
                    lower_res[largest_feasible_resolution] += 1
                s.resolution_options = {largest_feasible_resolution: s.resolution_options[largest_feasible_resolution]}

        if lower_res:
            logging.info(f" * --use_only_largest_resolution_per_image reduced resolutions: {dict(lower_res)}")

    if args.skip_undersized_images:
        pre_count = len(source_items)
        source_items = [s for s in source_items if s.is_feasible_for_any_resolution()]
        dropped = pre_count - len(source_items)
        if dropped:
            logging.info(
                f" * Dropped {dropped} images that are undersized at all configured "
                f"resolutions ({len(source_items)} remaining)."
            )

    # Drop images with empty JSON captions
    source_items = _drop_empty_json_caption_sources(source_items)

    # Verify divisibility
    have_invalid_size = False
    for s in source_items:
        for r, opt in s.resolution_options.items():
            if opt.target_wh[0] % divisible_by != 0:
                logging.error(
                    f" * image {s.pathname} at resolution {r} has width {opt.target_wh[0]} "
                    f"which is not divisible by {divisible_by}"
                )
                have_invalid_size = True
            if opt.target_wh[1] % divisible_by != 0:
                logging.error(
                    f" * image {s.pathname} at resolution {r} has height {opt.target_wh[1]} "
                    f"which is not divisible by {divisible_by}"
                )
                have_invalid_size = True
    if have_invalid_size:
        raise RuntimeError(
            f"One or more training images have width or height not divisible by {divisible_by}. "
            "This is a code error - check the errors above for matching values in "
            "`data/aspects.py` and fix."
        )

    if not source_items:
        raise RuntimeError(
            f"No training images found in '{args.data_root}'. "
            "Check --data_root and that your folder contains supported image files."
        )

    logging.info(f" * Found {len(source_items)} valid source images in '{args.data_root}'")
    gc.collect()

    # Stamp base_multiplier now that all user-configured multiplier changes are done.
    # DifficultyEstimator schedulers scale relative to this value, not the mutated one.
    for s in source_items:
        s.base_multiplier = s.multiplier
    for s in source_items:
        if s.cond_dropout is None:
            s.cond_dropout = args.cond_dropout
    if args.cond_dropout_global is not None:
        for s in source_items:
            s.cond_dropout *= args.cond_dropout_global

    return source_items


def _drop_empty_json_caption_sources(source_items: list) -> list:
    """Remove source items whose caption is a JSON dict with all-empty values."""
    result = []
    for item in source_items:
        caption = item.caption.get_caption()
        if caption.startswith("<<json>>"):
            caption_data = json.loads(caption.replace("<<json>>", ""))
            if caption_data is None or all(
                v is None or len(v.strip()) == 0 for v in caption_data.values()
            ):
                continue
        result.append(item)
    dropped = len(source_items) - len(result)
    if dropped:
        logging.info(
            f" * Dropped {dropped} images with empty JSON captions ({len(result)} remaining)."
        )
    return result
