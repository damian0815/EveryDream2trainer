"""
Adaptive per-resolution assignment for ImageSourceItem objects.

Public API
----------
assign_resolutions(source_items, global_resolution_weights, randomizer)
    -> list[ImageTrainItem]

Algorithm overview
------------------
1. For each source image, compute the eligible resolutions (feasible + non-zero weight)
   and normalise those weights into initial per-image per-resolution probabilities.
2. Compute per-resolution target counts: the expected number of images each resolution
   would receive if every image sampled independently from its initial probabilities.
3. Iterate over images in random order.  For each image, scale its initial probabilities
   by (target_count[r] / max(1, actual_count[r])).  This naturally boosts
   under-represented resolutions and suppresses over-represented ones.  Normalise and
   sample.
4. Corner case: any resolution that ends up with zero images after the main pass gets
   one filler image duplicated from the eligible pool, so the existing runt-filling
   logic in DataLoaderMultiAspect has something to work with.

!! MUTATION NOTE !!
assign_resolutions calls ImageSourceItem.make_resolved_item(), which mutates the
ImageSourceItem's internal ImageTrainItem.  When the same ImageSourceItem appears
more than once in source_items (multiplier > 1 case), assign_resolutions deep-copies
it for every appearance after the first, so each copy mutates a distinct object.
"""
import copy
import logging
import random
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.image_train_item import ImageSourceItem, ImageTrainItem


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_resolutions(
    source_items: list,
    global_resolution_weights: dict,
    randomizer: random.Random,
) -> list:
    """
    Assign each source image to exactly one resolution using adaptive probability
    sampling, then return the list of resolved ImageTrainItems.

    :param source_items: list[ImageSourceItem] — images already selected for this epoch.
        The same ImageSourceItem may appear more than once (multiplier > 1); this
        function handles that by deep-copying duplicates automatically.
    :param global_resolution_weights: {resolution_int: float} — the global per-resolution
        multiplier (from args.resolution_multiplier).  Applied on top of each image's
        per_resolution_multiply.  Pass {r: 1.0} for uniform weighting.
    :param randomizer: seeded Random instance for reproducibility.
    :return: list[ImageTrainItem], one per source image (plus any fillers for empty
        resolution buckets).
    """
    # Deduplicate to avoid the mutation hazard: make a fresh deep copy for every
    # duplicate occurrence of the same uid beyond the first.
    source_items = _deduplicate_by_deep_copy(source_items)

    eligible_sources, initial_probs = _compute_initial_probs(
        source_items, global_resolution_weights
    )
    if not eligible_sources:
        return []

    target_counts = _compute_target_counts(eligible_sources, initial_probs)
    resolved_items, source_by_resolved_uid = _adaptive_sample(
        eligible_sources, initial_probs, target_counts, randomizer
    )

    all_resolutions = list(global_resolution_weights.keys()) or list(
        next(iter(source_items)).resolution_options.keys()
    )
    actual_counts = _count_by_resolution(resolved_items)
    resolved_items = _fill_empty_resolutions(
        resolved_items, source_by_resolved_uid, all_resolutions, actual_counts,
        global_resolution_weights, randomizer
    )

    return resolved_items


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deduplicate_by_deep_copy(source_items: list) -> list:
    """
    Return a new list where the first occurrence of each uid is kept as-is and
    every subsequent occurrence is replaced with a deep copy.

    This is necessary because make_resolved_item() mutates the source's internal
    ImageTrainItem in place, so two assignments to the same source would corrupt
    each other.
    """
    seen: set[str] = set()
    result = []
    for item in source_items:
        if item.uid in seen:
            result.append(copy.deepcopy(item))
        else:
            seen.add(item.uid)
            result.append(item)
    return result


def _compute_initial_probs(
    source_items: list,
    global_resolution_weights: dict,
) -> tuple:
    """
    Step A+B: compute eligible resolutions and normalised initial probabilities per image.

    Returns (eligible_sources, initial_probs) where:
      eligible_sources — source items that have at least one feasible resolution with
                         non-zero weight; items with no eligible resolution are skipped
                         with a warning.
      initial_probs    — dict[source_uid -> dict[resolution_int -> float]] giving the
                         normalised probability of each eligible resolution for that image.
    """
    eligible_sources = []
    initial_probs: dict[str, dict[int, float]] = {}

    for source in source_items:
        eligible = {
            r: opt
            for r, opt in source.resolution_options.items()
            if opt.is_feasible and opt.unnormalised_weight > 0
        }
        if not eligible:
            logging.warning(
                f"Image {source.pathname} has no eligible resolution (too small or "
                f"all weights zero) — skipping for this epoch."
            )
            continue

        raw_weights: dict[int, float] = {
            r: opt.unnormalised_weight * global_resolution_weights.get(r, 1.0)
            for r, opt in eligible.items()
        }
        # Drop resolutions zeroed out by global_resolution_weights
        raw_weights = {r: w for r, w in raw_weights.items() if w > 0}
        if not raw_weights:
            logging.warning(
                f"Image {source.pathname} has no eligible resolution after applying "
                f"global_resolution_weights — skipping for this epoch."
            )
            continue

        total = sum(raw_weights.values())
        initial_probs[source.uid] = {r: w / total for r, w in raw_weights.items()}
        eligible_sources.append(source)

    return eligible_sources, initial_probs


def _compute_target_counts(
    eligible_sources: list,
    initial_probs: dict,
) -> dict:
    """
    Step B: sum the initial probabilities across all images to get per-resolution
    target counts (floating-point).

    target_counts[r] is the expected number of images that would be assigned to
    resolution r if each image sampled independently from initial_probs.
    """
    target_counts: dict[int, float] = defaultdict(float)
    for source in eligible_sources:
        for r, p in initial_probs[source.uid].items():
            target_counts[r] += p
    return dict(target_counts)


def _adaptive_sample(
    eligible_sources: list,
    initial_probs: dict,
    target_counts: dict,
    randomizer: random.Random,
) -> tuple:
    """
    Step C: assign each image to a resolution by adaptive weighted sampling.

    The weight for resolution r when processing image i is:
        initial_probs[i][r]  *  (target_counts[r] / max(1, actual_counts[r]))

    This feedback term boosts under-represented resolutions and suppresses
    over-represented ones while preserving each image's per-resolution preferences.

    Returns (resolved_items, source_by_resolved_uid).
    """
    actual_counts: dict[int, int] = defaultdict(int)
    resolved_items: list = []
    source_by_resolved_uid: dict[str, object] = {}

    order = list(eligible_sources)
    randomizer.shuffle(order)

    for source in order:
        eligible_r = list(initial_probs[source.uid].keys())

        adjusted: dict[int, float] = {
            r: initial_probs[source.uid][r]
               * (target_counts[r] / max(1, actual_counts[r]))
            for r in eligible_r
        }
        total = sum(adjusted.values())
        weights = [adjusted[r] / total for r in eligible_r]

        chosen_r = randomizer.choices(eligible_r, weights=weights)[0]

        # !! MUTATION: make_resolved_item mutates source.item in place !!
        resolved = source.make_resolved_item(chosen_r)
        actual_counts[chosen_r] += 1
        resolved_items.append(resolved)
        source_by_resolved_uid[resolved.uid] = source

    return resolved_items, source_by_resolved_uid


def _fill_empty_resolutions(
    resolved_items: list,
    source_by_resolved_uid: dict,
    all_resolutions: list,
    actual_counts: dict,
    global_resolution_weights: dict,
    randomizer: random.Random,
) -> list:
    """
    Step D: for any resolution with zero assigned images AND non-zero global weight,
    seed it with one duplicate so the runt-filling logic in DataLoaderMultiAspect
    has something to work with.

    Resolutions with global_resolution_weight == 0 are intentionally empty and
    should NOT be seeded.
    """
    for r in all_resolutions:
        # Skip resolutions that were explicitly zeroed out by global_resolution_weights
        if global_resolution_weights.get(r, 1.0) == 0.0:
            continue
        if actual_counts.get(r, 0) > 0:
            continue

        fill_pool = [
            item for item in resolved_items
            if _source_is_feasible_for(item, r, source_by_resolved_uid)
        ]

        if not fill_pool:
            logging.error(
                f"Resolution {r} has no eligible images and cannot be trained. "
                f"It will be excluded from this epoch.  Add images that are large "
                f"enough for resolution {r} to fix this."
            )
            continue

        logging.warning(
            f"Resolution {r} received 0 images after adaptive sampling; duplicating "
            f"1 eligible image to seed the bucket.  Add more images of this size to "
            f"avoid over-fitting."
        )
        donor = randomizer.choice(fill_pool)
        donor_source = source_by_resolved_uid[donor.uid]
        # deep-copy the source so we don't corrupt the already-resolved item
        filler_source = copy.deepcopy(donor_source)
        # !! MUTATION on the copy !!
        filler = filler_source.make_resolved_item(r)
        resolved_items.append(filler)

    return resolved_items


def _source_is_feasible_for(resolved_item, resolution: int, source_by_uid: dict) -> bool:
    """Return True if the source of resolved_item is feasible for the given resolution."""
    source = source_by_uid.get(resolved_item.uid)
    if source is None:
        return False
    opt = source.resolution_options.get(resolution)
    return opt is not None and opt.is_feasible


def _count_by_resolution(resolved_items: list) -> dict:
    """Count resolved items per resolution using their source_resolution attribute."""
    counts: dict[int, int] = defaultdict(int)
    for item in resolved_items:
        r = getattr(item, 'source_resolution', None)
        if r is not None:
            counts[r] += 1
    return dict(counts)



