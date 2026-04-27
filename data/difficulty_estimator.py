"""
DifficultyEstimator — per-image difficulty scoring and adaptive scheduling.

Difficulty score is the EMA of  mean( log( normalised_loss ) )  over all
(timestep, loss) observations for a given image.  A positive score means the
image is harder than average; a negative score means it is easier.

Two scheduling policies are provided:

  TypeAScheduler — adjusts item.multiplier once per epoch so hard images appear
                   more often and easy images appear less often.

  TypeBScheduler — spaced-repetition scheduler: maintains a next_due_slab
                   counter per image and re-builds the item list every
                   slab_size batches.  Hard images have shorter intervals
                   (appear sooner) and easy images have longer intervals.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

import torch

from core.mean_loss_per_timestep import normalize_loss_by_timestep
from data.image_train_item import ImageTrainItem


# ---------------------------------------------------------------------------
# Scheduler interface
# ---------------------------------------------------------------------------

class DifficultyScheduler(ABC):
    """Abstract policy: translates difficulty scores into scheduling decisions."""

    @abstractmethod
    def update_multipliers(
        self,
        items: list[ImageTrainItem],
        scores: dict[str, float],
        obs_counts: dict[str, int],
        min_obs_count: int,
    ) -> None:
        """Called once per epoch-end.  Mutates item.multiplier in-place."""
        ...


# ---------------------------------------------------------------------------
# Type A: per-epoch multiplier adjustment
# ---------------------------------------------------------------------------

class TypeAScheduler(DifficultyScheduler):
    """
    Maps  exp(score * expand_factor) * base_multiplier  onto item.multiplier,
    clamped to [min_multiplier, max_multiplier] * base_multiplier.
    Images below min_obs_count keep base_multiplier unchanged.
    """

    def __init__(
        self,
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
        expand_factor: float = 1.0,
    ):
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.expand_factor = expand_factor

    def update_multipliers(
        self,
        items: list[ImageTrainItem],
        scores: dict[str, float],
        obs_counts: dict[str, int],
        min_obs_count: int,
    ) -> None:
        for item in items:
            key = os.path.realpath(item.pathname)
            if obs_counts.get(key, 0) < min_obs_count:
                item.multiplier = item.base_multiplier
                continue
            score = scores.get(key, 0.0)
            raw = math.exp(score * self.expand_factor)
            clamped = max(self.min_multiplier, min(self.max_multiplier, raw))
            item.multiplier = item.base_multiplier * clamped


# ---------------------------------------------------------------------------
# Type B: spaced-repetition via slab reconstruction
# ---------------------------------------------------------------------------

class TypeBScheduler(DifficultyScheduler):
    """
    Spaced-repetition scheduler.

    Each image has a next_due_slab counter.  At each slab boundary,
    build_next_slab() picks which images appear in the next slab:

    1. Due *unscored* images fill first (cold-start guarantee, priority).
    2. Due *scored* images fill remaining capacity, most-overdue first.
    3. Any leftover slots are filled by weighted resampling from all due images
       (weight = exp(score * expand_factor)), so hard images earn extra copies.

    Interval formula:
        interval = clamp( round( base_interval / exp(score * expand_factor) ),
                          1, max_interval )

    With base_interval = epoch_size_in_slabs:
        score  0  → interval = base_interval  (neutral: once per epoch)
        score >0  → interval < base_interval  (hard: more frequent)
        score <0  → interval > base_interval  (easy: less frequent)

    base_interval MUST be > 1 to differentiate hard from neutral.
    """

    def __init__(
        self,
        base_interval: int,
        slab_size: int,
        expand_factor: float = 1.0,
        max_interval_factor: float = 10.0,
    ):
        if base_interval <= 1:
            logging.warning(
                "TypeBScheduler: base_interval=%d is <=1.  Hard and neutral images "
                "will receive identical scheduling intervals (both clamp to 1).  "
                "Set base_interval = ceil(dataset_size / slab_size) for meaningful "
                "differentiation.",
                base_interval,
            )
        self.base_interval = base_interval
        self.slab_size = slab_size
        self.expand_factor = expand_factor
        self.max_interval = max(1, round(base_interval * max_interval_factor))

        # realpath -> global slab index at which the image is next due
        self._next_due_slab: dict[str, int] = {}
        self._current_slab: int = 0

    # -- helpers -------------------------------------------------------------

    def interval_for_score(self, score: float) -> int:
        """Compute inter-slab interval for a given difficulty score."""
        raw = self.base_interval / math.exp(score * self.expand_factor)
        return max(1, min(self.max_interval, round(raw)))

    def reset_new_items(self, items: list[ImageTrainItem]) -> None:
        """Ensure every item has a next_due_slab entry (call at epoch start)."""
        for item in items:
            key = os.path.realpath(item.pathname)
            if key not in self._next_due_slab:
                self._next_due_slab[key] = self._current_slab

    # -- main API ------------------------------------------------------------

    def build_next_slab(
        self,
        items: list[ImageTrainItem],
        scores: dict[str, float],
        obs_counts: dict[str, int],
        min_obs_count: int,
        n_slots: int,
        rng: Optional[random.Random] = None,
    ) -> list[ImageTrainItem]:
        """
        Build the item list for the current slab and advance the slab counter.

        Returns a list of exactly n_slots ImageTrainItems (may contain
        duplicates when hard images earn extra copies).
        """
        if rng is None:
            rng = random.Random()

        s = self._current_slab
        self.reset_new_items(items)

        unscored_due: list[ImageTrainItem] = []
        scored_due: list[ImageTrainItem] = []

        for item in items:
            key = os.path.realpath(item.pathname)
            if self._next_due_slab.get(key, s) > s:
                continue  # not yet due
            if obs_counts.get(key, 0) < min_obs_count:
                unscored_due.append(item)
            else:
                scored_due.append(item)

        # randomise unscored order, sort scored by overdue-ness (most overdue first)
        rng.shuffle(unscored_due)
        scored_due.sort(
            key=lambda it: self._next_due_slab.get(os.path.realpath(it.pathname), s)
        )

        selected: list[ImageTrainItem] = []

        def _select(item: ImageTrainItem) -> None:
            key = os.path.realpath(item.pathname)
            is_scored = obs_counts.get(key, 0) >= min_obs_count
            score = scores.get(key, 0.0) if is_scored else 0.0
            interval = self.interval_for_score(score)
            self._next_due_slab[key] = s + interval
            selected.append(item)

        # 1) unscored first (cold-start priority)
        for item in unscored_due:
            if len(selected) >= n_slots:
                break
            _select(item)

        # 2) most-overdue scored items
        for item in scored_due:
            if len(selected) >= n_slots:
                break
            _select(item)

        # 3) weighted resample to fill any leftover slots
        all_due = unscored_due + scored_due
        if len(selected) < n_slots and all_due:
            weights = []
            for item in all_due:
                key = os.path.realpath(item.pathname)
                is_scored = obs_counts.get(key, 0) >= min_obs_count
                score = scores.get(key, 0.0) if is_scored else 0.0
                weights.append(math.exp(score * self.expand_factor))
            total_w = sum(weights)
            while len(selected) < n_slots:
                r = rng.uniform(0.0, total_w)
                cumulative = 0.0
                chosen = all_due[-1]
                for item, w in zip(all_due, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = item
                        break
                selected.append(chosen)

        self._current_slab += 1
        return selected[:n_slots]

    def update_multipliers(
        self,
        items: list[ImageTrainItem],
        scores: dict[str, float],
        obs_counts: dict[str, int],
        min_obs_count: int,
    ) -> None:
        """
        Called at epoch-end to ensure any new items get a next_due_slab entry.
        (The live scheduling state is maintained inside build_next_slab; this
        method exists to satisfy the DifficultyScheduler interface so that the
        epoch-end wiring in train.py does not need to branch on scheduler type.)
        """
        self.reset_new_items(items)


# ---------------------------------------------------------------------------
# DifficultyEstimator
# ---------------------------------------------------------------------------

class DifficultyEstimator:
    """
    Accumulates per-image EMA difficulty scores from training losses and
    delegates scheduling decisions to a DifficultyScheduler.

    DB format (JSON):
    {
        "<realpath>": { "score": <float>, "obs_count": <int> },
        ...
    }
    """

    def __init__(
        self,
        model_type_identifier: str,
        scheduler: DifficultyScheduler,
        min_obs_count: int = 10,
        ema_alpha: float = 0.1,
    ):
        self.model_type_identifier = model_type_identifier
        self.scheduler = scheduler
        self.min_obs_count = min_obs_count
        self.ema_alpha = ema_alpha
        self._scores: dict[str, float] = {}
        self._obs_counts: dict[str, int] = {}

    # -- properties ----------------------------------------------------------

    @property
    def scores(self) -> dict[str, float]:
        return self._scores

    @property
    def obs_counts(self) -> dict[str, int]:
        return self._obs_counts

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str,
        model_type_identifier: str,
        scheduler: DifficultyScheduler,
        min_obs_count: int = 10,
        ema_alpha: float = 0.1,
    ) -> "DifficultyEstimator":
        inst = cls(model_type_identifier, scheduler, min_obs_count, ema_alpha)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                for realpath, entry in db.items():
                    inst._scores[realpath] = float(entry.get("score", 0.0))
                    inst._obs_counts[realpath] = int(entry.get("obs_count", 0))
                logging.info(
                    "DifficultyEstimator: loaded %d entries from %s",
                    len(inst._scores), path,
                )
            except Exception as e:
                logging.warning(
                    "DifficultyEstimator: failed to load DB from %s: %s", path, e
                )
        else:
            logging.info(
                "DifficultyEstimator: no existing DB at %s, starting fresh", path
            )
        return inst

    def save(self, path: str) -> None:
        db = {
            realpath: {
                "score": self._scores[realpath],
                "obs_count": self._obs_counts.get(realpath, 0),
            }
            for realpath in self._scores
        }
        dirpath = os.path.dirname(os.path.abspath(path))
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)

    # -- ingestion -----------------------------------------------------------

    def ingest_epoch_losses(self, log_data) -> None:
        """
        Read log_data.loss_per_image_and_timestep, normalise losses via the
        per-timestep reference curve, and update EMA scores + observation counts.

        log_data.loss_per_image_and_timestep:
            dict[batch_resolution, dict[image_path, list[tuple[timestep, loss]]]]
        """
        for _resolution, path_dict in log_data.loss_per_image_and_timestep.items():
            for image_path, timestep_loss_pairs in path_dict.items():
                if not timestep_loss_pairs:
                    continue

                realpath = os.path.realpath(image_path)
                timesteps_t = torch.tensor(
                    [t for t, _l in timestep_loss_pairs], dtype=torch.long
                )
                losses_t = torch.tensor(
                    [l for _t, l in timestep_loss_pairs], dtype=torch.float32
                )

                try:
                    normalised = normalize_loss_by_timestep(
                        losses_t, timesteps_t, self.model_type_identifier
                    )
                except (ValueError, IndexError) as e:
                    logging.warning(
                        "DifficultyEstimator: could not normalise loss for %s: %s",
                        realpath, e,
                    )
                    continue

                # score = mean( log(normalised_loss) )
                # > 0 → harder than average; < 0 → easier than average
                epoch_score = normalised.clamp(min=1e-8).log().mean().item()

                prev_score = self._scores.get(realpath, epoch_score)
                self._scores[realpath] = (
                    self.ema_alpha * epoch_score
                    + (1.0 - self.ema_alpha) * prev_score
                )
                self._obs_counts[realpath] = (
                    self._obs_counts.get(realpath, 0) + len(timestep_loss_pairs)
                )

    # -- epoch-end hook ------------------------------------------------------

    def update_item_schedule(self, items: list[ImageTrainItem]) -> None:
        """
        Epoch-end: update item.multiplier (Type A) or slab state (Type B)
        via the scheduler.
        """
        self.scheduler.update_multipliers(
            items, self._scores, self._obs_counts, self.min_obs_count
        )

