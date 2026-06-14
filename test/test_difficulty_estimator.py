"""
Comprehensive integration tests for DifficultyEstimator, TypeAScheduler,
and TypeBScheduler.
Tests cover:
  - TypeA: multiplier increases for hard images, decreases for easy images
  - TypeA: images below min_obs_count are unaffected
  - TypeA: base_multiplier anchors scaling
  - TypeB: cold-start — all images seen before scored images crowd out unseen
  - TypeB: after scoring, hard images appear more often than easy images
  - TypeB: 10-epoch simulation — appearance frequency correlates with difficulty
  - TypeB: base_interval=1 degeneracy warning
  - DifficultyEstimator: score ingestion & EMA accumulation
  - DifficultyEstimator: save/load round-trip
  - DifficultyEstimator: unknown model_type raises ValueError
"""
from __future__ import annotations
import json
import math
import os
import random
import tempfile
import types
import unittest
from collections import Counter
from unittest.mock import MagicMock
import torch
# ---------------------------------------------------------------------------
# Minimal stub for ImageTrainItem (avoids disk I/O)
# ---------------------------------------------------------------------------
class _FakeItem:
    """Lightweight stand-in for ImageTrainItem."""
    def __init__(self, name: str, multiplier: float = 1.0):
        self.pathname = f"/fake/{name}.png"
        self.multiplier = multiplier
        self.base_multiplier = multiplier
    @property
    def _key(self) -> str:
        return os.path.realpath(self.pathname)
def _make_items(names, multiplier=1.0):
    return [_FakeItem(n, multiplier) for n in names]
def _make_log_data(losses_by_name, resolution=512):
    """Build a minimal log_data object for ingest_epoch_losses."""
    path_dict = {f"/fake/{name}.png": pairs for name, pairs in losses_by_name.items()}
    ld = types.SimpleNamespace()
    ld.loss_per_image_and_timestep = {resolution: path_dict}
    return ld
from data.difficulty_estimator import (
    DifficultyEstimator,
    TypeAScheduler,
    TypeBScheduler,
)
# ===========================================================================
# TypeAScheduler
# ===========================================================================
class TestTypeAScheduler(unittest.TestCase):
    def test_hard_image_gets_higher_multiplier(self):
        item = _FakeItem("hard", 1.0)
        s = TypeAScheduler(min_multiplier=0.1, max_multiplier=10.0, expand_factor=1.0)
        s.update_multipliers([item], {item._key: 1.5}, {item._key: 5}, 1)
        self.assertGreater(item.multiplier, 1.0)
    def test_easy_image_gets_lower_multiplier(self):
        item = _FakeItem("easy", 1.0)
        s = TypeAScheduler(min_multiplier=0.1, max_multiplier=10.0, expand_factor=1.0)
        s.update_multipliers([item], {item._key: -1.5}, {item._key: 5}, 1)
        self.assertLess(item.multiplier, 1.0)
    def test_neutral_image_keeps_base_multiplier(self):
        item = _FakeItem("neutral", 1.0)
        s = TypeAScheduler(min_multiplier=0.1, max_multiplier=10.0)
        s.update_multipliers([item], {item._key: 0.0}, {item._key: 5}, 1)
        self.assertAlmostEqual(item.multiplier, 1.0, places=6)
    def test_unscored_image_unchanged(self):
        item = _FakeItem("cold", 1.0)
        s = TypeAScheduler()
        s.update_multipliers([item], {item._key: 2.0}, {item._key: 0}, min_obs_count=5)
        self.assertAlmostEqual(item.multiplier, 1.0, places=6)
    def test_max_clamp(self):
        item = _FakeItem("xh", 1.0)
        s = TypeAScheduler(min_multiplier=0.5, max_multiplier=2.0, expand_factor=1.0)
        s.update_multipliers([item], {item._key: 100.0}, {item._key: 10}, 1)
        self.assertAlmostEqual(item.multiplier, 2.0, places=6)
    def test_min_clamp(self):
        item = _FakeItem("xe", 1.0)
        s = TypeAScheduler(min_multiplier=0.5, max_multiplier=2.0, expand_factor=1.0)
        s.update_multipliers([item], {item._key: -100.0}, {item._key: 10}, 1)
        self.assertAlmostEqual(item.multiplier, 0.5, places=6)
    def test_scales_relative_to_base_multiplier(self):
        item = _FakeItem("weighted", 2.0)
        item.base_multiplier = 2.0
        s = TypeAScheduler(min_multiplier=0.5, max_multiplier=3.0, expand_factor=1.0)
        s.update_multipliers([item], {item._key: 0.0}, {item._key: 5}, 1)
        self.assertAlmostEqual(item.multiplier, 2.0, places=6)
    def test_ordering_hard_neutral_easy(self):
        hard = _FakeItem("h", 1.0); neutral = _FakeItem("n", 1.0); easy = _FakeItem("e", 1.0)
        s = TypeAScheduler(min_multiplier=0.1, max_multiplier=10.0, expand_factor=1.0)
        scores = {hard._key: 1.0, neutral._key: 0.0, easy._key: -1.0}
        obs = {k: 5 for k in scores}
        s.update_multipliers([hard, neutral, easy], scores, obs, 1)
        self.assertGreater(hard.multiplier, neutral.multiplier)
        self.assertGreater(neutral.multiplier, easy.multiplier)
# ===========================================================================
# TypeBScheduler
# ===========================================================================
class TestTypeBScheduler(unittest.TestCase):
    def _sched(self, base_interval=5, slab_size=4, expand_factor=1.0):
        return TypeBScheduler(base_interval=base_interval, slab_size=slab_size,
                               expand_factor=expand_factor)
    def test_returns_correct_slot_count(self):
        items = _make_items([f"img{i}" for i in range(20)])
        sched = self._sched()
        result = sched.build_next_slab(items, {}, {}, 1, 4, rng=random.Random(0))
        self.assertEqual(len(result), 4)
    def test_cold_start_all_images_seen(self):
        items = _make_items([f"img{i:02d}" for i in range(20)])
        sched = self._sched(base_interval=5, slab_size=4)
        rng = random.Random(0)
        seen = set()
        for _ in range(5):
            for i in sched.build_next_slab(items, {}, {}, 1, 4, rng=rng):
                seen.add(os.path.realpath(i.pathname))
        self.assertEqual(len(seen), 20)
    def test_unscored_priority_over_scored(self):
        unscored = _make_items(["u0", "u1", "u2"])
        scored   = _make_items(["s0", "s1", "s2"])
        all_items = unscored + scored
        scores    = {s._key: 2.0 for s in scored}
        obs       = {s._key: 99 for s in scored}
        sched = self._sched()
        slab  = sched.build_next_slab(all_items, scores, obs, 1, 3, rng=random.Random(7))
        slab_keys = {os.path.realpath(i.pathname) for i in slab}
        for u in unscored:
            self.assertIn(u._key, slab_keys)
    def test_hard_do_not_crowd_out_unscored(self):
        unscored   = _make_items(["u0", "u1"])
        hard_scored = _make_items([f"h{i}" for i in range(10)])
        all_items   = unscored + hard_scored
        scores = {i._key: 3.0 for i in hard_scored}
        obs    = {i._key: 50  for i in hard_scored}
        sched  = self._sched()
        slab   = sched.build_next_slab(all_items, scores, obs, 1, 4, rng=random.Random(1))
        slab_keys = [os.path.realpath(i.pathname) for i in slab]
        for u in unscored:
            self.assertIn(u._key, slab_keys)
    def test_interval_formula(self):
        sched = self._sched(base_interval=10, slab_size=4, expand_factor=1.0)
        self.assertEqual(sched.interval_for_score(0.0), 10)
        self.assertEqual(sched.interval_for_score(math.log(2)), 5)
        self.assertEqual(sched.interval_for_score(100.0), 1)
        self.assertEqual(sched.interval_for_score(-100.0), sched.max_interval)
    def test_next_due_advances(self):
        items = _make_items(["a"])
        sched = TypeBScheduler(base_interval=7, slab_size=1)
        sched.build_next_slab(items, {items[0]._key: 0.0}, {items[0]._key: 5},
                              1, 1, rng=random.Random(0))
        self.assertEqual(sched._next_due_slab[items[0]._key], 7)
    def test_base_interval_one_warns(self):
        with self.assertLogs(level="WARNING"):
            TypeBScheduler(base_interval=1, slab_size=4)
    def test_update_multipliers_noop_for_existing(self):
        items = _make_items(["x"])
        sched = self._sched()
        sched._next_due_slab[items[0]._key] = 99
        sched.update_multipliers(items, {}, {}, 1)
        self.assertEqual(sched._next_due_slab[items[0]._key], 99)
# ===========================================================================
# 10-epoch simulation
# ===========================================================================
class TestTypeBTenEpochSimulation(unittest.TestCase):
    N_HARD = 5; N_NEUTRAL = 5; N_EASY = 5; N_VERY_EASY = 5
    SLAB_SIZE = 4; N_SLABS_PER_EPOCH = 5; N_EPOCHS = 10; BASE_INTERVAL = 5
    MIN_OBS = 1; EXPAND_FACTOR = 1.0
    SCORE_HARD = +1.5; SCORE_NEUTRAL = 0.0; SCORE_EASY = -1.5; SCORE_VERY_EASY = -3.0
    def setUp(self):
        self.hard    = _make_items([f"h{i}" for i in range(self.N_HARD)])
        self.neutral = _make_items([f"n{i}" for i in range(self.N_NEUTRAL)])
        self.easy    = _make_items([f"e{i}" for i in range(self.N_EASY)])
        self.veryeasy= _make_items([f"v{i}" for i in range(self.N_VERY_EASY)])
        self.all_items = self.hard + self.neutral + self.easy + self.veryeasy
        self.scores:     dict[str, float] = {}
        self.obs_counts: dict[str, int]   = {}
        for item, score in (
            [(i, self.SCORE_HARD)      for i in self.hard]    +
            [(i, self.SCORE_NEUTRAL)   for i in self.neutral] +
            [(i, self.SCORE_EASY)      for i in self.easy]    +
            [(i, self.SCORE_VERY_EASY) for i in self.veryeasy]
        ):
            self.scores[item._key]     = score
            self.obs_counts[item._key] = self.MIN_OBS
        self.sched = TypeBScheduler(
            base_interval=self.BASE_INTERVAL,
            slab_size=self.SLAB_SIZE,
            expand_factor=self.EXPAND_FACTOR,
            max_interval_factor=20.0,
        )
    def _simulate(self, seed=42):
        rng = random.Random(seed)
        counts: Counter = Counter()
        for _ in range(self.N_EPOCHS):
            for _ in range(self.N_SLABS_PER_EPOCH):
                slab = self.sched.build_next_slab(
                    self.all_items, self.scores, self.obs_counts,
                    min_obs_count=self.MIN_OBS, n_slots=self.SLAB_SIZE, rng=rng)
                for item in slab:
                    counts[item._key] += 1
        return counts
    def _mean(self, counts, group):
        return sum(counts[i._key] for i in group) / len(group)
    def test_cold_start_all_seen(self):
        sched = TypeBScheduler(base_interval=self.BASE_INTERVAL, slab_size=self.SLAB_SIZE)
        rng = random.Random(0); seen = set()
        for _ in range(self.N_SLABS_PER_EPOCH):
            for i in sched.build_next_slab(self.all_items, {}, {}, self.MIN_OBS,
                                            self.SLAB_SIZE, rng):
                seen.add(os.path.realpath(i.pathname))
        self.assertEqual(len(seen), len(self.all_items))
    def test_strict_ordering(self):
        counts = self._simulate()
        mh = self._mean(counts, self.hard); mn = self._mean(counts, self.neutral)
        me = self._mean(counts, self.easy); mv = self._mean(counts, self.veryeasy)
        self.assertGreater(mh, mn, f"hard={mh:.2f} neutral={mn:.2f}")
        self.assertGreater(mn, me, f"neutral={mn:.2f} easy={me:.2f}")
        self.assertGreater(me, mv, f"easy={me:.2f} veryeasy={mv:.2f}")
    def test_hard_appears_multiple_times_per_epoch(self):
        counts = self._simulate()
        for item in self.hard:
            avg = counts[item._key] / self.N_EPOCHS
            self.assertGreater(avg, 1.5, f"{item.pathname} avg={avg:.2f}, expected >1.5")
    def test_very_easy_appears_less_than_once_per_epoch(self):
        counts = self._simulate()
        for item in self.veryeasy:
            avg = counts[item._key] / self.N_EPOCHS
            self.assertLess(avg, 1.0, f"{item.pathname} avg={avg:.2f}, expected <1.0")
    def test_total_slots_filled(self):
        counts = self._simulate()
        expected = self.N_EPOCHS * self.N_SLABS_PER_EPOCH * self.SLAB_SIZE
        self.assertEqual(sum(counts.values()), expected)
    def test_monotonic_rank_correlation(self):
        counts = self._simulate(seed=999)
        means = [self._mean(counts, g) for g in
                 [self.hard, self.neutral, self.easy, self.veryeasy]]
        for i in range(len(means) - 1):
            self.assertGreater(means[i], means[i+1],
                               f"Group means not monotone: {[f'{m:.2f}' for m in means]}")
# ===========================================================================
# DifficultyEstimator
# ===========================================================================
class TestDifficultyEstimatorIngestion(unittest.TestCase):
    def _est(self, ema_alpha=1.0, min_obs=1, model="flowmatching-sd2"):
        return DifficultyEstimator(model, TypeAScheduler(), min_obs, ema_alpha)
    def test_unknown_model_type_logs_warning_and_skips(self):
        """An unrecognised model_type_identifier should log a warning and not
        update scores, but must NOT crash (training should continue)."""
        est = self._est(model="unknown-model")
        key = os.path.realpath("/fake/img.png")
        with self.assertLogs(level="WARNING"):
            est.ingest_epoch_losses(_make_log_data({"img": [(100, 0.5)]}))
        # Score should not have been set
        self.assertNotIn(key, est.scores)
    def test_obs_count_accumulates(self):
        est = self._est()
        key = os.path.realpath("/fake/img.png")
        est.ingest_epoch_losses(_make_log_data({"img": [(100, 0.5), (200, 0.4)]}))
        self.assertEqual(est.obs_counts.get(key, 0), 2)
        est.ingest_epoch_losses(_make_log_data({"img": [(100, 0.5), (200, 0.4)]}))
        self.assertEqual(est.obs_counts.get(key, 0), 4)
    def test_score_sign_hard_vs_easy(self):
        ref = 0.4275
        est_hard = self._est(); est_easy = self._est()
        est_hard.ingest_epoch_losses(_make_log_data({"img": [(0, ref * 3.0)]}))
        est_easy.ingest_epoch_losses(_make_log_data({"img": [(0, ref * 0.2)]}))
        key = os.path.realpath("/fake/img.png")
        self.assertGreater(est_hard.scores[key], 0)
        self.assertLess(est_easy.scores[key], 0)
    def test_ema_smoothing(self):
        est = self._est(ema_alpha=0.1)
        key = os.path.realpath("/fake/img.png")
        est.ingest_epoch_losses(_make_log_data({"img": [(0, 10.0)]}))
        s1 = est.scores.get(key, 0.0)
        est.ingest_epoch_losses(_make_log_data({"img": [(0, 0.001)]}))
        s2 = est.scores.get(key, 0.0)
        self.assertGreater(s1, s2)
        self.assertGreater(s2, s1 - 5.0)
    def test_save_load_roundtrip(self):
        est = self._est()
        est.ingest_epoch_losses(_make_log_data({
            "hard": [(100, 2.0)], "neutral": [(100, 0.4)], "easy": [(100, 0.1)],
        }))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            est.save(tmp)
            with open(tmp) as f:
                db = json.load(f)
            self.assertIn(os.path.realpath("/fake/hard.png"), db)
            est2 = DifficultyEstimator.load(tmp, "flowmatching-sd2", TypeAScheduler())
            for key in est.scores:
                self.assertAlmostEqual(est.scores[key], est2.scores[key], places=5)
                self.assertEqual(est.obs_counts[key], est2.obs_counts.get(key))
        finally:
            os.unlink(tmp)
    def test_load_nonexistent_starts_fresh(self):
        est = DifficultyEstimator.load("/no/such/file.json", "flowmatching-sd2",
                                       TypeAScheduler())
        self.assertEqual(len(est.scores), 0)
    def test_delegates_to_scheduler(self):
        mock_sched = MagicMock(spec=TypeAScheduler)
        est = DifficultyEstimator("flowmatching-sd2", mock_sched, min_obs_count=3)
        items = _make_items(["a", "b"])
        est.update_item_schedule(items)
        mock_sched.update_multipliers.assert_called_once_with(
            items, est.scores, est.obs_counts, 3)
# ===========================================================================
# Full end-to-end: DifficultyEstimator + TypeA
# ===========================================================================
class TestTypeAEndToEnd(unittest.TestCase):
    def test_multipliers_track_difficulty(self):
        hard = _FakeItem("hard", 1.0); easy = _FakeItem("easy", 1.0)
        sched = TypeAScheduler(min_multiplier=0.1, max_multiplier=10.0, expand_factor=1.0)
        est   = DifficultyEstimator("flowmatching-sd2", sched, 1, ema_alpha=0.5)
        ref   = 0.4275
        for _ in range(5):
            est.ingest_epoch_losses(_make_log_data({
                "hard": [(0, ref * 4.0)], "easy": [(0, ref * 0.1)]}))
            est.update_item_schedule([hard, easy])
        self.assertGreater(hard.multiplier, 1.0)
        self.assertLess(easy.multiplier, 1.0)
        self.assertGreater(hard.multiplier, easy.multiplier)
# ===========================================================================
# Full end-to-end: DifficultyEstimator + TypeB
# ===========================================================================
class TestTypeBEndToEnd(unittest.TestCase):
    def test_frequency_ordering(self):
        hard_items = _make_items([f"h{i}" for i in range(5)])
        easy_items = _make_items([f"e{i}" for i in range(5)])
        all_items  = hard_items + easy_items
        sched = TypeBScheduler(base_interval=5, slab_size=2, expand_factor=1.0)
        est   = DifficultyEstimator("flowmatching-sd2", sched, 1, ema_alpha=0.5)
        ref   = 0.4275
        rng   = random.Random(123)
        counts: Counter = Counter()
        for _ in range(10):
            est.ingest_epoch_losses(_make_log_data(
                {os.path.basename(i.pathname).replace(".png", ""): [(100, ref * 5.0)]
                 for i in hard_items} |
                {os.path.basename(i.pathname).replace(".png", ""): [(100, ref * 0.05)]
                 for i in easy_items}
            ))
            est.update_item_schedule(all_items)
            for _ in range(5):
                for item in sched.build_next_slab(all_items, est.scores, est.obs_counts,
                                                   1, 2, rng=rng):
                    counts[os.path.realpath(item.pathname)] += 1
        hard_total = sum(counts[i._key] for i in hard_items)
        easy_total = sum(counts[i._key] for i in easy_items)
        self.assertGreater(hard_total, easy_total,
                           f"hard={hard_total} should exceed easy={easy_total}")
if __name__ == "__main__":
    unittest.main()
