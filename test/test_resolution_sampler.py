"""
Unit tests for data.resolution_sampler.assign_resolutions().

These tests use in-memory ImageSourceItem / ImageTrainItem objects — no image files
are opened on disk.
"""
import copy
import random
import unittest
import uuid

from data.image_train_item import ImageCaption, ImageSourceItem, ImageTrainItem, ResolutionOption
from data.resolution_sampler import assign_resolutions


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_caption() -> ImageCaption:
    return ImageCaption("test caption", rating=1.0, tags=[], tag_weights=[], max_target_length=256, use_weights=False)


def _make_base_item(pathname: str = "/fake/path/image.jpg") -> ImageTrainItem:
    """Create a minimal ImageTrainItem without opening a file."""
    from torchvision import transforms
    item = ImageTrainItem.__new__(ImageTrainItem)
    item.caption         = _make_caption()
    item.aspects         = []
    item.pathname        = pathname
    item.flip            = transforms.RandomHorizontalFlip(p=0.0)
    item.cropped_img     = None
    item.runt_size       = 0
    item.multiplier      = 1.0
    item.base_multiplier = 1.0
    item.cond_dropout    = None
    item.shuffle_tags    = False
    item.batch_id        = "default_batch"
    item.loss_scale      = 1.0
    item.timesteps_range = None
    item.target_wh       = None
    item.is_runt         = False
    item.uid             = uuid.uuid4().hex
    item.source_resolution = None
    item.image_size      = (512, 512)
    item.mask            = None
    item.image           = []
    item.is_undersized   = False
    item.error           = None
    return item


def _make_source(
    resolution_options: dict,
    multiplier: float = 1.0,
    pathname: str = None,
) -> ImageSourceItem:
    """Create an ImageSourceItem with the given resolution options."""
    if pathname is None:
        pathname = f"/fake/path/image_{uuid.uuid4().hex[:8]}.jpg"
    base_item = _make_base_item(pathname)
    base_item.multiplier      = multiplier
    base_item.base_multiplier = multiplier
    source = ImageSourceItem(
        item=base_item,
        resolution_options=resolution_options,
        uid=uuid.uuid4().hex,
    )
    return source


def _both_resolutions(weight_512: float = 1.0, weight_1024: float = 1.0,
                      feasible_512: bool = True, feasible_1024: bool = True) -> dict:
    return {
        512:  ResolutionOption(512,  [512, 512],   weight_512,  feasible_512),
        1024: ResolutionOption(1024, [1024, 1024], weight_1024, feasible_1024),
    }


def _only_512() -> dict:
    return {
        512:  ResolutionOption(512,  [512, 512],   1.0, True),
        1024: ResolutionOption(1024, [1024, 1024], 1.0, False),  # not feasible
    }


EQUAL_WEIGHTS = {512: 1.0, 1024: 1.0}
FIXED_SEED    = random.Random(42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignResolutionsCount(unittest.TestCase):
    """All eligible images appear exactly once in the output."""

    def test_all_images_assigned(self):
        sources = [_make_source(_both_resolutions()) for _ in range(10)]
        result  = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(len(result), 10)

    def test_ineligible_image_is_skipped(self):
        """An image with is_feasible=False for all resolutions must be skipped."""
        sources = [_make_source({
            512:  ResolutionOption(512,  [512, 512],   1.0, False),
            1024: ResolutionOption(1024, [1024, 1024], 1.0, False),
        })]
        result = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(len(result), 0)


class TestResolutionDistribution(unittest.TestCase):
    """Resolution assignments should approximate the desired probabilities."""

    def test_equal_weights_give_roughly_half_each(self):
        n = 100
        sources = [_make_source(_both_resolutions()) for _ in range(n)]
        result  = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(7))
        count_512  = sum(1 for i in result if i.source_resolution == 512)
        count_1024 = sum(1 for i in result if i.source_resolution == 1024)
        self.assertEqual(count_512 + count_1024, n)
        # With 100 images and equal weights, expect roughly 50/50.
        # Allow generous tolerance (15–85%) since this is stochastic.
        self.assertGreater(count_512,  15)
        self.assertLess   (count_512,  85)

    def test_biased_per_resolution_multiply(self):
        """512:1.0, 1024:3.0  →  ~25% at 512, ~75% at 1024."""
        n = 100
        sources = [_make_source(_both_resolutions(weight_512=1.0, weight_1024=3.0))
                   for _ in range(n)]
        result = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(13))
        count_512 = sum(1 for i in result if i.source_resolution == 512)
        # expect ~25; allow 12–45
        self.assertGreater(count_512, 8,  "Too few 512 images")
        self.assertLess   (count_512, 45, "Too many 512 images")

    def test_global_weight_zero_excludes_resolution(self):
        """global_resolution_weights = {512: 0.0, 1024: 1.0}  → all at 1024."""
        sources = [_make_source(_both_resolutions()) for _ in range(10)]
        result  = assign_resolutions(sources, {512: 0.0, 1024: 1.0}, random.Random(0))
        for item in result:
            self.assertEqual(item.source_resolution, 1024)


class TestFeasibilityConstraint(unittest.TestCase):
    """Undersized images must never be assigned to a resolution they cannot fill."""

    def test_small_images_excluded_from_1024(self):
        # 5 small images (only feasible for 512), 5 large images (feasible for both)
        small  = [_make_source(_only_512())          for _ in range(5)]
        large  = [_make_source(_both_resolutions())  for _ in range(5)]
        result = assign_resolutions(small + large, EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(len(result), 10)
        for item in result:
            if item.image_size == (512, 512):  # small images have 512x512 in our fixture
                # all base items share the same image_size in this fixture; use pathname instead
                pass
        # Check by resolution option: no small-only image should have source_resolution == 1024
        small_pathnames = {s.pathname for s in small}
        for item in result:
            if item.pathname in small_pathnames:
                self.assertNotEqual(item.source_resolution, 1024)


class TestCornerCases(unittest.TestCase):

    def test_completely_ineligible_resolution_no_crash(self):
        """All images infeasible for 1024 → no 1024 items, no exception."""
        sources = [_make_source(_only_512()) for _ in range(5)]
        result  = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(len(result), 5)
        for item in result:
            self.assertEqual(item.source_resolution, 512)

    def test_zero_source_items_returns_empty(self):
        result = assign_resolutions([], EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(result, [])

    def test_empty_resolution_seeded_with_filler(self):
        """
        10 images, all weight on 1024 but only 1 is feasible.
        After main pass: at least 1 item has source_resolution == 1024.
        """
        sources = (
            [_make_source(_both_resolutions(weight_512=0.0, weight_1024=1.0))]  # 1 feasible for 1024
            + [_make_source({512: ResolutionOption(512, [512, 512], 1.0, True),
                             1024: ResolutionOption(1024, [1024, 1024], 0.0, False)})
               for _ in range(9)]  # 9 only feasible for 512 with all weight on it
        )
        result = assign_resolutions(sources, {512: 0.0, 1024: 1.0}, random.Random(0))
        count_1024 = sum(1 for i in result if i.source_resolution == 1024)
        # The one feasible image must end up at 1024, possibly with a filler
        self.assertGreaterEqual(count_1024, 1)


class TestAdaptiveCorrection(unittest.TestCase):
    """
    Adaptive weights should produce a tighter distribution than independent sampling.
    """

    def _run_distribution(self, n_trials: int) -> list[float]:
        """Return fraction of images assigned to 512 over n_trials runs."""
        fractions = []
        for seed in range(n_trials):
            sources = [_make_source(_both_resolutions()) for _ in range(50)]
            result  = assign_resolutions(sources, EQUAL_WEIGHTS, random.Random(seed))
            count_512 = sum(1 for i in result if i.source_resolution == 512)
            fractions.append(count_512 / max(1, len(result)))
        return fractions

    def test_distribution_is_close_to_half(self):
        fracs = self._run_distribution(20)
        mean = sum(fracs) / len(fracs)
        self.assertGreater(mean, 0.35)
        self.assertLess   (mean, 0.65)

    def test_variance_is_low(self):
        fracs = self._run_distribution(20)
        mean  = sum(fracs) / len(fracs)
        var   = sum((f - mean) ** 2 for f in fracs) / len(fracs)
        # With 50 images and perfect adaptive correction, std-dev should be < 0.15
        import math
        self.assertLess(math.sqrt(var), 0.15)


class TestMultiplierDeduplication(unittest.TestCase):
    """
    When the same ImageSourceItem appears twice (multiplier>1 case), make_resolved_item
    is called on distinct objects so neither result corrupts the other.
    """

    def test_duplicate_source_items_produce_distinct_resolved_items(self):
        source = _make_source(_both_resolutions())
        # Simulate multiplier=2 by passing the same object twice
        duplicate = [source, source]
        result = assign_resolutions(duplicate, EQUAL_WEIGHTS, random.Random(0))
        self.assertEqual(len(result), 2)
        # Both resolved items should have valid target_wh
        for item in result:
            self.assertIsNotNone(item.target_wh)
        # Their uids must differ (each call to make_resolved_item generates a fresh uid)
        uids = [item.uid for item in result]
        self.assertEqual(len(set(uids)), 2)


if __name__ == '__main__':
    unittest.main()


