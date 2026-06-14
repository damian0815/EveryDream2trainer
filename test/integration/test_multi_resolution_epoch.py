"""
Integration tests for the multi-resolution epoch pipeline.

These tests create real (tiny) JPEG images on disk and exercise the full
Dataset → ImageSourceItem → assign_resolutions path.
"""
import os
import pathlib
import random
import shutil
import tempfile
import unittest

import PIL.Image

import data.aspects as aspects_module
from data.dataset import Dataset
from data.resolution_sampler import assign_resolutions


BUCKETS_512  = aspects_module.get_aspect_buckets(512)
BUCKETS_1024 = aspects_module.get_aspect_buckets(1024)

ASPECTS_PER_RESOLUTION = {
    512:  BUCKETS_512,
    1024: BUCKETS_1024,
}
EQUAL_GLOBAL_WEIGHTS = {512: 1.0, 1024: 1.0}


class TestMultiResolutionEpoch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ed2_inttest_"))

        # 10 "small" images (512×512) — too small for 1024
        cls.small_paths = []
        for i in range(10):
            p = cls.tmpdir / f"small_{i:02d}.jpg"
            PIL.Image.new("RGB", (512, 512), color=(i * 20, 0, 0)).save(p)
            (cls.tmpdir / f"small_{i:02d}.txt").write_text(f"small image {i}")
            cls.small_paths.append(str(p))

        # 10 "large" images (1024×1024) — feasible for both 512 and 1024
        cls.large_paths = []
        for i in range(10):
            p = cls.tmpdir / f"large_{i:02d}.jpg"
            PIL.Image.new("RGB", (1024, 1024), color=(0, i * 20, 0)).save(p)
            (cls.tmpdir / f"large_{i:02d}.txt").write_text(f"large image {i}")
            cls.large_paths.append(str(p))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _load_sources(self):
        dataset = Dataset.from_path(str(self.tmpdir))
        return dataset.image_source_items(ASPECTS_PER_RESOLUTION)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_all_images_appear_each_epoch(self):
        """Every source image should appear exactly once in the resolved output."""
        sources = self._load_sources()
        self.assertEqual(len(sources), 20, "Expected 20 source images")
        result = assign_resolutions(sources, EQUAL_GLOBAL_WEIGHTS, random.Random(0))
        resolved_paths = {item.pathname for item in result}
        source_paths   = {s.pathname for s in sources}
        self.assertEqual(resolved_paths, source_paths)

    def test_small_images_not_assigned_to_1024(self):
        """512×512 images must never appear with source_resolution == 1024."""
        sources = self._load_sources()
        result  = assign_resolutions(sources, EQUAL_GLOBAL_WEIGHTS, random.Random(0))
        small_path_set = set(os.path.abspath(p) for p in self.small_paths)
        for item in result:
            if os.path.abspath(item.pathname) in small_path_set:
                self.assertNotEqual(
                    item.source_resolution, 1024,
                    f"Small image {item.pathname} was assigned to 1024"
                )

    def test_large_images_feasible_for_both(self):
        """Large images should appear in both resolution buckets across multiple runs."""
        sources = self._load_sources()
        large_path_set = set(os.path.abspath(p) for p in self.large_paths)
        seen_512  = set()
        seen_1024 = set()
        for seed in range(20):
            result = assign_resolutions(sources, EQUAL_GLOBAL_WEIGHTS, random.Random(seed))
            for item in result:
                p = os.path.abspath(item.pathname)
                if p in large_path_set:
                    if item.source_resolution == 512:
                        seen_512.add(p)
                    elif item.source_resolution == 1024:
                        seen_1024.add(p)
        # With 20 runs and equal weights, every large image should hit both resolutions
        self.assertGreater(len(seen_512),  0, "No large image ever went to 512")
        self.assertGreater(len(seen_1024), 0, "No large image ever went to 1024")

    def test_per_resolution_multiply_respected(self):
        """
        Images with a strong 1024 preference (weight 1:9) should land at 1024
        more often than those with equal weights.
        """
        # Write 5 biased images with a local.yaml declaring per_resolution_multiply
        biased_dir = self.tmpdir / "biased"
        biased_dir.mkdir(exist_ok=True)
        for i in range(5):
            PIL.Image.new("RGB", (1024, 1024), color=(0, 0, i * 40)).save(
                biased_dir / f"biased_{i:02d}.jpg"
            )
            (biased_dir / f"biased_{i:02d}.txt").write_text(f"biased image {i}")

        import yaml
        (biased_dir / "local.yaml").write_text(
            yaml.dump({"per_resolution_multiply": {512: 1, 1024: 9}})
        )

        dataset = Dataset.from_path(str(self.tmpdir))
        sources = dataset.image_source_items(ASPECTS_PER_RESOLUTION)
        biased_paths = {
            os.path.abspath(str(biased_dir / f"biased_{i:02d}.jpg"))
            for i in range(5)
        }

        counts_1024 = 0
        n_runs      = 10
        for seed in range(n_runs):
            result = assign_resolutions(sources, EQUAL_GLOBAL_WEIGHTS, random.Random(seed))
            for item in result:
                if os.path.abspath(item.pathname) in biased_paths:
                    if item.source_resolution == 1024:
                        counts_1024 += 1

        # Expect biased images to land at 1024 in more than 70% of appearances
        total_biased_appearances = 5 * n_runs
        fraction_1024 = counts_1024 / total_biased_appearances
        self.assertGreater(
            fraction_1024, 0.60,
            f"Biased images landed at 1024 only {fraction_1024:.0%} of the time"
        )

        # cleanup
        shutil.rmtree(biased_dir, ignore_errors=True)

    def test_resolved_items_have_target_wh(self):
        """Every resolved ImageTrainItem must have a non-None target_wh."""
        sources = self._load_sources()
        result  = assign_resolutions(sources, EQUAL_GLOBAL_WEIGHTS, random.Random(0))
        for item in result:
            self.assertIsNotNone(item.target_wh, f"target_wh is None for {item.pathname}")
            self.assertEqual(len(item.target_wh), 2)


if __name__ == '__main__':
    unittest.main()

