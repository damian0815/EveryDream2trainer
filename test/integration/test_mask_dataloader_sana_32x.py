"""
Integration test: masks survive the training dataloader with SANA's 32x VAE downscale.

Exercises the real Dataset → ImageSourceItem → DataLoaderMultiAspect → EveryDreamBatch
pipeline on disk with real image + mask pairs.  During hydrate() we record the exact
transform parameters (crop jitter, trim offset, rotation, flip) that were applied to
each image, then independently replay them onto the *original* mask, downscale by
``vae_resize_factor`` (32 for SANA), and assert the result equals the mask that came
out of the dataloader.  This proves the mask is transformed in lock-step with the image.

Set ED_MASK_DEBUG=1 to dump all involved images to /tmp/ed_mask_debug/ for inspection.
"""
import json
import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import PIL.Image

import data.image_train_item as iti
from data.aspects import get_aspect_buckets
from data.data_loader import DataLoaderMultiAspect
from data.dataset import Dataset
from data.every_dream import EveryDreamBatch
from data.image_train_item import ImageTrainItem

VAE_RESIZE_FACTOR = 32


def _gradient_rgb(width, height, seed=0):
    """Diagonal RGB gradient that is sensitive to any crop/flip displacement."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    channel = (xx + yy).astype(np.float32) * 0.15
    r = (channel + rng.uniform(0, 60)).astype(np.uint8)
    g = (channel * 1.4).astype(np.uint8)
    b = (255 - channel).astype(np.uint8)
    return PIL.Image.fromarray(np.stack([r, g, b], axis=-1))


def _gradient_mask(width, height, seed=1):
    """Different gradient than the image, so any misalignment is caught."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    v = (xx * 0.2 - yy * 0.1 + rng.uniform(0, 40)).astype(np.uint8)
    return PIL.Image.fromarray(v, mode="L")


class _TransformRecorder:
    """Captures the transform parameters hydrate() uses, per hydrated item."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.jitter = None
        self.trim_offsets = None
        self.rotation_angles = []
        self.safe_boxes = []
        self.flipped = False


class TestMaskThroughDataloaderSana32x(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ed2_mask_inttest_"))

        # 3 image/mask pairs with different aspect ratios, plus one flip-forced pair.
        cls.data_dir = cls.tmpdir / "data"
        cls.data_dir.mkdir()
        pairs = [
            ("square_768", 768, 768),
            ("landscape_1024x576", 1024, 576),
            ("portrait_576x1024", 576, 1024),
        ]
        for name, w, h in pairs:
            img_path = cls.data_dir / f"{name}.png"
            _gradient_rgb(w, h).save(img_path)
            _gradient_mask(w, h).save(img_path.parent / f"{img_path.name}.mask.png")

        # flip_p.txt forces every image in this subdir through TF.hflip.
        flip_dir = cls.data_dir / "flip"
        flip_dir.mkdir()
        (flip_dir / "flip_p.txt").write_text("1.0")
        img_path = flip_dir / "flip_landscape.png"
        _gradient_rgb(1024, 576, seed=7).save(img_path)
        _gradient_mask(1024, 576, seed=8).save(img_path.parent / f"{img_path.name}.mask.png")

        # Mask whose size differs from the image must be rejected by hydrate().
        cls.wrongsize_dir = cls.tmpdir / "wrongsize"
        cls.wrongsize_dir.mkdir()
        wrong_img = cls.wrongsize_dir / "wrongsize.png"
        _gradient_rgb(768, 768, seed=9).save(wrong_img)
        _gradient_mask(512, 512, seed=10).save(wrong_img.parent / f"{wrong_img.name}.mask.png")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        self.rec = _TransformRecorder()
        self._patched = {
            "_get_random_jitter_amounts": ImageTrainItem._get_random_jitter_amounts,
            "_get_trim_offsets": ImageTrainItem._get_trim_offsets,
            "_apply_rotation": ImageTrainItem._apply_rotation,
            "plan_rotation_and_crop": iti.plan_rotation_and_crop,
            "hflip": iti.TF.hflip,
        }
        ImageTrainItem._get_random_jitter_amounts = self._record_jitter
        ImageTrainItem._get_trim_offsets = self._record_trim
        ImageTrainItem._apply_rotation = staticmethod(self._record_rotate)
        iti.plan_rotation_and_crop = self._record_plan
        iti.TF.hflip = self._record_hflip

    def tearDown(self):
        ImageTrainItem._get_random_jitter_amounts = self._patched["_get_random_jitter_amounts"]
        ImageTrainItem._get_trim_offsets = self._patched["_get_trim_offsets"]
        ImageTrainItem._apply_rotation = staticmethod(self._patched["_apply_rotation"])
        iti.plan_rotation_and_crop = self._patched["plan_rotation_and_crop"]
        iti.TF.hflip = self._patched["hflip"]

    # ------------------------------------------------------------------
    # Recorder wrappers
    # ------------------------------------------------------------------

    def _record_jitter(self, image, crop_jitter=0.02, rng=None):
        out = self._patched["_get_random_jitter_amounts"](
            self, image, crop_jitter=crop_jitter, rng=rng
        )
        self.rec.jitter = out
        return out

    def _record_trim(self, image, target_wh, rng=None):
        out = self._patched["_get_trim_offsets"](self, image, target_wh, rng=rng)
        self.rec.trim_offsets = out
        return out

    def _record_rotate(self, image, angle_degrees):
        out = self._patched["_apply_rotation"](image, angle_degrees)
        self.rec.rotation_angles.append(angle_degrees)
        return out

    def _record_plan(self, *args, **kwargs):
        out = self._patched["plan_rotation_and_crop"](*args, **kwargs)
        self.rec.safe_boxes.append(out[1])
        return out

    def _record_hflip(self, img):
        self.rec.flipped = True
        return self._patched["hflip"](img)

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _build_dataloader(self, root):
        aspects = {512: get_aspect_buckets(512)}
        sources = Dataset.from_path(str(root)).image_source_items(
            aspects, vae_resize_factor=VAE_RESIZE_FACTOR
        )
        dlma = DataLoaderMultiAspect(
            sources, seed=555, batch_size=1, global_resolution_weights={512: 1.0}
        )
        edb = EveryDreamBatch(dlma, tokenizer=None, mask_p=1.0, crop_jitter=0.02, seed=555)
        return dlma, edb

    def _reconstruct_mask(self, item):
        """Replay the recorded transforms onto the original mask + 32x downscale."""
        mask = PIL.Image.open(item.pathname_mask).convert("L")

        for angle in self.rec.rotation_angles:
            mask = ImageTrainItem._apply_rotation(mask, angle)
        for box_w, box_h in self.rec.safe_boxes:
            w, h = mask.size
            left, top = (w - box_w) // 2, (h - box_h) // 2
            mask = mask.crop((left, top, left + box_w, top + box_h))
        if self.rec.jitter is not None:
            mask = item._apply_crop_jitter(mask, precomputed_jitter=self.rec.jitter)
        if self.rec.trim_offsets is not None:
            mask, _ = item._apply_trim(mask, item.target_wh, self.rec.trim_offsets)

        mask = mask.resize(
            (item.target_wh[0] // item.vae_resize_factor, item.target_wh[1] // item.vae_resize_factor)
        )
        if self.rec.flipped:
            mask = iti.TF.hflip(mask)
        return np.array(mask.convert("L")).astype(np.float32) / 255.0

    def _debug_dump(self, example, item, expected, idx):
        outdir = pathlib.Path("/tmp") / "ed_mask_debug" / f"item_{idx}"
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(item.pathname, outdir / "image_original.png")
        shutil.copy(item.pathname_mask, outdir / "mask_original.png")

        img = example["image"].numpy().transpose(1, 2, 0)
        img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(img).save(outdir / "image_hydrated.png")

        mask = (example["mask"].numpy()[0] * 255.0).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(mask).save(outdir / "mask_hydrated.png")

        PIL.Image.fromarray((expected * 255.0).clip(0, 255).astype(np.uint8)).save(
            outdir / "mask_reconstructed.png"
        )
        (outdir / "params.json").write_text(json.dumps({
            "pathname": item.pathname,
            "target_wh": list(item.target_wh),
            "vae_resize_factor": item.vae_resize_factor,
            "jitter": list(self.rec.jitter) if self.rec.jitter is not None else None,
            "trim_offsets": list(self.rec.trim_offsets) if self.rec.trim_offsets is not None else None,
            "rotation_angles": self.rec.rotation_angles,
            "safe_boxes": [list(b) for b in self.rec.safe_boxes],
            "flipped": self.rec.flipped,
        }, indent=2))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_mask_survives_dataloader_with_32x_downscale(self):
        _, edb = self._build_dataloader(self.data_dir)
        self.assertEqual(len(edb), 4)
        self.assertTrue(all(i.vae_resize_factor == VAE_RESIZE_FACTOR for i in edb.image_train_items))

        flipped_covered = False
        for i in range(len(edb)):
            self.rec.reset()
            example = edb[i]
            item = edb.image_train_items[i]

            self.assertIsNotNone(example["mask"], f"no mask surfaced for {item.pathname}")
            self.assertEqual(item.vae_resize_factor, VAE_RESIZE_FACTOR)
            self.assertEqual(
                example["mask"].shape,
                (1, item.target_wh[1] // VAE_RESIZE_FACTOR, item.target_wh[0] // VAE_RESIZE_FACTOR),
                f"mask shape wrong for {item.pathname}",
            )

            expected = self._reconstruct_mask(item)
            actual = example["mask"].numpy()[0]
            np.testing.assert_array_equal(
                actual, expected,
                err_msg=f"mask mismatch for {item.pathname} (target {item.target_wh})",
            )

            if self.rec.flipped:
                flipped_covered = True
            if os.environ.get("ED_MASK_DEBUG") == "1":
                self._debug_dump(example, item, expected, i)

        self.assertTrue(
            flipped_covered,
            "flip path never exercised - set flip_p=1.0 in the test data dir",
        )

    def test_wrong_size_mask_is_rejected(self):
        _, edb = self._build_dataloader(self.wrongsize_dir)
        example = edb[0]
        self.assertIsNone(example["mask"], "wrong-size mask should have been dropped")


if __name__ == '__main__':
    unittest.main()
