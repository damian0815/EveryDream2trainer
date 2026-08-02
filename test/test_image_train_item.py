import unittest
import os
import pathlib
import random
import PIL.Image as Image

from data.image_train_item import ImageCaption, ImageTrainItem
from utils.fs_helpers import is_image
import data.aspects as aspects

DATA_PATH = pathlib.Path('./test/data')

class TestImageCaption(unittest.TestCase):
    
    def setUp(self) -> None:
        with open(DATA_PATH / "test1.txt", encoding='utf-8', mode='w') as f:
            f.write("caption for test1")
            
        Image.new("RGB", (512,512)).save(DATA_PATH / "test1.jpg")
        Image.new("RGB", (512,512)).save(DATA_PATH / "test2.jpg")
        
        with open(DATA_PATH / "test_caption.caption", encoding='utf-8', mode='w') as f:
            f.write("caption for test2")
            
        return super().setUp()
    
    def tearDown(self) -> None:
        for file in DATA_PATH.glob("test*"):
            file.unlink()

        return super().tearDown()
    
    def test_constructor(self):
        caption = ImageCaption("hello world", 1.0, ["one", "two", "three"], [1.0]*3, 2048, False)
        self.assertEqual(caption.get_caption(), "hello world, one, two, three")
        
        caption = ImageCaption("hello world", 1.0, [], [], 2048, False)
        self.assertEqual(caption.get_caption(), "hello world")

class TestImageTrainItemConstructor(unittest.TestCase):
    
    def tearDown(self) -> None:
        for file in DATA_PATH.glob("img_*"):
            file.unlink()

        return super().tearDown()

    @staticmethod 
    def image_with_size(width, height):
        filename = DATA_PATH / "img_{}x{}.jpg".format(width, height)
        Image.new("RGB", (width, height)).save(filename)
        caption = ImageCaption("hello world", 1.0, [], [], 2048, False)
        return ImageTrainItem(None, caption, aspects.ASPECTS_512, filename, 0.0, 1.0, False, False, 0)
           
    def test_target_size_computation(self):
        # Square images
        image = self.image_with_size(30, 30)
        self.assertEqual(image.target_wh, [512,512])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (30,30))

        image = self.image_with_size(512, 512)
        self.assertEqual(image.target_wh, [512,512])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (512,512))

        image = self.image_with_size(580, 580)
        self.assertEqual(image.target_wh, [512,512])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (580,580))

        # Landscape images
        image = self.image_with_size(64, 38)
        self.assertEqual(image.target_wh, [640,384])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (64,38))

        image = self.image_with_size(640, 384)
        self.assertEqual(image.target_wh, [640,384])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (640,384))

        image = self.image_with_size(704, 422)
        self.assertEqual(image.target_wh, [640,384])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (704,422))

        # Portrait images
        image = self.image_with_size(38, 64)
        self.assertEqual(image.target_wh, [384,640])
        self.assertTrue(image.is_undersized)
        self.assertEqual(image.image_size, (38,64))

        image = self.image_with_size(384, 640)
        self.assertEqual(image.target_wh, [384,640])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (384,640))

        image = self.image_with_size(422, 704)
        self.assertEqual(image.target_wh, [384,640])
        self.assertFalse(image.is_undersized)
        self.assertEqual(image.image_size, (422,704))


class TestIsImageDpoBad(unittest.TestCase):
    def test_regular_image_is_image(self):
        self.assertTrue(is_image("photo.jpg"))
        self.assertTrue(is_image("photo.png"))
        self.assertTrue(is_image("photo.webp"))

    def test_dpobad_excluded(self):
        self.assertFalse(is_image("photo.dpobad.jpg"))
        self.assertFalse(is_image("photo.dpobad.png"))
        self.assertFalse(is_image("photo.dpobad.webp"))

    def test_mask_still_excluded(self):
        self.assertFalse(is_image("photo.mask.jpg"))
        self.assertFalse(is_image("photo.mask.png"))


class TestDpoSpatialAlignment(unittest.TestCase):
    def setUp(self):
        self.data_dir = DATA_PATH / "dpo_test"
        self.data_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for file in self.data_dir.rglob("*"):
            if file.is_file():
                file.unlink()
        self.data_dir.rmdir()

    def test_hydrate_dpo_bad_applies_same_transforms(self):
        good_path = self.data_dir / "photo.jpg"
        bad_path = self.data_dir / "photo.dpobad.jpg"

        Image.new("RGB", (800, 600), color=(255, 0, 0)).save(good_path)
        Image.new("RGB", (800, 600), color=(0, 0, 255)).save(bad_path)

        caption = ImageCaption("test caption", 1.0, [], [], 2048, False)
        item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(good_path), 0.5, 1.0, False, False, 0)

        rng = random.Random(42)
        item.hydrate(crop_jitter=0.02, load_dpo_bad=True, rng=rng)

        self.assertIsNotNone(item.dpo_bad_image)
        self.assertEqual(item.image.shape, item.dpo_bad_image.shape)
        self.assertEqual(item.image.dtype, item.dpo_bad_image.dtype)

    def test_hydrate_no_dpobad_file(self):
        good_path = self.data_dir / "photo.jpg"
        Image.new("RGB", (800, 600)).save(good_path)

        caption = ImageCaption("test caption", 1.0, [], [], 2048, False)
        item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(good_path), 0.0, 1.0, False, False, 0)

        rng = random.Random(42)
        item.hydrate(crop_jitter=0.02, load_dpo_bad=True, rng=rng)

        self.assertIsNone(item.dpo_bad_image)

    def test_hydrate_dpo_disabled(self):
        good_path = self.data_dir / "photo.jpg"
        bad_path = self.data_dir / "photo.dpobad.jpg"

        Image.new("RGB", (800, 600)).save(good_path)
        Image.new("RGB", (800, 600)).save(bad_path)

        caption = ImageCaption("test caption", 1.0, [], [], 2048, False)
        item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(good_path), 0.0, 1.0, False, False, 0)

        rng = random.Random(42)
        item.hydrate(crop_jitter=0.02, load_dpo_bad=False, rng=rng)

        self.assertIsNone(item.dpo_bad_image)

    def test_trim_precompute_matches_original(self):
        good_path = self.data_dir / "photo.jpg"
        Image.new("RGB", (800, 600)).save(good_path)

        caption = ImageCaption("test caption", 1.0, [], [], 2048, False)
        item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(good_path), 0.0, 1.0, False, False, 0)

        img = item.load_image()
        target_wh = item.target_wh
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result_old, offsets_old = item._trim_to_aspect(img.copy(), target_wh, rng=rng1)
        offsets_new = item._get_trim_offsets(img, target_wh, rng=rng2)
        result_new, _ = self._apply_trim_standalone(img.copy(), target_wh, offsets_new)

        self.assertEqual(offsets_old, offsets_new)
        self.assertEqual(result_old.size, result_new.size)

    @staticmethod
    def _apply_trim_standalone(image, target_wh, trim_offsets):
        width, height = image.size
        target_aspect = target_wh[0] / target_wh[1]
        image_aspect = width / height
        l, t = trim_offsets
        if image_aspect > target_aspect:
            target_width = int(height * target_aspect)
            overwidth = width - target_width
            r = width - overwidth + l
            return image.crop((l, 0, r, height)), (l, 0)
        elif target_aspect > image_aspect:
            target_height = int(width / target_aspect)
            overheight = height - target_height
            b = height - overheight + t
            return image.crop((0, t, width, b)), (0, t)
        else:
            return image, (0, 0)


class TestRotationAugmentation(unittest.TestCase):

    def test_compute_max_rotation_angle_unconstrained(self):
        angle = ImageTrainItem._compute_max_rotation_angle(2000, 2000, 1024, 1024, 30.0)
        self.assertAlmostEqual(angle, 30.0, places=2)

    def test_compute_max_rotation_angle_constrained(self):
        angle = ImageTrainItem._compute_max_rotation_angle(1280, 1280, 1024, 1024, 30.0)
        self.assertAlmostEqual(angle, 17.11, places=1)
        self.assertGreater(angle, 0)
        self.assertLess(angle, 30.0)

    def test_compute_max_rotation_angle_too_small(self):
        angle = ImageTrainItem._compute_max_rotation_angle(1024, 1024, 1024, 1024, 30.0)
        self.assertEqual(angle, 0.0)

    def test_compute_max_rotation_angle_crop_larger_than_image(self):
        angle = ImageTrainItem._compute_max_rotation_angle(512, 512, 1024, 1024, 30.0)
        self.assertEqual(angle, 0.0)

    def test_apply_rotation_preserves_size(self):
        img = Image.new("RGB", (800, 600))
        rotated = ImageTrainItem._apply_rotation(img, 15.0)
        self.assertEqual(rotated.size, img.size)

    def test_hydrate_with_rotation(self):
        filename = DATA_PATH / "img_rotation_test.jpg"
        Image.new("RGB", (2000, 2000)).save(filename)
        caption = ImageCaption("test", 1.0, [], [], 2048, False)
        item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(filename), 0.0, 1.0, False, False, 0)

        rng = random.Random(42)
        item.hydrate(crop_jitter=0.02, rotation_degrees=30.0, rng=rng)

        self.assertEqual(item.image.shape[1], item.target_wh[0])
        self.assertEqual(item.image.shape[0], item.target_wh[1])

    def test_hydrate_rotation_disabled_by_default(self):
        filename = DATA_PATH / "img_rotation_nodefault.jpg"
        Image.new("RGB", (2000, 2000)).save(filename)
        caption = ImageCaption("test", 1.0, [], [], 2048, False)
        item1 = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(filename), 0.0, 1.0, False, False, 0)
        item2 = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(filename), 0.0, 1.0, False, False, 0)

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        item1.hydrate(crop_jitter=0.0, rotation_degrees=0.0, rng=rng1)
        item2.hydrate(crop_jitter=0.0, rng=rng2)

        self.assertTrue((item1.image == item2.image).all())

    def test_hydrate_rotation_applied_to_mask_and_dpobad(self):
        data_dir = DATA_PATH / "rotation_dpo_test"
        data_dir.mkdir(exist_ok=True)
        try:
            good_path = data_dir / "photo.jpg"
            bad_path = data_dir / "photo.dpobad.jpg"

            Image.new("RGB", (2000, 2000), color=(255, 0, 0)).save(good_path)
            Image.new("RGB", (2000, 2000), color=(0, 0, 255)).save(bad_path)

            caption = ImageCaption("test", 1.0, [], [], 2048, False)
            item = ImageTrainItem(None, caption, aspects.ASPECTS_512, str(good_path), 0.5, 1.0, False, False, 0)

            rng = random.Random(42)
            item.hydrate(crop_jitter=0.02, load_dpo_bad=True, rotation_degrees=30.0, rng=rng)

            self.assertEqual(item.image.shape, item.dpo_bad_image.shape)
        finally:
            for f in data_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
            data_dir.rmdir()
