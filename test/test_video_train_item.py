import sys
import unittest
import os
import pathlib
import tempfile
from unittest.mock import MagicMock

import numpy as np

DATA_PATH = pathlib.Path('./test/data')


def _video_libs_available():
    try:
        import decord
        import cv2
        return True
    except ImportError:
        return False


def _touch_file(path):
    with open(path, 'wb') as f:
        f.write(b'')


def _install_mock_modules():
    """Insert mock modules for decord and cv2 when real ones are unavailable.
    Returns a dict of mocked module names for cleanup."""
    mocked = {}
    for mod_name in ('decord', 'cv2'):
        if mod_name not in sys.modules:
            mock = MagicMock()
            sys.modules[mod_name] = mock
            mocked[mod_name] = mock
    return mocked


def _cleanup_mock_modules(mocked):
    for mod_name in mocked:
        if mod_name in sys.modules and isinstance(sys.modules[mod_name], MagicMock):
            del sys.modules[mod_name]


def _make_mock_vr(path):
    """Return a mock decord.VideoReader that yields frames sized per-path."""
    path = str(path)
    if 'big' in path:
        w, h, n_frames = 128, 128, 8
    else:
        w, h, n_frames = 64, 64, 16
    mock_vr = MagicMock()
    mock_vr.__len__.return_value = n_frames
    mock_batch = MagicMock()
    mock_batch.asnumpy.return_value = np.random.randint(
        0, 256, (min(n_frames, 8), h, w, 3), dtype=np.uint8,
    )
    mock_vr.get_batch.return_value = mock_batch
    return mock_vr


class TestVideoTrainItem(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.test_dir, "test_video.mp4")
        _touch_file(self.video_path)
        self._mocked = _install_mock_modules()
        self._configure_mocks()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        _cleanup_mock_modules(self._mocked)

    def _configure_mocks(self):
        if 'cv2' in self._mocked:
            self._mocked['cv2'].resize.side_effect = (
                lambda img, dsize, **kw: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
            )
        if 'decord' in self._mocked:
            self._mocked['decord'].VideoReader.side_effect = _make_mock_vr

    def test_hydrate_shape(self):
        from data.video_train_item import VideoTrainItem
        from data.image_train_item import ImageCaption

        caption = ImageCaption("a test video", 1.0, [], [], 2048, False)
        item = VideoTrainItem(
            pathname=self.video_path,
            caption=caption,
            target_wh=(64, 64),
            video_frames=8,
        )
        item.hydrate()
        frames = item.frames
        self.assertEqual(frames.ndim, 4)
        self.assertEqual(frames.shape[0], 8)
        self.assertEqual(frames.shape[1], 64)
        self.assertEqual(frames.shape[2], 64)
        self.assertEqual(frames.shape[3], 3)

    def test_hydrate_pixel_range(self):
        from data.video_train_item import VideoTrainItem
        from data.image_train_item import ImageCaption

        caption = ImageCaption("a test video", 1.0, [], [], 2048, False)
        item = VideoTrainItem(
            pathname=self.video_path,
            caption=caption,
            target_wh=(64, 64),
            video_frames=8,
        )
        item.hydrate()
        frames = item.frames
        self.assertGreaterEqual(frames.min(), 0)
        self.assertLessEqual(frames.max(), 255)

    def test_is_video_property(self):
        from data.video_train_item import VideoTrainItem
        from data.image_train_item import ImageCaption

        caption = ImageCaption("a test video", 1.0, [], [], 2048, False)
        item = VideoTrainItem(
            pathname=self.video_path,
            caption=caption,
            target_wh=(64, 64),
            video_frames=8,
        )
        self.assertTrue(item.is_video)

    def test_hydrate_crop_jitter_noop_when_small(self):
        """When source == target resolution, crop jitter should be a no-op."""
        from data.video_train_item import VideoTrainItem
        from data.image_train_item import ImageCaption

        caption = ImageCaption("test", 1.0, [], [], 2048, False)
        item = VideoTrainItem(
            pathname=self.video_path,
            caption=caption,
            target_wh=(64, 64),
            video_frames=8,
        )
        result, crop_info = item.hydrate(crop_jitter=0.5, return_crop_info=True)
        self.assertEqual(crop_info, (0, 0, 64, 64))
        self.assertEqual(result.frames.shape, (8, 64, 64, 3))

    def test_hydrate_crop_jitter_larger_source(self):
        """When source > target and crop_jitter > 0, crop should reduce area."""
        from data.video_train_item import VideoTrainItem
        from data.image_train_item import ImageCaption

        big_path = os.path.join(self.test_dir, "big_video.mp4")
        _touch_file(big_path)

        caption = ImageCaption("test", 1.0, [], [], 2048, False)
        item = VideoTrainItem(
            pathname=big_path,
            caption=caption,
            target_wh=(64, 64),
            video_frames=8,
        )
        result, crop_info = item.hydrate(crop_jitter=1.0, return_crop_info=True)
        crop_x, crop_y, uncropped_w, uncropped_h = crop_info
        # source was 128x128, so uncropped should reflect that
        self.assertEqual(uncropped_w, 128)
        self.assertEqual(uncropped_h, 128)
        # crop_topleft should be non-zero because we had spare pixels
        self.assertGreater(crop_x + crop_y, 0)
        # final frames should be 64x64
        self.assertEqual(result.frames.shape, (8, 64, 64, 3))


class TestInjectMotionScore(unittest.TestCase):

    def test_inject_motion_score(self):
        from data.every_dream import _inject_motion_score
        result = _inject_motion_score("A cat baking a cake.", 30)
        self.assertEqual(result, "A cat baking a cake. motion score: 30.")

    def test_inject_motion_score_empty(self):
        from data.every_dream import _inject_motion_score
        result = _inject_motion_score("", 15)
        self.assertEqual(result, " motion score: 15.")

    def test_inject_motion_score_zero(self):
        from data.every_dream import _inject_motion_score
        result = _inject_motion_score("a video.", 0)
        self.assertEqual(result, "a video. motion score: 0.")


if __name__ == "__main__":
    unittest.main()
