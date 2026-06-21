import logging
import os
import uuid
from copy import deepcopy

import numpy as np


class VideoTrainItem:
    """
    Represents a video file for training, analogous to ImageTrainItem.

    hydrate() loads frames via decord, applies optional spatial crop jitter,
    resizes to target_wh, and stores a numpy array of shape (F, H, W, C)
    with uint8 values [0, 255].
    """
    def __init__(self,
                 pathname: str,
                 caption,
                 target_wh: tuple,
                 video_frames: int = 81,
                 flip_p: float = 0.0,
                 multiplier: float = 1.0,
                 cond_dropout=None,
                 shuffle_tags=False,
                 batch_id: str = "default_batch",
                 loss_scale: float = 1.0,
                 timesteps_range=None):
        self.pathname = pathname
        self.caption = caption
        self.target_wh = target_wh
        self.train_num_frames = video_frames
        self.flip_p = flip_p
        self.multiplier = multiplier
        self.base_multiplier = multiplier
        self.cond_dropout = cond_dropout
        self.shuffle_tags = shuffle_tags
        self.batch_id = batch_id
        self.loss_scale = loss_scale
        self.timesteps_range = timesteps_range
        self.runt_size = 0
        self.uid = uuid.uuid4().hex
        self.source_resolution = None
        self.is_undersized = False
        self.error = None
        self.image_size = None
        self.frames = None

    @property
    def is_video(self) -> bool:
        return True

    @property
    def flip(self):
        class _NoFlip:
            p = 0.0
        return _NoFlip()

    def _load_frames(self, rng) -> np.ndarray:
        """Load raw frames from video at original resolution.

        Returns array of shape (F, H, W, C), uint8, range [0, 255].
        """
        import decord

        vr = decord.VideoReader(self.pathname)
        source_video_frame_count = len(vr)

        if source_video_frame_count == 0:
            raise ValueError(f"Video {self.pathname} has no frames")

        if source_video_frame_count < self.train_num_frames:
            indices = np.linspace(0, source_video_frame_count - 1, self.train_num_frames, dtype=int)
        else:
            start_idx = rng.randint(0, source_video_frame_count - self.train_num_frames)
            indices = np.arange(start_idx, start_idx + self.train_num_frames)

        return vr.get_batch(indices).asnumpy()

    @staticmethod
    def _resize_frames(frames: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize all frames to target_w x target_h."""
        import cv2

        resized = []
        for i in range(frames.shape[0]):
            frame = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        return np.stack(resized, axis=0)

    def _get_random_jitter_amounts(self, frame_w, frame_h, img_jitter, rng=None):
        """Return (left, right, top, bottom) crop pixel amounts."""
        import random as _random_module

        max_crop_pixels = int(min(512, frame_w, frame_h) * img_jitter)
        _rng = rng if rng is not None else _random_module
        left = int(round(_rng.uniform(0, max_crop_pixels)))
        right = int(round(_rng.uniform(0, max_crop_pixels)))
        top = int(round(_rng.uniform(0, max_crop_pixels)))
        bottom = int(round(_rng.uniform(0, max_crop_pixels)))
        return left, right, top, bottom

    def load_video(self, rng) -> np.ndarray:
        """Load and resize video frames.

        Returns array of shape (F, H, W, C), uint8, range [0, 255],
        already resized to target_wh.  This method exists for backward
        compatibility; new code should prefer hydrate() which also
        applies crop jitter.
        """
        frames = self._load_frames(rng)
        return self._resize_frames(frames, *self.target_wh)

    def hydrate(self, save=False, crop_jitter=0.02, load_mask=False, invert_mask=False,
                return_crop_info=False, rng=None):
        import random as _random_module

        _rng = rng if rng is not None else _random_module

        raw_frames = self._load_frames(_rng)
        frame_h, frame_w = raw_frames.shape[1:3]
        target_w, target_h = self.target_wh
        uncropped_w, uncropped_h = frame_w, frame_h

        img_jitter = min(
            (frame_w - target_w) / target_w,
            (frame_h - target_h) / target_h,
            crop_jitter,
        )
        img_jitter = max(img_jitter, 0.0)

        if img_jitter > 0.0:
            left, right, top, bottom = self._get_random_jitter_amounts(
                frame_w, frame_h, img_jitter, rng=_rng,
            )
            raw_frames = raw_frames[:, top:frame_h - bottom, left:frame_w - right, :]
            crop_topleft = (left, top)
        else:
            crop_topleft = (0, 0)

        self.frames = self._resize_frames(raw_frames, target_w, target_h)

        if return_crop_info:
            return self, (crop_topleft[0], crop_topleft[1], uncropped_w, uncropped_h)
        return self

    def copy_with_new_uid(self):
        copy = deepcopy(self)
        copy.uid = uuid.uuid4().hex
        return copy
