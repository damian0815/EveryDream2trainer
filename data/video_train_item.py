import itertools
import logging
import math
import os
import uuid
from copy import deepcopy
import av
import cv2

import numpy as np
import torch

_debug_output_path = None
_debug_counter = itertools.count(1)
_debug_dumped = False


def set_debug_video_path(path: str | None):
    global _debug_output_path
    _debug_output_path = path


def _maybe_dump_frames(pathname: str, frames: np.ndarray):
    global _debug_dumped
    if _debug_output_path is None or _debug_dumped:
        return
    info = torch.utils.data.get_worker_info()
    worker_id = info.id if info is not None else 0
    if worker_id != 0:
        return
    n = next(_debug_counter)
    if n != 3:
        return

    dump_dir = os.path.join(_debug_output_path, f"worker_{worker_id}", f"video_{n:03d}")
    os.makedirs(dump_dir, exist_ok=True)
    import cv2
    for i in range(frames.shape[0]):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dump_dir, f"frame_{i:05d}.png"), bgr)
    with open(os.path.join(dump_dir, "info.txt"), "w") as f:
        f.write(f"source: {pathname}\nshape: {list(frames.shape)}\n")
    logging.info(f"Dumped {frames.shape[0]} frames of {pathname} to {dump_dir}")
    _debug_dumped = True


def largest_valid_frame_count(
    source_frames: int,
    avg_fps: float,
    native_fps: float = 16.0,
    max_train_frames: int = 81,
    scale_factor_temporal: int = 4,
) -> int:
    """
    The largest number of training frames this source video can yield
    without duplicating frames (i.e. by repeating the source).

    If the source is long enough, returns max_train_frames.
    Otherwise snaps down to the nearest value of the form
    scale_factor_temporal * k + 1 (matching the temporal VAE's
    constraint).  Returns at least 1.
    """
    needed = round(max_train_frames / native_fps * avg_fps)
    if source_frames >= needed:
        return max_train_frames
    max_k = int(source_frames * native_fps / avg_fps)
    k = (max_k - 1) // scale_factor_temporal
    k = max(k, 1)
    return scale_factor_temporal * k + 1


class VideoTrainItem:
    """
    Represents a video file for training, analogous to ImageTrainItem.

    hydrate() loads frames via PyAV, applies optional spatial crop jitter,
    resizes to target_wh, and stores a numpy array of shape (F, H, W, C)
    with uint8 values [0, 255].
    """
    def __init__(self,
                 pathname: str,
                 caption,
                 target_wh: tuple,
                 video_frames: int = 81,
                 video_model_native_fps: float = 16.0,
                 flip_p: float = 0.0,
                 multiplier: float = 1.0,
                 cond_dropout=None,
                 shuffle_tags=False,
                 batch_id: str = "default_batch",
                  loss_scale: float = 1.0,
                  timesteps_range=None,
                  start_time: float = None,
                  end_time: float = None,
                  largest_valid_frame_count: int = 0):
        self.pathname = pathname
        self.caption = caption
        self.target_wh = target_wh
        self.train_num_frames = largest_valid_frame_count if largest_valid_frame_count > 0 else video_frames
        self.train_native_fps = video_model_native_fps
        self.flip_p = flip_p
        self.multiplier = multiplier
        self.base_multiplier = multiplier
        self.cond_dropout = cond_dropout
        self.shuffle_tags = shuffle_tags
        self.batch_id = batch_id
        self.loss_scale = loss_scale
        self.timesteps_range = timesteps_range
        self.start_time = start_time
        self.end_time = end_time
        self.runt_size = 0
        self.uid = uuid.uuid4().hex
        self.source_resolution = None
        self.is_undersized = False
        self.error = None
        self.image_size = None
        self.frames = None
        self.largest_valid_frame_count = largest_valid_frame_count

    @property
    def is_video(self) -> bool:
        return True

    @property
    def flip(self):
        class _NoFlip:
            p = 0.0
        return _NoFlip()

    def _load_frames(self, rng) -> np.ndarray:
        """Load raw frames from video at original resolution via PyAV.

        Returns array of shape (F, H, W, C), uint8, range [0, 255].
        """
        container = av.open(self.pathname, options={
            'fflags': 'ignidx',
            'err_detect': 'ignore_err',
        })
        stream = container.streams.video[0]
        avg_fps = float(stream.average_rate) if stream.average_rate else 30.0
        fps_inv = 1.0 / max(avg_fps, 1.0)

        logging.info(f"Loading video frames from {self.pathname} - [{self.start_time}-{self.end_time}] @ {avg_fps} fps")

        train_duration_seconds = self.train_num_frames / self.train_native_fps
        decode_deadline = (self.start_time or 0.0) + 2 * train_duration_seconds

        frames = []
        pts_list = []
        for packet in container.demux(stream):
            try:
                decoded = packet.decode()
            except Exception:
                continue
            for frame in decoded:
                pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
                if pts is None and frames:
                    pts = pts_list[-1] + fps_inv
                elif pts is None:
                    pts = 0.0
                if self.start_time is not None and pts < self.start_time:
                    continue
                pts_list.append(pts)
                frames.append(frame.to_ndarray(format='rgb24'))
                if pts >= decode_deadline:
                    break
            else:
                continue
            break

        container.close()

        source_video_frame_count = len(frames)
        if source_video_frame_count == 0:
            raise ValueError(f"Video {self.pathname} has no frames")
        train_duration_frames = round(train_duration_seconds * avg_fps)

        if source_video_frame_count < train_duration_frames:
            train_duration_frames = source_video_frame_count

        pts_all = np.array(pts_list)

        segment_start_frame = 0

        if self.end_time is not None:
            segment_end_frame = int(np.searchsorted(pts_all, self.end_time, side="right"))
        else:
            segment_end_frame = source_video_frame_count

        segment_end_frame = min(segment_end_frame, source_video_frame_count)
        segment_start_frame = min(segment_start_frame, max(0, segment_end_frame - 1))

        if train_duration_frames > (segment_end_frame - segment_start_frame):
            train_duration_frames = max(1, segment_end_frame - segment_start_frame)

        last_possible_start_index = max(segment_start_frame, segment_end_frame - train_duration_frames - 1)
        jitter_start_index = False
        if jitter_start_index:
            start_index = rng.randint(segment_start_frame, max(segment_start_frame, last_possible_start_index + 1))
        else:
            # always start 1s into the clip
            start_index = max(segment_start_frame, min(last_possible_start_index, round(1.0*avg_fps)))

        indices_real = np.linspace(start_index, start_index + train_duration_frames, self.train_num_frames)
        frame_interval = indices_real[1] - indices_real[0]
        time_jitter_pct = 0.2
        time_jitter = [rng.uniform(-time_jitter_pct * frame_interval, time_jitter_pct * frame_interval)
                       for _ in range(self.train_num_frames)]
        indices_real[1:-1] += time_jitter[1:-1]
        indices_int = np.round(indices_real).astype(int)
        indices_int = np.clip(indices_int, 0, source_video_frame_count - 1)

        result = np.stack([frames[i] for i in indices_int], axis=0)
        return result

    @staticmethod
    def _resize_frames(frames: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize all frames to target_w x target_h."""

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

    def _trim_to_aspect_frames(self, frames: np.ndarray, target_w: int, target_h: int, rng=None) -> tuple:
        """Crop frames to match target aspect ratio, preventing stretch on resize.

        Mirrors ImageTrainItem._trim_to_aspect. Operates on (F, H, W, C) arrays.
        Returns (cropped_frames, (crop_x, crop_y)).
        """
        import random as _random_module

        _rng = rng if rng is not None else _random_module
        frame_h, frame_w = frames.shape[1:3]
        target_aspect = target_w / target_h
        frame_aspect = frame_w / frame_h

        if frame_aspect > target_aspect:
            target_width = int(frame_h * target_aspect)
            overwidth = frame_w - target_width
            l = _rng.triangular(0, overwidth)
            l = max(0, l)
            l = int(min(l, overwidth))
            r = frame_w - overwidth + l
            frames = frames[:, :, l:r, :]
            return frames, (l, 0)
        elif target_aspect > frame_aspect:
            target_height = int(frame_w / target_aspect)
            overheight = frame_h - target_height
            t = _rng.triangular(0, overheight)
            t = max(0, t)
            t = int(min(t, overheight))
            b = frame_h - overheight + t
            frames = frames[:, t:b, :, :]
            return frames, (0, t)
        else:
            return frames, (0, 0)

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

        raw_frames, trim_offset = self._trim_to_aspect_frames(raw_frames, target_w, target_h, rng=_rng)
        crop_topleft = (crop_topleft[0] + trim_offset[0], crop_topleft[1] + trim_offset[1])

        self.frames = self._resize_frames(raw_frames, target_w, target_h)

        _maybe_dump_frames(self.pathname, self.frames)

        if return_crop_info:
            return self, (crop_topleft[0], crop_topleft[1], uncropped_w, uncropped_h)
        return self

    def copy_with_new_uid(self):
        copy = deepcopy(self)
        copy.uid = uuid.uuid4().hex
        return copy
