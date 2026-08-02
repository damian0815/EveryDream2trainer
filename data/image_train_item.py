"""
Copyright [2022] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import bisect
import json
import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass

import math
import os
import random
import typing
import yaml

import PIL
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

OptionalImageCaption = typing.Optional['ImageCaption']

def check_caption_json(caption_str: str):
    if '<<json>>' in caption_str:
        try:
            captions = json.loads(caption_str.replace("<<json>>", ""))
        except Exception as e:
            logging.error(f"caught {e} loading caption from {caption_str}")
            raise


class ImageCaption:
    """
    Represents the various parts of an image caption
    """
    def __init__(self, main_prompt: str, rating: float, tags: list[str], tag_weights: list[float], max_target_length: int, use_weights: bool):
        """
        :param main_prompt: The part of the caption which should always be included
        :param tags: list of tags to pick from to fill the caption
        :param tag_weights: weights to indicate which tags are more desired and should be picked preferably
        :param max_target_length: The desired maximum length of a generated caption
        :param use_weights: if ture, weights are considered when shuffling tags
        """
        self.__main_prompt = main_prompt
        self.__rating = rating
        self.__tags = tags
        self.__tag_weights = tag_weights
        self.__max_target_length = max_target_length or 2048
        self.__use_weights = use_weights
        if use_weights and len(tags) > len(tag_weights):
            self.__tag_weights.extend([1.0] * (len(tags) - len(tag_weights)))

        if use_weights and len(tag_weights) > len(tags):
            self.__tag_weights = tag_weights[:len(tags)]

        #check_caption_json(", ".join([self.__main_prompt] + self.__tags))

    def rating(self) -> float:
        return self.__rating

    def get_shuffled_caption(self, seed: int, keep_tags: int) -> str:
        """
        returns the caption a string with a random selection of the tags in random order
        :param seed used to initialize the randomizer
        :return: generated caption string
        """
        if self.__tags:
            try:
                max_target_tag_length = self.__max_target_length - len(self.__main_prompt or 0)
            except Exception as e:
                print()
                logging.error(f"Error determining length for: {e} on {self.__main_prompt}")
                print()
                max_target_tag_length = 2048

            if self.__use_weights:
                tags_caption = self.__get_weighted_shuffled_tags(seed, self.__tags, self.__tag_weights, max_target_tag_length)
            else:
                tags_caption = self.__get_shuffled_tags(seed, self.__tags, keep_tags)

            return self.__main_prompt + ", " + tags_caption
        return self.__main_prompt

    def get_caption(self) -> str:
        if self.__tags:
            return self.__main_prompt + ", " + ", ".join(self.__tags)
        return self.__main_prompt

    @staticmethod
    def __get_weighted_shuffled_tags(seed: int, tags: list[str], weights: list[float], max_target_tag_length: int) -> str:
        picker = random.Random(seed)
        tags_copy = tags.copy()
        weights_copy = weights.copy()

        caption = ""
        while len(tags_copy) != 0 and len(caption) < max_target_tag_length:
            cum_weights = []
            weight_sum = 0.0
            for weight in weights_copy:
                weight_sum += weight
                cum_weights.append(weight_sum)

            point = picker.uniform(0, weight_sum)
            pos = bisect.bisect_left(cum_weights, point)

            weights_copy.pop(pos)
            tag = tags_copy.pop(pos)

            if caption:
                caption += ", "
            caption += tag

        return caption

    @staticmethod
    def __get_shuffled_tags(seed: int, tags: list[str], keep_tags: int) -> str:
        tags = tags.copy()
        keep_tags = min(keep_tags, 0)

        if len(tags) > keep_tags:
            fixed_tags = tags[:keep_tags]
            rest = tags[keep_tags:]
            random.Random(seed).shuffle(rest)
            tags = fixed_tags + rest

        return ", ".join(tags)

@dataclass
class ResolutionOption:
    """Candidate resolution assignment for one (image, resolution) pair."""
    resolution: int                 # e.g. 512 or 1024
    target_wh: list                 # selected aspect-ratio bucket, e.g. [512, 768]
    unnormalised_weight: float      # from per_resolution_multiply, or 1.0 if absent
    is_feasible: bool               # True if image is large enough for this resolution


class ImageSourceItem:
    """
    Wraps one ImageTrainItem (the 'base item') together with per-resolution options.

    Resolution assignment is deferred to shuffle time.  Call make_resolved_item(r) to
    mutate the base item for resolution r and receive it as a resolved ImageTrainItem
    ready for training.

    !! MUTATION WARNING !!
    make_resolved_item() mutates self.item IN PLACE and returns it directly (not a
    copy).  This is intentional — it avoids heap allocation — but it means:

      1. The returned reference IS self.item.  A subsequent call to make_resolved_item
         will overwrite self.item's resolution fields AND be visible through any
         reference previously returned.

      2. The same ImageSourceItem must NOT be passed to make_resolved_item more than
         once per epoch without deep-copying the source first.  assign_resolutions()
         in resolution_sampler.py enforces this for multiplier > 1 cases.
    """

    def __init__(self, item: 'ImageTrainItem', resolution_options: dict, uid: str):
        self.item = item
        self.resolution_options: dict[int, ResolutionOption] = resolution_options
        # Stable source-level identifier used as dict key in multiplier dicts.
        # Never reassigned after construction, unlike self.item.uid which changes
        # each time make_resolved_item() is called.
        self.uid = uid

    # ------------------------------------------------------------------
    # Attribute delegation — allows DifficultyEstimator, DataLoaderMultiAspect
    # and other callers to read/write item fields without knowing about the wrapper.
    # ------------------------------------------------------------------

    @property
    def multiplier(self): return self.item.multiplier
    @multiplier.setter
    def multiplier(self, v): self.item.multiplier = v

    @property
    def base_multiplier(self): return self.item.base_multiplier
    @base_multiplier.setter
    def base_multiplier(self, v): self.item.base_multiplier = v

    @property
    def caption(self): return self.item.caption
    @property
    def pathname(self): return self.item.pathname
    @property
    def error(self): return self.item.error
    @property
    def image_size(self): return self.item.image_size
    @property
    def batch_id(self): return self.item.batch_id
    @property
    def cond_dropout(self): return self.item.cond_dropout
    @cond_dropout.setter
    def cond_dropout(self, v): self.item.cond_dropout = v
    @property
    def shuffle_tags(self): return self.item.shuffle_tags
    @property
    def loss_scale(self): return self.item.loss_scale
    @property
    def timesteps_range(self): return self.item.timesteps_range
    @property
    def largest_valid_frame_count(self): return self.item.largest_valid_frame_count

    def is_feasible_for_any_resolution(self) -> bool:
        """True if at least one resolution is large enough for this image."""
        return any(opt.is_feasible for opt in self.resolution_options.values())

    def make_resolved_item(self, resolution: int) -> 'ImageTrainItem':
        """
        !! MUTATES self.item — see class docstring !!

        Assigns target_wh, is_undersized, uid, and source_resolution on self.item for
        the chosen resolution, then returns self.item directly (NOT a copy).

        Calling this method a second time on the same instance will overwrite the
        fields set by the first call, including on any reference previously returned.
        deep-copy the ImageSourceItem first if you need multiple resolved variants.
        """
        opt = self.resolution_options[resolution]
        self.item.target_wh         = opt.target_wh
        self.item.is_undersized     = not opt.is_feasible
        self.item.uid               = uuid.uuid4().hex   # fresh uid per resolution assignment
        self.item.source_resolution = resolution
        return self.item


class ImageTrainItem:
    """
    image: PIL.Image
    identifier: caption,
    target_aspect: (width, height),
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    rating: the relative rating of the images. The rating is measured in comparison to the other images.
    """
    def __init__(self,
                 image: PIL.Image, 
                 caption: ImageCaption, 
                 aspects: list[float], 
                 pathname: str, 
                 flip_p=0.0, 
                 multiplier: float=1.0,
                 cond_dropout=None,
                 shuffle_tags=False,
                 batch_id: str=None,
                 loss_scale: float=None,
                 timesteps_range: tuple[int]=None
                 ):
        self.caption = caption
        self.aspects = aspects
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.runt_size = 0
        self.multiplier = multiplier
        self.base_multiplier = multiplier  # preserved so difficulty schedulers can scale relative to user intent
        self.cond_dropout = cond_dropout
        self.shuffle_tags = shuffle_tags
        self.batch_id = batch_id or DEFAULT_BATCH_ID
        self.loss_scale = 1 if loss_scale is None else loss_scale
        self.timesteps_range = timesteps_range
        self.target_wh = None
        self.is_runt = False
        self.uid = uuid.uuid4().hex
        # Set by ImageSourceItem.make_resolved_item(); None for items created
        # directly via the old ImageTrainItem constructor path.
        self.source_resolution: int = None

        self.image_size = None
        if image is None or len(image) == 0:
            self.image = []
        else:
            self.image = image
            self.image_size = image.size
        self.mask = None

        self.is_undersized = False
        self.error = None
        self.largest_valid_frame_count = 0
        self.__compute_target_width_height()

    @property
    def pathname_mask(self):
        for extension in [".png", ".jpg", ".jpeg", ".bmp", ".jfif", ".webp"]:
            candidate = self.pathname + f".mask{extension}"
            if os.path.exists(candidate):
                return candidate
        return None

    @property
    def pathname_dpobad(self):
        base, _ = os.path.splitext(self.pathname)
        for extension in [".png", ".jpg", ".jpeg", ".bmp", ".jfif", ".webp"]:
            candidate = base + f".dpobad{extension}"
            if os.path.exists(candidate):
                return candidate
        return None

    def load_image(self) -> PIL.Image:
        try:
            image = PIL.Image.open(self.pathname).convert('RGB')
            image = self._try_transpose(image, print_error=False)
        except SyntaxError as e:
            pass
        except OSError as e:
            logging.error(f"fatal error loading image {self.pathname}: {e}")
            raise e
        return image

    def load_mask(self) -> PIL.Image:
        if self.pathname_mask is None:
            return None
        try:
            mask = PIL.Image.open(self.pathname_mask).convert('L')
            mask = self._try_transpose(mask, print_error=False)
        except OSError as e:
            logging.error(f"fatal error loading mask {self.pathname}: {e}")
            raise e
        except SyntaxError as e:
            pass
        return mask

    def load_dpobad(self) -> PIL.Image:
        if self.pathname_dpobad is None:
            return None
        try:
            image = PIL.Image.open(self.pathname_dpobad).convert('RGB')
            image = self._try_transpose(image, print_error=False)
        except OSError as e:
            logging.error(f"fatal error loading dpobad image {self.pathname_dpobad}: {e}")
            raise e
        except SyntaxError:
            pass
        return image

    
    def _try_transpose(self, image, print_error=False):
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            logging.warning(F"Error rotating image: {e} on {self.pathname}, image will be loaded as is, EXIF may be corrupt") if print_error else None
            pass
        return image

    def _get_random_jitter_amounts(self, image, crop_jitter=0.02, rng=None):
        """
        randomly crops the image by a percentage of the image size on each of the four sides
        """
        width, height = image.size
        max_crop_pixels = int(min(512, width, height) * crop_jitter)

        _rng = rng if rng is not None else random

        left_crop_pixels = int(round(_rng.uniform(0, max_crop_pixels)))
        right_crop_pixels = int(round(_rng.uniform(0, max_crop_pixels)))
        top_crop_pixels = int(round(_rng.uniform(0, max_crop_pixels)))
        bottom_crop_pixels = int(round(_rng.uniform(0, max_crop_pixels)))

        return left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels

    def _apply_crop_jitter(self, image, crop_jitter=0.02, precomputed_jitter: tuple[int, int, int, int]=None, rng=None):
        """
        crops the image by a percentage of the image size on each of the four sides.
        """
        width, height = image.size
        if precomputed_jitter is not None:
            left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels = precomputed_jitter
        else:
            left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels = self._get_random_jitter_amounts(image, crop_jitter=crop_jitter, rng=rng)

        # print(f"{left_crop_pixels}, {right_crop_pixels}, {top_crop_pixels}, {bottom_crop_pixels}, ")

        left = left_crop_pixels
        right = width - right_crop_pixels
        top = top_crop_pixels
        bottom = height - bottom_crop_pixels

        cropped = image.crop((left, top, right, bottom))

        return cropped
    
    def _debug_save_image(self, image, folder=""):
        base_name = os.path.basename(self.pathname)
        target_dir = os.path.join('test/output', folder)
        target_file = os.path.join(target_dir, base_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        try:
            #print(f"saving to test/output: {os.path.join('test/output', folder, base_name)}")
            image.save(target_file)
        except Exception as e:
            print(f"error for debug saving image of {self.pathname}: {e}")
            pass

    def _get_trim_offsets(self, image, target_wh, rng=None):
        """Precompute trim offsets without modifying the image."""
        try:
            width, height = image.size
            target_aspect = target_wh[0] / target_wh[1]
            image_aspect = width / height
            _rng = rng if rng is not None else random

            if image_aspect > target_aspect:
                target_width = int(height * target_aspect)
                overwidth = width - target_width
                l = _rng.triangular(0, overwidth)
                l = max(0, int(min(l, overwidth)))
                return (l, 0)
            elif target_aspect > image_aspect:
                target_height = int(width / target_aspect)
                overheight = height - target_height
                t = _rng.triangular(0, overheight)
                t = max(0, int(min(t, overheight)))
                return (0, t)
            else:
                return (0, 0)
        except Exception as e:
            logging.error(f"fatal error computing trim offsets for {self.pathname}: {e}")
            raise e

    def _apply_trim(self, image, target_wh, trim_offsets):
        """Apply precomputed trim offsets to an image."""
        try:
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
        except Exception as e:
            logging.error(f"fatal error applying trim for {self.pathname}: {e}")
            raise e

    def _trim_to_aspect(self, image, target_wh, rng=None) -> tuple[PIL.Image, tuple[int, int]]:
        offsets = self._get_trim_offsets(image, target_wh, rng)
        return self._apply_trim(image, target_wh, offsets)



    @staticmethod
    def _apply_rotation(image: PIL.Image.Image, angle_degrees: float) -> PIL.Image.Image:
        """
        Rotate a PIL image by `angle_degrees` (positive = counterclockwise)
        around its center, using fill color (0, 0, 0) for exposed regions.
        """
        return image.rotate(angle_degrees, resample=PIL.Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    def hydrate(self, save=False, crop_jitter=0.02, load_mask=False, invert_mask=False, return_crop_info=False, rng=None,
                load_dpo_bad=False, rotation_degrees=0.0,
                ) -> typing.Union['ImageTrainItem',
                                  tuple['ImageTrainItem', tuple[int, int, int, int]]]:
        try:
            image = self.load_image()
        except Exception as e:
            err = f"Unable to load image for {self.pathname}: {e}"
            logging.error(err)
            print(err)
            image = None
        mask = self.load_mask() if load_mask else None
        dpo_bad_image = self.load_dpobad() if load_dpo_bad else None

        if image is None:
            uncropped_width, uncropped_height = self.target_wh
            crop_topleft = (0, 0)
        else:
            if image is not None and rotation_degrees > 0:
                width, height = image.size
                max_angle = _compute_max_rotation_angle(
                    width, height,
                    self.target_wh[0], self.target_wh[1],
                    rotation_degrees,
                )
                if max_angle > 0:
                    _rng = rng if rng is not None else random
                    angle = _rng.uniform(-max_angle, max_angle)
                    image = self._apply_rotation(image, angle)
                    if mask is not None:
                        mask = self._apply_rotation(mask, angle)
                    if dpo_bad_image is not None:
                        dpo_bad_image = self._apply_rotation(dpo_bad_image, angle)

            def _center_crop(img, box):
                bw, bh = box
                w, h = img.size
                left, top = (w - bw) // 2, (h - bh) // 2
                return img.crop((left, top, left + bw, top + bh))

            if image is not None and rotation_degrees > 0:
                width, height = image.size
                max_angle, safe_box = plan_rotation_and_crop(
                    width, height, self.target_wh[0], self.target_wh[1],
                    rotation_degrees, min_box_scale=1.0,
                )
                if max_angle > 0:
                    _rng = rng if rng is not None else random
                    angle = _rng.uniform(-max_angle, max_angle)
                    image = self._apply_rotation(image, angle)
                    image = _center_crop(image, safe_box)
                    if mask is not None:
                        mask = self._apply_rotation(mask, angle)
                        mask = _center_crop(mask, safe_box)
                    if dpo_bad_image is not None:
                        dpo_bad_image = self._apply_rotation(dpo_bad_image, angle)
                        dpo_bad_image = _center_crop(dpo_bad_image, safe_box)

            width, height = image.size
            if mask is not None:
                if mask.size[0] != width or mask.size[1] != height:
                    logging.error(f"found a mask at {self.pathname_mask} but it was the wrong size (image size {image.size}, mask size {mask.size}) - ignoring mask")
                    mask = None

            img_jitter = min((width-self.target_wh[0])/self.target_wh[0], (height-self.target_wh[1])/self.target_wh[1])
            img_jitter = min(img_jitter, crop_jitter)
            img_jitter = max(img_jitter, 0.0)

            uncropped_width, uncropped_height = image.size
            if img_jitter > 0.0:
                jitter_amounts = self._get_random_jitter_amounts(image, img_jitter, rng=rng)
                image = self._apply_crop_jitter(image, precomputed_jitter=jitter_amounts)
                if mask is not None:
                    mask = self._apply_crop_jitter(mask, precomputed_jitter=jitter_amounts)
                if dpo_bad_image is not None:
                    dpo_bad_image = self._apply_crop_jitter(dpo_bad_image, precomputed_jitter=jitter_amounts)
            else:
                jitter_amounts = (0, 0, 0, 0)
            crop_topleft = (jitter_amounts[0], jitter_amounts[2])

            trim_offsets = self._get_trim_offsets(image, self.target_wh, rng=rng)
            image, trim_crop_offset = self._apply_trim(image, self.target_wh, trim_offsets)
            if mask is not None:
                mask, _ = self._apply_trim(mask, self.target_wh, trim_offsets)
            if dpo_bad_image is not None:
                dpo_bad_image, _ = self._apply_trim(dpo_bad_image, self.target_wh, trim_offsets)
            crop_topleft = (crop_topleft[0] + trim_crop_offset[0], crop_topleft[1] + trim_crop_offset[1])
            cropped_width = image.size[0]

            # resize
            image = image.resize(self.target_wh)
            if mask:
                mask = mask.resize((self.target_wh[0]//8, self.target_wh[1]//8))
            if dpo_bad_image is not None:
                dpo_bad_image = dpo_bad_image.resize(self.target_wh)

            # sdxl: add_time_embeds crop tracking
            resized_cropped_width = image.size[0]
            image_scale_factor = resized_cropped_width / cropped_width
            # apply scale factor to crop_topleft to put it in the resized image space
            crop_topleft = (crop_topleft[0] * image_scale_factor, crop_topleft[1] * image_scale_factor)
            # if original image is larger than resize, discard the original size and just use the resized size
            if max(math.log(image.size[1] / uncropped_height), math.log(image.size[0] / uncropped_width)) < -0.1:
                uncropped_width, uncropped_height = uncropped_width * image_scale_factor, uncropped_height * image_scale_factor

            _rng = rng if rng is not None else random
            if _rng.random() < self.flip.p:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)
                if dpo_bad_image is not None:
                    dpo_bad_image = TF.hflip(dpo_bad_image)

            if save:
                self._debug_save_image(image, "final")

            image = np.array(image).astype(np.uint8)
            if dpo_bad_image is not None:
                dpo_bad_image = np.array(dpo_bad_image).astype(np.uint8)

        self.image = image
        self.mask = mask
        self.dpo_bad_image = dpo_bad_image

        if self.mask is not None:
            self.mask = np.array(self.mask.convert('L')).astype(np.float32) / 255
            if invert_mask:
                self.mask = 1 - self.mask
            if np.count_nonzero(self.mask) == 0:
                logging.warning(f"mask for {self.pathname} has no non-black pixels - setting to None")
                self.mask = None

        if return_crop_info:
            return self, (crop_topleft[0], crop_topleft[1], uncropped_width, uncropped_height)
        else:
            return self

    def __compute_target_width_height(self):
        self.target_wh = None
        try:
            # check if image can be opened
            with PIL.Image.open(self.pathname) as image:
                if _needs_transpose(image):
                    height, width = image.size
                else:
                    width, height = image.size

                image_aspect = width / height
                target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

                self.is_undersized = (width != target_wh[0] and height != target_wh[1]) and (width * height) < (target_wh[0]*1.02 * target_wh[1]*1.02)

                self.target_wh = target_wh
                self.image_size = image.size
        except Exception as e:
            self.error = e

    @staticmethod
    def __autocrop(image: PIL.Image, q=.404):
        """
        crops image to a random square inside small axis using a truncated gaussian distribution across the long axis
        """
        x, y = image.size

        if x != y:
            if (x > y):
                rand_x = x - y
                sigma = max(rand_x * q, 1)
            else:
                rand_y = y - x
                sigma = max(rand_y * q, 1)

            if (x > y):
                x_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = min(x_crop_gauss, (x - y) / 2)
                x_crop = math.trunc(x_crop)
                y_crop = 0
            else:
                y_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = 0
                y_crop = min(y_crop_gauss, (y - x) / 2)
                y_crop = math.trunc(y_crop)

            min_xy = min(x, y)
            image = image.crop((x_crop, y_crop, x_crop + min_xy, y_crop + min_xy))

        return image

    def copy_with_new_uid(self):
        copy = deepcopy(self)
        copy.uid = uuid.uuid4().hex
        return copy


def _needs_transpose(image, print_error=False):
    try:
        exif = image.getexif()
        orientation = exif.get(0x0112)
        """
            https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html#exif_transpose
            method = {
                2: Image.Transpose.FLIP_LEFT_RIGHT,
                3: Image.Transpose.ROTATE_180,
                4: Image.Transpose.FLIP_TOP_BOTTOM,
                5: Image.Transpose.TRANSPOSE,
                6: Image.Transpose.ROTATE_270,
                7: Image.Transpose.TRANSVERSE,
                8: Image.Transpose.ROTATE_90,
            }.get(orientation)
        """
        return orientation in [5, 6, 7, 8]
    except Exception as e:
        logging.warning(F"Error rotating image: {e} on {self.pathname}, image will be loaded as is, EXIF may be corrupt") if print_error else None
        pass
    return False


def _max_ar_box_height(native_w, native_h, ar, angle_deg):
    """Tallest centered box of aspect ratio ar that survives rotation by angle_deg."""
    t = math.radians(abs(angle_deg))
    c, s = math.cos(t), math.sin(t)
    return min(native_w / (ar * c + s), native_h / (c + ar * s))

def plan_rotation_and_crop(native_w, native_h, target_w, target_h,
                           max_rotation_deg, min_box_scale=1.0, margin_px=2):
    """(theta_max, (box_w, box_h)) -- rotate by any |a| <= theta_max, center-crop
    to box, and you're fill-free. Box keeps the target aspect ratio and is as
    large as the angle allows, floored at min_box_scale * target."""
    ar = target_w / target_h
    floor_h = target_h * min_box_scale
    theta = max(0.0, float(max_rotation_deg))

    # If the requested angle would shrink the clean box below the floor, back off.
    if theta > 0.0 and _max_ar_box_height(native_w, native_h, ar, theta) - 2 * margin_px < floor_h:
        theta = _compute_max_rotation_angle(
            native_w, native_h,
            math.ceil(ar * floor_h) + 2 * margin_px,
            math.ceil(floor_h) + 2 * margin_px,
            max_degrees=theta,
        )

    m = 2 * margin_px if theta > 0.0 else 0
    box_h = max(1, min(int(_max_ar_box_height(native_w, native_h, ar, theta)) - m, native_h))
    box_w = max(1, min(int(box_h * ar), native_w))
    return theta, (box_w, box_h)

def _compute_max_rotation_angle(img_w: int, img_h: int, box_w: int, box_h: int, max_degrees: float) -> float:
    """
    Compute the maximum safe rotation angle (in degrees) such that a
    center axis-aligned crop of `box_w x box_h` after rotating the
    `img_w x img_h` image by that angle contains no border fill.

    The safe-rotation constraints are:
      box_w * cos(θ) + box_h * sin(θ) ≤ img_w
      box_h * cos(θ) + box_w * sin(θ) ≤ img_h

    Each is of the form A·cos(θ) + B·sin(θ) ≤ C, which rewrites as
    R·sin(θ + φ) ≤ C with R = √(A²+B²), φ = atan2(A, B).

    Returns the smaller of the two bound angles, capped at max_degrees.
    If both constraints are satisfied for all θ in [0, max_degrees],
    returns max_degrees.
    """
    if box_w > img_w or box_h > img_h:
        return 0.0

    max_rad = math.radians(max_degrees)
    bounds = []

    for a, b, c in [(box_w, box_h, img_w), (box_h, box_w, img_h)]:
        r = math.sqrt(a * a + b * b)
        if c >= r:
            continue
        phi = math.atan2(a, b)
        bound = math.asin(c / r) - phi
        bounds.append(bound)

    if not bounds:
        return max_degrees

    result = min(bounds)
    result_degrees = math.degrees(result)
    return max(0.0, min(result_degrees, max_degrees))



DEFAULT_BATCH_ID = "default_batch"
