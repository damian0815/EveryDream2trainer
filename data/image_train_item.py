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
            json.loads(caption_str.replace("<<json>>", ""))
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
        self.cond_dropout = cond_dropout
        self.shuffle_tags = shuffle_tags
        self.batch_id = batch_id or DEFAULT_BATCH_ID
        self.loss_scale = 1 if loss_scale is None else loss_scale
        self.timesteps_range = timesteps_range
        self.target_wh = None
        self.is_runt = False

        self.image_size = None
        if image is None or len(image) == 0:
            self.image = []
        else:
            self.image = image
            self.image_size = image.size
            #self.target_size = None

        self.is_undersized = False
        self.error = None
        self.__compute_target_width_height()

    @property
    def pathname_mask(self):
        return self.pathname + ".mask.png"

    def load_image(self) -> PIL.Image:
        try:
            image = PIL.Image.open(self.pathname).convert('RGB')
            image = self._try_transpose(image, print_error=False)
        except SyntaxError as e:
            pass
        return image

    def load_mask(self) -> PIL.Image:
        if not os.path.exists(self.pathname_mask):
            return None
        try:
            mask = PIL.Image.open(self.pathname_mask).convert('L')
            mask = self._try_transpose(mask, print_error=False)
        except SyntaxError as e:
            pass
        return mask

    
    def _try_transpose(self, image, print_error=False):
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            logging.warning(F"Error rotating image: {e} on {self.pathname}, image will be loaded as is, EXIF may be corrupt") if print_error else None
            pass
        return image
    
    def _needs_transpose(self, image, print_error=False):
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

    def _get_random_jitter_amounts(self, image, crop_jitter=0.02):
        """
        randomly crops the image by a percentage of the image size on each of the four sides
        """
        width, height = image.size
        max_crop_pixels = int(min(width, height) * crop_jitter)

        left_crop_pixels = int(round(random.uniform(0, max_crop_pixels)))
        right_crop_pixels = int(round(random.uniform(0, max_crop_pixels)))
        top_crop_pixels = int(round(random.uniform(0, max_crop_pixels)))
        bottom_crop_pixels = int(round(random.uniform(0, max_crop_pixels)))

        return left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels

    def _apply_crop_jitter(self, image, crop_jitter=0.02, precomputed_jitter: tuple[int, int, int, int]=None):
        """
        crops the image by a percentage of the image size on each of the four sides.
        """
        width, height = image.size
        if precomputed_jitter is None:
            left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels = precomputed_jitter
        else:
            left_crop_pixels, right_crop_pixels, top_crop_pixels, bottom_crop_pixels = self._get_random_jitter_amounts(image, crop_jitter=crop_jitter)

        # print(f"{left_crop_pixels}, {right_crop_pixels}, {top_crop_pixels}, {bottom_crop_pixels}, ")

        left = left_crop_pixels
        right = width - right_crop_pixels
        top = top_crop_pixels
        bottom = height - bottom_crop_pixels

        crop_size = image.crop((left, top, right, bottom))

        return crop_size
    
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

    def _trim_to_aspect(self, image, target_wh):
        try:
            width, height = image.size
            target_aspect = target_wh[0] / target_wh[1] # 0.60
            image_aspect = width / height # 0.5865
            #self._debug_save_image(image, "precrop")
            if image_aspect > target_aspect:
                target_width = int(height * target_aspect)
                overwidth = width - target_width
                l = random.triangular(0, overwidth)
                #print(f"l: {l}, overwidth: {overwidth}")
                l = max(0, l)
                l = int(min(l, overwidth))
                r = width - overwidth + l
                #print(f"\n_trim_to_aspect actual ar: {image_aspect}, target ar:{target_aspect:.2f}, {image.size}, cropping with box: {l}, 0, {r}, {height}, {self.pathname}")
                image = image.crop((l, 0, r, height))
            elif target_aspect > image_aspect:
                target_height = int(width / target_aspect)
                overheight = height - target_height
                t = random.triangular(0, overheight)
                #print(f"t: {t}, overheight: {overheight}")
                t = max(0, t)
                t = int(min(t, overheight))
                b = height - overheight + t
                #print(f"\n_trim_to_aspect actual ar: {image_aspect}, target ar:{target_aspect:.2f}, {image.size}, cropping with box: 0, {t}, {width}, {b}, {self.pathname}")
                image = image.crop((0, t, width, b))
                
        except Exception as e:
            logging.error(f"fatal error trimming image {self.pathname}: {e}")
            raise e
        return image

    def hydrate(self, save=False, crop_jitter=0.02, load_mask=False) -> 'ImageTrainItem':
        image = self.load_image()
        mask = self.load_mask() if load_mask else None

        width, height = image.size
        if mask is not None:
            if mask.size[0] != width or mask.size[1] != height:
                logging.error(f"found a mask at {self.pathname_mask} but it was the wrong size (image size {image.size}, mask size {mask.size}) - ignoring mask")
                mask = None

        img_jitter = min((width-self.target_wh[0])/self.target_wh[0], (height-self.target_wh[1])/self.target_wh[1])
        img_jitter = min(img_jitter, crop_jitter)
        img_jitter = max(img_jitter, 0.0)
        
        if img_jitter > 0.0:
            jitter_amounts = self._get_random_jitter_amounts(image, img_jitter)
            image = self._apply_crop_jitter(image, precomputed_jitter=jitter_amounts)
            if mask is not None:
                mask = self._apply_crop_jitter(mask, precomputed_jitter=jitter_amounts)

        image = self._trim_to_aspect(image, self.target_wh)
        if mask is not None:
            mask = self._trim_to_aspect(mask, self.target_wh)

        image = image.resize(self.target_wh)
        if mask:
            mask = mask.resize((self.target_wh[0]//8, self.target_wh[1]//8))

        if random.random() < self.flip.p:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        self.image = image
        self.mask = mask

        if save:
            self._debug_save_image(self.image, "final")

        self.image = np.array(self.image).astype(np.uint8)
        if self.mask is not None:
            self.mask = np.array(self.mask.convert('L')).astype(np.float32) / 255
            if np.count_nonzero(self.mask) == 0:
                logging.warning(f"mask for {self.pathname} has no non-black pixels - setting to None")
                self.mask = None

        return self

    def __compute_target_width_height(self):
        self.target_wh = None
        try:
            with PIL.Image.open(self.pathname) as image:
                if self._needs_transpose(image):
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


DEFAULT_BATCH_ID = "default_batch"
