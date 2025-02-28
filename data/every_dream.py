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
import logging
import os
import statistics
import traceback
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from data.data_loader import DataLoaderMultiAspect
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID, check_caption_json
import random
from torchvision import transforms
from transformers import CLIPTokenizer
import torch.nn.functional as F

from data.perlin import rand_perlin_2d_octaves
from plugins.plugins import PluginRunner

class EveryDreamBatch(Dataset):
    """
    data_loader: `DataLoaderMultiAspect` object
    debug_level: 0=none, 1=print drops due to unfilled batches on aspect ratio buckets, 2=debug info per image, 3=save crops to disk for inspection
    conditional_dropout: probability of dropping the caption for a given image
    crop_jitter: percent of maximum cropping for crop jitter (ex 0.02 is two percent)
    seed: random seed
    """
    def __init__(self,
                 data_loader: DataLoaderMultiAspect,
                 debug_level=0,
                 conditional_dropout=0.02,
                 crop_jitter=0.02,
                 seed=555,
                 tokenizer=None,
                 shuffle_tags=False,
                 keep_tags=0,
                 plugin_runner:PluginRunner=None,
                 rated_dataset=False,
                 rated_dataset_dropout_target=0.5,
                 name='train',
                 contrastive_learning_batch_ids=None,
                 use_masks=False,
                 contrastive_learning_dropout_p=0,
                 cond_dropout_noise_p=0
                 ):
        self.contrastive_learning_batch_ids = contrastive_learning_batch_ids or []
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        self.tokenizer = tokenizer
        self.max_token_length = self.tokenizer.model_max_length
        self.shuffle_tags = shuffle_tags
        self.keep_tags = keep_tags
        self.plugin_runner = plugin_runner
        self.seed = seed
        self.rated_dataset = rated_dataset
        self.rated_dataset_dropout_target = rated_dataset_dropout_target
        # First epoch always trains on all images
        self.image_train_items = []
        self.__update_image_train_items(1.0)
        self.name = name
        self.use_masks = use_masks
        self.contrastive_learning_dropout_p = contrastive_learning_dropout_p
        self.cond_dropout_noise_p = cond_dropout_noise_p
        self.random_instance = random.Random(seed)

        num_images = len(self.image_train_items)
        logging.info(f" ** Dataset '{name}': {num_images / self.batch_size:.0f} batches, num_images: {num_images}, batch_size: {self.batch_size}")

    def shuffle(self, epoch_n: int, max_epochs: int):
        self.seed += 1

        if self.rated_dataset:
            dropout_fraction = (max_epochs - (epoch_n * self.rated_dataset_dropout_target)) / max_epochs
        else:
            dropout_fraction = 1.0

        self.__update_image_train_items(dropout_fraction)

    def __len__(self):
        return len(self.image_train_items)

    def __getitem__(self, i):
        example = {}

        train_item = self.__get_image_for_trainer(self.image_train_items[i], self.debug_level)

        std_dev = 0.5
        mean = 0.5

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([mean], [std_dev]),
            ]
        )

        if (self.shuffle_tags or train_item["shuffle_tags"]) and not train_item["do_contrastive_learning"]:
            example["caption"] = train_item["caption"].get_shuffled_caption(self.seed, keep_tags=self.keep_tags)
        else:
            example["caption"] = train_item["caption"].get_caption()
        check_caption_json(example["caption"])

        if self.random_instance.random() <= self.contrastive_learning_dropout_p:
            example["do_contrastive_learning"] = False

        example["image"] = self.plugin_runner.run_transform_pil_image(train_item["image"])
        example["image"] = image_transforms(example["image"])
        if train_item["mask"] is not None:
            mask_transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
            example["mask"] = mask_transforms(train_item["mask"])
        else:
            example["mask"] = None
        example["untransformed_caption"] = example["caption"]
        example["caption"] = self.plugin_runner.run_transform_caption(example["caption"])
        if type(example["caption"]) is dict:
            caption_dict = example["caption"]
        else:
            caption_dict = {"default": example["caption"]}

        for k in caption_dict.keys():
            if self.random_instance.random() <= (train_item.get("cond_dropout", self.conditional_dropout)):
                caption_dict[k] = " "

        if self.random_instance.random() <= self.cond_dropout_noise_p:
            #example["image"] = torch.randn_like(example["image"])
            perlin_shape = example["image"].shape[1:]
            perlin3 = torch.stack([rand_perlin_2d_octaves(perlin_shape, (1, 1), 7)
                                   for _ in range(3)]
                                  ).to(example["image"].device, dtype=example["image"].dtype)
            example["image"] = transforms.Normalize(mean=0.5, std=0.5)(perlin3)

        example["caption"] = caption_dict
        try:
            example["tokens"] = {k: torch.tensor(self.tokenizer(caption_dict[k] or '',
                                                truncation=True,
                                                padding="max_length",
                                                max_length=self.tokenizer.model_max_length,
                                              ).input_ids)
                                 for k in caption_dict.keys()}
        except ValueError as e:
            traceback.print_exc()
            print('caption_dict:', caption_dict)
            print('train_item:', train_item)
            raise

        example["runt_size"] = train_item["runt_size"]

        additional_scale_factor = 1
        if train_item["do_contrastive_learning"]:
            additional_scale_factor = 1
        example["loss_scale"] = train_item["loss_scale"] * additional_scale_factor
        example["do_contrastive_learning"] = train_item["do_contrastive_learning"]

        return example

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}
        save = debug_level > 2

        do_contrastive_learning = (image_train_item.batch_id in self.contrastive_learning_batch_ids)
        crop_jitter = (0.0
                       if image_train_item.runt_size > 0 # and do_contrastive_learning
                       else self.crop_jitter)
        image_train_tmp = image_train_item.hydrate(save=save, crop_jitter=crop_jitter, load_mask=self.use_masks)

        example["image"] = image_train_tmp.image.copy() # hack for now to avoid memory leak
        example["mask"] = None if image_train_tmp.mask is None else image_train_tmp.mask.copy() # hack for now to avoid memory leak
        image_train_tmp.image = None # hack for now to avoid memory leak
        image_train_tmp.mask = None # hack for now to avoid memory leak
        example["caption"] = image_train_tmp.caption
        if image_train_tmp.cond_dropout is not None:
            example["cond_dropout"] = image_train_tmp.cond_dropout
        example["runt_size"] = image_train_tmp.runt_size
        example["shuffle_tags"] = image_train_tmp.shuffle_tags
        example["loss_scale"] = image_train_tmp.loss_scale
        example["batch_id"] = image_train_tmp.batch_id
        example["do_contrastive_learning"] = do_contrastive_learning

        return example

    def __update_image_train_items(self, dropout_fraction: float):
        self.image_train_items = self.data_loader.get_shuffled_image_buckets(dropout_fraction)



class DataLoaderWithFixedBuffer(torch.utils.data.DataLoader):
    def __init__(self, dataset, buffer_tensor, batch_size:int, max_pixels: int, buffer_dtype: torch.dtype, device="cuda"):
        color_channels = 3
        buffer_size = batch_size * color_channels * max_pixels
        self.buffer_size = buffer_size

        buffer_tensor = torch.empty(buffer_size, dtype=buffer_dtype, device=device).pin_memory()
        self.buffer_tensor = buffer_tensor
        logging.info(f"buffer_tensor created with shape: {buffer_tensor.shape}")

        super().__init__(dataset, batch_size=batch_size, shuffle=False, num_workers=min(batch_size, os.cpu_count()), collate_fn=self.fixed_collate_fn)

    def fixed_collate_fn(self, batch):
        """
        Collates images to a pinned buffer returned as a view using actual resolution shape
        """
        images = [example["image"] for example in batch]

        # map the image data to the fixed buffer view
        w, h = images[0].size
        for i in range(self.batch_size):
            image = batch["image"][i]
            self.buffer_tensor[i*self.buffer_size//self.batch_size:(i+1)*self.buffer_size//self.batch_size] = image.view(-1)
            images = self.buffer_tensor.view(self.batch_size, 3, w, h)

        captions = [example["caption"] for example in batch]
        tokens = [example["tokens"] for example in batch]
        runt_size = batch[0]["runt_size"]

        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()

        ret = {
            "tokens": torch.stack(tuple(tokens)),
            "image": images,
            "captions": captions,
            "runt_size": runt_size,
        }
        del batch
        return ret

def build_torch_dataloader2(dataset, batch_size, max_pixels) -> torch.utils.data.DataLoader:    
    dataloader = DataLoaderWithFixedBuffer(
        dataset,
        max_pixels=max_pixels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(batch_size, os.cpu_count()),
        collate_fn=collate_fn
    )
    return dataloader

def build_torch_dataloader(dataset, batch_size) -> torch.utils.data.DataLoader:    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=False,
        #num_workers=0,
        num_workers=min(batch_size, os.cpu_count()),
        collate_fn=collate_fn
    )
    return dataloader

def collate_fn(batch):
    """
    Collates batches
    """
    images = [example["image"] for example in batch]
    masks = [example["mask"] for example in batch]
    do_contrastive_learning = all(example["do_contrastive_learning"] for example in batch)

    #captions = [example["untransformed_caption" if do_contrastive_learning else "caption"] for example in batch]
    caption_variants = list(batch[0]["caption"].keys())
    captions = {k: [example["caption"][k] for example in batch]
                for k in caption_variants}
    tokens = {k: torch.stack([example["tokens"][k] for example in batch])
              for k in caption_variants}

    runt_size = batch[0]["runt_size"]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    if any(m is not None for m in masks):
        masks = [m if m is not None
                 else torch.ones([1, images[0].shape[1]//8, images[0].shape[2]//8], dtype=torch.float32).to(images.device)
                 for m in masks]
        masks = torch.stack(masks)
        masks = masks.to(memory_format=torch.contiguous_format).float()
    else:
        masks = None

    loss_scale = torch.tensor([example.get("loss_scale", 1) for example in batch])

    ret = {
        "tokens": tokens,
        "image": images,
        "mask": masks,
        "captions": captions,
        "runt_size": runt_size,
        "loss_scale": loss_scale,
        "do_contrastive_learning": do_contrastive_learning,
    }
    del batch
    return ret


