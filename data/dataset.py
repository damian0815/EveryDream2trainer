import logging
import random
import traceback
import uuid

import yaml
import json

from attrs import define, field
from data.image_train_item import (
    ImageCaption, ImageTrainItem, ImageSourceItem, ResolutionOption,
    check_caption_json, _needs_transpose, DEFAULT_BATCH_ID,
)
from data.video_train_item import VideoTrainItem
import PIL.Image
from torchvision import transforms
from utils.fs_helpers import *
from typing import Iterable, Counter

from tqdm import tqdm

DEFAULT_MAX_CAPTION_LENGTH = 2048

def overlay(overlay, base):
    return overlay if overlay is not None else base

def safe_set(val):
    if isinstance(val, str):
        return dict.fromkeys([val]) if val else dict()

    if isinstance(val, Iterable):
        return dict.fromkeys((i for i in val if i is not None))
    
    return val or dict() 

@define(frozen=True)
class Tag:
    value: str
    weight: float = field(default=1.0, converter=lambda x: x if x is not None else 1.0)

    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            return Tag(data)

        if isinstance(data, dict):
            value = str(data.get("tag"))
            weight = data.get("weight")
            if value:
                return Tag(value, weight)

        return None

@define
class ImageConfig:
    # Captions
    main_prompts: dict[str, None] = field(factory=dict, converter=safe_set)
    rating: float = None
    max_caption_length: int = None
    tags: dict[Tag, None] = field(factory=dict, converter=safe_set)
    batch_id: str = None
    
    # Options
    multiply: float = None
    per_resolution_multiply: dict[int, float] = None
    cond_dropout: float = None
    flip_p: float = None
    shuffle_tags: bool = False
    loss_scale: float = None
    timesteps_range: tuple[int, int] = None

    def merge(self, other):
        if other is None:
            return self

        other_multiply = 1.0 if other.multiply is None else other.multiply
        self_multiply = 1.0 if self.multiply is None else self.multiply
        self_per_resolution_multiply = self.per_resolution_multiply or dict()
        other_per_resolution_multiply = other.per_resolution_multiply or dict()
        return ImageConfig(
            main_prompts=other.main_prompts | self.main_prompts,
            rating=overlay(other.rating, self.rating),
            max_caption_length=overlay(other.max_caption_length, self.max_caption_length),
            tags= other.tags | self.tags,
            multiply=other_multiply * self_multiply,
            per_resolution_multiply={**self_per_resolution_multiply, **other_per_resolution_multiply},
            cond_dropout=overlay(other.cond_dropout, self.cond_dropout),
            flip_p=overlay(other.flip_p, self.flip_p),
            shuffle_tags=overlay(other.shuffle_tags, self.shuffle_tags),
            batch_id=overlay(other.batch_id, self.batch_id),
            loss_scale=overlay(other.loss_scale, self.loss_scale),
            timesteps_range=overlay(other.timesteps_range, self.timesteps_range),
        )

    @classmethod
    def from_dict(cls, data: dict):
        if dict is None:
            raise ValueError("Cannot parse ImageConfig from None")
        # Parse standard yaml tag file (with options)
        parsed_cfg = ImageConfig(
            main_prompts=safe_set(data.get("main_prompt")), 
            rating=data.get("rating"), 
            max_caption_length=data.get("max_caption_length"), 
            tags=safe_set(map(Tag.parse, data.get("tags", []))),
            multiply=data.get("multiply"),
            per_resolution_multiply=data.get("per_resolution_multiply"),
            cond_dropout=data.get("cond_dropout"),
            flip_p=data.get("flip_p"),
            shuffle_tags=data.get("shuffle_tags"),
            batch_id=data.get("batch_id"),
            loss_scale=data.get("loss_scale"),
            timesteps_range=data.get("timesteps_range"),
        )

        # Alternatively parse from dedicated `caption` attribute
        if cap_attr := data.get('caption'):
            parsed_cfg = parsed_cfg.merge(ImageConfig.parse(cap_attr))

        return parsed_cfg

    @classmethod
    def fold(cls, configs):
        acc = ImageConfig()
        for cfg in configs:
            acc = acc.merge(cfg)
            
        acc.shuffle_tags = any(cfg.shuffle_tags for cfg in configs)
        #print(f"accum shuffle:{acc.shuffle_tags}")
        return acc

    def ensure_caption(self):
        return self

    @classmethod
    def from_caption_text(cls, text: str):
        if not text:
            return ImageConfig()
        if os.path.isfile(text):
            return ImageConfig.from_file(text)

        if text.startswith("<<json>>"):
            return ImageConfig(main_prompts=text, tags=[])
        else:
            split_caption = list(map(str.strip, text.split(",")))
            main_prompt = ' ' if len(text.strip()) == 0 else split_caption[0]
            return ImageConfig(
                main_prompts=main_prompt,
                tags=map(Tag.parse, split_caption[1:])
                )

    @classmethod    
    def from_file(cls, file: str):
        try:
            match ext(file):
                case '.jpg' | '.jpeg' | '.png' | '.bmp' | '.webp' | '.jfif':
                    return ImageConfig(image=file)
                case ".json":
                    json_dict = json.load(read_text(file))
                    if json_dict is None:
                        logging.warning(f" *** JSON file {file} is empty, treating as no config")
                        return None
                    return ImageConfig.from_dict(json_dict)
                case ".yaml" | ".yml":
                    yaml_dict = yaml.safe_load(read_text(file))
                    if yaml_dict is None:
                        logging.warning(f" *** YAML file {file} is empty, treating as no config")
                        return None
                    return ImageConfig.from_dict(yaml_dict)
                case ".txt" | ".caption":
                    return ImageConfig.from_caption_text(read_text(file))
                case _:
                    logging.warning(" *** Unrecognized config extension {ext}")
                    return None
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f" *** Error parsing prompt/config file (is it empty?): {file}: {repr(e)}")

    @classmethod
    def parse(cls, input):
        if isinstance(input, str):
            if os.path.isfile(input):
                return ImageConfig.from_file(input)
            else:
                return ImageConfig.from_caption_text(input)
        elif isinstance(input, dict):
            return ImageConfig.from_dict(input)
    

@define()
class Dataset:
    image_configs: dict[str, ImageConfig]

    def __global_cfg(fileset):
        cfgs = []
        
        for cfgfile in ['global.yaml', 'global.yml']:
            if cfgfile in fileset:
                cfgs.append(ImageConfig.from_file(fileset[cfgfile]))
        return ImageConfig.fold(cfgs)

    def __local_cfg(fileset):
        cfgs = []

        if 'multiply.txt' in fileset:
            cfgs.append(ImageConfig(multiply=read_float(fileset['multiply.txt'])))
        if 'cond_dropout.txt' in fileset:
            cfgs.append(ImageConfig(cond_dropout=read_float(fileset['cond_dropout.txt'])))
        if 'flip_p.txt' in fileset:
            cfgs.append(ImageConfig(flip_p=read_float(fileset['flip_p.txt'])))
        if 'local.yaml' in fileset:
            cfgs.append(ImageConfig.from_file(fileset['local.yaml']))
        if 'local.yml' in fileset:
            cfgs.append(ImageConfig.from_file(fileset['local.yml']))
        if 'batch_id.txt' in fileset:
            cfgs.append(ImageConfig(batch_id=read_text(fileset['batch_id.txt']).strip()))
        if 'loss_scale.txt' in fileset:
            cfgs.append(ImageConfig(loss_scale=read_float(fileset['loss_scale.txt'])))
        if 'timesteps_range.txt' in fileset:
            cfgs.append(ImageConfig(timesteps_range=read_int_pair(fileset['timesteps_range.txt'])))

        
        result = ImageConfig.fold(cfgs)
        if 'shuffle_tags.txt' in fileset:
            result.shuffle_tags = True

        return result

    def __sidecar_cfg(imagepath, fileset):
        cfgs = []
        for cfgext in ['.txt', '.caption', '.yml', '.yaml']:
            cfgfile = barename(imagepath) + cfgext
            if cfgfile in fileset:
                cfgs.append(ImageConfig.from_file(fileset[cfgfile]))
        return ImageConfig.fold(cfgs)

    # Use file name for caption only as a last resort
    @classmethod
    def __ensure_caption(cls, cfg: ImageConfig, file: str):
        if cfg.main_prompts:
            return cfg
        cap_cfg = ImageConfig.from_caption_text(barename(file).split("_")[0])
        return cfg.merge(cap_cfg)

    @classmethod
    def from_path(cls, data_root):
        # Create a visitor that maintains global config stack 
        # and accumulates image configs as it traverses dataset
        image_configs = {}
        def process_dir(files, parent_globals):
            fileset = {os.path.basename(f): f for f in files}
            global_cfg = parent_globals.merge(Dataset.__global_cfg(fileset))
            local_cfg = Dataset.__local_cfg(fileset)
            for img in filter(is_image, files):
                img_cfg = Dataset.__sidecar_cfg(img, fileset)
                resolved_cfg = ImageConfig.fold([global_cfg, local_cfg, img_cfg])
                ensured = Dataset.__ensure_caption(resolved_cfg, img)
                image_configs[img] = ensured

            for vid in filter(is_video, files):
                vid_cfg = Dataset.__sidecar_cfg(vid, fileset)
                resolved_cfg = ImageConfig.fold([global_cfg, local_cfg, vid_cfg])
                ensured = Dataset.__ensure_caption(resolved_cfg, vid)
                image_configs[vid] = ensured

            return global_cfg

        walk_and_visit(data_root, process_dir, ImageConfig())
        return Dataset(image_configs)

    @classmethod
    def from_json(cls, json_path):
        """
        Import a dataset definition from a JSON file
        """
        image_configs = {}
        with open(json_path, encoding='utf-8', mode='r') as stream:
            for data in json.load(stream):
                img = data.get("image")
                cfg = Dataset.__ensure_caption(ImageConfig.parse(data), img)
                if not img:
                    logging.warning(f" *** Error parsing json image entry in {json_path}: {data}")
                    continue
                image_configs[img] = cfg
        return Dataset(image_configs)    
    
    def image_train_items(self, aspects, resolution):
        items = []
        for image in tqdm(self.image_configs, desc=f"preloading [{resolution}]", dynamic_ncols=True):
            config = self.image_configs[image]
            #print(f" ********* shuffle: {config.shuffle_tags}")

            if len(config.main_prompts) > 1:
                logging.warning(f" *** Found multiple multiple main_prompts for image {image}, but only one will be applied: {config.main_prompts}")

            if len(config.main_prompts) < 1:
                logging.warning(f" *** No main_prompts for image {image}")
                continue

            tags = []
            tag_weights = []
            for tag in sorted(config.tags, key=lambda x: x.weight or 1.0, reverse=True):
                tags.append(tag.value)
                tag_weights.append(tag.weight)
            use_weights = len(set(tag_weights)) > 1 

            try:            
                caption = ImageCaption(
                    main_prompt=next(iter(config.main_prompts)),
                    rating=config.rating or 1.0,
                    tags=tags,
                    tag_weights=tag_weights,
                    max_target_length=config.max_caption_length or DEFAULT_MAX_CAPTION_LENGTH,
                    use_weights=use_weights)

                multiply = config.multiply or 1.0
                if config.per_resolution_multiply is not None and resolution in config.per_resolution_multiply:
                    multiply *= config.per_resolution_multiply[resolution]

                item = ImageTrainItem(
                    image=None,
                    caption=caption,
                    aspects=aspects,
                    pathname=os.path.abspath(image),
                    flip_p=config.flip_p or 0.0,
                    multiplier=multiply,
                    cond_dropout=config.cond_dropout,
                    shuffle_tags=config.shuffle_tags,
                    batch_id=config.batch_id,
                    loss_scale=config.loss_scale,
                    timesteps_range=config.timesteps_range,
                )
                items.append(item)
            except Exception as e:
                logging.error(f" *** Error preloading image or caption for: {image}, error: {e}")
                raise e
        return items

    def image_source_items(self, aspects_per_resolution: dict) -> list:
        """
        Create one ImageSourceItem per image, computing resolution options for all
        resolutions in a single pass (one file-open per image).

        :param aspects_per_resolution: maps each resolution integer to its list of
            (w, h) aspect-ratio bucket pairs.
        :return: list[ImageSourceItem], one per image (including items with errors so
            callers can log the error and skip them).
        """
        items = []
        for image_path in tqdm(
            self.image_configs,
            desc="preloading (multi-resolution)",
            dynamic_ncols=True,
        ):
            config = self.image_configs[image_path]

            if len(config.main_prompts) > 1:
                logging.warning(
                    f" *** Multiple main_prompts for {image_path}; only the first is used."
                )
            if len(config.main_prompts) < 1:
                logging.warning(f" *** No main_prompts for {image_path} — skipping.")
                continue

            caption = _build_caption(config)
            abs_path = os.path.abspath(image_path)

            if is_video(image_path):
                resolution_options, image_size, error = _compute_video_resolution_options(
                    abs_path, aspects_per_resolution, config.per_resolution_multiply or {},
                )
                base_item = VideoTrainItem(
                    pathname=abs_path,
                    caption=caption,
                    target_wh=list(aspects_per_resolution.values())[0][0] if aspects_per_resolution else None,
                    video_frames=81,
                    flip_p=config.flip_p or 0.0,
                    multiplier=config.multiply or 1.0,
                    cond_dropout=config.cond_dropout,
                    shuffle_tags=config.shuffle_tags or False,
                    batch_id=config.batch_id or DEFAULT_BATCH_ID,
                    loss_scale=config.loss_scale or 1.0,
                    timesteps_range=config.timesteps_range,
                )
                base_item.target_wh       = None  # assigned by make_resolved_item()
                base_item.is_undersized   = False
                base_item.runt_size       = 0
                base_item.source_resolution = None
                base_item.image_size      = image_size
                base_item.error           = error
            else:
                image_size, error, resolution_options = _compute_image_resolution_options(
                    abs_path,
                    aspects_per_resolution,
                    config.per_resolution_multiply or {},
                )
                # Build the base ImageTrainItem via __new__ to avoid re-opening
                base_item = ImageTrainItem.__new__(ImageTrainItem)
                base_item.caption         = caption
                base_item.aspects         = []
                base_item.pathname        = abs_path
                flip_p                    = config.flip_p or 0.0
                base_item.flip            = transforms.RandomHorizontalFlip(p=flip_p)
                base_item.cropped_img     = None
                base_item.runt_size       = 0
                mult                      = config.multiply or 1.0
                base_item.multiplier      = mult
                base_item.base_multiplier = mult
                base_item.cond_dropout    = config.cond_dropout
                base_item.shuffle_tags    = config.shuffle_tags or False
                base_item.batch_id        = config.batch_id or DEFAULT_BATCH_ID
                base_item.loss_scale      = config.loss_scale or 1.0
                base_item.timesteps_range = config.timesteps_range
                base_item.target_wh       = None
                base_item.is_runt         = False
                base_item.uid             = uuid.uuid4().hex
                base_item.source_resolution = None
                base_item.image_size      = image_size
                base_item.mask            = None
                base_item.image           = []
                base_item.is_undersized   = False
                base_item.error           = error

            source_item = ImageSourceItem(
                item=base_item,
                resolution_options=resolution_options,
                uid=uuid.uuid4().hex,
            )
            items.append(source_item)
        return items


# ---------------------------------------------------------------------------
# Module-level helpers (used by Dataset.image_source_items)
# ---------------------------------------------------------------------------

def _build_caption(config: 'ImageConfig') -> ImageCaption:
    """Build an ImageCaption from a resolved ImageConfig."""
    tags = []
    tag_weights = []
    for tag in sorted(config.tags, key=lambda x: x.weight or 1.0, reverse=True):
        tags.append(tag.value)
        tag_weights.append(tag.weight)
    use_weights = len(set(tag_weights)) > 1
    return ImageCaption(
        main_prompt=next(iter(config.main_prompts)),
        rating=config.rating or 1.0,
        tags=tags,
        tag_weights=tag_weights,
        max_target_length=config.max_caption_length or DEFAULT_MAX_CAPTION_LENGTH,
        use_weights=use_weights,
    )


def _compute_image_resolution_options(
    pathname: str,
    aspects_per_resolution: dict,
    per_resolution_multiply: dict,
) -> tuple:
    """
    Open the image once and compute a ResolutionOption for every resolution.

    :param pathname: absolute path to the image file.
    :param aspects_per_resolution: {resolution_int: list of [w, h] buckets}.
    :param per_resolution_multiply: {resolution_int: float} from per_resolution_multiply
        config field; may be empty.
    :return: (image_size, error, resolution_options)
        image_size  — (width, height) tuple, or None if the file could not be opened.
        error       — Exception instance, or None.
        resolution_options — dict[int, ResolutionOption].
    """
    image_size = None
    error = None
    resolution_options: dict[int, ResolutionOption] = {}
    try:
        with PIL.Image.open(pathname) as img:
            if _needs_transpose(img):
                height, width = img.size
            else:
                width, height = img.size
        resolution_options = _compute_resolution_options(width, height, aspects_per_resolution, per_resolution_multiply)
    except Exception as e:
        error = e
    return image_size, error, resolution_options


def _compute_video_resolution_options(
    pathname: str,
    aspects_per_resolution: dict,
    per_resolution_multiply: dict,
) -> tuple:
    """
    Compute resolution options for a video file.
    Uses the first resolution from aspects_per_resolution directly,
    since videos don't do aspect-ratio bucketing.
    """
    import decord

    resolution_options: dict[int, ResolutionOption] = {}
    image_size = None
    error = None
    try:
        vr = decord.VideoReader(pathname)
        frame = vr[0].asnumpy()
        height, width = frame.shape[:2]
        resolution_options = _compute_resolution_options(width, height, aspects_per_resolution, per_resolution_multiply)
    except Exception as e:
        error = e
    return resolution_options, image_size, error

def _compute_resolution_options(width, height, aspects_per_resolution, per_resolution_multiply):
    image_size = (width, height)
    image_aspect = width / height
    resolution_options = {}
    for resolution, aspects in aspects_per_resolution.items():
        target_wh = min(aspects, key=lambda wh: abs(wh[0] / wh[1] - image_aspect))
        is_feasible = not (
            (width != target_wh[0] and height != target_wh[1])
            and (width * height) < (target_wh[0] * 1.02 * target_wh[1] * 1.02)
        )
        weight = per_resolution_multiply.get(resolution, 1.0)
        resolution_options[resolution] = ResolutionOption(
            resolution=resolution,
            target_wh=target_wh,
            unnormalised_weight=weight,
            is_feasible=is_feasible,
        )
    return resolution_options

def select_caption_variants(
    captions_dict: dict[str, list[str]],
    requested_variants: list[str],
    caption_cross_concatenation_p: float = 0,
    caption_cross_concatenation_empty_half_p: float = 0
) -> list[str|tuple[str, str]]:
    available_non_default_variants = set(k for k in captions_dict if k != "default")
    if requested_variants:
        available_requested = available_non_default_variants.intersection(set(requested_variants))
        # add wildcards for each missing - this takes care of '*' as variant too
        missing = max(0, len(requested_variants) - len(available_requested))
        for _ in range(missing):
            remaining = available_non_default_variants - available_requested
            if not remaining:
                break
            available_requested.add(random.choice(list(remaining)))
        caption_candidates = list(available_requested)
    else:
        caption_candidates = list(available_non_default_variants)

    if len(caption_candidates) == 0:
        caption_candidates.append("default")
        if "default" not in captions_dict:
            raise RuntimeError("No captions found for batch, even default. This should not happen. Check your dataset and caption files.")
    selected_primary_variants = [random.choice(caption_candidates)]

    remaining_unused = set(available_non_default_variants) - set(selected_primary_variants) - set(requested_variants)

    final_variants = []
    for i, left in enumerate(selected_primary_variants):
        others = set(c for c in available_non_default_variants if c != left)
        if random.random() > caption_cross_concatenation_p or len(others) == 0:
            final_variants.append(left)
            continue

        right_candidates = others.intersection(remaining_unused)
        if not right_candidates:
            remaining_unused = set(available_non_default_variants) - set(selected_primary_variants)
            if not remaining_unused:
                remaining_unused = set(available_non_default_variants)
        if not right_candidates:
            final_variants.append(left)
            continue

        right = random.choice(list(right_candidates))
        remaining_unused.remove(right)

        if random.random() < 0.5:
            left, right = right, left
        concat_pair = [left, right]
        if random.random() < caption_cross_concatenation_empty_half_p:
            concat_pair[random.randint(0, 1)] = None

        final_variants.append(tuple(concat_pair))

    return final_variants
