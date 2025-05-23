import yaml
import json

from attrs import define, field
from data.image_train_item import ImageCaption, ImageTrainItem, check_caption_json
from utils.fs_helpers import *
from typing import Iterable

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
    cond_dropout: float = None
    flip_p: float = None
    shuffle_tags: bool = False
    loss_scale: float = None
    timesteps_range: tuple[int, int] = None

    def merge(self, other):
        if other is None:
            return self

        return ImageConfig(
            main_prompts=other.main_prompts | self.main_prompts,
            rating=overlay(other.rating, self.rating),
            max_caption_length=overlay(other.max_caption_length, self.max_caption_length),
            tags= other.tags | self.tags,
            multiply=overlay(other.multiply, self.multiply),
            cond_dropout=overlay(other.cond_dropout, self.cond_dropout),
            flip_p=overlay(other.flip_p, self.flip_p),
            shuffle_tags=overlay(other.shuffle_tags, self.shuffle_tags),
            batch_id=overlay(other.batch_id, self.batch_id),
            loss_scale=overlay(other.loss_scale, self.loss_scale),
            timesteps_range=overlay(other.timesteps_range, self.timesteps_range),
        )

    @classmethod
    def from_dict(cls, data: dict):
        # Parse standard yaml tag file (with options)
        parsed_cfg = ImageConfig(
            main_prompts=safe_set(data.get("main_prompt")), 
            rating=data.get("rating"), 
            max_caption_length=data.get("max_caption_length"), 
            tags=safe_set(map(Tag.parse, data.get("tags", []))),
            multiply=data.get("multiply"),
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
            return ImageConfig(
                main_prompts=split_caption[0],
                tags=map(Tag.parse, split_caption[1:])
                )

    @classmethod    
    def from_file(cls, file: str):
        match ext(file):
            case '.jpg' | '.jpeg' | '.png' | '.bmp' | '.webp' | '.jfif':
                return ImageConfig(image=file)
            case ".json":
                return ImageConfig.from_dict(json.load(read_text(file)))
            case ".yaml" | ".yml":
                return ImageConfig.from_dict(yaml.safe_load(read_text(file)))
            case ".txt" | ".caption":
                return ImageConfig.from_caption_text(read_text(file))
            case _:
                return logging.warning(" *** Unrecognized config extension {ext}")

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
                #for m in ensured.main_prompts.keys():
                #    check_caption_json(m)
                image_configs[img] = ensured

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
    
    def image_train_items(self, aspects):
        items = []
        for image in tqdm(self.image_configs, desc="preloading", dynamic_ncols=True):
            config = self.image_configs[image]
            #print(f" ********* shuffle: {config.shuffle_tags}")

            if len(config.main_prompts) > 1:
                logging.warning(f" *** Found multiple multiple main_prompts for image {image}, but only one will be applied: {config.main_prompts}")

            if len(config.main_prompts) < 1:
                logging.warning(f" *** No main_prompts for image {image}")

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

                item = ImageTrainItem(
                    image=None,
                    caption=caption,
                    aspects=aspects,
                    pathname=os.path.abspath(image),
                    flip_p=config.flip_p or 0.0,
                    multiplier=config.multiply or 1.0,
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