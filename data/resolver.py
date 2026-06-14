import json
import logging
import os
import typing
import zipfile
import argparse
from data.dataset import Dataset

import tqdm
from colorama import Fore, Style

from data.image_train_item import ImageCaption, ImageTrainItem

class DataResolver:
    def __init__(self, args: argparse.Namespace, aspects, resolution):
        """
        :param args: EveryDream configuration, an `argparse.Namespace` object.
        """
        self.aspects = aspects
        self.flip_p = args.flip_p
        self.resolution = resolution

    def image_train_items(self, data_root: str) -> list[ImageTrainItem]:
        """
        Get the list of `ImageTrainItem` for the given data root.

        :param data_root: The data root, a directory, a file, etc..
        :return: The list of `ImageTrainItem`.
        """
        raise NotImplementedError()
    
class JSONResolver(DataResolver):
    def image_train_items(self, json_path: str) -> list[ImageTrainItem]:
        """
        Create `ImageTrainItem` objects with metadata for hydration later.
        Extracts images and captions from a JSON file.

        :param json_path: The path to the JSON file.
        """
        return Dataset.from_json(json_path).image_train_items(self.aspects, resolution=self.resolution)
    
class DirectoryResolver(DataResolver):    
    def image_train_items(self, data_root: str) -> list[ImageTrainItem]:
        """
        Create `ImageTrainItem` objects with metadata for hydration later.
        Unzips all zip files in `data_root` and then recursively searches the
        `data_root` for images and captions.

        :param data_root: The root directory to recurse through
        """
        DirectoryResolver.unzip_all(data_root)
        return Dataset.from_path(data_root).image_train_items(self.aspects, resolution=self.resolution)
        
    @staticmethod
    def unzip_all(path):
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.zip'):
                        logging.info(f"Unzipping {file}")
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            zip_ref.extractall(path)
        except Exception as e:
            logging.error(f"Error unzipping files {e}")
    
def strategy(data_root: str) -> typing.Type[DataResolver]:
    """
    Determine the strategy to use for resolving the data.
    :param data_root: The root directory or JSON file to resolve.
    """
    if os.path.isfile(data_root) and data_root.endswith('.json'):
        return JSONResolver
    
    if os.path.isdir(data_root):
        return DirectoryResolver
        
    raise ValueError(f"data_root '{data_root}' is not a valid directory or JSON file.")
                    
def resolve_root(path: str, args: argparse.Namespace, resolution=None, aspects=None) -> list[ImageTrainItem]:
    """
    Resolve the training data from the root path.
    :param path: The root path to resolve.
    :param args: EveryDream configuration, an `argparse.Namespace` object.
    :param aspects: Aspect ratio buckets. Falls back to args.aspects when not supplied.
    """
    if aspects is None:
        aspects = getattr(args, 'aspects', None)
    resolver = strategy(path)
    return resolver(args, aspects, resolution).image_train_items(path)

def resolve(value: typing.Union[dict, str], args: argparse.Namespace, resolution=None, aspects=None) -> list[ImageTrainItem]:
    """
    Resolve the training data from the value.
    :param value: The value to resolve, either a dict, an array, or a string.
    :param args: EveryDream configuration, an `argparse.Namespace` object.
    :param aspects: Aspect ratio buckets. Falls back to args.aspects when not supplied.
    """
    if aspects is None:
        aspects = getattr(args, 'aspects', None)
    if isinstance(value, str):
        return resolve_root(value, args, resolution, aspects)
    
    if isinstance(value, dict):
        resolver = value.get('resolver', None)
        match resolver:
            case 'directory' | 'json':
                path = value.get('path', None)
                return resolve_root(path, args, resolution, aspects)
            case 'multi':
                return resolve(value.get('resolvers', []), args, resolution, aspects)
            case _:
                raise ValueError(f"Cannot resolve training data for resolver value '{resolver}'")

    if isinstance(value, list):
        items = []
        for item in value:
            items += resolve(item, args, resolution, aspects)
        return items 

def resolve_sources(
    value,
    args: argparse.Namespace,
    aspects_per_resolution: dict,
) -> list:
    """
    Like resolve(), but returns a list of ImageSourceItem covering all resolutions
    at once.  Each image file is opened exactly once.

    :param value: same as resolve() — a path string, dict, or list.
    :param args: EveryDream configuration namespace.
    :param aspects_per_resolution: {resolution_int: list of (w, h) buckets}.
    :return: list[ImageSourceItem].
    """
    if isinstance(value, str):
        return _resolve_sources_root(value, args, aspects_per_resolution)

    if isinstance(value, dict):
        resolver_name = value.get('resolver', None)
        match resolver_name:
            case 'directory' | 'json':
                return _resolve_sources_root(
                    value.get('path', None), args, aspects_per_resolution
                )
            case 'multi':
                return resolve_sources(
                    value.get('resolvers', []), args, aspects_per_resolution
                )
            case _:
                raise ValueError(
                    f"Cannot resolve training data for resolver '{resolver_name}'"
                )

    if isinstance(value, list):
        items = []
        for item in value:
            items += resolve_sources(item, args, aspects_per_resolution)
        return items

    raise ValueError(f"Unsupported value type for resolve_sources: {type(value)}")


def _resolve_sources_root(
    path: str,
    args: argparse.Namespace,
    aspects_per_resolution: dict,
) -> list:
    """Resolve a single data root (directory or JSON file) into ImageSourceItems."""
    if os.path.isfile(path) and path.endswith('.json'):
        dataset = Dataset.from_json(path)
    elif os.path.isdir(path):
        DirectoryResolver.unzip_all(path)
        dataset = Dataset.from_path(path)
    else:
        raise ValueError(f"data_root '{path}' is not a valid directory or JSON file.")
    return dataset.image_source_items(aspects_per_resolution)
