import argparse
import math
import os.path
import random
import shutil
from typing import Optional

from tqdm.auto import tqdm

IMAGE_EXTENSIONS =  ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']
CAPTION_EXTENSIONS = ['.txt', '.caption', '.yaml', '.yml']

def gather_captioned_images(root_dir: str) -> list[tuple[str,Optional[str]]]:
    for directory, _, filenames in os.walk(root_dir):
        image_filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        for image_filename in image_filenames:
            image_path = os.path.join(directory, image_filename)
            image_path_without_extension = os.path.splitext(image_path)[0]
            caption_path = None
            for caption_extension in CAPTION_EXTENSIONS:
                possible_caption_path = image_path_without_extension + caption_extension
                if os.path.exists(possible_caption_path):
                    caption_path = possible_caption_path
                    break
            yield image_path, caption_path


def copy_captioned_image(image_caption_pair: tuple[str, Optional[str]], source_root: str, target_root: str):
    image_path = image_caption_pair[0]
    caption_path = image_caption_pair[1]

    # make target folder if necessary
    relative_folder = os.path.dirname(os.path.relpath(image_path, source_root))
    target_folder = os.path.join(target_root, relative_folder)
    os.makedirs(target_folder, exist_ok=True)

    # copy files
    shutil.copy2(image_path, os.path.join(target_folder, os.path.basename(image_path)), follow_symlinks=False)
    if caption_path is not None:
        shutil.copy2(caption_path, os.path.join(target_folder, os.path.basename(caption_path)), follow_symlinks=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=str, help='Source root folder containing images')
    parser.add_argument('--output_folder_base', type=str, required=False, help="Output folder base name.")
    parser.add_argument('--split_count', type=int, required=True, help="Number of splits")
    parser.add_argument('--seed', type=int, required=False, default=555, help='Random seed for shuffling')
    args = parser.parse_args()

    images = list(gather_captioned_images(args.source_root))

    print(f"Found {len(images)} captioned images in {args.source_root}")

    random.seed(args.seed)
    random.shuffle(images)

    split_size = math.ceil(len(images)/args.split_count)
    splits = [images[k*split_size:(k+1)*split_size] for k in range(0, args.split_count)] 

    print(f"Split to {len(splits)} splits each with {split_size} images")
    for i, s in enumerate(tqdm(splits)):
        split_folder = args.output_folder_base + '_' + str(i)
        for v in s:
            copy_captioned_image(v, args.source_root,  split_folder)
        subfolders = set([os.path.dirname(v[0]) for v in s])
        for f in subfolders:
            if os.path.exists(os.path.join(f, 'multiply.txt')):
                shutil.copy2(os.path.join(f, 'multiply.txt'), os.path.join(split_folder, os.path.basename(f)))
            if os.path.exists(os.path.join(f, 'batch_id.txt')):
                shutil.copy2(os.path.join(f, 'batch_id.txt'), os.path.join(split_folder, os.path.basename(f)))
    print("Done.")
