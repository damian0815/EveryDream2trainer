import argparse
import os
import zipfile
from typing import Generator


def collect_files_and_sizes(root: str) -> Generator[tuple[str, int], None, None]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            yield path, os.path.getsize(path)

def split_zip_independent(roots: list[str], output_prefix: str, max_size: int = 2 * 1024 * 1024 * 1024):

    common_prefix = os.path.commonpath([os.path.dirname(r) for r in roots])

    def make_zip(index: int) -> zipfile.ZipFile:
        return zipfile.ZipFile(f"{output_prefix}_{index:03d}.zip", 'w', zipfile.ZIP_DEFLATED)

    current_zip_index = 0
    current_zip_size = 0

    current_zip = make_zip(current_zip_index)
    for root in roots:
        for filepath, filesize in collect_files_and_sizes(root):
            if current_zip_size + filesize > max_size:
                current_zip.close()
                current_zip_index += 1
                current_zip_size = 0
                current_zip = make_zip(current_zip_index)

            rel_path = os.path.relpath(filepath, common_prefix)
            current_zip.write(filepath, rel_path)
            current_zip_size += filesize

    current_zip.close()





if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--root', nargs='+', type=str, required=True)
    arg_parser.add_argument('--out_prefix', type=str, required=True)

    args = arg_parser.parse_args()

    split_zip_independent(roots = args.root, output_prefix=args.out_prefix)
