# adapted from https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb#scrollTo=36026528
import os
import random
import argparse

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont


# get_wrapped_text adapted from https://stackoverflow.com/a/67203353
def get_wrapped_text(text: str, font: ImageFont.ImageFont,
                     line_length: int):
    lines = []
    words = text.split()
    current_line = None
    while len(words) > 0:
        next_word = words.pop(0)
        current_line_with_next_word = next_word if current_line is None else f'{current_line} {next_word}'
        if font.getlength(current_line_with_next_word) <= line_length:
            current_line = current_line_with_next_word
        else:
            # break here
            if current_line is not None:
                lines.append(current_line)
                words.insert(0, next_word)
                current_line = None
            else:
                # single word is too long, need to truncate
                truncated_word = next_word
                while len(truncated_word) > 0 and font.getlength(truncated_word) > line_length:
                    truncated_word = truncated_word[:-1]
                if len(truncated_word) > 0:
                    lines.append(truncated_word)
                    words.insert(0, next_word[len(truncated_word):])
                else:
                    # give up
                    lines.append(next_word)
    if current_line is not None:
        lines.append(current_line)
    return lines

    lines = ['']
    for word in text.split():

        line = f'{lines[-1]} {word}'.strip()
        if font.getlength(line) <= line_length:
            lines[-1] = line
        else:
            lines.append(word)
    return '\n'.join(lines)


def make_label_image(label: str, font: ImageFont, width: int, height: int, margins=None):
    if margins is None:
        margins = [10, 10, 10, 10]
    line_spacing = 5  # pixels between lines
    wrapped_text = '\n'.join(get_wrapped_text(label, font, line_length=width - (margins[0] + margins[2])))
    label_image = Image.new('RGB', size=(width, height), color=(247, 247, 247))
    draw = ImageDraw.Draw(label_image)
    text_bbox = draw.multiline_textbbox((width / 2, 0),
                                        text=wrapped_text,
                                        font=font,
                                        anchor='ma',
                                        spacing=line_spacing,
                                        )  # anchor = horizontal middle, ascender
    text_bbox_height = text_bbox[3] - text_bbox[1]
    # print("got text_bbox ", text_bbox)
    y_offset = (height - text_bbox_height) / 2
    x_offset = (width - margins[2] - margins[0]) // 2
    draw.multiline_text((margins[0] + x_offset, margins[1] + y_offset),
                        text=wrapped_text, font=font, anchor='ma', fill=(64, 48, 32))
    return label_image


def make_image_grid(imgs, rows, cols, row_labels: list[str], col_labels: list[str]):
    assert len(imgs) == rows * cols
    assert len(row_labels) == rows
    assert len(col_labels) == cols

    w, h = imgs[0].size
    margin = int(0.125 * w)
    spacing = int(0.125 * w)
    grid = Image.new('RGB', size=(margin + (cols + 1) * (w + spacing), margin + (rows + 1) * (h + spacing)),
                     color=(255, 255, 255))
    # grid_w, grid_h = grid.size

    # font = ImageFont.load_default()
    font_path = os.path.join(os.path.dirname(__file__), "LibreBaskerville-DpdE.ttf")
    col_font = ImageFont.truetype(font_path, size=32)
    row_font = ImageFont.truetype(font_path, size=48)

    for i, l in enumerate(col_labels):
        label_image = make_label_image(l, col_font, w, h, margins=[10, 10, 10, 10])
        grid.paste(label_image, box=(margin + (1 + i) * (w + spacing), margin))

    for i, l in enumerate(row_labels):
        label_image = make_label_image(l, row_font, w, h, margins=[20, 10, 20, 10])
        grid.paste(label_image, box=(margin, margin + (1 + i) * (h + spacing)))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(margin + (1 + (i % cols)) * (w + spacing), margin + (1 + (i // cols)) * (h + spacing)))

    return grid


def chunk_list(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i+batch_size]


def render_row(prompts: list[str],
               repo_id_or_path: str,
               device: str='cuda',
               batch_size=1) -> list[Image]:
    sample_w = 512
    sample_h = 512

    pipeline = StableDiffusionPipeline.from_pretrained(repo_id_or_path)
    # ddpm++2m
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                                 algorithm_type="dpmsolver++")
    pipeline = pipeline.to("cuda")
    images = []
    for batch in chunk_list(prompts, batch_size):
        images.extend(pipeline(batch, num_inference_steps=15))

    return images


def render_all(prompts: list[str], repo_ids_or_paths: list[str]) -> Image:
    all_images = []
    print(f"{len(prompts)} prompts")
    for repo_id_or_path in repo_ids_or_paths:
        print(f"rendering with {repo_id_or_path}...")
        row_images = render_row(prompts, repo_id_or_path)
        all_images.extend(row_images)

    grid_image = make_image_grid(all_images, len(repo_ids_or_paths), len(prompts), repo_ids_or_paths, prompts)
    return grid_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts",
                        nargs='*',
                        required=True,
                        help="prompts to render, as strings enclosed in \"\" and separated by spaces")
    parser.add_argument("--repo_ids_or_paths",
                        nargs='*',
                        required=True,
                        help="repository ids or paths to models, as strings enclosed in \"\" and separated by spaces")
    parser.add_argument("--output_path",
                        required=True,
                        help="where to save the resulting image")
    args = parser.parse_args()

    grid: Image = render_all(args.prompts, args.repo_ids_or_paths)
    grid.save(args.output_path)
