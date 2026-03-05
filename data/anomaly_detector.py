"""
Inference script for anomaly segmentation.
Applies the trained model to new images and generates segmentation masks.
"""
import math
import os
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
import json

"""
Model architecture for anomaly segmentation.
Consists of a frozen DINOv2 backbone and trainable MLP decoder.
"""
import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Pixel-wise MLP decoder for binary segmentation.
    Takes patch embeddings and outputs logits per patch.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (N, C) where C=1024
        Returns:
            Logits of shape (B, 1, H, W) or (N, 1)
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Reshape to (B*H*W, C)
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            logits = self.decoder(x)
            # Reshape back to (B, H, W, 1) then (B, 1, H, W)
            logits = logits.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        else:
            logits = self.decoder(x)

        return logits


class SegmentationModel(nn.Module):
    """
    Complete segmentation model with frozen DINOv2 backbone and trainable MLP decoder.
    """
    def __init__(self, device='mps'):
        super().__init__()
        self.device = device

        # Load frozen DINOv2 backbone
        print("Loading DINOv2 backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.backbone.eval()

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable MLP decoder
        self.decoder = MLPDecoder(input_dim=1024, hidden_dim=256, dropout=0.1)

        self.to(device)

    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (B, 3, H, W) where H and W are divisible by 14
        Returns:
            Logits of shape (B, 1, H/14, W/14)
        """
        B, C, H, W = x.shape

        # Extract features with frozen backbone
        with torch.no_grad():
            # Get patch tokens (exclude CLS token)
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # (B, N_patches, 1024)

            # Reshape to spatial grid
            h, w = H // 14, W // 14
            patch_tokens = patch_tokens.reshape(B, h, w, 1024).permute(0, 3, 1, 2)  # (B, 1024, h, w)

        # Decode with trainable MLP
        logits = self.decoder(patch_tokens)  # (B, 1, h, w)

        return logits

    def get_trainable_parameters(self):
        """
        Returns only the trainable parameters (decoder).
        Use this for optimizer initialization to ensure backbone stays frozen.
        """
        return self.decoder.parameters()


@dataclass
class ImageFile:
    path: str
    target_wh: tuple


def get_images_with_target_wh(pil_images_or_paths, resolution, patch_size) -> list[ImageFile]:
    """
    Open all image_files, get next-smallest width/height based on patch_size, and bucket images by that size

    On complete, collapse buckets until every bucket has at least 4*batch_size images. Return collated by batch_size
    """
    target_pixel_count = resolution ** 2
    def _get_basic_wh(pil_image_or_path):
        if isinstance(pil_image_or_path, Image.Image):
            w, h = pil_image_or_path.size
        elif type(pil_image_or_path) is str:
            with Image.open(pil_image_or_path) as img:
                w, h = img.size
        else:
            raise ValueError(f"image_or_path must be PIL Image or file path string (got {type(pil_image_or_path)})")

        scale_factor = math.sqrt(target_pixel_count / (w * h))

        new_h = int((h * scale_factor + patch_size - 1) // patch_size) * patch_size
        new_w = int((w * scale_factor + patch_size - 1) // patch_size) * patch_size

        return (new_w, new_h)

    image_file_objects = [ImageFile(path=p, target_wh=_get_basic_wh(p))
                          for p in pil_images_or_paths]
    return image_file_objects

# adapted from Victor C. Hall's EveryDream2trainer
def trim_to_aspect(image: Image.Image, target_wh: tuple) -> Image.Image:
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
        return image
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
        return image
    else:
        return image


patch_size = 14

def get_device():
    """
    Get the best available device (MPS for Apple Silicon, otherwise CPU).
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS acceleration")
    else:
        device = torch.device('cpu')
        print("MPS not available, using CPU")

    return device


def preprocess_image(image_pil, resolution):
    """
    Preprocess image for model input.

    Args:
        image_pil: PIL Image (RGB)

    Returns:
        image_tensor: Preprocessed tensor (1, 3, H, W)
        original_size: (width, height) of original image
        resized_size: (width, height) of resized image
    """
    original_size = image_pil.size
    image_obj = get_images_with_target_wh([image_pil], resolution=resolution, patch_size=patch_size)
    resized_size = image_obj[0].target_wh

    image_pil = trim_to_aspect(image_pil, resized_size)
    image_pil = image_pil.resize(resized_size, Image.LANCZOS)

    # Convert to tensor and normalize
    image_tensor = TF.to_tensor(image_pil)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_tensor = normalize(image_tensor)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)

    return image_tensor, original_size, resized_size


def postprocess_mask(mask_logits, original_size, resized_size, threshold=0.5, invert=False, smooth_mask=False) -> np.ndarray:
    """
    Convert model output to binary mask and upscale to original resolution.

    Args:
        mask_logits: Model output (1, 1, H/14, W/14)
        original_size: (width, height) of original image
        resized_size: (width, height) of resized image
        threshold: Threshold for binarization

    Returns:
        mask_np: Binary mask as numpy array (H, W) with values 0 or 255
    """
    # Convert logits to probabilities
    mask_probs = torch.sigmoid(mask_logits)

    # Remove batch and channel dimensions
    mask_probs = mask_probs.squeeze(0).squeeze(0)  # (H/14, W/14)

    # Move to CPU and convert to numpy
    mask_np = mask_probs.cpu().numpy()

    # Binarize
    if invert:
        mask_binary = (mask_np <= threshold).astype(np.uint8)
    else:
        mask_binary = (mask_np > threshold).astype(np.uint8)

    # Upscale to resized image size using nearest neighbor
    resized_w, resized_h = resized_size
    mask_upscaled = cv2.resize(
        mask_binary,
        (resized_w, resized_h),
        interpolation=cv2.INTER_CUBIC
    )

    # Resize to original size if needed
    original_w, original_h = original_size
    if (resized_w, resized_h) != (original_w, original_h):
        mask_upscaled = cv2.resize(
            mask_upscaled,
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC

        )

    # Convert to 0-255 range
    mask_upscaled = (mask_upscaled * 255).astype(np.uint8)
    if smooth_mask:
        blur_radius = 2 * int(2 * (patch_size * original_w / args.resolution) // 2) + 1
        mask_upscaled = dilate_erode_mask(mask_upscaled, pre_erode=0, dilate=blur_radius//4, erode=0, kernel_size=3)
        mask_upscaled = cv2.GaussianBlur(mask_upscaled, (blur_radius, blur_radius), 0)

    return mask_upscaled


def create_overlay(image_pil, mask_np, alpha=0.5, color=(255, 0, 0)):
    """
    Create an overlay of the mask on the original image.

    Args:
        image_pil: Original PIL Image (RGB)
        mask_np: Binary mask (H, W) with values 0 or 255
        alpha: Transparency of overlay (0-1)
        color: RGB color for the mask overlay

    Returns:
        overlay_pil: PIL Image with mask overlay
    """
    # Convert image to numpy
    image_np = np.array(image_pil)

    # Convert 0..255 mask to graduated tint overlay
    #tint_overlay = mask_np / 255.0 * np.array(color)  # (H, W, 3) with values in range [0, color]
    tint_overlay = (mask_np / 255.0)[..., np.newaxis] * np.array(color)

    # Create colored mask
    #image_color = mask_np * color
    #mask_colored = np.zeros_like(image_np)

    #mask_colored += mask_np * color

    # Blend
    #overlay_np = cv2.addWeighted(image_np, 1 - alpha, mask_colored, alpha, 0)
    overlay_np = cv2.addWeighted(image_np, 1 - alpha, tint_overlay.astype(np.uint8), alpha, 0)

    # Convert back to PIL
    overlay_pil = Image.fromarray(overlay_np)

    return overlay_pil


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = SegmentationModel(device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.decoder.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    print(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")

    return model


def segment_image(model, image_pil, resolution, device, threshold=0.5, smooth_mask=False) -> np.ndarray:
    """
    Segment a single image.

    Args:
        model: Trained segmentation model
        image_pil: PIL Image (RGB)
        device: Device to run inference on
        threshold: Threshold for binarization

    Returns:
        mask_np: Binary mask (H, W) with values 0 or 255
    """
    # Preprocess
    image_tensor, original_size, resized_size = preprocess_image(image_pil, resolution=resolution)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        mask_logits = model(image_tensor)

    # Postprocess
    mask_np = postprocess_mask(mask_logits, original_size, resized_size, threshold, smooth_mask=smooth_mask)

    return mask_np


def dilate_erode_mask(mask_np, pre_erode: int, dilate: int, erode: int, kernel_size: int=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = cv2.erode(mask_np, kernel, iterations=pre_erode)
    mask_np = cv2.dilate(mask_np, kernel, iterations=dilate)
    mask_np = cv2.erode(mask_np, kernel, iterations=erode)
    return mask_np



def main(args):
    # Set device
    device = get_device()

    # Load model
    model = load_model(args.checkpoint, device)

    # Get list of images
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        print("finding files...")
        image_files = []
        for directory, _, files in input_path.walk():
            image_files.extend(sorted([
                directory / f for f in files
                if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']
            ]))
    else:
        raise ValueError(f"Input path not found: {input_path}")

    print(f"\nFound {len(image_files)} image(s) to process")

    if args.save_anomaly_percentages_json:
        print(f"✓ Anomaly percentages will be saved to: {args.save_anomaly_percentages_json}")
    elif args.output_dir is None:
        raise ValueError("No output directory specified. Please provide --output_dir or --save_anomaly_percentages_json to save results.")

    if args.output_dir is None:
        output_dir = None
    else:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.save_overlay:
            overlay_dir = output_dir.parent / (output_dir.name + '_overlays')
            overlay_dir.mkdir(exist_ok=True)

    # Process each image
    print("\nProcessing images...")
    anomaly_percentages = {}
    for image_file in tqdm(image_files):
        try:
            # Load image
            image_pil = Image.open(image_file).convert('RGB')

            # Segment
            mask_np = segment_image(model, image_pil, resolution=args.resolution, device=device, threshold=args.threshold,
                                    smooth_mask = args.smooth_mask)

            invert = False
            if invert:
                mask_np = 255 - mask_np
            anomaly_percentages[image_file.name] = int(mask_np.sum() / 255) / (mask_np.shape[0] * mask_np.shape[1])

            if output_dir is not None:
                # Save mask
                if mask_np.sum() == 0:
                    print(f"mask for {image_file} is empty - not saving")
                    continue
                relative_dir = image_file.parent.relative_to(input_path) if input_path.is_dir() else Path('.')
                mask_filename = image_file.name + '.mask.png'
                (output_dir / relative_dir).mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / relative_dir / mask_filename
                Image.fromarray(mask_np).save(mask_path)

                # Save overlay if requested
                if args.save_overlay:
                    overlay_pil = create_overlay(
                        image_pil,
                        mask_np,
                        alpha=args.overlay_alpha,
                        color=tuple(args.overlay_color)
                    )
                    overlay_filename = image_file.name + '.overlay.png'
                    overlay_path = overlay_dir / relative_dir / overlay_filename
                    (overlay_dir / relative_dir).mkdir(parents=True, exist_ok=True)
                    overlay_pil.save(overlay_path)


        except Exception as e:
            print(f"✗ Failed to process {image_file}: {repr(e)}")
            continue

    print(f"\n✓ Processing complete!")
    if output_dir is not None:
        print(f"  Masks saved to: {output_dir}")
        if args.save_overlay:
            print(f"  Overlays saved to: {overlay_dir}")
    if args.save_anomaly_percentages_json:
        with open(args.save_anomaly_percentages_json, 'wt') as f:
            json.dump(anomaly_percentages, f)
        print(f"  Anomaly percentages saved to: {args.save_anomaly_percentages_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Segment anomalies in images'
    )

    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output masks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Resolution to run the model, should match training resolution')

    parser.add_argument('--save_anomaly_percentages_json', type=str, default=None, help='Write anomaly percentage to this path')

    # Inference arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions (0-1)')
    parser.add_argument('--save_overlay', action='store_true',
                        help='Save overlay visualization of mask on image')
    parser.add_argument('--overlay_alpha', type=float, default=0.5,
                        help='Transparency of overlay (0-1)')
    parser.add_argument('--overlay_color', type=int, nargs=3, default=[255, 0, 0],
                        help='RGB color for mask overlay (e.g., 255 0 0 for red)')
    parser.add_argument('--smooth_mask', action=argparse.BooleanOptionalAction,
                        help='Make mask smoother by applying erosion, dilation, and blur')

    args = parser.parse_args()

    main(args)

