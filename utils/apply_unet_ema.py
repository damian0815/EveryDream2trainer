from diffusers import UNet2DConditionModel
from safetensors.torch import load_file
import os

def apply_unet_ema(
    unet: UNet2DConditionModel,
    ema_path: str,
) -> None:
    """
    Load EMA weights from *ema_path* and apply them in-place to *unet*.

    The *ema_path* should be a safetensors file containing only the UNet EMA
    weights, with keys matching the UNet's state dict.  This is the format
    produced by the default training loop when ``--save-ema`` is enabled.

    Example usage::
        apply_unet_ema(model.unet, "checkpoints/ema_unet.safetensors")
    """
    ema_state = load_file(ema_path)
    unet.load_state_dict(ema_state, strict=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply UNet EMA weights from a safetensors file.")
    parser.add_argument("unet_path", type=str, help="Path to the UNet model (e.g. a diffusers pipeline checkpoint).")
    parser.add_argument("ema_path", type=str, help="Path to the EMA weights safetensors file.")
    parser.add_argument("--inplace", action="store_true", help="Whether to overwrite the original UNet checkpoint (default: False, saves to a new directory).")
    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(args.unet_path)
    apply_unet_ema(unet, args.ema_path)
    print(f"Applied EMA weights from {args.ema_path} to UNet loaded from {args.unet_path}")
    unet.save_pretrained(args.unet_path + "_with_ema")
    if args.inplace:
        os.rename(args.unet_path, args.unet_path + "_backup")
        os.rename(args.unet_path + "_with_ema", args.unet_path)
        shutil.rmtree(args.unet_path + "_backup")
        print(f"Overwrote original UNet checkpoint at {args.unet_path} with EMA weights")
    else:
        print(f"Saved UNet with EMA weights to {args.unet_path + '_with_ema'}")
