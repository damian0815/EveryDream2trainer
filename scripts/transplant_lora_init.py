"""
transplant_lora_init.py

Standalone script (only needs torch / diffusers / peft / safetensors — no
dependency on any custom trainer package).

Computes the domain-adaptation delta between your finetuned SANA-1.5 1.6B
checkpoint and the base 1.6B pretrained weights, then uses an SVD-truncated
version of that delta to *initialize* a LoRA adapter on the 4.8B model,
instead of random LoRA init.

Only blocks that were verifiably copied 1:1 during NVIDIA's 1.6B->4.8B
"constant" growth strategy get a real transplant. Based on your own diff:
  - blocks 0..17  : clean copy (bf16 noise floor only)      -> full-weight transplant
  - block 18      : copied but drifted during further pretraining -> transplant, downweighted
  - blocks 19+    : brand-new blocks, no source counterpart  -> left at default LoRA init
                     (these should get their domain signal via --teacher distillation
                     in your existing trainer, not from this script)

Usage:
    python transplant_lora_init.py \
        --base_1_6b_id Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
        --finetuned_1_6b_path /path/to/your/transformer_gsNNNNN.safetensors \
        --base_4_8b_id Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers \
        --output_dir ./sana_4_8b_lora_init \
        --rank 32 --alpha 32

Then in your trainer:
    --model_id Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers
    --lora --lora_resume ./sana_4_8b_lora_init
    --teacher /path/to/your/finetuned_1.6B/transformer_gsNNNNN.safetensors
"""
import argparse
import logging
import re

import torch
from safetensors.torch import load_model, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Modules we treat as LoRA-eligible. attn q/k/v/out are always safe (plain
# nn.Linear). ff.conv_point / ff.conv_inverted are 1x1 Conv2d in SANA's
# glumbconv FFN and CAN be LoRA'd, but are riskier to auto-target (shape
# handling, peft version support) so they're opt-in via --include_ffn.
# ff.conv_depth is a depthwise conv — low-rank decomposition doesn't make
# sense for a grouped/depthwise conv, so it's never targeted.
CORE_TARGETS = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
]
FFN_TARGETS = ["ff.conv_point", "ff.conv_inverted"]

BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def get_num_blocks(state_dict):
    return max(int(m.group(1)) for k in state_dict if (m := BLOCK_RE.search(k))) + 1


def svd_lora_init(delta: torch.Tensor, rank: int, scaling: float):
    """
    Given a weight delta (out, in) [or (out, in, 1, 1) for 1x1 conv],
    return (lora_A, lora_B) such that lora_B @ lora_A * scaling ~= delta
    (truncated to `rank`).
    """
    orig_shape = delta.shape
    is_conv = delta.dim() == 4
    if is_conv:
        assert orig_shape[2] == 1 and orig_shape[3] == 1, "only 1x1 conv supported"
        mat = delta[:, :, 0, 0].float()
    else:
        mat = delta.float()

    out_dim, in_dim = mat.shape
    r = min(rank, out_dim, in_dim)

    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    # target: B @ A * scaling ~= mat  =>  B @ A ~= mat / scaling
    sqrt_S = torch.sqrt(S_r / scaling)
    B = U_r * sqrt_S.unsqueeze(0)          # (out, r)
    A = sqrt_S.unsqueeze(1) * Vh_r         # (r, in)

    if is_conv:
        B = B.unsqueeze(-1).unsqueeze(-1)  # (out, r, 1, 1)
        A = A.unsqueeze(-1).unsqueeze(-1)  # (r, in, 1, 1)

    return A.to(delta.dtype), B.to(delta.dtype), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_1_6b_id", default="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers")
    ap.add_argument("--finetuned_1_6b_path", required=True,
                     help="local safetensors path to YOUR finetuned 1.6B transformer state dict")
    ap.add_argument("--base_4_8b_id", default="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--full_confidence_upto", type=int, default=18,
                     help="blocks [0, this) get full-weight transplant (exclusive)")
    ap.add_argument("--transition_block", type=int, default=18,
                     help="single block transplanted at reduced confidence")
    ap.add_argument("--transition_scale", type=float, default=0.3,
                     help="delta multiplier for the transition block")
    ap.add_argument("--include_ffn", action="store_true",
                     help="also transplant/LoRA the ff.conv_point / ff.conv_inverted 1x1 convs")
    args = ap.parse_args()

    from diffusers import SanaTransformer2DModel
    from peft import LoraConfig, get_peft_model

    # ---- 1. load base 1.6B and your finetuned 1.6B, compute deltas ----
    log.info("Loading base 1.6B from HF: %s", args.base_1_6b_id)
    base_1_6b = SanaTransformer2DModel.from_pretrained(args.base_1_6b_id, subfolder="transformer")
    base_1_6b_sd = base_1_6b.state_dict()

    log.info("Loading your finetuned 1.6B from %s", args.finetuned_1_6b_path)
    finetuned_1_6b = SanaTransformer2DModel.from_pretrained(args.base_1_6b_id, subfolder="transformer")
    load_model(finetuned_1_6b, args.finetuned_1_6b_path)
    finetuned_1_6b_sd = finetuned_1_6b.state_dict()

    n_blocks_1_6b = get_num_blocks(base_1_6b_sd)
    log.info("1.6B has %d transformer blocks", n_blocks_1_6b)

    deltas = {}  # key -> delta_tensor (already confidence-scaled)
    target_names = CORE_TARGETS + (FFN_TARGETS if args.include_ffn else [])
    for i in range(min(args.transition_block + 1, n_blocks_1_6b)):
        scale = 1.0 if i < args.full_confidence_upto else args.transition_scale
        for name in target_names:
            key = f"transformer_blocks.{i}.{name}.weight"
            if key not in base_1_6b_sd or key not in finetuned_1_6b_sd:
                continue
            delta = (finetuned_1_6b_sd[key] - base_1_6b_sd[key]) * scale
            deltas[key] = delta

    log.info("Computed %d weight deltas across blocks 0..%d (block %d downweighted x%.2f)",
              len(deltas), args.transition_block, args.transition_block, args.transition_scale)
    del base_1_6b, finetuned_1_6b, base_1_6b_sd, finetuned_1_6b_sd

    # ---- 2. load base 4.8B and attach LoRA ----
    log.info("Loading base 4.8B from HF: %s", args.base_4_8b_id)
    model_4_8b = SanaTransformer2DModel.from_pretrained(args.base_4_8b_id, subfolder="transformer")
    n_blocks_4_8b = get_num_blocks(model_4_8b.state_dict())
    log.info("4.8B has %d transformer blocks", n_blocks_4_8b)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=list(target_names),  # peft matches on module name suffix
        init_lora_weights=True,  # default init for everything not explicitly overwritten below
    )
    peft_model = get_peft_model(model_4_8b, lora_config)

    named_modules = dict(peft_model.named_modules())

    transplanted, skipped = 0, 0
    for key, delta in deltas.items():
        # key like "transformer_blocks.18.attn1.to_q.weight" -> module path
        module_path = key.rsplit(".weight", 1)[0]
        full_module_name = None
        for mod_name in named_modules:
            if mod_name.endswith(module_path):
                full_module_name = mod_name
                break
        if full_module_name is None:
            log.warning("no LoRA module found for %s — skipping", key)
            skipped += 1
            continue

        module = named_modules[full_module_name]
        if not hasattr(module, "lora_A"):
            log.warning("%s has no lora_A (not a LoRA layer?) — skipping", full_module_name)
            skipped += 1
            continue

        adapter_name = list(module.lora_A.keys())[0]
        scaling = module.scaling[adapter_name]

        A, B, used_rank = svd_lora_init(delta, args.rank, scaling)
        with torch.no_grad():
            module.lora_A[adapter_name].weight.copy_(A)
            module.lora_B[adapter_name].weight.copy_(B)
        transplanted += 1

    log.info("Transplanted %d LoRA modules from task-vector SVD init (%d skipped)", transplanted, skipped)
    log.info("Blocks %d..%d were left at default LoRA init — feed domain signal via --teacher instead",
              args.transition_block + 1, n_blocks_4_8b - 1)

    # ---- 3. save adapter in diffusers-compatible format ----
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(peft_model))

    try:
        from diffusers import SanaPipeline
        SanaPipeline.save_lora_weights(save_directory=args.output_dir, transformer_lora_layers=lora_state_dict)
        # todo: postprocess: state_dict = {k.replace('transformer.base_model.model.', 'transformer.'): v for k,v in state_dict.items()}
        log.info("Saved diffusers-format LoRA adapter to %s", args.output_dir)
    except Exception as e:
        log.warning("SanaPipeline.save_lora_weights failed (%s) — falling back to raw safetensors dump. "
                    "If your trainer's load_lora_weights() rejects this, build the adapter via "
                    "model.build_inference_pipeline().save_lora_weights(...) instead, matching your "
                    "own save_sana_lora() implementation.", e)
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        save_file(lora_state_dict, os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))


if __name__ == "__main__":
    main()