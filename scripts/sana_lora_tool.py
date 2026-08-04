"""
sana_lora_tool.py

Standalone script (only needs torch / safetensors — diffusers is imported
lazily) for the three common LoRA bookkeeping operations on SANA
transformers:

  bake    Merge a LoRA adapter into a full transformer checkpoint:
          W += (lora_alpha / r) * (B @ A).

  extract Recover a LoRA adapter from a finetuned full transformer
          checkpoint by taking the delta against the base transformer and
          SVD-decomposing it.

  resize  Change the rank and/or alpha of an existing LoRA adapter:
          reconstruct the effective delta, re-SVD at the new rank, and
          write a new adapter. (Named `resize` deliberately — `rescale`
          would misleadingly imply a plain multiply on the weights.)

All math is pure state-dict arithmetic. No peft and no diffusers model
instantiation is needed for bake/resize; the base state dict (bake/extract)
is loaded via the diffusers model classes. This gives us full control over
the output file keys so the result is loadable by
`SanaPipeline.load_lora_weights()` without any key post-processing.

The written adapter uses the trainer-compatible key form:

    transformer.transformer_blocks.N.attn1.to_q.lora.down.weight
    transformer.transformer_blocks.N.attn1.to_q.lora.up.weight

(with `lora.down` = peft `lora_A`, `lora.up` = peft `lora_B`), plus a
`lora_adapter_metadata` file-level JSON carrying `r`, `lora_alpha` and
`target_modules` so an `alpha != r` survives a save/load round-trip.

Usage:
    python scripts/sana_lora_tool.py bake \\
        --base Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers \\
        --lora /path/to/lora_dir --output ./baked_transformer.safetensors

    python scripts/sana_lora_tool.py extract \\
        --base Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers \\
        --finetuned /path/to/transformer_gsNNNN.safetensors \\
        --output ./recovered_lora --rank 32 --alpha 32

    python scripts/sana_lora_tool.py resize \\
        --lora /path/to/lora_dir --output ./resized_lora --rank 16 --alpha 16
"""
import argparse
import glob
import json
import logging
import os
import re

import torch
import safetensors.torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BASE_ID = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"

# Modules we treat as LoRA-eligible. attn q/k/v/out are always safe (plain
# nn.Linear). ff.conv_point / ff.conv_inverted are 1x1 Conv2d in SANA's
# glumbconv FFN and CAN be LoRA'd, but are riskier to auto-target so they're
# opt-in via --include_ffn. ff.conv_depth is a depthwise conv — low-rank
# decomposition doesn't make sense for a grouped/depthwise conv, so it's
# never targeted.
CORE_TARGETS = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
]
FFN_TARGETS = ["ff.conv_point", "ff.conv_inverted"]

BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")

# Matches any of the supported LoRA key forms and captures the canonical
# module path (e.g. "transformer_blocks.0.attn1.to_q"):
#   transformer.transformer_blocks.0.attn1.to_q.lora.down.weight
#   transformer_blocks.0.attn1.to_q.lora_A.default.weight
#   base_model.model.transformer_blocks.0.attn1.to_q.lora_A.default.weight
# The optional adapter-name segment between the lora part and .weight is
# tolerated and dropped.
LORA_WEIGHT_RE = re.compile(
    r"^(?:transformer\.)?(?:base_model\.model\.)?"
    r"(?P<module>.*?)"
    r"\.lora(?P<dot>\.|_)(?P<part>down|up|A|B)"
    r"(?:\.[^.]+)?\.weight$"
)
LORA_BIAS_RE = re.compile(r"\.lora(?P<dot>\.|_)(?:down|up|A|B)(?:\.[^.]+)?\.bias$")


def get_num_blocks(state_dict):
    return max(int(m.group(1)) for k in state_dict if (m := BLOCK_RE.search(k))) + 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_base_state_dict(base_id, is_video=False):
    """Load the base SANA transformer state dict via diffusers model classes."""
    if is_video:
        from diffusers import SanaVideoTransformer3DModel as ModelClass
    else:
        from diffusers import SanaTransformer2DModel as ModelClass
    log.info("Loading base transformer %s from %s",
             ModelClass.__name__, base_id)
    model = ModelClass.from_pretrained(base_id, subfolder="transformer")
    sd = model.state_dict()
    n_blocks = get_num_blocks(sd)
    del model
    log.info("Base transformer has %d blocks", n_blocks)
    return sd


def _resolve_lora_path(path):
    if os.path.isdir(path):
        resolved = os.path.join(path, "pytorch_lora_weights.safetensors")
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"no pytorch_lora_weights.safetensors found in directory {path}")
        return resolved
    return path


def _lora_metadata_r_alpha(path):
    """Return (r, lora_alpha) from file-level lora_adapter_metadata, or (None, None)."""
    with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata()
    if not meta or "lora_adapter_metadata" not in meta:
        return None, None
    try:
        parsed = json.loads(meta["lora_adapter_metadata"])
    except json.JSONDecodeError as e:
        log.warning("could not parse lora_adapter_metadata JSON: %s", e)
        return None, None
    component = parsed.get("transformer")
    if not isinstance(component, dict):
        return None, None
    return component.get("r"), component.get("lora_alpha")


def load_lora_adapter(path):
    """
    Load a LoRA adapter from any of the supported key styles and normalize it
    into a canonical form.

    Returns {module_path: (A, B, layer_rank, alpha)} where
    A (r, in[, 1, 1]), B (out, r[, 1, 1]).
    """
    path = _resolve_lora_path(path)
    log.info("Loading LoRA adapter from %s", path)
    sd = safetensors.torch.load_file(path, device="cpu")

    meta_r, meta_alpha = _lora_metadata_r_alpha(path)

    raw = {}  # module_path -> {"down"/"up": tensor}
    unparsed = []
    for key, tensor in sd.items():
        if LORA_BIAS_RE.search(key):
            raise ValueError(
                f"bias LoRA tensors are not supported (found {key!r}); "
                f"refusing to continue")
        m = LORA_WEIGHT_RE.match(key)
        if m is None:
            unparsed.append(key)
            continue
        part = m.group("part")
        if m.group("dot") == ".":
            part = "down" if part == "down" else "up"
        else:
            part = "A" if part == "A" else "B"
        entry = raw.setdefault(m.group("module"), {})
        entry[part] = tensor

    if unparsed:
        log.warning("skipping %d tensor(s) with unrecognized keys: %s",
                    len(unparsed), ", ".join(sorted(unparsed)))

    adapters = {}
    for module_path, halves in raw.items():
        if "A" not in halves or "B" not in halves:
            raise ValueError(
                f"module {module_path!r} is missing lora_A or lora_B — cannot load")
        A, B = halves["A"], halves["B"]
        if A.shape[0] != B.shape[1]:
            raise ValueError(
                f"module {module_path!r}: A.shape[0]={A.shape[0]} != B.shape[1]={B.shape[1]}")
        layer_rank = int(A.shape[0])
        alpha = meta_alpha if meta_alpha is not None else layer_rank
        if meta_r is not None and layer_rank != meta_r:
            log.warning("module %s rank %d differs from adapter metadata r=%d",
                        module_path, layer_rank, meta_r)
        adapters[module_path] = (A, B, layer_rank, alpha)

    if not adapters:
        raise ValueError("no LoRA weights found in adapter")

    ranks = {a[2] for a in adapters.values()}
    log.info("Loaded %d LoRA layers (ranks=%s, alpha=%s)",
             len(adapters), sorted(ranks), meta_alpha if meta_alpha is not None else "=rank")
    return adapters


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def compute_delta(A, B, scaling):
    """
    Effective LoRA contribution `scaling * (B @ A)` in float32.

    Handles both plain linear layers (A (r, in), B (out, r)) and the 1x1 conv
    form (A (r, in, 1, 1), B (out, r, 1, 1)) used by SANA's ff convs.
    """
    if B.dim() == 4:
        delta = B[:, :, 0, 0].float() @ A[:, :, 0, 0].float()
        delta = delta.unsqueeze(-1).unsqueeze(-1)
    else:
        delta = B.float() @ A.float()
    return scaling * delta


def svd_extract(delta, rank, scaling):
    """
    delta: (out, in) or (out, in, 1, 1). Returns (A, B, used_rank) such that
    scaling * (B @ A) ~= delta (best rank-`used_rank` least-squares
    approximation, computed in float32 and cast back to delta's dtype).
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


# ---------------------------------------------------------------------------
# Core operations (pure state-dict math, unit-testable without a model)
# ---------------------------------------------------------------------------

def bake_lora_into_state_dict(base_sd, adapters):
    """Merge adapters into a copy of base_sd. Returns (sd, merged, skipped)."""
    sd = dict(base_sd)
    merged, skipped = 0, 0
    for module_path, (A, B, layer_rank, alpha) in adapters.items():
        weight_key = f"{module_path}.weight"
        if weight_key not in sd:
            log.warning("no %s in base checkpoint — skipping", weight_key)
            skipped += 1
            continue
        delta = compute_delta(A, B, alpha / layer_rank)
        sd[weight_key] = (sd[weight_key] + delta.to(sd[weight_key].dtype)).contiguous()
        merged += 1
    return sd, merged, skipped


def extract_lora_from_state_dicts(base_sd, finetuned_sd, rank, alpha, include_ffn=False):
    """
    Compute the finetuned-vs-base deltas over the LoRA target modules and
    SVD-factor them. Returns (adapters, skipped_keys, used_ranks) where
    adapters maps module_path -> (A, B).
    """
    n_blocks = get_num_blocks(base_sd)
    targets = CORE_TARGETS + (FFN_TARGETS if include_ffn else [])
    adapters = {}
    skipped = []
    used_ranks = []
    for i in range(n_blocks):
        for name in targets:
            key = f"transformer_blocks.{i}.{name}.weight"
            if key not in base_sd or key not in finetuned_sd:
                skipped.append(key)
                continue
            delta = finetuned_sd[key].float() - base_sd[key].float()
            A, B, used = svd_extract(delta, rank, alpha / rank)
            module_path = key.rsplit(".weight", 1)[0]
            adapters[module_path] = (A, B)
            used_ranks.append(used)
    return adapters, skipped, used_ranks


def resize_adapter(adapters, rank, alpha):
    """
    Reconstruct each adapter's effective delta and re-SVD it at the new
    rank/alpha. Returns (new_adapters, used_ranks) where new_adapters maps
    module_path -> (A, B).
    """
    new_adapters = {}
    used_ranks = []
    for module_path, (A, B, r_old, alpha_old) in adapters.items():
        delta = compute_delta(A, B, alpha_old / r_old)
        A2, B2, used = svd_extract(delta, rank, alpha / rank)
        new_adapters[module_path] = (A2, B2)
        used_ranks.append(used)
    return new_adapters, used_ranks


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_lora_adapter(out_dir, adapters, rank, alpha, target_modules, base_id=None):
    """
    Write a diffusers/trainer-loadable adapter: `transformer.<path>.lora.down/up.weight`
    keys plus `lora_adapter_metadata` file-level JSON. Also writes model_id.txt
    when base_id is given.
    """
    os.makedirs(out_dir, exist_ok=True)
    state_dict = {}
    for module_path, (A, B) in adapters.items():
        state_dict[f"transformer.{module_path}.lora.down.weight"] = A
        state_dict[f"transformer.{module_path}.lora.up.weight"] = B

    lora_adapter_metadata = {
        "transformer": {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": sorted(target_modules),
        }
    }
    metadata = {
        "format": "pt",
        "lora_adapter_metadata": json.dumps(lora_adapter_metadata, indent=2, sort_keys=True),
    }
    out_path = os.path.join(out_dir, "pytorch_lora_weights.safetensors")
    safetensors.torch.save_file(state_dict, out_path, metadata=metadata)
    log.info("Saved %d LoRA layers to %s", len(state_dict) // 2, out_path)

    if base_id is not None:
        with open(os.path.join(out_dir, "model_id.txt"), "w") as f:
            f.write(base_id)


def _target_suffixes(module_paths):
    """Reduce full module paths to the block-index-free suffix (e.g. attn1.to_q)."""
    return sorted({BLOCK_RE.sub("", p) for p in module_paths})


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_bake(args):
    base_sd = load_base_state_dict(args.base, is_video=args.is_video)
    adapters = load_lora_adapter(args.lora)
    sd, merged, skipped = bake_lora_into_state_dict(base_sd, adapters)
    safetensors.torch.save_file(sd, args.output)
    log.info("Baked %d LoRA layers into %s (%d layers skipped)",
             merged, args.output, skipped)


def cmd_extract(args):
    base_sd = load_base_state_dict(args.base, is_video=args.is_video)
    finetuned_path = _resolve_full_checkpoint(args.finetuned)
    log.info("Loading finetuned checkpoint from %s", finetuned_path)
    finetuned_sd = safetensors.torch.load_file(finetuned_path, device="cpu")
    finetuned_sd, stripped = _strip_known_prefixes(finetuned_sd)
    if stripped:
        log.info("stripped transformer./base_model.model. prefixes from finetuned keys")

    adapters, skipped, used_ranks = extract_lora_from_state_dicts(
        base_sd, finetuned_sd, args.rank, args.alpha, include_ffn=args.include_ffn)
    if skipped:
        log.warning("%d target key(s) missing from base or finetuned checkpoint: %s",
                    len(skipped), ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""))

    target_modules = CORE_TARGETS + (FFN_TARGETS if args.include_ffn else [])
    save_lora_adapter(args.output, adapters, args.rank, args.alpha,
                      target_modules=target_modules, base_id=args.base)

    min_used = min(used_ranks) if used_ranks else 0
    if min_used < args.rank:
        log.info("Extracted %d layers; min used rank = %d (below requested %d)",
                 len(adapters), min_used, args.rank)
    else:
        log.info("Extracted %d layers at rank %d", len(adapters), args.rank)


def cmd_resize(args):
    adapters = load_lora_adapter(args.lora)
    new_adapters, used_ranks = resize_adapter(adapters, args.rank, args.alpha)

    target_modules = (list(args.target_modules)
                      if args.target_modules
                      else _target_suffixes(new_adapters.keys()))
    save_lora_adapter(args.output, new_adapters, args.rank, args.alpha,
                      target_modules=target_modules)

    for module_path, (A, B, r_old, alpha_old) in adapters.items():
        log.info("resized %s: (rank=%d, alpha=%d) -> (rank=%d, alpha=%d)",
                 module_path, r_old, alpha_old, B.shape[1], args.alpha)
    min_used = min(used_ranks) if used_ranks else 0
    log.info("Resized %d layers to rank %d (min used rank %d)",
             len(new_adapters), args.rank, min_used)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_full_checkpoint(path):
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "transformer_gs*.safetensors")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one .safetensors transformer checkpoint in {path}, "
                f"found {len(candidates)}")
        return candidates[0]
    return path


def _strip_known_prefixes(sd):
    stripped = False
    new_sd = {}
    for key, value in sd.items():
        new_key = key
        if new_key.startswith("transformer.base_model.model."):
            new_key = new_key[len("transformer.base_model.model."):]
        elif new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        elif new_key.startswith("transformer."):
            new_key = new_key[len("transformer."):]
        if new_key != key:
            stripped = True
        new_sd[new_key] = value
    return new_sd, stripped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

RESIZE_HELP = (
    "NOTE: upscaling rank beyond the source rank cannot recover lost information — "
    "an SVD of an already-rank-r matrix yields zeros for components beyond r, so the "
    "result is functionally identical to the input. The real value of this operation "
    "is rank reduction and alpha adjustment."
)


def main():
    ap = argparse.ArgumentParser(prog="sana_lora_tool")
    sub = ap.add_subparsers(dest="command", required=True)

    for sp in (
        _build_bake_parser(sub),
        _build_extract_parser(sub),
        _build_resize_parser(sub),
    ):
        sp.add_argument("--is_video", action="store_true",
                        help="use SanaVideoTransformer3DModel instead of SanaTransformer2DModel")

    args = ap.parse_args()
    dispatch = {"bake": cmd_bake, "extract": cmd_extract, "resize": cmd_resize}
    dispatch[args.command](args)


def _build_bake_parser(sub):
    sp = sub.add_parser("bake", help="merge a LoRA adapter into a full transformer checkpoint")
    sp.add_argument("--base", default=DEFAULT_BASE_ID,
                    help="base SANA model id (default: %(default)s)")
    sp.add_argument("--lora", required=True,
                    help="LoRA adapter directory or .safetensors file")
    sp.add_argument("--output", required=True,
                    help="output path of the merged transformer .safetensors checkpoint")
    return sp


def _build_extract_parser(sub):
    sp = sub.add_parser("extract",
                        help="recover a LoRA adapter from a finetuned full checkpoint")
    sp.add_argument("--base", default=DEFAULT_BASE_ID,
                    help="base SANA model id (default: %(default)s)")
    sp.add_argument("--finetuned", required=True,
                    help="finetuned transformer .safetensors file (or directory containing "
                         "a transformer_gs*.safetensors)")
    sp.add_argument("--output", required=True,
                    help="output directory for the recovered adapter")
    sp.add_argument("--rank", type=int, default=32, help="target rank (default: %(default)s)")
    sp.add_argument("--alpha", type=int, default=32, help="target lora_alpha (default: %(default)s)")
    sp.add_argument("--include_ffn", action="store_true",
                    help="also LoRA the ff.conv_point / ff.conv_inverted 1x1 convs")
    return sp


def _build_resize_parser(sub):
    sp = sub.add_parser("resize", help="change the rank and/or alpha of an existing LoRA adapter",
                        epilog=RESIZE_HELP)
    sp.add_argument("--lora", required=True,
                    help="LoRA adapter directory or .safetensors file")
    sp.add_argument("--output", required=True,
                    help="output directory for the resized adapter")
    sp.add_argument("--rank", type=int, default=32, help="target rank (default: %(default)s)")
    sp.add_argument("--alpha", type=int, default=32, help="target lora_alpha (default: %(default)s)")
    sp.add_argument("--target_modules", nargs="+", default=None,
                    help="override target_modules recorded in metadata "
                         "(default: derived from adapter module paths)")
    return sp


if __name__ == "__main__":
    main()
