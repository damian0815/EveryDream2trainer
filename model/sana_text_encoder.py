"""
SANA text encoding utilities using 🤗 diffusers.

Adapted from SanaPipeline._get_gemma_prompt_embeds (diffusers).
No SANA repo clone required.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def encode_prompts(
    tokenizer,
    text_encoder: nn.Module,
    captions: list[str],
    device: torch.device,
    *,
    max_sequence_length: int = 300,
    complex_human_instruction: Optional[list[str]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a list of captions using the Gemma text encoder.

    Args:
        tokenizer           : GemmaTokenizerFast from the SanaPipeline.
        text_encoder        : Gemma2 model from the SanaPipeline (frozen).
        captions            : List of B caption strings.
        device              : Target device for the returned tensors.
        max_sequence_length : Gemma token budget (default 300, matching SANA defaults).
        complex_human_instruction : Optional list of system-prompt lines prepended to
                                    every caption (mirrors SANA's chi_prompt feature).
        dtype               : Float dtype for the output embeddings.

    Returns:
        prompt_embeds         (B, N, C) — text embeddings ready for
                              SanaTransformer2DModel.forward(encoder_hidden_states=...)
        prompt_attention_mask (B, N)    — boolean-style mask (1=keep, 0=pad) ready for
                              SanaTransformer2DModel.forward(encoder_attention_mask=...)
    """
    tokenizer.padding_side = "right"

    if complex_human_instruction:
        chi_prompt = "\n".join(complex_human_instruction)
        texts = [chi_prompt + c for c in captions]
        num_chi_tokens = len(tokenizer.encode(chi_prompt))
        max_length_all = num_chi_tokens + max_sequence_length - 2  # [bos] + [pad]
    else:
        texts = captions
        max_length_all = max_sequence_length

    text_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length_all,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, attention_mask


def encode_null_prompt(
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device,
    *,
    max_sequence_length: int = 300,
    dtype: torch.dtype = torch.bfloat16,
    cache_path: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes an empty caption for classifier-free guidance (null conditioning).

    If cache_path is given the result is written to disk on first call and read
    back on all subsequent calls, avoiding redundant encoder passes.

    Returns:
        null_embeds (1, N, C) and null_mask (1, N).
    """
    if cache_path and os.path.exists(cache_path):
        saved = torch.load(cache_path, map_location=device)
        return saved["embeds"], saved["mask"]

    with torch.no_grad():
        null_embeds, null_mask = encode_prompts(
            tokenizer, text_encoder, [""], device,
            max_sequence_length=max_sequence_length, dtype=dtype,
        )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"embeds": null_embeds, "mask": null_mask}, cache_path)

    return null_embeds, null_mask
