"""
SANA text encoding utilities.
Supports T5, Gemma, and Qwen text encoder branches as used in the SANA reference implementation.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def encode_sana_text(
    tokenizer,
    text_encoder: nn.Module,
    captions: list[str],
    config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a list of caption strings using the appropriate text encoder branch
    (T5, Gemma, or Qwen — selected by config.text_encoder.text_encoder_name).

    Returns:
        y      : (B, 1, N, C) text embeddings
        y_mask : (B, 1, 1, N) attention mask
    """
    encoder_name = config.text_encoder.text_encoder_name
    max_length = config.text_encoder.model_max_length

    if "T5" in encoder_name:
        txt_tokens = tokenizer(
            captions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
        y_mask = txt_tokens.attention_mask[:, None, None]

    elif "gemma" in encoder_name.lower() or "Qwen" in encoder_name:
        if not config.text_encoder.chi_prompt:
            max_length_all = max_length
            prompts = captions
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            prompts = [chi_prompt + c for c in captions]
            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_sys_prompt_tokens + max_length - 2  # [bos], [_]

        txt_tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Select: first bos token and the last (max_length - 1) tokens
        select_index = [0] + list(range(-max_length + 1, 0))
        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
            :, :, select_index
        ]
        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]

    else:
        raise ValueError(f"Unsupported text encoder: {encoder_name}. Expected T5, Gemma, or Qwen.")

    return y, y_mask


def encode_sana_null_text(
    tokenizer,
    text_encoder: nn.Module,
    config,
    device: torch.device,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Encodes an empty/null caption for unconditional guidance.
    If cache_path is given, writes the result to disk on first call and reads
    from disk on subsequent calls (avoids re-encoding every validation pass).
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)

    with torch.no_grad():
        null_embedding = encode_sana_text(tokenizer, text_encoder, [""], config, device)[0]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(null_embedding, cache_path)

    return null_embedding

