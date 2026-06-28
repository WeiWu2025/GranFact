#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Common helpers for local multimodal LLM backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def get_torch_dtype(name: str):
    name = (name or "").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


def get_eos_token_ids(model, processor) -> List[int]:
    eos_ids: List[int] = []
    for obj in [getattr(model, "generation_config", None), getattr(model, "config", None)]:
        try:
            v = getattr(obj, "eos_token_id", None)
            if isinstance(v, int):
                eos_ids.append(v)
            elif isinstance(v, (list, tuple)):
                eos_ids.extend([int(x) for x in v if isinstance(x, int)])
        except Exception:
            pass
    try:
        tokenizer = getattr(processor, "tokenizer", None)
        tid = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        if isinstance(tid, int):
            eos_ids.append(tid)
    except Exception:
        pass
    return sorted(set(eos_ids))


def get_pad_token_id(model, processor) -> Optional[int]:
    for obj in [getattr(model, "generation_config", None), getattr(model, "config", None)]:
        try:
            v = getattr(obj, "pad_token_id", None)
            if isinstance(v, int):
                return v
        except Exception:
            pass
    try:
        tokenizer = getattr(processor, "tokenizer", None)
        tid = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if isinstance(tid, int):
            return tid
    except Exception:
        pass
    eos_ids = get_eos_token_ids(model, processor)
    return eos_ids[0] if eos_ids else None


def processor_has_usable_chat_template(processor) -> bool:
    if not hasattr(processor, "apply_chat_template"):
        return False
    direct_template = getattr(processor, "chat_template", None)
    if isinstance(direct_template, str) and direct_template.strip():
        return True
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    return isinstance(tokenizer_template, str) and bool(tokenizer_template.strip())


def normalize_vlm_processor_outputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(inputs)
    pixel_values = out.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 5:
        try:
            if int(pixel_values.shape[1]) == 1:
                out["pixel_values"] = pixel_values.squeeze(1)
        except Exception:
            pass
    return out


def move_inputs_to_device(inputs: Dict[str, Any], device) -> Dict[str, Any]:
    if device is None:
        return inputs
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def compute_finish_meta(gen_ids, model, processor, max_new_tokens: int) -> Dict[str, Any]:
    out_len = int(gen_ids.shape[-1]) if hasattr(gen_ids, "shape") else 0
    eos_ids = get_eos_token_ids(model, processor)
    last_id = None
    try:
        if out_len > 0:
            last_id = int(gen_ids[0, -1].item()) if gen_ids.ndim >= 2 else int(gen_ids[-1].item())
    except Exception:
        last_id = None
    ended_with_eos = (last_id in eos_ids) if (last_id is not None and eos_ids) else False
    truncated = (out_len >= int(max_new_tokens)) and (not ended_with_eos)
    return {
        "output_token_count": out_len,
        "finish_reason": "eos" if ended_with_eos else ("length" if truncated else "unknown"),
        "truncated_by_max_new_tokens": bool(truncated),
    }


def sanitize_generation_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    model=None,
    processor=None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if bool(do_sample):
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    eos_ids = get_eos_token_ids(model, processor) if model is not None else []
    pad_id = get_pad_token_id(model, processor) if model is not None else None
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    if pad_id is not None:
        kwargs["pad_token_id"] = int(pad_id)
    return kwargs
