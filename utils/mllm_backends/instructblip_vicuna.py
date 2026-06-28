#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Special backend for InstructBLIP-Vicuna models.

InstructBLIP is older than chat-template based MLLMs. Routing it through the
generic chat/generic VLM path can lead to degenerate repeated-token outputs
such as long runs of ``666...``. This backend uses the model-specific HF classes
and avoids prompt-length slicing that is unsafe for encoder-decoder style VLMs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PIL import Image
import torch

from .common import compute_finish_meta, get_pad_token_id, get_torch_dtype, move_inputs_to_device, sanitize_generation_kwargs

try:
    from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
    _HAS_INSTRUCTBLIP = True
    _INSTRUCTBLIP_IMPORT_ERROR = None
except Exception as _e:
    InstructBlipForConditionalGeneration = None
    InstructBlipProcessor = None
    _HAS_INSTRUCTBLIP = False
    _INSTRUCTBLIP_IMPORT_ERROR = repr(_e)


def is_instructblip_vicuna(model_name: str, model_ckpt: str) -> bool:
    s = f"{model_name or ''} {model_ckpt or ''}".lower()
    return "instructblip" in s and "vicuna" in s


def _normalize_for_prefix_match(s: str) -> str:
    return " ".join(str(s or "").strip().split()).casefold()


def _strip_prompt_echo(text: str, prompt: str) -> str:
    """Remove a leading prompt echo from decoded InstructBLIP output.

    InstructBLIP-Vicuna may decode the textual instruction together with the
    answer. Token-level slicing is not safe enough for this backend, so we do a
    conservative text-level prefix cleanup only. The function intentionally does
    not remove prompt-like text in the middle of the response.
    """
    raw = str(text or "").strip()
    p = str(prompt or "").strip()
    if not raw or not p:
        return raw

    # Fast path: exact prefix with the original prompt string.
    if raw.startswith(p):
        return raw[len(p):].lstrip(" \t\r\n:：-–—")

    # Conservative normalized-prefix path: tolerate case and whitespace drift.
    norm_prompt = _normalize_for_prefix_match(p)
    norm_chars = []
    raw_cut = 0
    for idx, ch in enumerate(raw):
        if ch.isspace():
            if norm_chars and norm_chars[-1] != " ":
                norm_chars.append(" ")
        else:
            norm_chars.append(ch.casefold())
        cur = "".join(norm_chars).strip()
        if cur == norm_prompt:
            raw_cut = idx + 1
            break
        if len(cur) > len(norm_prompt) + 8 or (cur and not norm_prompt.startswith(cur)):
            break

    if raw_cut > 0:
        return raw[raw_cut:].lstrip(" \t\r\n:：-–—")

    return raw


class InstructBlipVicunaBackend:
    name = "instructblip_vicuna"
    supports_batch = False

    def load_model_and_processor(
        self,
        model_ckpt: str,
        torch_dtype: str,
        device: torch.device,
        base_model_ckpt: Optional[str] = None,
        processor_ckpt: Optional[str] = None,
    ):
        if not _HAS_INSTRUCTBLIP:
            raise RuntimeError(
                "InstructBLIP backend selected, but transformers does not expose "
                "InstructBlipForConditionalGeneration/InstructBlipProcessor. "
                f"Import error: {_INSTRUCTBLIP_IMPORT_ERROR}"
            )
        dtype = get_torch_dtype(torch_dtype)
        proc_ckpt = (processor_ckpt or "").strip() or model_ckpt
        processor = InstructBlipProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_ckpt,
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        pad_id = get_pad_token_id(model, processor)
        if pad_id is not None:
            try:
                model.generation_config.pad_token_id = int(pad_id)
            except Exception:
                pass
        model.to(device)
        model.eval()
        return model, processor

    @torch.inference_mode()
    def generate_one(
        self,
        model,
        processor,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_pixels: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        # InstructBLIP processors generally do not support max_pixels. Image
        # resizing is handled by the caller's OOM policy if needed.
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, getattr(model, "device", None))
        gen_kwargs = sanitize_generation_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            model=model,
            processor=processor,
        )
        gen_ids = model.generate(**inputs, **gen_kwargs)
        # Do NOT slice by input prompt length here. InstructBLIP generation is
        # not the same as decoder-only chat models where output = prompt + new.
        if hasattr(processor, "batch_decode"):
            out = processor.batch_decode(gen_ids, skip_special_tokens=True)
            text = (out[0] if out else "").strip()
        else:
            text = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        text = _strip_prompt_echo(text, prompt)
        return text, compute_finish_meta(gen_ids, model, processor, max_new_tokens)


def instructblip_runtime_flags() -> Dict[str, Any]:
    return {
        "has_instructblip_class": _HAS_INSTRUCTBLIP,
        "instructblip_import_error": _INSTRUCTBLIP_IMPORT_ERROR,
    }
