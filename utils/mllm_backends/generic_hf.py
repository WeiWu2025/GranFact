#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic HuggingFace VLM backend used by current working models."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor

from .common import (
    compute_finish_meta,
    get_torch_dtype,
    move_inputs_to_device,
    normalize_vlm_processor_outputs,
    processor_has_usable_chat_template,
    sanitize_generation_kwargs,
)

_QWEN3_VL_IMPORT_ERROR = None
try:
    from transformers import Qwen3VLForConditionalGeneration
    _HAS_QWEN3_VL_CLASS = True
except Exception as _e:
    Qwen3VLForConditionalGeneration = None
    _HAS_QWEN3_VL_CLASS = False
    _QWEN3_VL_IMPORT_ERROR = repr(_e)

_AUTO_V2S_IMPORT_ERROR = None
try:
    from transformers import AutoModelForVision2Seq
    _HAS_AUTO_V2S = True
except Exception as _e:
    AutoModelForVision2Seq = None
    _HAS_AUTO_V2S = False
    _AUTO_V2S_IMPORT_ERROR = repr(_e)

_AUTO_ITTT_IMPORT_ERROR = None
try:
    from transformers import AutoModelForImageTextToText
    _HAS_AUTO_ITTT = True
except Exception as _e:
    AutoModelForImageTextToText = None
    _HAS_AUTO_ITTT = False
    _AUTO_ITTT_IMPORT_ERROR = repr(_e)

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_VL_UTILS = True
except Exception:
    process_vision_info = None
    _HAS_QWEN_VL_UTILS = False


def is_qwen3_vl(ckpt_or_name: str) -> bool:
    s = (ckpt_or_name or "").lower()
    return ("qwen3" in s and "vl" in s)


def looks_like_qwen3_vl_model(model_ckpt: str) -> bool:
    if is_qwen3_vl(model_ckpt):
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        if "qwen3" in model_type and "vl" in model_type:
            return True
        archs = getattr(cfg, "architectures", None)
        if isinstance(archs, (list, tuple)):
            joined = " ".join(str(x).lower() for x in archs)
            if "qwen3" in joined and "vl" in joined:
                return True
    except Exception:
        pass
    return False


def load_auto_vision_language_model(ckpt: str, dtype):
    errors: List[str] = []
    if _HAS_AUTO_ITTT:
        try:
            return AutoModelForImageTextToText.from_pretrained(ckpt, dtype=dtype, device_map=None, trust_remote_code=True)
        except Exception as e:
            errors.append(f"AutoModelForImageTextToText.from_pretrained failed: {repr(e)}")
    else:
        errors.append(f"AutoModelForImageTextToText import unavailable: {_AUTO_ITTT_IMPORT_ERROR}")
    if _HAS_AUTO_V2S:
        try:
            return AutoModelForVision2Seq.from_pretrained(ckpt, dtype=dtype, device_map=None, trust_remote_code=True)
        except Exception as e:
            errors.append(f"AutoModelForVision2Seq.from_pretrained failed: {repr(e)}")
    else:
        errors.append(f"AutoModelForVision2Seq import unavailable: {_AUTO_V2S_IMPORT_ERROR}")
    raise RuntimeError(
        "Failed to load model via both AutoModelForImageTextToText and AutoModelForVision2Seq. "
        f"ckpt={ckpt}. Diagnostics: " + " | ".join(errors)
    )


def is_lora_adapter_dir(path: str) -> bool:
    return bool(path and os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")) and os.path.exists(os.path.join(path, "adapter_model.safetensors")))


def read_lora_base_model_from_adapter(path: str) -> Optional[str]:
    import json

    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = obj.get("base_model_name_or_path") if isinstance(obj, dict) else None
        return v.strip() if isinstance(v, str) and v.strip() else None
    except Exception:
        return None


class GenericHFBackend:
    name = "generic_hf"
    supports_batch = True

    def load_model_and_processor(
        self,
        model_ckpt: str,
        torch_dtype: str,
        device: torch.device,
        base_model_ckpt: Optional[str] = None,
        processor_ckpt: Optional[str] = None,
    ):
        dtype = get_torch_dtype(torch_dtype)
        use_lora = is_lora_adapter_dir(model_ckpt)
        if use_lora:
            if not _HAS_PEFT:
                raise RuntimeError("LoRA checkpoint detected, but peft is not installed. Please install peft.")
            base_ckpt = (base_model_ckpt or "").strip() or read_lora_base_model_from_adapter(model_ckpt)
            if not base_ckpt:
                raise RuntimeError("LoRA checkpoint detected but base model is unknown. Please pass --base-model-ckpt.")
            proc_ckpt = (processor_ckpt or "").strip() or base_ckpt
            processor = AutoProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)
            if looks_like_qwen3_vl_model(base_ckpt) and _HAS_QWEN3_VL_CLASS:
                model = Qwen3VLForConditionalGeneration.from_pretrained(base_ckpt, dtype=dtype, device_map=None, trust_remote_code=True)
            else:
                model = load_auto_vision_language_model(base_ckpt, dtype=dtype)
            model = PeftModel.from_pretrained(model, model_ckpt)
            model = model.merge_and_unload()
        else:
            proc_ckpt = (processor_ckpt or "").strip() or model_ckpt
            processor = AutoProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)
            if looks_like_qwen3_vl_model(model_ckpt) and _HAS_QWEN3_VL_CLASS:
                model = Qwen3VLForConditionalGeneration.from_pretrained(model_ckpt, dtype=dtype, device_map=None, trust_remote_code=True)
            else:
                model = load_auto_vision_language_model(model_ckpt, dtype=dtype)
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
        has_template = processor_has_usable_chat_template(processor)
        use_qwen_path = is_qwen3_vl(getattr(model, "name_or_path", "")) or has_template
        if use_qwen_path and has_template:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            proc_kwargs = dict(padding=True, return_tensors="pt")
            if max_pixels is not None:
                proc_kwargs["max_pixels"] = int(max_pixels)
            if _HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, **proc_kwargs)
            else:
                inputs = processor(text=[text], images=[image], **proc_kwargs)
        else:
            proc_kwargs = dict(return_tensors="pt")
            if max_pixels is not None:
                proc_kwargs["max_pixels"] = int(max_pixels)
            inputs = normalize_vlm_processor_outputs(processor(images=image, text=prompt, **proc_kwargs))
        inputs = move_inputs_to_device(inputs, getattr(model, "device", None))
        gen_kwargs = sanitize_generation_kwargs(
            max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, model=model, processor=processor
        )
        gen_ids = model.generate(**inputs, **gen_kwargs)
        if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            in_len = inputs["input_ids"].shape[-1]
            if gen_ids.shape[-1] >= in_len:
                gen_ids = gen_ids[:, in_len:]
        if hasattr(processor, "batch_decode"):
            out = processor.batch_decode(gen_ids, skip_special_tokens=True)
            text = (out[0] if out else "").strip()
        else:
            text = processor.decode(gen_ids[0], skip_special_tokens=True).strip()
        return text, compute_finish_meta(gen_ids, model, processor, max_new_tokens)


def generic_runtime_flags() -> Dict[str, Any]:
    return {
        "has_qwen_vl_utils": _HAS_QWEN_VL_UTILS,
        "has_qwen3_vl_class": _HAS_QWEN3_VL_CLASS,
        "has_auto_model_for_image_text_to_text": _HAS_AUTO_ITTT,
        "has_auto_model_for_vision2seq": _HAS_AUTO_V2S,
        "qwen3_vl_import_error": _QWEN3_VL_IMPORT_ERROR,
        "auto_vision2seq_import_error": _AUTO_V2S_IMPORT_ERROR,
        "auto_image_text_to_text_import_error": _AUTO_ITTT_IMPORT_ERROR,
    }
