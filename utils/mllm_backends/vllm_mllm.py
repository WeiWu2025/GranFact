#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Embedded vLLM backend for local multimodal generation.

This backend does not start an OpenAI-compatible server. It creates a local
``vllm.LLM`` engine inside the benchmark process and feeds PIL images through
vLLM's offline multimodal request format.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _processor_has_usable_chat_template(processor) -> bool:
    if processor is None or not hasattr(processor, "apply_chat_template"):
        return False
    direct_template = getattr(processor, "chat_template", None)
    if isinstance(direct_template, str) and direct_template.strip():
        return True
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    return isinstance(tokenizer_template, str) and bool(tokenizer_template.strip())


def _build_qwen_vl_fallback_prompt(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _finish_meta_from_vllm_output(output, max_new_tokens: int) -> Dict[str, Any]:
    token_ids = getattr(output, "token_ids", None) or []
    out_len = len(token_ids)
    finish_reason = str(getattr(output, "finish_reason", None) or "unknown")
    return {
        "output_token_count": int(out_len),
        "finish_reason": finish_reason,
        "truncated_by_max_new_tokens": bool(finish_reason == "length" or out_len >= int(max_new_tokens)),
    }


class VLLMMLLMBackend:
    name = "vllm_mllm"
    supports_batch = False

    def __init__(
        self,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        max_model_len: int = 0,
        limit_mm_per_prompt_image: int = 1,
        gpus: Optional[List[int]] = None,
    ):
        self.tensor_parallel_size = max(1, int(tensor_parallel_size))
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.dtype = str(dtype or "auto")
        self.max_model_len = int(max_model_len or 0)
        self.limit_mm_per_prompt_image = max(1, int(limit_mm_per_prompt_image or 1))
        self.gpus = list(gpus or [])
        self.processor = None

    def load_model_and_processor(
        self,
        model_ckpt: str,
        torch_dtype: str,
        device=None,
        base_model_ckpt: Optional[str] = None,
        processor_ckpt: Optional[str] = None,
    ):
        # vLLM reads visible devices at engine init time. In vLLM local-engine
        # mode, cfg.gpus denotes devices owned by one tensor-parallel engine.
        if self.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in self.gpus)

        from transformers import AutoProcessor  # noqa: PLC0415
        from vllm import LLM  # noqa: PLC0415

        proc_ckpt = (processor_ckpt or "").strip() or model_ckpt
        try:
            self.processor = AutoProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)
        except Exception:
            self.processor = None

        kwargs: Dict[str, Any] = {
            "model": model_ckpt,
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "limit_mm_per_prompt": {"image": self.limit_mm_per_prompt_image},
        }
        if self.max_model_len > 0:
            kwargs["max_model_len"] = self.max_model_len

        llm = LLM(**kwargs)
        return llm, self.processor

    def _build_prompt(self, processor, image: Image.Image, prompt: str) -> str:
        if _processor_has_usable_chat_template(processor):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return _build_qwen_vl_fallback_prompt(prompt)

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
        from vllm import SamplingParams  # noqa: PLC0415

        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature) if bool(do_sample) else 0.0,
            "top_p": float(top_p),
        }
        sampling_params = SamplingParams(**sampling_kwargs)
        prompt_text = self._build_prompt(processor, image, prompt)
        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        }
        try:
            outputs = model.generate([request], sampling_params, use_tqdm=False)
        except TypeError:
            outputs = model.generate([request], sampling_params)

        if not outputs or not getattr(outputs[0], "outputs", None):
            return "", {
                "output_token_count": 0,
                "finish_reason": "empty",
                "truncated_by_max_new_tokens": False,
            }
        out = outputs[0].outputs[0]
        text = str(getattr(out, "text", "") or "").strip()
        return text, _finish_meta_from_vllm_output(out, max_new_tokens)


def vllm_runtime_flags() -> Dict[str, Any]:
    return {"has_vllm_mllm_backend": True}
