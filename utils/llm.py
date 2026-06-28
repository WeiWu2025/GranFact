#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.common import _eprint

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class LocalChatModel:
    def __init__(self, ckpt: str, logical_name: str):
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed; cannot load local HF model.")
        if not ckpt:
            raise RuntimeError(f"checkpoint path is empty for model {logical_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.ckpt = ckpt
        self.logical_name = logical_name

        _eprint(f"[model:{logical_name}] loading tokenizer + model from: {ckpt}")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        _eprint(f"[model:{logical_name}] ready.")

    def _apply_chat_template(self, messages: List[Dict[str, str]]):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        guided_json: Optional[Dict[str, Any]] = None,
    ) -> str:
        if guided_json is not None:
            _eprint(
                f"[model:{self.logical_name}] guided_json requested but transformers backend "
                "does not support constrained decoding here; falling back to normal generation."
            )
        inputs = self._apply_chat_template(messages)
        device = next(self.model.parameters()).device

        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(device)
            input_len = input_ids.shape[1]
            gen_kw: Dict[str, Any] = {"input_ids": input_ids}
        else:
            gen_kw = {k: v.to(device) for k, v in inputs.items()}
            input_len = gen_kw["input_ids"].shape[1]

        pad_id = getattr(self.tokenizer, "pad_token_id", None) or self.tokenizer.eos_token_id

        with torch.no_grad():
            out_ids = self.model.generate(
                **gen_kw,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0,
                temperature=float(temperature),
                top_p=float(top_p),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )

        return self.tokenizer.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()


class VLLMChatModel:
    def __init__(
        self,
        ckpt: str,
        logical_name: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        max_model_len: int = 0,
    ):
        if not ckpt:
            raise RuntimeError(f"checkpoint path is empty for model {logical_name}")

        from transformers import AutoTokenizer  # noqa: PLC0415
        from vllm import LLM  # noqa: PLC0415

        self.ckpt = ckpt
        self.logical_name = logical_name

        _eprint(f"[model:{logical_name}] loading tokenizer from: {ckpt}")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

        kwargs: Dict[str, Any] = {
            "model": ckpt,
            "trust_remote_code": True,
            "tensor_parallel_size": max(1, int(tensor_parallel_size)),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": str(dtype or "auto"),
        }
        if int(max_model_len or 0) > 0:
            kwargs["max_model_len"] = int(max_model_len)

        _eprint(
            f"[model:{logical_name}] loading vLLM engine from: {ckpt} "
            f"tp={kwargs['tensor_parallel_size']} dtype={kwargs['dtype']}"
        )
        self.llm = LLM(**kwargs)
        self.last_generation_info: Dict[str, Any] = {}
        _eprint(f"[model:{logical_name}] vLLM ready.")

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        guided_json: Optional[Dict[str, Any]] = None,
    ) -> str:
        from vllm import SamplingParams  # noqa: PLC0415

        prompt = self._apply_chat_template(messages)
        self.last_generation_info = {
            "guided_json_requested": bool(guided_json is not None),
            "guided_json_effective": False,
            "guided_json_backend": None,
            "guided_json_warning": None,
        }
        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if guided_json is not None:
            try:
                from vllm.sampling_params import StructuredOutputsParams  # noqa: PLC0415

                sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=guided_json)
                self.last_generation_info.update({
                    "guided_json_effective": True,
                    "guided_json_backend": "vllm_structured_outputs_json",
                })
            except Exception as e:
                warning = (
                    "vLLM StructuredOutputsParams(json=...) is unavailable; "
                    f"falling back to normal generation: {e!r}"
                )
                _eprint(f"[model:{self.logical_name}] {warning}")
                self.last_generation_info.update({
                    "guided_json_effective": False,
                    "guided_json_backend": "vllm_structured_outputs_json",
                    "guided_json_warning": warning,
                })
        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError as e:
            if guided_json is None:
                raise
            warning = (
                "vLLM SamplingParams does not accept structured_outputs; "
                f"falling back to normal generation: {e!r}"
            )
            _eprint(
                f"[model:{self.logical_name}] {warning}"
            )
            sampling_kwargs.pop("structured_outputs", None)
            self.last_generation_info.update({
                "guided_json_effective": False,
                "guided_json_warning": warning,
            })
            sampling_params = SamplingParams(**sampling_kwargs)
        try:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        except TypeError:
            outputs = self.llm.generate([prompt], sampling_params)

        if not outputs or not getattr(outputs[0], "outputs", None):
            return ""
        return str(outputs[0].outputs[0].text or "").strip()


def build_chat_model(cfg: Any):
    backend = str(getattr(cfg, "llm_backend", "transformers") or "transformers").strip().lower()
    if backend == "vllm":
        return VLLMChatModel(
            cfg.judge_ckpt,
            cfg.judge_model,
            tensor_parallel_size=getattr(cfg, "vllm_tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(cfg, "vllm_gpu_memory_utilization", 0.9),
            dtype=getattr(cfg, "vllm_dtype", "auto"),
            max_model_len=getattr(cfg, "vllm_max_model_len", 0),
        )
    if backend == "transformers":
        return LocalChatModel(cfg.judge_ckpt, cfg.judge_model)
    raise ValueError(f"unknown llm_backend: {backend!r}")
