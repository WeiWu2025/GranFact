#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Backend selection for local multimodal generation."""

from __future__ import annotations

from typing import Any, Dict

from .generic_hf import GenericHFBackend, generic_runtime_flags
from .instructblip_vicuna import InstructBlipVicunaBackend, instructblip_runtime_flags, is_instructblip_vicuna
from .vllm_mllm import VLLMMLLMBackend, vllm_runtime_flags


def get_backend(model_name: str, model_ckpt: str):
    if is_instructblip_vicuna(model_name, model_ckpt):
        return InstructBlipVicunaBackend()
    return GenericHFBackend()


def backend_runtime_flags() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(generic_runtime_flags())
    out.update(instructblip_runtime_flags())
    out.update(vllm_runtime_flags())
    return out
