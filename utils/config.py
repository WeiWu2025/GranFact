#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.common import _eprint, load_json

EXTRACTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    # Release builds intentionally avoid machine-specific checkpoint paths.
    # Deprecated extractor args are retained only for CLI compatibility.
}

JUDGE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Add local judge-model presets here if desired, or pass --judge-ckpt explicitly.
}

_MANIFEST_CORE_FIELDS = [
    ("extractor_model", lambda m: m.get("extractor_model")),
    ("extractor_ckpt", lambda m: m.get("extractor_ckpt")),
    ("judge_model", lambda m: m.get("judge_model")),
    ("judge_ckpt", lambda m: m.get("judge_ckpt")),
    ("temperature", lambda m: m.get("extractor_params", {}).get("temperature")),
    ("top_p", lambda m: m.get("extractor_params", {}).get("top_p")),
    ("max_new_tokens", lambda m: m.get("extractor_params", {}).get("max_new_tokens")),
    ("judge_temperature", lambda m: m.get("judge_params", {}).get("temperature")),
    ("judge_top_p", lambda m: m.get("judge_params", {}).get("top_p")),
    ("judge_max_new_tokens", lambda m: m.get("judge_params", {}).get("max_new_tokens")),
    ("gt_mode", lambda m: m.get("gt_mode")),
    ("candidate_strategy", lambda m: m.get("candidate_strategy")),
    ("llm_backend", lambda m: m.get("llm_backend")),
    ("vllm_tensor_parallel_size", lambda m: m.get("vllm_params", {}).get("tensor_parallel_size")),
    ("vllm_gpu_memory_utilization", lambda m: m.get("vllm_params", {}).get("gpu_memory_utilization")),
    ("vllm_dtype", lambda m: m.get("vllm_params", {}).get("dtype")),
    ("vllm_max_model_len", lambda m: m.get("vllm_params", {}).get("max_model_len")),
    ("guided_json_mode", lambda m: m.get("guided_json", {}).get("mode")),
]


def resolve_ckpt(model_name: str, ckpt: Optional[str], presets: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if ckpt:
        return ckpt
    if not model_name:
        return None
    preset = presets.get(model_name, {})
    return preset.get("ckpt")


def _load_priority_image_paths(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    cli_items = getattr(args, "priority_image_path", None) or []
    for x in cli_items:
        s = str(x).strip()
        if s:
            out.append(s)

    file_path = getattr(args, "priority_image_paths_file", None)
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s)

    deduped: List[str] = []
    seen = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


@dataclass
class ExtractAndMatchConfig:
    run_root: str
    run_id: str
    dataset_root_override: Optional[str]

    extractor_model: str
    extractor_ckpt: str

    judge_model: str
    judge_ckpt: str

    temperature: float
    top_p: float
    max_new_tokens: int

    judge_temperature: float
    judge_top_p: float
    judge_max_new_tokens: int

    debug: bool
    mode: str
    num_workers: int

    priority_image_paths: List[str]
    priority_image_paths_file: Optional[str]
    only_priority: bool
    progress_interval_sec: float
    gt_mode: str

    # Candidate selection strategy
    candidate_strategy: str  # Release supports only "pairwise".

    llm_backend: str
    vllm_tensor_parallel_size: int
    vllm_gpu_memory_utilization: float
    vllm_dtype: str
    vllm_max_model_len: int
    guided_json_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument(
        "--dataset-root-override",
        type=str,
        default=None,
        help=(
            "Optional current-machine testset root. When set, Stage1 source_json/image_path "
            "absolute paths are relocated by preserving their path relative to the old testset root."
        ),
    )

    # kept for backward compatibility only
    p.add_argument(
        "--extractor-model",
        type=str,
        default=None,
        help="Deprecated compatibility arg. Ignored. All stages use --judge-model.",
    )
    p.add_argument(
        "--extractor-ckpt",
        type=str,
        default=None,
        help="Deprecated compatibility arg. Used only as fallback if --judge-ckpt is not provided.",
    )

    p.add_argument("--judge-model", type=str, required=True)
    p.add_argument("--judge-ckpt", type=str, default=None)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=2048)

    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--judge-top-p", type=float, default=1.0)
    p.add_argument("--judge-max-new-tokens", type=int, default=2048)

    p.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mode", type=str, default="resume", choices=["resume", "force"])
    p.add_argument("--num-workers", type=int, default=1,
                   help="compatibility arg; current implementation runs single-process.")

    p.add_argument(
        "--priority-image-path",
        action="append",
        default=[],
        help="Image path to prioritize; can be passed multiple times.",
    )
    p.add_argument(
        "--priority-image-paths-file",
        type=str,
        default=None,
        help="Text file with one image path per line. Earlier lines have higher priority.",
    )
    p.add_argument(
        "--only-priority",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--progress-interval-sec", type=float, default=5.0)
    p.add_argument(
        "--llm-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="LLM inference backend for Stage-2 judge/extractor calls.",
    )
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=4)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--vllm-dtype", type=str, default="auto")
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="0 means let vLLM infer/use its default max_model_len.",
    )
    p.add_argument(
        "--gt-mode",
        type=str,
        default="bucket",
        choices=["bucket", "atomic_with_contraction"],
        help=(
            "GT interpretation mode. 'bucket' is the default and only implemented mode. "
            "'atomic_with_contraction' is a reserved placeholder and will raise NotImplementedError."
        ),
    )
    p.add_argument(
        "--guided-json-mode",
        type=str,
        default="fallback",
        choices=["off", "fallback", "on"],
        help=(
            "Guided JSON mode for Stage-2 LLM calls with closed/semi-closed schemas. "
            "off=current behavior; fallback=retry with guided JSON after parse/schema failure; "
            "on=use guided JSON on the first attempt. Extraction is intentionally excluded."
        ),
    )
    p.add_argument(
        "--candidate-strategy",
        type=str,
        default="pairwise",
        choices=["pairwise"],
        help=(
            "Candidate selection strategy. "
            "This release supports pairwise matching for each (pred, gt) pair."
        ),
    )

    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExtractAndMatchConfig:
    judge_model = str(args.judge_model)
    judge_ckpt = resolve_ckpt(judge_model, args.judge_ckpt or args.extractor_ckpt, JUDGE_PRESETS)
    if not judge_ckpt:
        raise ValueError("Cannot resolve judge_ckpt.")

    # All stages use judge_model/judge_ckpt.
    # extractor_* fields are kept in config/manifest only for compatibility.
    extractor_model = judge_model
    extractor_ckpt = judge_ckpt

    return ExtractAndMatchConfig(
        run_root=str(args.run_root),
        run_id=str(args.run_id),
        dataset_root_override=(str(args.dataset_root_override) if args.dataset_root_override else None),
        extractor_model=str(extractor_model),
        extractor_ckpt=str(extractor_ckpt),
        judge_model=str(judge_model),
        judge_ckpt=str(judge_ckpt),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        judge_temperature=float(args.judge_temperature),
        judge_top_p=float(args.judge_top_p),
        judge_max_new_tokens=int(args.judge_max_new_tokens),
        debug=bool(args.debug),
        mode=str(args.mode),
        num_workers=1,
        priority_image_paths=_load_priority_image_paths(args),
        priority_image_paths_file=(str(args.priority_image_paths_file) if args.priority_image_paths_file else None),
        only_priority=bool(args.only_priority),
        progress_interval_sec=float(args.progress_interval_sec),
        gt_mode=str(args.gt_mode),
        candidate_strategy=str(args.candidate_strategy),
        llm_backend=str(args.llm_backend),
        vllm_tensor_parallel_size=max(1, int(args.vllm_tensor_parallel_size)),
        vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        vllm_dtype=str(args.vllm_dtype),
        vllm_max_model_len=max(0, int(args.vllm_max_model_len)),
        guided_json_mode=str(args.guided_json_mode),
    )


def _cfg_core_values(cfg: ExtractAndMatchConfig) -> Dict[str, Any]:
    return {
        "extractor_model": cfg.extractor_model,
        "extractor_ckpt": cfg.extractor_ckpt,
        "judge_model": cfg.judge_model,
        "judge_ckpt": cfg.judge_ckpt,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_new_tokens": cfg.max_new_tokens,
        "judge_temperature": cfg.judge_temperature,
        "judge_top_p": cfg.judge_top_p,
        "judge_max_new_tokens": cfg.judge_max_new_tokens,
        "gt_mode": cfg.gt_mode,
        "candidate_strategy": cfg.candidate_strategy,
        "llm_backend": cfg.llm_backend,
        "vllm_tensor_parallel_size": cfg.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
        "vllm_dtype": cfg.vllm_dtype,
        "vllm_max_model_len": cfg.vllm_max_model_len,
        "guided_json_mode": cfg.guided_json_mode,
    }


def validate_manifest_params(manifest_path: str, cfg: ExtractAndMatchConfig) -> None:
    saved = load_json(manifest_path)
    current = _cfg_core_values(cfg)
    mismatches = []
    for field, getter in _MANIFEST_CORE_FIELDS:
        saved_val = getter(saved)
        cur_val = current[field]
        if saved_val is None and field in {
            "llm_backend",
            "vllm_tensor_parallel_size",
            "vllm_gpu_memory_utilization",
            "vllm_dtype",
            "vllm_max_model_len",
            "guided_json_mode",
            "candidate_strategy",
        }:
            continue
        if saved_val != cur_val:
            mismatches.append(f"  {field}: saved={saved_val!r}  current={cur_val!r}")
    if mismatches:
        raise RuntimeError(
            "[manifest mismatch] Core parameters differ from existing run.\n"
            + "\n".join(mismatches)
            + "\nUse --mode force or a new --run-id."
        )
    _eprint(f"[manifest] Core params match existing run '{cfg.run_id}'. Resuming safely.")
