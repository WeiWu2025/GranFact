#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import multiprocessing as mp
from types import SimpleNamespace

from utils.generate_answers_api import build_api_config_from_args, run_generation_api
from utils.generate_answers_local import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_RESUME_POLICY,
    DEFAULT_RESUME_WHEN_MISSING_FIELD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TORCH_DTYPE,
    build_config as build_local_config,
    default_run_id,
    run_generation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--backend", type=str, default="local", choices=["local", "api"])
    p.add_argument(
        "--dataset-root",
        type=str,
        default="./data/testset",
        help="Benchmark dataset root. Defaults to ./data/testset for release portability.",
    )
    p.add_argument("--output-root", type=str, default="./root")
    p.add_argument("--model-name", type=str, required=True, help="Logical model name used in output dir.")
    p.add_argument("--run-id", type=str, default=None, help="e.g. 20260226; default is today YYYYMMDD")

    # local / common model args
    p.add_argument("--model-path", type=str, default=None, help="Generic local model path alias.")
    p.add_argument("--model-ckpt", type=str, default=None, help="HF repo or local path. If omitted, use preset.")
    p.add_argument("--base-model-ckpt", type=str, default=None)
    p.add_argument("--processor-ckpt", type=str, default=None)

    # generation args shared by both backends
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", "--max-new-token", dest="max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--do-sample", type=int, default=int(DEFAULT_DO_SAMPLE), help="1/0")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--mode", type=str, default="resume", choices=["resume", "force"])
    p.add_argument("--resume-policy", type=str, default=DEFAULT_RESUME_POLICY, choices=["done_only", "json_mtime", "json_hash"])
    p.add_argument("--resume-when-missing-field", type=str, default=DEFAULT_RESUME_WHEN_MISSING_FIELD, choices=["skip", "rerun"])
    p.add_argument("--dedup-after-run", type=int, default=1)

    # local-only runtime args (kept for backward compatibility)
    p.add_argument("--torch-dtype", type=str, default=DEFAULT_TORCH_DTYPE, help="bfloat16/float16/float32")
    p.add_argument("--local-engine", type=str, default="transformers", choices=["transformers", "vllm"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batch-sort-by-image-size", type=int, default=1)
    p.add_argument("--gpus", type=str, default="")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=0, help="0 means use len(--gpus).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-dtype", type=str, default="auto")
    p.add_argument("--vllm-max-model-len", type=int, default=0)
    p.add_argument("--vllm-limit-mm-per-prompt-image", type=int, default=1)
    p.add_argument("--oom-max-attempts", type=int, default=5)
    p.add_argument("--oom-retry-empty-cache", type=int, default=1)
    p.add_argument("--oom-limit-max-pixels", type=int, default=0)
    p.add_argument("--oom-reduce-max-new-tokens-factor", type=float, default=0.5)
    p.add_argument("--oom-resize-long-edge", type=int, default=1024)
    p.add_argument("--progress-print-interval-sec", type=float, default=5.0)
    p.add_argument("--watchdog-stall-sec", type=float, default=3600.0)
    p.add_argument("--watchdog-kill-on-stall", type=int, default=1)
    p.add_argument("--per-sample-timeout-sec", type=float, default=600.0)

    # api-only args
    p.add_argument("--api-model", type=str, default=None, help="Model name sent to OpenAI-compatible API.")
    p.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL or /v1 endpoint.")
    p.add_argument("--api-key", type=str, default=None, help="Optional explicit API key; falls back to env var.")
    p.add_argument(
        "--api-key-env-name",
        type=str,
        default="DEERAPI_KEY_WW",
        help="Environment variable name used to read the API key when --api-key is omitted.",
    )
    p.add_argument("--request-timeout-sec", type=float, default=600.0)
    p.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="Max attempts for retryable API/network errors, including the first attempt.",
    )
    p.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional OpenAI-compatible reasoning effort control, e.g. none/low/medium/high.",
    )

    args = p.parse_args()
    args.run_id = args.run_id or default_run_id()
    args.do_sample = bool(int(args.do_sample))
    if args.model_path and (not args.model_ckpt):
        args.model_ckpt = args.model_path
    return args


def _build_local_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        model_name=args.model_name,
        model_ckpt=args.model_ckpt,
        base_model_ckpt=args.base_model_ckpt,
        processor_ckpt=args.processor_ckpt,
        run_id=args.run_id,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=int(bool(args.do_sample)),
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        local_engine=args.local_engine,
        batch_size=args.batch_size,
        batch_sort_by_image_size=args.batch_sort_by_image_size,
        gpus=args.gpus,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_limit_mm_per_prompt_image=args.vllm_limit_mm_per_prompt_image,
        mode=args.mode,
        dedup_after_run=args.dedup_after_run,
        resume_policy=args.resume_policy,
        resume_when_missing_field=args.resume_when_missing_field,
        oom_max_attempts=args.oom_max_attempts,
        oom_retry_empty_cache=args.oom_retry_empty_cache,
        oom_limit_max_pixels=args.oom_limit_max_pixels,
        oom_reduce_max_new_tokens_factor=args.oom_reduce_max_new_tokens_factor,
        oom_resize_long_edge=args.oom_resize_long_edge,
        progress_print_interval_sec=args.progress_print_interval_sec,
        watchdog_stall_sec=args.watchdog_stall_sec,
        watchdog_kill_on_stall=args.watchdog_kill_on_stall,
        per_sample_timeout_sec=args.per_sample_timeout_sec,
    )


def main() -> None:
    args = parse_args()
    if args.backend == "api":
        cfg = build_api_config_from_args(args)
        run_generation_api(cfg)
        return

    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    cfg = build_local_config(_build_local_namespace(args))
    run_generation(cfg)


if __name__ == "__main__":
    main()
