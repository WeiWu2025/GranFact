#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import io
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()  # allow PIL to open HEIC/HEIF even with misleading extensions

from utils.generate_answers_local import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_RESUME_POLICY,
    DEFAULT_RESUME_WHEN_MISSING_FIELD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FileLock,
    RunMeta,
    build_sample_id,
    check_resume_manifest,
    collect_tasks,
    dedup_jsonl_keep_last_inplace,
    ensure_dir,
    load_latest_results_by_sample_id,
    now_iso,
    safe_jsonl_append,
    should_rerun_task,
)


@dataclass
class ApiGenConfig:
    dataset_root: str
    output_root: str
    model_name: str
    run_id: str
    prompt: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    mode: str
    dedup_after_run: bool
    resume_policy: str
    resume_when_missing_field: str
    api_model: str
    base_url: str
    api_key: str
    api_key_env_name: str
    request_timeout_sec: float
    api_max_retries: int
    reasoning_effort: Optional[str]


def _sanitize_base_url(base_url: str) -> str:
    s = str(base_url or "").strip()
    while len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.rstrip("'\"")
    if " " in s or "(see below" in s.lower():
        raise ValueError(f"Invalid base_url: {base_url!r}")
    return s


def _image_to_data_url(image_path: str) -> str:
    """
    Robust image serialization for API upload.
    Always decode via PIL (HEIC/HEIF opener registered) and re-encode to JPEG,
    so MIME and bytes are guaranteed consistent even when file extension is wrong.
    """
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        # Use JPEG to significantly reduce request body size vs PNG.
        img.save(buf, format="JPEG", quality=90, optimize=True)
        raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _normalize_base_url(base_url: str) -> str:
    s = str(base_url).rstrip("/")
    if s.endswith("/chat/completions"):
        return s
    if s.endswith("/v1"):
        return s + "/chat/completions"
    return s + "/chat/completions"


def _api_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _split_message_content(content: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "final_text": "",
        "reasoning_text": "",
        "raw_content": content,
        "content_items": [],
    }
    if isinstance(content, str):
        out["final_text"] = content.strip()
        return out

    if not isinstance(content, list):
        out["final_text"] = str(content).strip()
        return out

    final_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    items: List[Dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            txt = str(item).strip()
            if txt:
                final_chunks.append(txt)
            continue

        typ = str(item.get("type") or "").strip().lower()
        text_val = item.get("text")
        if text_val is None:
            text_val = item.get("content")
        if text_val is None:
            text_val = item.get("value")
        txt = text_val if isinstance(text_val, str) else (str(text_val).strip() if text_val is not None else "")

        items.append({"type": typ, "text": txt})

        if typ in {"reasoning", "thinking", "thought", "analysis"}:
            if txt:
                reasoning_chunks.append(txt)
        elif typ in {"text", "output_text", "answer", "final"}:
            if txt:
                final_chunks.append(txt)
        else:
            if txt:
                final_chunks.append(txt)

    out["final_text"] = "\n".join(final_chunks).strip()
    out["reasoning_text"] = "\n".join(reasoning_chunks).strip()
    out["content_items"] = items
    return out


def _extract_response_parts(payload: Dict[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {
            "final_text": "",
            "reasoning_text": "",
            "raw_content": None,
            "content_items": [],
        }
    msg = choices[0].get("message", {})
    if not isinstance(msg, dict):
        msg = {}
    content = msg.get("content", "")
    out = _split_message_content(content)

    # DeepSeek reasoning models commonly return reasoning text as a sibling
    # field on the assistant message, e.g. message.reasoning_content, rather
    # than as a typed content item. Preserve it when present.
    extra_reasoning_chunks: List[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            extra_reasoning_chunks.append(val.strip())
    if extra_reasoning_chunks:
        existing = str(out.get("reasoning_text") or "").strip()
        out["reasoning_text"] = "\n".join([x for x in [existing, *extra_reasoning_chunks] if x]).strip()
    return out


def _extract_finish_meta(payload: Dict[str, Any], cfg: ApiGenConfig) -> Dict[str, Any]:
    choices = payload.get("choices")
    finish_reason = "unknown"
    if isinstance(choices, list) and choices:
        finish_reason = str(choices[0].get("finish_reason") or "unknown")
    usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens")
    try:
        completion_tokens = int(completion_tokens) if completion_tokens is not None else None
    except Exception:
        completion_tokens = None
    return {
        "output_token_count": completion_tokens,
        "finish_reason": finish_reason,
        "truncated_by_max_new_tokens": bool(finish_reason == "length"),
        "api_usage": usage,
        "requested_max_new_tokens": int(cfg.max_new_tokens),
    }


def call_openai_compatible_api(image_path: str, cfg: ApiGenConfig) -> Dict[str, Any]:
    data_url = _image_to_data_url(image_path)
    payload = {
        "model": cfg.api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cfg.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": int(cfg.max_new_tokens),
        "temperature": float(cfg.temperature),
        "top_p": float(cfg.top_p),
    }
    if not cfg.do_sample:
        payload["temperature"] = 0.0
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort

    url = _normalize_base_url(cfg.base_url)
    timeout = max(1.0, float(cfg.request_timeout_sec))
    max_attempts = max(1, int(getattr(cfg, "api_max_retries", 3) or 1))
    retryable_http_codes = {408, 429, 500, 502, 503, 504}
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_api_headers(cfg.api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"API HTTPError {e.code}: {detail}")
            if e.code not in retryable_http_codes or attempt >= max_attempts:
                raise last_error from e
        except urllib.error.URLError as e:
            last_error = RuntimeError(f"API URLError: {e}")
            if attempt >= max_attempts:
                raise last_error from e

        # Retry only transient network/rate-limit/server errors. Keep backoff
        # short but increasing to avoid hammering the provider.
        time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    else:
        raise RuntimeError(f"API request failed after {max_attempts} attempts: {last_error!r}")

    try:
        return json.loads(body)
    except Exception as e:
        raise RuntimeError(f"API returned non-JSON response: {body[:1000]}") from e


def build_api_config_from_args(args) -> ApiGenConfig:
    env_name = str(getattr(args, "api_key_env_name", None) or "DEERAPI_KEY_WW")
    api_key = str(getattr(args, "api_key", None) or os.environ.get(env_name, ""))
    if not api_key:
        raise ValueError(
            f"API key is empty. Pass --api-key explicitly or set environment variable '{env_name}'."
        )
    api_model = str(getattr(args, "api_model", None) or getattr(args, "model_name", None) or "").strip()
    if not api_model:
        raise ValueError("api_model is empty. Pass --api-model or --model-name.")
    base_url = _sanitize_base_url(getattr(args, "base_url", None) or "")
    if not base_url:
        raise ValueError("base_url is empty. Pass --base-url for API backend.")

    return ApiGenConfig(
        dataset_root=str(args.dataset_root),
        output_root=str(args.output_root),
        model_name=str(args.model_name),
        run_id=str(args.run_id),
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        mode=str(args.mode),
        dedup_after_run=bool(args.dedup_after_run),
        resume_policy=str(args.resume_policy),
        resume_when_missing_field=str(args.resume_when_missing_field),
        api_model=api_model,
        base_url=base_url,
        api_key=api_key,
        api_key_env_name=env_name,
        request_timeout_sec=float(getattr(args, "request_timeout_sec", 600.0)),
        api_max_retries=max(1, int(getattr(args, "api_max_retries", 3) or 1)),
        reasoning_effort=(str(getattr(args, "reasoning_effort", "")).strip() or None),
    )


def run_generation_api(cfg: ApiGenConfig) -> None:
    run_dir = os.path.join(cfg.output_root, cfg.model_name, f"stage1_answers-{cfg.run_id}")
    ensure_dir(run_dir)
    stage_dir = os.path.join(run_dir, "stage1_outputs")
    ensure_dir(stage_dir)

    results_jsonl = os.path.join(stage_dir, "results.jsonl")
    errors_jsonl = os.path.join(stage_dir, "errors.jsonl")
    meta_json = os.path.join(stage_dir, "meta.json")
    lock_path = os.path.join(stage_dir, ".write.lock")

    manifest_cfg = type(
        "ManifestCfg",
        (),
        {
            "dataset_root": cfg.dataset_root,
            "model_name": cfg.model_name,
            "model_ckpt": f"api::{cfg.api_model}@{cfg.base_url}",
            "prompt": cfg.prompt,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "torch_dtype": "api",
        },
    )()
    if cfg.mode == "resume":
        check_resume_manifest(meta_json, manifest_cfg)

    if not os.path.exists(results_jsonl):
        open(results_jsonl, "a", encoding="utf-8").close()
    if not os.path.exists(errors_jsonl):
        open(errors_jsonl, "a", encoding="utf-8").close()

    meta = RunMeta(
        run_id=cfg.run_id,
        created_at=now_iso(),
        dataset_root=cfg.dataset_root,
        model_name=cfg.model_name,
        model_ckpt=f"api::{cfg.api_model}@{cfg.base_url}",
        prompt=cfg.prompt,
        decoding_params={
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        runtime={
            "backend": "api",
            "api_model": cfg.api_model,
            "base_url": cfg.base_url,
            "api_key_env_name": cfg.api_key_env_name,
            "request_timeout_sec": cfg.request_timeout_sec,
            "api_max_retries": cfg.api_max_retries,
            "reasoning_effort": cfg.reasoning_effort,
        },
        parallel={
            "gpus": [],
            "num_workers": 1,
            "strategy": "single-process API requests",
            "batch_size_per_worker": 1,
            "batch_sort_by_image_size": False,
        },
        resume={
            "mode": cfg.mode,
            "resume_policy": cfg.resume_policy,
            "resume_when_missing_field": cfg.resume_when_missing_field,
            "dedup_after_run": cfg.dedup_after_run,
        },
        oom_policy={
            "oom_max_attempts": None,
            "oom_retry_empty_cache": False,
            "oom_limit_max_pixels": None,
            "oom_reduce_max_new_tokens_factor": None,
            "oom_resize_long_edge": None,
        },
    )
    if not os.path.exists(meta_json):
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    tasks_all = collect_tasks(cfg.dataset_root, cfg.prompt)
    resume_stats: Dict[str, int] = {}
    if cfg.mode == "force":
        tasks_todo = tasks_all
    else:
        with FileLock(lock_path):
            latest_by_sid = load_latest_results_by_sample_id(results_jsonl)
        tasks_todo = []
        for t in tasks_all:
            sid = t["sample_id"]
            rerun, reason = should_rerun_task(t, latest_by_sid.get(sid), cfg)
            resume_stats[reason] = int(resume_stats.get(reason, 0)) + 1
            if rerun:
                tasks_todo.append(t)

    for t in tasks_todo:
        try:
            if not os.path.exists(t["image_path"]):
                raise FileNotFoundError(f"image not found: {t['image_path']}")
            with Image.open(t["image_path"]) as img:
                img.verify()
            payload = call_openai_compatible_api(t["image_path"], cfg)
            finish_meta = _extract_finish_meta(payload, cfg)
            resp_parts = _extract_response_parts(payload)
            warnings: List[str] = []
            if (not resp_parts.get("final_text")) and resp_parts.get("reasoning_text"):
                warnings.append("final_response_empty_but_reasoning_present")
            usage = finish_meta.get("api_usage", {}) if isinstance(finish_meta.get("api_usage"), dict) else {}
            ctd = usage.get("completion_tokens_details", {}) if isinstance(usage.get("completion_tokens_details"), dict) else {}
            reasoning_tokens = ctd.get("reasoning_tokens")
            try:
                reasoning_tokens = int(reasoning_tokens) if reasoning_tokens is not None else None
            except Exception:
                reasoning_tokens = None
            if (not resp_parts.get("final_text")) and reasoning_tokens and reasoning_tokens > 0:
                warnings.append("final_response_empty_with_reasoning_tokens")
            rec = {
                "sample_id": t["sample_id"],
                "domain": t.get("domain"),
                "generated_at": now_iso(),
                "model_name": cfg.model_name,
                "model_ckpt": f"api::{cfg.api_model}@{cfg.base_url}",
                "prompt": cfg.prompt,
                "decoding_params": {
                    "max_new_tokens": cfg.max_new_tokens,
                    "do_sample": cfg.do_sample,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                },
                "image_path": t["image_path"],
                "source_json": t["source_json"],
                "index_in_source_json": t["index_in_source_json"],
                "source_json_mtime": t.get("source_json_mtime"),
                "source_json_sha1": t.get("source_json_sha1"),
                "latency_sec": None,
                "worker": {"worker_id": 0, "gpu_id": None},
                "degradation": {
                    "used": False,
                    "steps": [],
                    "attempts": [{"attempt": 0, "backend": "api"}],
                    "final_settings": {
                        "max_new_tokens": cfg.max_new_tokens,
                        "max_pixels": None,
                        "resized_to": None,
                    },
                    "image_info": {"orig_size": None, "final_size": None},
                    "gen_time_sec": None,
                    "finish": finish_meta,
                },
                "output_token_count": finish_meta.get("output_token_count"),
                "finish_reason": finish_meta.get("finish_reason"),
                "truncated_by_max_new_tokens": bool(finish_meta.get("truncated_by_max_new_tokens", False)),
                "response": resp_parts.get("final_text", ""),
                "reasoning_content": resp_parts.get("reasoning_text", ""),
                "raw_response_content": resp_parts.get("raw_content"),
                "warnings": warnings,
                "api_response_meta": {
                    "id": payload.get("id"),
                    "model": payload.get("model"),
                    "usage": finish_meta.get("api_usage", {}),
                    "content_items": resp_parts.get("content_items", []),
                },
            }
            with FileLock(lock_path):
                safe_jsonl_append(results_jsonl, rec)
        except Exception as e:
            err = {
                "generated_at": now_iso(),
                "model_name": cfg.model_name,
                "model_ckpt": f"api::{cfg.api_model}@{cfg.base_url}",
                "prompt": cfg.prompt,
                "worker": {"worker_id": 0, "gpu_id": None},
                "sample_id": t.get("sample_id"),
                "domain": t.get("domain"),
                "source_json": t.get("source_json"),
                "index_in_source_json": t.get("index_in_source_json"),
                "image_path": t.get("image_path"),
                "error": repr(e),
                "traceback": traceback.format_exc(limit=20),
            }
            with FileLock(lock_path):
                safe_jsonl_append(errors_jsonl, err)

    if cfg.dedup_after_run:
        with FileLock(lock_path):
            dedup_jsonl_keep_last_inplace(results_jsonl, key="sample_id")
