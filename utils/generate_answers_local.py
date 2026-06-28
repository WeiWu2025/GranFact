#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1 generation script for VLM grain-hallucination benchmark (multi-GPU / multi-process).

Output layout:
  OUTPUT_ROOT/{model_name}/stage1_answers-{run_id}/stage1_outputs/
    meta.json
    results.jsonl
    errors.jsonl

Key features:
- English prompt + enforce English-only responses
- Prefer no manual resize (rely on Qwen3 dynamic resolution)
- OOM-safe: retry + degradation strategy; record which degradations happened
- Multi-GPU data-parallel (1 process per GPU); user-select GPUs
- Multi-process safe append using file lock
- Mode: resume (skip done) or force (regenerate all)
- Progress bar (tqdm) in parent process; worker reports progress via queue
- Show success/failure counters in progress display
- Deduplicate results.jsonl at the end (keep last by order)

Note:
- File locking uses fcntl (Linux). If you need Windows, tell me and I’ll switch to a cross-platform lock.
"""

import os
import sys
import json
import time
import argparse
import hashlib
import traceback
import multiprocessing as mp
import threading
import queue as _thread_queue
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import AutoConfig
from utils.mllm_backends import VLLMMLLMBackend, backend_runtime_flags, get_backend

#处理heif图片
from pillow_heif import register_heif_opener
register_heif_opener()  # 程序启动时注册一次


# tqdm (optional)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

# --- Optional imports (Qwen3-VL recommended path) ---
_QWEN3_VL_IMPORT_ERROR = None
try:
    from transformers import Qwen3VLForConditionalGeneration
    _HAS_QWEN3_VL_CLASS = True
except Exception as _e:
    _HAS_QWEN3_VL_CLASS = False
    _QWEN3_VL_IMPORT_ERROR = repr(_e)

_AUTO_V2S_IMPORT_ERROR = None
try:
    from transformers import AutoModelForVision2Seq
    _HAS_AUTO_V2S = True
except Exception as _e:
    _HAS_AUTO_V2S = False
    _AUTO_V2S_IMPORT_ERROR = repr(_e)

_AUTO_ITTT_IMPORT_ERROR = None
try:
    # Newer naming in recent transformers versions.
    from transformers import AutoModelForImageTextToText
    _HAS_AUTO_ITTT = True
except Exception as _e:
    _HAS_AUTO_ITTT = False
    _AUTO_ITTT_IMPORT_ERROR = repr(_e)

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_VL_UTILS = True
except Exception:
    _HAS_QWEN_VL_UTILS = False

# Linux file lock
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

# per-sample timeout (Linux)
try:
    import signal
    _HAS_SIGNAL = True
except Exception:
    signal = None
    _HAS_SIGNAL = False


# =========================
# 0) Presets
# =========================
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # Release builds intentionally avoid machine-specific checkpoint paths.
    # Add local presets here if desired, or pass --model-ckpt explicitly.
}


# =========================
# 1) Defaults
# =========================
DEFAULT_PROMPT = (
    "Describe the image in as much detail as possible. Respond in English only."
)
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.9
DEFAULT_TORCH_DTYPE = "bfloat16"   # bf16 preferred on A40; fallback to fp16 if needed
DEFAULT_RESUME_POLICY = "json_hash"
DEFAULT_RESUME_WHEN_MISSING_FIELD = "rerun"


# =========================
# 2) Data structures
# =========================
@dataclass
class RunMeta:
    run_id: str
    created_at: str
    dataset_root: str
    model_name: str
    model_ckpt: str
    prompt: str
    decoding_params: Dict[str, Any]
    runtime: Dict[str, Any]
    parallel: Dict[str, Any]
    resume: Dict[str, Any]
    oom_policy: Dict[str, Any]


@dataclass
class GenConfig:
    dataset_root: str
    output_root: str
    model_name: str
    model_ckpt: str
    base_model_ckpt: Optional[str]
    processor_ckpt: Optional[str]
    run_id: str

    prompt: str

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float

    torch_dtype: str

    # local engine
    local_engine: str  # "transformers" | "vllm"
    vllm_tensor_parallel_size: int
    vllm_gpu_memory_utilization: float
    vllm_dtype: str
    vllm_max_model_len: int
    vllm_limit_mm_per_prompt_image: int

    # batching
    batch_size: int
    batch_sort_by_image_size: bool

    # parallel
    gpus: List[int]

    # mode
    mode: str  # "resume" | "force"
    dedup_after_run: bool
    resume_policy: str  # "done_only" | "json_mtime" | "json_hash"
    resume_when_missing_field: str  # "skip" | "rerun"

    # OOM degradation policy
    oom_retry_empty_cache: bool
    oom_limit_max_pixels: Optional[int]       # if not None, try to pass max_pixels to processor
    oom_reduce_max_new_tokens_factor: float   # e.g., 0.5
    oom_resize_long_edge: Optional[int]       # last resort; if None, never resize
    oom_max_attempts: int                     # total attempts (including first)

    # progress / watchdog
    progress_print_interval_sec: float
    watchdog_stall_sec: float
    watchdog_kill_on_stall: bool

    # sample timeout
    per_sample_timeout_sec: float


# =========================
# 3) Utilities
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


def iter_items_from_json_file(json_path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    Supports:
      1) dict with "image_path"
      2) list[dict] each with "image_path"
    Yields: (index_in_file, item_dict)
    """
    obj = load_json(json_path)
    if isinstance(obj, dict):
        yield (0, obj)
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                yield (i, it)


def resolve_image_path(json_path: str, image_path_field: str) -> str:
    p = str(image_path_field).strip()
    if os.path.isabs(p):
        return p
    base = os.path.dirname(json_path)
    return os.path.normpath(os.path.join(base, p))


def resolve_domain(json_path: str, item: Dict[str, Any], image_abs_path: str) -> str:
    domain = str(item.get("domain") or "").strip()
    if domain:
        return domain
    # Import lazily to keep worker process startup lighter and avoid importing
    # evaluation-side dependencies before each worker's CUDA_VISIBLE_DEVICES is
    # narrowed by the parent process.
    from evaluate import classify  # noqa: PLC0415

    for candidate in (image_abs_path, json_path):
        inferred = classify(candidate)
        if inferred:
            return inferred
    return "unknown"


def build_sample_id(image_abs_path: str, prompt: str) -> str:
    # stable join key for stage-2; prompt is part of ID by design
    return sha1_short(image_abs_path + "||" + prompt)


def get_torch_dtype(name: str):
    name = (name or "").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


def is_qwen3_vl(ckpt_or_name: str) -> bool:
    s = (ckpt_or_name or "").lower()
    return ("qwen3" in s and "vl" in s)


def _looks_like_qwen3_vl_model(model_ckpt: str) -> bool:
    """
    Prefer config-based detection; fall back to string heuristic.
    """
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


def _load_auto_vision_language_model(
    ckpt: str,
    dtype,
):
    """
    Load a generic vision-language generation model with robust class fallback.
    Supports both legacy and newer transformers class names.
    """
    errors: List[str] = []

    if _HAS_AUTO_ITTT:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                ckpt,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
        except Exception as e:
            errors.append(f"AutoModelForImageTextToText.from_pretrained failed: {repr(e)}")
    else:
        errors.append(f"AutoModelForImageTextToText import unavailable: {_AUTO_ITTT_IMPORT_ERROR}")

    if _HAS_AUTO_V2S:
        try:
            return AutoModelForVision2Seq.from_pretrained(
                ckpt,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
        except Exception as e:
            errors.append(f"AutoModelForVision2Seq.from_pretrained failed: {repr(e)}")
    else:
        errors.append(f"AutoModelForVision2Seq import unavailable: {_AUTO_V2S_IMPORT_ERROR}")

    raise RuntimeError(
        "Failed to load model via both AutoModelForImageTextToText and AutoModelForVision2Seq. "
        f"ckpt={ckpt}. Diagnostics: " + " | ".join(errors)
    )


def is_lora_adapter_dir(path: str) -> bool:
    if not path or (not os.path.isdir(path)):
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json")) and os.path.exists(os.path.join(path, "adapter_model.safetensors"))


def read_lora_base_model_from_adapter(path: str) -> Optional[str]:
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        obj = load_json(cfg_path)
        if isinstance(obj, dict):
            v = obj.get("base_model_name_or_path")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        return None
    return None


def read_done_ids(results_jsonl: str) -> set:
    done = set()
    if not os.path.exists(results_jsonl):
        return done
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sample_id")
                if sid:
                    done.add(sid)
            except Exception:
                pass
    return done


def load_latest_results_by_sample_id(results_jsonl: str) -> Dict[str, Dict[str, Any]]:
    """Load results.jsonl and keep only the latest record for each sample_id."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(results_jsonl):
        return out
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("sample_id")
            if sid:
                out[str(sid)] = obj
    return out


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _iso_to_epoch_or_none(v: Any) -> Optional[float]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _missing_field_should_rerun(cfg: "GenConfig") -> bool:
    return str(cfg.resume_when_missing_field).strip().lower() == "rerun"


def should_rerun_task(
    task: Dict[str, Any],
    latest_result: Optional[Dict[str, Any]],
    cfg: "GenConfig",
) -> Tuple[bool, str]:
    """
    Decide whether this task should be regenerated in resume mode.
    Returns (should_rerun, reason).
    """
    if latest_result is None:
        return True, "new_sample"

    if not str(latest_result.get("response") or "").strip():
        if _missing_field_should_rerun(cfg):
            return True, "missing_response_rerun"
        return False, "missing_response_skip"

    policy = str(cfg.resume_policy).strip().lower()
    if policy == "done_only":
        return False, "already_done"

    if policy == "json_hash":
        cur_hash = task.get("source_json_sha1")
        old_hash = latest_result.get("source_json_sha1")
        if not cur_hash or not old_hash:
            if _missing_field_should_rerun(cfg):
                return True, "missing_json_hash_rerun"
            return False, "missing_json_hash_skip"
        if str(cur_hash) != str(old_hash):
            return True, "json_hash_changed"
        return False, "json_hash_same"

    if policy == "json_mtime":
        cur_mtime = _to_float_or_none(task.get("source_json_mtime"))
        old_mtime = _to_float_or_none(latest_result.get("source_json_mtime"))
        old_generated = _iso_to_epoch_or_none(latest_result.get("generated_at"))
        baseline = old_mtime if old_mtime is not None else old_generated
        if cur_mtime is None or baseline is None:
            if _missing_field_should_rerun(cfg):
                return True, "missing_json_mtime_rerun"
            return False, "missing_json_mtime_skip"
        if cur_mtime > (baseline + 1e-6):
            return True, "json_mtime_newer"
        return False, "json_mtime_not_newer"

    return False, f"unknown_policy_{policy}_skip"


class FileLock:
    """
    Simple cross-process lock using fcntl.flock (Linux).
    Locks a separate lockfile path, not the data file itself.
    """
    def __init__(self, lock_path: str):
        if not _HAS_FCNTL:
            raise RuntimeError("fcntl is not available; cannot use FileLock on this platform.")
        self.lock_path = lock_path
        self._fd = None

    def __enter__(self):
        ensure_dir(os.path.dirname(self.lock_path))
        self._fd = open(self.lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self._fd.close()
            except Exception:
                pass
            self._fd = None


def safe_jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def dedup_jsonl_keep_last_inplace(path: str, key: str = "sample_id") -> Dict[str, Any]:
    """
    Deduplicate JSONL by `key`, keep the last occurrence (by file order).
    Rewrites the file atomically.
    Returns stats dict.
    """
    if not os.path.exists(path):
        return {"path": path, "existed": False, "kept": 0, "dropped": 0}

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                # keep malformed lines as-is? Here we drop them (could also keep in a side file).
                pass

    last_idx = {}
    for i, r in enumerate(records):
        k = r.get(key)
        if k is not None:
            last_idx[k] = i

    keep_set = set(last_idx.values())
    kept = [records[i] for i in range(len(records)) if i in keep_set]

    tmp = path + ".tmp"
    bak = path + ".bak"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # backup then replace
    try:
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(path, bak)
    except Exception:
        # if backup fails, still try replace
        pass
    os.replace(tmp, path)

    return {
        "path": path,
        "existed": True,
        "original": len(records),
        "kept": len(kept),
        "dropped": max(0, len(records) - len(kept)),
        "backup": bak,
    }


def _best_effort_unbuffered_io() -> None:
    """
    Make progress prints visible immediately even when stdout/stderr are block-buffered.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _eprint(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


class _Timeout(Exception):
    pass


class SampleTimeout:
    """
    Linux-only timeout based on signal.alarm.
    Only works in the main thread of a process.
    """
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._old_handler = None

    def __enter__(self):
        if (not _HAS_SIGNAL) or (self.seconds <= 0):
            return self

        def _handler(signum, frame):
            raise _Timeout(f"per-sample timeout after {self.seconds}s")

        self._old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if (not _HAS_SIGNAL) or (self.seconds <= 0):
            return False
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        try:
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        except Exception:
            pass
        return False


# =========================
# 3b) Manifest validation
# =========================
_MANIFEST_MISSING = object()  # sentinel for missing keys

# Stage-1 resume manifest policy:
#   - Required-match fields define the generation/data protocol and must stay
#     identical when resuming into the same run-id.
#   - Runtime/resource fields are intentionally ignored. This allows resuming a
#     run with different GPUs, worker counts, batch size, local engine
#     (transformers/vLLM), tensor parallel size, vLLM memory settings, dtype, or
#     library versions. Those changes may affect implementation details, but
#     should not block continuing a partially completed benchmark run.
_STAGE1_MANIFEST_REQUIRED_MATCH_FIELDS: List[Tuple[str, ...]] = [
    ("dataset_root",),
    ("model_name",),
    ("model_ckpt",),
    ("prompt",),
    ("decoding_params", "max_new_tokens"),
    ("decoding_params", "do_sample"),
    ("decoding_params", "temperature"),
    ("decoding_params", "top_p"),
]

# Documentation-only list for the resume policy above.  The implementation
# intentionally checks only _STAGE1_MANIFEST_REQUIRED_MATCH_FIELDS; these
# runtime/resource sections are recorded in meta.json for provenance but must
# not block resume when changed between partial runs.
_STAGE1_MANIFEST_IGNORED_FIELD_PREFIXES: List[Tuple[str, ...]] = [
    ("parallel",),
    ("runtime",),
    ("resume",),
    ("oom_policy",),
]


def _get_nested(d: Any, keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return _MANIFEST_MISSING
        d = d[k]
    return d


def check_resume_manifest(meta_json: str, cfg: "GenConfig") -> None:
    """
    In resume mode, compare current cfg against the stored meta.json on all
    core fields. Aborts (sys.exit 1) with a clear diff if any mismatch is found.
    If meta.json does not exist yet this is a no-op (first run).
    """
    if not os.path.exists(meta_json):
        return

    try:
        stored = load_json(meta_json)
    except Exception as e:
        _eprint(f"[manifest] WARNING: could not read meta.json for validation: {e}")
        return

    current: Dict[str, Any] = {
        "dataset_root": cfg.dataset_root,
        "model_name": cfg.model_name,
        "model_ckpt": cfg.model_ckpt,
        "prompt": cfg.prompt,
        "decoding_params": {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        "runtime": {
            "torch_dtype": cfg.torch_dtype,
        },
    }

    mismatches = []
    for keys in _STAGE1_MANIFEST_REQUIRED_MATCH_FIELDS:
        stored_val = _get_nested(stored, keys)
        if stored_val is _MANIFEST_MISSING:
            continue  # field absent in older meta.json — skip rather than false-alarm
        current_val = _get_nested(current, keys)
        if stored_val != current_val:
            mismatches.append((".".join(keys), stored_val, current_val))

    if not mismatches:
        return

    lines = [
        "[manifest] ERROR: resume parameter mismatch — current args differ from stored meta.json!",
        f"  meta.json: {meta_json}",
        "",
        "  Field                              Stored                    Current",
        "  " + "-" * 72,
    ]
    for field, stored_val, current_val in mismatches:
        lines.append(f"  {field:<34} {repr(stored_val):<25} {repr(current_val)}")
    lines += [
        "",
        "  Resuming with different generation/data protocol fields would silently contaminate results.",
        "  Runtime/resource changes are allowed and ignored by this check, including:",
        "    GPUs, num_workers, batch size, transformers/vLLM local engine, vLLM TP/memory settings, dtype, library versions.",
        "  Options:",
        "    (a) Restore original params to match stored meta.json, then re-run.",
        "    (b) Use --run-id <new-id> to start a separate fresh run.",
        "    (c) Use --mode force to regenerate ALL samples with the new params.",
    ]
    _eprint("\n".join(lines))
    sys.exit(1)


# =========================
# 4) Model / generation
# =========================
def load_model_and_processor(
    model_name: str,
    model_ckpt: str,
    torch_dtype: str,
    device: torch.device,
    base_model_ckpt: Optional[str] = None,
    processor_ckpt: Optional[str] = None,
):
    backend = get_backend(model_name, model_ckpt)
    model, processor = backend.load_model_and_processor(
        model_ckpt=model_ckpt,
        torch_dtype=torch_dtype,
        device=device,
        base_model_ckpt=base_model_ckpt,
        processor_ckpt=processor_ckpt,
    )
    return backend, model, processor


def _legacy_load_model_and_processor(
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
            raise RuntimeError(
                "LoRA checkpoint detected but base model is unknown. "
                "Please pass --base-model-ckpt or ensure adapter_config.json has base_model_name_or_path."
            )

        proc_ckpt = (processor_ckpt or "").strip() or base_ckpt
        processor = AutoProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)

        if _looks_like_qwen3_vl_model(base_ckpt) and _HAS_QWEN3_VL_CLASS:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_ckpt,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
        else:
            model = _load_auto_vision_language_model(base_ckpt, dtype=dtype)

        model = PeftModel.from_pretrained(model, model_ckpt)
        model = model.merge_and_unload()
    else:
        proc_ckpt = (processor_ckpt or "").strip() or model_ckpt
        processor = AutoProcessor.from_pretrained(proc_ckpt, trust_remote_code=True)

        # Prefer passing dtype (newer transformers); keep torch_dtype for compatibility if needed
        if _looks_like_qwen3_vl_model(model_ckpt) and _HAS_QWEN3_VL_CLASS:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_ckpt,
                dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
        else:
            model = _load_auto_vision_language_model(model_ckpt, dtype=dtype)

    model.to(device)
    model.eval()
    return model, processor


def _is_oom_error(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg)


def _maybe_resize_long_edge(img: Image.Image, long_edge: int) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
    if long_edge is None:
        return img, None
    w, h = img.size
    le = max(w, h)
    if le <= long_edge:
        return img, None
    scale = float(long_edge) / float(le)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.BICUBIC)
    return resized, (nw, nh)


def _get_eos_token_ids(model, processor) -> List[int]:
    eos_ids: List[int] = []
    try:
        v = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(v, int):
            eos_ids.append(v)
        elif isinstance(v, (list, tuple)):
            eos_ids.extend([int(x) for x in v if isinstance(x, int)])
    except Exception:
        pass
    try:
        v2 = getattr(processor, "tokenizer", None)
        if v2 is not None:
            tid = getattr(v2, "eos_token_id", None)
            if isinstance(tid, int):
                eos_ids.append(tid)
    except Exception:
        pass
    return sorted(set(eos_ids))


def _processor_has_usable_chat_template(processor) -> bool:
    """
    Some processors expose apply_chat_template() but still do not carry an
    actual chat template, so calling it raises ValueError. We only route to the
    chat-template path when a non-empty template is really present.

    This preserves current behavior for Qwen-family models that already work,
    while allowing models like InstructBLIP to fall back to the generic
    processor(images=..., text=...) path.
    """
    if not hasattr(processor, "apply_chat_template"):
        return False

    direct_template = getattr(processor, "chat_template", None)
    if isinstance(direct_template, str) and direct_template.strip():
        return True

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if isinstance(tokenizer_template, str) and tokenizer_template.strip():
        return True

    return False


def _normalize_vlm_processor_outputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize processor outputs for generic VLM backends.

    Some processors (observed with certain InstructBLIP setups) may return
    pixel_values with an extra singleton dimension for single-image inputs,
    e.g. [B, 1, C, H, W] instead of [B, C, H, W]. Remove only this highly
    specific redundant dimension. Leave all other shapes untouched so current
    Qwen-family behavior is preserved.
    """
    out = dict(inputs)
    pixel_values = out.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 5:
        try:
            if int(pixel_values.shape[1]) == 1:
                out["pixel_values"] = pixel_values.squeeze(1)
        except Exception:
            pass
    return out


@torch.inference_mode()
def generate_one(
    backend,
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
    if backend is not None:
        return backend.generate_one(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_pixels=max_pixels,
        )

    """
    Qwen3-VL preferred:
      messages + apply_chat_template + process_vision_info (qwen-vl-utils)
    Fallback:
      processor(images=..., text=...)
    """
    has_usable_chat_template = _processor_has_usable_chat_template(processor)
    use_qwen_path = is_qwen3_vl(getattr(model, "name_or_path", "")) or has_usable_chat_template
    # Qwen-family models that already work should keep their current path.
    # For non-Qwen models, only use chat-template when a real template exists.

    if use_qwen_path and has_usable_chat_template:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Some processors accept max_pixels/min_pixels in kwargs. We'll try to pass it if provided.
        proc_kwargs = dict(padding=True, return_tensors="pt")
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = int(max_pixels)

        if _HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                **proc_kwargs,
            )
        else:
            inputs = processor(
                text=[text],
                images=[image],
                **proc_kwargs,
            )
    else:
        proc_kwargs = dict(return_tensors="pt")
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = int(max_pixels)
        inputs = processor(images=image, text=prompt, **proc_kwargs)
        inputs = _normalize_vlm_processor_outputs(inputs)

    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    # Try to remove prompt part for chatty decodes
    if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
        in_len = inputs["input_ids"].shape[-1]
        if gen_ids.shape[-1] >= in_len:
            gen_ids = gen_ids[:, in_len:]

    if hasattr(processor, "batch_decode"):
        out = processor.batch_decode(gen_ids, skip_special_tokens=True)
        text = (out[0] if out else "").strip()
    else:
        text = processor.decode(gen_ids[0], skip_special_tokens=True).strip()

    out_len = int(gen_ids.shape[-1]) if hasattr(gen_ids, "shape") else 0
    eos_ids = _get_eos_token_ids(model, processor)
    last_id = None
    try:
        if out_len > 0:
            last_id = int(gen_ids[0, -1].item())
    except Exception:
        last_id = None

    ended_with_eos = (last_id in eos_ids) if (last_id is not None and eos_ids) else False
    truncated = (out_len >= int(max_new_tokens)) and (not ended_with_eos)
    finish_reason = "eos" if ended_with_eos else ("length" if truncated else "unknown")

    return text, {
        "output_token_count": out_len,
        "finish_reason": finish_reason,
        "truncated_by_max_new_tokens": bool(truncated),
    }


def generate_with_oom_policy(backend, model, processor, image: Image.Image, cfg: GenConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Try generation with staged degradation.
    Records which degradations were applied.
    """
    w0, h0 = image.size
    degradation_steps: List[str] = []
    attempts: List[Dict[str, Any]] = []

    # attempt state
    cur_image = image
    cur_max_new_tokens = cfg.max_new_tokens
    cur_max_pixels = None  # None means "do not constrain"
    resized_to = None

    for attempt in range(cfg.oom_max_attempts):
        attempts.append({
            "attempt": attempt,
            "max_new_tokens": cur_max_new_tokens,
            "max_pixels": cur_max_pixels,
            "resized_to": resized_to,
        })

        try:
            t0 = time.time()
            text, finish_meta = generate_one(
                backend=backend,
                model=model,
                processor=processor,
                image=cur_image,
                prompt=cfg.prompt,
                max_new_tokens=cur_max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_pixels=cur_max_pixels,
            )
            t1 = time.time()
            info = {
                "used": len(degradation_steps) > 0,
                "steps": degradation_steps,
                "attempts": attempts,
                "final_settings": {
                    "max_new_tokens": cur_max_new_tokens,
                    "max_pixels": cur_max_pixels,
                    "resized_to": resized_to,
                },
                "image_info": {
                    "orig_size": [w0, h0],
                    "final_size": list(cur_image.size),
                },
                "gen_time_sec": round(t1 - t0, 4),
                "finish": finish_meta,
            }
            return text, info

        except Exception as e:
            if not _is_oom_error(e):
                raise

            # OOM handling
            if cfg.oom_retry_empty_cache and (attempt == 0):
                degradation_steps.append("oom_empty_cache_retry")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            # Next: constrain max_pixels (if configured and not already)
            if cfg.oom_limit_max_pixels is not None and cur_max_pixels is None:
                degradation_steps.append(f"limit_max_pixels:{cfg.oom_limit_max_pixels}")
                cur_max_pixels = int(cfg.oom_limit_max_pixels)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            # Next: reduce max_new_tokens
            if cfg.oom_reduce_max_new_tokens_factor < 1.0 and cur_max_new_tokens > 64:
                new_tokens = max(64, int(cur_max_new_tokens * cfg.oom_reduce_max_new_tokens_factor))
                if new_tokens < cur_max_new_tokens:
                    degradation_steps.append(f"reduce_max_new_tokens:{cur_max_new_tokens}->{new_tokens}")
                    cur_max_new_tokens = new_tokens
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue

            # Last resort: resize long edge
            if cfg.oom_resize_long_edge is not None and resized_to is None:
                degradation_steps.append(f"resize_long_edge:{cfg.oom_resize_long_edge}")
                cur_image, resized_to = _maybe_resize_long_edge(image, cfg.oom_resize_long_edge)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            # If no more degradation steps available, re-raise
            raise


# =========================
# 5) Args / config
# =========================
def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d")


def resolve_model_ckpt(model_name: str, model_ckpt: Optional[str]) -> str:
    if model_ckpt:
        return model_ckpt
    preset = MODEL_PRESETS.get(model_name, {})
    ckpt = preset.get("ckpt", None)
    if not ckpt:
        raise ValueError(
            f"model_ckpt is None and no preset found for model_name='{model_name}'. "
            f"Please pass --model-ckpt or add to MODEL_PRESETS."
        )
    return ckpt


def parse_gpus(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        # default: all visible GPUs
        n = torch.cuda.device_count()
        return list(range(n))
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(x) for x in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset-root", type=str, default="./data/testset")
    p.add_argument("--output-root", type=str, default="./root")

    p.add_argument("--model-name", type=str, required=True, help="Logical model name used in output dir.")
    p.add_argument("--model-ckpt", type=str, default=None, help="HF repo or local path. If omitted, use preset.")
    p.add_argument("--base-model-ckpt", type=str, default=None,
                   help="Optional base model checkpoint for LoRA adapter checkpoints.")
    p.add_argument("--processor-ckpt", type=str, default=None,
                   help="Optional processor checkpoint path/repo. Default: model_ckpt (full) or base_model_ckpt (LoRA).")
    p.add_argument("--run-id", type=str, default=None, help="e.g. 20260226; default is today YYYYMMDD")

    # Prompt / decoding
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", "--max-new-token", dest="max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--do-sample", type=int, default=0, help="1/0")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--torch-dtype", type=str, default=DEFAULT_TORCH_DTYPE, help="bfloat16/float16/float32")
    p.add_argument("--local-engine", type=str, default="transformers", choices=["transformers", "vllm"],
                   help="Local generation engine. transformers keeps legacy data-parallel workers; vllm uses one embedded vLLM engine.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=0,
                   help="vLLM tensor parallel size. 0 means len(--gpus).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-dtype", type=str, default="auto")
    p.add_argument("--vllm-max-model-len", type=int, default=0)
    p.add_argument("--vllm-limit-mm-per-prompt-image", type=int, default=1)

    # Batch inference
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size per worker process. 1 keeps legacy behavior.")
    p.add_argument("--batch-sort-by-image-size", type=int, default=1,
                   help="1=sort each batch by image area before generation to reduce padding waste.")

    # Parallel
    p.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (e.g. '0,1,2,3'). Empty => all.")

    # Mode: only resume/force (removed "all")
    p.add_argument(
        "--mode",
        type=str,
        default="resume",
        choices=["resume", "force"],
        help="resume=skip done sample_id; force=regenerate all samples",
    )
    p.add_argument(
        "--resume-policy",
        type=str,
        default=DEFAULT_RESUME_POLICY,
        choices=["done_only", "json_mtime", "json_hash"],
        help="Resume filtering policy. json_hash is robust to cross-machine file mtime changes.",
    )
    p.add_argument(
        "--resume-when-missing-field",
        type=str,
        default=DEFAULT_RESUME_WHEN_MISSING_FIELD,
        choices=["skip", "rerun"],
        help="When required provenance fields are missing in old results: skip or rerun.",
    )
    p.add_argument("--dedup-after-run", type=int, default=1, help="1=dedup results.jsonl at end (keep last).")

    # OOM policy
    p.add_argument("--oom-max-attempts", type=int, default=5, help="Total attempts including first.")
    p.add_argument("--oom-retry-empty-cache", type=int, default=1, help="Retry once after torch.cuda.empty_cache().")
    p.add_argument("--oom-limit-max-pixels", type=int, default=0,
                   help="If >0, on OOM retry pass max_pixels to processor (tries to cap vision tokens).")
    p.add_argument("--oom-reduce-max-new-tokens-factor", type=float, default=0.5,
                   help="On OOM, reduce max_new_tokens by this factor (e.g. 0.5).")
    p.add_argument("--oom-resize-long-edge", type=int, default=1024,
                   help="Last resort: resize image long edge to this. Set 0 to disable resizing completely.")

    # progress / watchdog
    p.add_argument("--progress-print-interval-sec", type=float, default=5.0,
                   help="When tqdm can't render (non-TTY), print progress every N seconds.")
    p.add_argument("--watchdog-stall-sec", type=float, default=3600.0,
                   help="If no progress for N seconds, treat as stall.")
    p.add_argument("--watchdog-kill-on-stall", type=int, default=1,
                   help="1=terminate all workers and exit when stalled.")

    # sample timeout
    p.add_argument("--per-sample-timeout-sec", type=float, default=600.0,
                   help="If a single sample takes longer than N seconds, mark it as error and continue. 0 disables.")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> GenConfig:
    run_id = args.run_id or default_run_id()
    ckpt = resolve_model_ckpt(args.model_name, args.model_ckpt)
    gpus = parse_gpus(args.gpus)

    if torch.cuda.is_available() and len(gpus) == 0:
        raise ValueError("No GPUs selected/found. Provide --gpus or check CUDA_VISIBLE_DEVICES.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script currently assumes GPU inference.")

    oom_limit_max_pixels = None if int(args.oom_limit_max_pixels) <= 0 else int(args.oom_limit_max_pixels)
    oom_resize_long_edge = None if int(args.oom_resize_long_edge) <= 0 else int(args.oom_resize_long_edge)
    local_engine = str(getattr(args, "local_engine", "transformers") or "transformers").strip().lower()
    vllm_tp = int(getattr(args, "vllm_tensor_parallel_size", 0) or 0)
    if local_engine == "vllm":
        if vllm_tp <= 0:
            vllm_tp = len(gpus)
        if vllm_tp <= 0:
            raise ValueError("vLLM local engine requires at least one GPU.")
        if vllm_tp > len(gpus):
            raise ValueError(f"vLLM tensor_parallel_size={vllm_tp} exceeds selected gpus={gpus}.")

    return GenConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        model_name=args.model_name,
        model_ckpt=ckpt,
        base_model_ckpt=args.base_model_ckpt,
        processor_ckpt=args.processor_ckpt,
        run_id=run_id,

        prompt=args.prompt,

        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(int(args.do_sample)),
        temperature=float(args.temperature),
        top_p=float(args.top_p),

        torch_dtype=args.torch_dtype,

        local_engine=local_engine,
        vllm_tensor_parallel_size=max(1, vllm_tp),
        vllm_gpu_memory_utilization=float(getattr(args, "vllm_gpu_memory_utilization", 0.9)),
        vllm_dtype=str(getattr(args, "vllm_dtype", "auto") or "auto"),
        vllm_max_model_len=max(0, int(getattr(args, "vllm_max_model_len", 0) or 0)),
        vllm_limit_mm_per_prompt_image=max(1, int(getattr(args, "vllm_limit_mm_per_prompt_image", 1) or 1)),

        batch_size=max(1, int(args.batch_size)),
        batch_sort_by_image_size=bool(int(args.batch_sort_by_image_size)),

        gpus=gpus,

        mode=str(args.mode),
        dedup_after_run=bool(int(args.dedup_after_run)),
        resume_policy=str(args.resume_policy),
        resume_when_missing_field=str(args.resume_when_missing_field),

        oom_retry_empty_cache=bool(int(args.oom_retry_empty_cache)),
        oom_limit_max_pixels=oom_limit_max_pixels,
        oom_reduce_max_new_tokens_factor=float(args.oom_reduce_max_new_tokens_factor),
        oom_resize_long_edge=oom_resize_long_edge,
        oom_max_attempts=int(args.oom_max_attempts),

        progress_print_interval_sec=float(args.progress_print_interval_sec),
        watchdog_stall_sec=float(args.watchdog_stall_sec),
        watchdog_kill_on_stall=bool(int(args.watchdog_kill_on_stall)),

        per_sample_timeout_sec=float(args.per_sample_timeout_sec),
    )


# =========================
# 6) worker / main
# =========================
def collect_tasks(dataset_root: str, prompt: str) -> List[Dict[str, Any]]:
    json_files = list_json_files(dataset_root)
    if not json_files:
        raise RuntimeError(f"No json files found under: {dataset_root}")

    tasks = []
    json_meta_cache: Dict[str, Dict[str, Any]] = {}
    for jp in json_files:
        if jp not in json_meta_cache:
            try:
                mtime = os.path.getmtime(jp)
            except Exception:
                mtime = None
            try:
                content_sha1 = sha1_file(jp)
            except Exception:
                content_sha1 = None
            json_meta_cache[jp] = {
                "source_json_mtime": mtime,
                "source_json_sha1": content_sha1,
            }

        meta = json_meta_cache[jp]
        for idx, it in iter_items_from_json_file(jp):
            if not (isinstance(it, dict) and "image_path" in it):
                continue
            image_abs = resolve_image_path(jp, it["image_path"])
            domain = resolve_domain(jp, it, image_abs)
            sample_id = build_sample_id(image_abs, prompt)
            tasks.append({
                "sample_id": sample_id,
                "domain": domain,
                "image_path": image_abs,
                "source_json": jp,
                "index_in_source_json": idx,
                "source_json_mtime": meta.get("source_json_mtime"),
                "source_json_sha1": meta.get("source_json_sha1"),
            })
    return tasks


def worker_entry(
    worker_id: int,
    physical_gpu_id: int,
    cfg: GenConfig,
    stage_dir: str,
    task_queue,
    progress_queue,
    lock_path: str,
):
    """Transformers worker entrypoint with per-process GPU visibility.

    Each data-parallel transformers worker should own exactly one physical GPU.
    Narrow CUDA_VISIBLE_DEVICES before running worker_main so model loading and
    any later CUDA initialization see a single logical cuda:0 device.  vLLM does
    not use this wrapper because it needs visibility of multiple GPUs for tensor
    parallelism.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    return worker_main(
        worker_id=worker_id,
        gpu_id=0,
        cfg=cfg,
        stage_dir=stage_dir,
        task_queue=task_queue,
        progress_queue=progress_queue,
        lock_path=lock_path,
        physical_gpu_id=physical_gpu_id,
    )


def worker_main(
    worker_id: int,
    gpu_id: int,
    cfg: GenConfig,
    stage_dir: str,
    task_queue,
    progress_queue,
    lock_path: str,
    physical_gpu_id: Optional[int] = None,
):
    _best_effort_unbuffered_io()

    if physical_gpu_id is None:
        physical_gpu_id = gpu_id

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    worker_info = {
        "worker_id": worker_id,
        # Keep gpu_id as the physical GPU id for backward-compatible logs.
        "gpu_id": physical_gpu_id,
        "physical_gpu_id": physical_gpu_id,
        "local_cuda_device": gpu_id,
    }

    results_jsonl = os.path.join(stage_dir, "results.jsonl")
    errors_jsonl = os.path.join(stage_dir, "errors.jsonl")
    events_jsonl = os.path.join(stage_dir, "worker_events.jsonl")

    backend, model, processor = load_model_and_processor(
        cfg.model_name,
        cfg.model_ckpt,
        cfg.torch_dtype,
        device,
        base_model_ckpt=cfg.base_model_ckpt,
        processor_ckpt=cfg.processor_ckpt,
    )

    processed = 0
    failures = 0

    def _write_start_event(t: Dict[str, Any]) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(
                events_jsonl,
                {
                    "time": now_iso(),
                    "event": "start",
                    "sample_id": t.get("sample_id"),
                    "worker_id": worker_id,
                    "gpu_id": physical_gpu_id,
                    "physical_gpu_id": physical_gpu_id,
                    "local_cuda_device": gpu_id,
                    "image_path": t.get("image_path"),
                },
            )

    def _write_finish_event(sid: str, ok: bool) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(
                events_jsonl,
                {
                    "time": now_iso(),
                    "event": "finish",
                    "sample_id": sid,
                    "ok": bool(ok),
                    "worker_id": worker_id,
                    "gpu_id": physical_gpu_id,
                    "physical_gpu_id": physical_gpu_id,
                    "local_cuda_device": gpu_id,
                },
            )

    def _append_result(rec: Dict[str, Any]) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(results_jsonl, rec)

    def _append_error(err: Dict[str, Any]) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(errors_jsonl, err)

    def _report_progress(ok: bool) -> None:
        try:
            progress_queue.put({"ok": 1, "err": 0} if ok else {"ok": 0, "err": 1})
        except Exception:
            pass

    def _record_success(t: Dict[str, Any], response: str, degradation: Dict[str, Any], latency_sec: float) -> None:
        nonlocal processed
        finish_meta = degradation.get("finish") if isinstance(degradation, dict) else None
        if not isinstance(finish_meta, dict):
            finish_meta = {
                "output_token_count": None,
                "finish_reason": "unknown",
                "truncated_by_max_new_tokens": False,
            }
        rec = {
            "sample_id": t["sample_id"],
            "domain": t.get("domain"),
            "generated_at": now_iso(),
            "model_name": cfg.model_name,
            "model_ckpt": cfg.model_ckpt,
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
            "latency_sec": round(float(latency_sec), 4),
            "worker": dict(worker_info),
            "degradation": degradation,
            "output_token_count": finish_meta.get("output_token_count"),
            "finish_reason": finish_meta.get("finish_reason"),
            "truncated_by_max_new_tokens": bool(finish_meta.get("truncated_by_max_new_tokens", False)),
            "response": response,
        }
        _append_result(rec)
        processed += 1

    def _record_error(t: Dict[str, Any], e: BaseException) -> None:
        nonlocal failures
        failures += 1
        err = {
            "generated_at": now_iso(),
            "model_name": cfg.model_name,
            "model_ckpt": cfg.model_ckpt,
            "prompt": cfg.prompt,
            "worker": dict(worker_info),
            "sample_id": t.get("sample_id"),
            "domain": t.get("domain"),
            "source_json": t.get("source_json"),
            "index_in_source_json": t.get("index_in_source_json"),
            "image_path": t.get("image_path"),
            "error": repr(e),
            "traceback": traceback.format_exc(limit=20),
        }
        _append_error(err)

    def _process_single_task(t: Dict[str, Any]) -> None:
        sid = t["sample_id"]
        ok = False
        try:
            with SampleTimeout(cfg.per_sample_timeout_sec):
                if not os.path.exists(t["image_path"]):
                    raise FileNotFoundError(f"image not found: {t['image_path']}")

                image = Image.open(t["image_path"]).convert("RGB")

                t0 = time.time()
                response, degradation = generate_with_oom_policy(backend, model, processor, image, cfg)
                t1 = time.time()
                _record_success(t, response, degradation, t1 - t0)
                ok = True
        except Exception as e:
            _record_error(t, e)
        finally:
            _write_finish_event(sid, ok)
            _report_progress(ok)

    def _process_batch(tasks: List[Dict[str, Any]]) -> None:
        if not tasks:
            return
        if len(tasks) <= 1 or not bool(getattr(backend, "supports_batch", True)):
            # Some legacy/model-specific backends (e.g. InstructBLIP-Vicuna)
            # are safer in per-sample mode because their generation output is
            # not compatible with generic prompt-length slicing/batch decode.
            for t in tasks:
                _process_single_task(t)
            return

        if len(tasks) <= 1:
            _process_single_task(tasks[0])
            return

        valid: List[Tuple[Dict[str, Any], Image.Image, int]] = []
        for t in tasks:
            try:
                if not os.path.exists(t["image_path"]):
                    raise FileNotFoundError(f"image not found: {t['image_path']}")
                img = Image.open(t["image_path"]).convert("RGB")
                area = int(img.size[0]) * int(img.size[1])
                valid.append((t, img, area))
            except Exception as e:
                _record_error(t, e)
                _write_finish_event(t.get("sample_id", ""), False)
                _report_progress(False)

        if not valid:
            return

        if cfg.batch_sort_by_image_size:
            valid.sort(key=lambda x: x[2])

        batch_tasks = [x[0] for x in valid]
        batch_images = [x[1] for x in valid]

        t0 = time.time()
        try:
            if _processor_has_usable_chat_template(processor):
                messages_list = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": cfg.prompt},
                            ],
                        }
                    ]
                    for img in batch_images
                ]
                texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
                inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt")
            else:
                texts = [cfg.prompt] * len(batch_images)
                inputs = processor(text=texts, images=batch_images, padding=True, return_tensors="pt")
                inputs = _normalize_vlm_processor_outputs(inputs)

            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg.max_new_tokens),
                do_sample=bool(cfg.do_sample),
                temperature=float(cfg.temperature),
                top_p=float(cfg.top_p),
            )

            per_item_ids = []
            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
                    in_lens = inputs["attention_mask"].sum(dim=-1).tolist()
                else:
                    in_lens = [inputs["input_ids"].shape[-1]] * inputs["input_ids"].shape[0]
                for i in range(gen_ids.shape[0]):
                    in_len = int(in_lens[i])
                    per_item_ids.append(gen_ids[i, in_len:])
            else:
                per_item_ids = [gen_ids[i] for i in range(gen_ids.shape[0])]

            responses: List[str] = []
            for gid in per_item_ids:
                if hasattr(processor, "decode"):
                    responses.append(str(processor.decode(gid, skip_special_tokens=True)).strip())
                else:
                    responses.append("")

            t1 = time.time()
            batch_latency = max(1e-6, t1 - t0)
            eos_ids = _get_eos_token_ids(model, processor)

            for t, resp, gid in zip(batch_tasks, responses, per_item_ids):
                out_len = int(gid.shape[-1]) if hasattr(gid, "shape") else 0
                last_id = None
                try:
                    if out_len > 0:
                        last_id = int(gid[-1].item())
                except Exception:
                    last_id = None
                ended_with_eos = (last_id in eos_ids) if (last_id is not None and eos_ids) else False
                truncated = (out_len >= int(cfg.max_new_tokens)) and (not ended_with_eos)
                finish_reason = "eos" if ended_with_eos else ("length" if truncated else "unknown")

                degradation = {
                    "used": False,
                    "steps": [],
                    "attempts": [{"attempt": 0, "batch_size": len(batch_tasks)}],
                    "final_settings": {
                        "max_new_tokens": cfg.max_new_tokens,
                        "max_pixels": None,
                        "resized_to": None,
                    },
                    "image_info": {
                        "orig_size": None,
                        "final_size": None,
                    },
                    "gen_time_sec": round(batch_latency / float(len(batch_tasks)), 4),
                    "batch_info": {
                        "batch_size": len(batch_tasks),
                        "sorted_by_image_size": bool(cfg.batch_sort_by_image_size),
                    },
                    "finish": {
                        "output_token_count": out_len,
                        "finish_reason": finish_reason,
                        "truncated_by_max_new_tokens": bool(truncated),
                    },
                }
                _record_success(t, resp, degradation, batch_latency / float(len(batch_tasks)))
                _write_finish_event(t["sample_id"], True)
                _report_progress(True)

        except Exception:
            # Batch failed: fallback to per-sample path to preserve robustness.
            for t in batch_tasks:
                _process_single_task(t)

    stop = False
    pending: List[Dict[str, Any]] = []
    while True:
        while (not stop) and (len(pending) < max(1, int(cfg.batch_size))):
            t = task_queue.get()
            if t is None:
                stop = True
                break
            pending.append(t)

        if not pending and stop:
            break

        batch = pending
        pending = []

        for t in batch:
            _write_start_event(t)

        _process_batch(batch)

    with FileLock(lock_path):
        safe_jsonl_append(
            os.path.join(stage_dir, "worker_log.jsonl"),
            {
                "time": now_iso(),
                "worker_id": worker_id,
                "gpu_id": physical_gpu_id,
                "physical_gpu_id": physical_gpu_id,
                "local_cuda_device": gpu_id,
                "processed": processed,
                "failures": failures,
            },
        )


def _dispatch_tasks(tasks_todo: List[Dict[str, Any]], task_queue, num_workers: int, dispatch_state: Dict[str, Any], cfg: GenConfig, stage_dir: str) -> None:
    dispatch_log = os.path.join(stage_dir, "dispatch_log.jsonl")
    total = len(tasks_todo)
    try:
        for i, t in enumerate(tasks_todo, start=1):
            task_queue.put(t)
            dispatch_state["dispatched"] = i
            dispatch_state["last_dispatch_ts"] = time.time()
            if (i == 1) or (i % 50 == 0) or (i == total):
                _eprint(f"[dispatch] {i}/{total}")
                try:
                    safe_jsonl_append(dispatch_log, {"time": now_iso(), "event": "dispatch", "dispatched": i, "total": total})
                except Exception:
                    pass

        for _ in range(num_workers):
            task_queue.put(None)

        dispatch_state["done"] = True
        dispatch_state["last_dispatch_ts"] = time.time()
        _eprint(f"[dispatch] sent sentinels: {num_workers}")
        try:
            safe_jsonl_append(dispatch_log, {"time": now_iso(), "event": "sentinels_sent", "num_workers": num_workers})
        except Exception:
            pass

    except Exception as e:
        dispatch_state["error"] = repr(e)
        dispatch_state["done"] = False
        dispatch_state["last_dispatch_ts"] = time.time()
        _eprint(f"[dispatch] error: {repr(e)}")
        try:
            safe_jsonl_append(dispatch_log, {"time": now_iso(), "event": "dispatch_error", "error": repr(e), "traceback": traceback.format_exc(limit=20)})
        except Exception:
            pass


def _monitor_progress(total: int, progress_queue, procs: List[mp.Process], cfg: GenConfig, stage_dir: str, dispatch_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if total <= 0:
        return {"done": 0, "ok": 0, "err": 0, "aborted": False, "reason": None}

    watchdog_log = os.path.join(stage_dir, "watchdog_log.jsonl")

    done = 0
    ok = 0
    err = 0

    last_progress_ts = time.time()
    last_print_ts = 0.0

    # Pump mp.SimpleQueue -> thread queue (so we can timeout reliably)
    tq: _thread_queue.Queue = _thread_queue.Queue()

    def _pump():
        while True:
            m = progress_queue.get()
            if m is None:
                break
            tq.put(m)

    pump_th = threading.Thread(target=_pump, daemon=True)
    pump_th.start()

    # Render tqdm only when stderr is a TTY; otherwise fall back to periodic prints.
    use_tqdm = bool(_HAS_TQDM and hasattr(sys.stderr, "isatty") and sys.stderr.isatty())

    if use_tqdm:
        pbar = tqdm(total=total, desc="Generating", unit="sample", dynamic_ncols=True, file=sys.stderr)
        pbar.set_postfix({"ok": ok, "err": err})
    else:
        pbar = None
        _eprint(f"[progress] total={total} (tqdm disabled or non-TTY; printing every {cfg.progress_print_interval_sec}s)")

    aborted = False
    reason = None

    try:
        while done < total:
            now = time.time()

            # watchdog: stalled
            if (now - last_progress_ts) > cfg.watchdog_stall_sec:
                alive = [{"pid": p.pid, "alive": p.is_alive(), "exitcode": p.exitcode} for p in procs]
                ds = dispatch_state or {}
                msg = {
                    "time": now_iso(),
                    "event": "stall_detected",
                    "stall_sec": round(now - last_progress_ts, 2),
                    "done": done,
                    "total": total,
                    "ok": ok,
                    "err": err,
                    "dispatched": int(ds.get("dispatched", -1)),
                    "dispatch_done": bool(ds.get("done", False)),
                    "dispatch_error": ds.get("error", None),
                    "workers": alive,
                }
                try:
                    safe_jsonl_append(watchdog_log, msg)
                except Exception:
                    pass

                _eprint(
                    f"[watchdog] stalled for {msg['stall_sec']}s: done={done}/{total} ok={ok} err={err} "
                    f"dispatched={msg['dispatched']} dispatch_done={msg['dispatch_done']} workers={alive}"
                )

                if cfg.watchdog_kill_on_stall:
                    aborted = True
                    reason = "stall_detected"
                    _eprint("[watchdog] terminating workers due to stall...")
                    for p in procs:
                        try:
                            if p.is_alive():
                                p.terminate()
                        except Exception:
                            pass
                    break
                else:
                    last_progress_ts = time.time()

            # print periodically if not using tqdm
            if (pbar is None) and (now - last_print_ts) >= cfg.progress_print_interval_sec:
                alive_cnt = sum(1 for p in procs if p.is_alive())
                ds = dispatch_state or {}
                disp = int(ds.get("dispatched", 0))
                _eprint(f"[progress] {done}/{total} ok={ok} err={err} alive_workers={alive_cnt} dispatched={disp}/{total}")
                last_print_ts = now

            # If all workers died unexpectedly, avoid hanging forever.
            if all((not p.is_alive()) for p in procs):
                alive = [{"pid": p.pid, "alive": p.is_alive(), "exitcode": p.exitcode} for p in procs]
                ds = dispatch_state or {}
                try:
                    safe_jsonl_append(
                        watchdog_log,
                        {
                            "time": now_iso(),
                            "event": "all_workers_exited_early",
                            "done": done,
                            "total": total,
                            "ok": ok,
                            "err": err,
                            "dispatched": int(ds.get("dispatched", -1)),
                            "dispatch_done": bool(ds.get("done", False)),
                            "dispatch_error": ds.get("error", None),
                            "workers": alive,
                        },
                    )
                except Exception:
                    pass

                _eprint(f"[progress] all workers exited early: done={done}/{total} ok={ok} err={err} workers={alive}")
                aborted = True
                reason = "all_workers_exited_early"
                break

            try:
                msg = tq.get(timeout=0.5)
            except Exception:
                continue

            if not msg:
                continue

            inc_ok = int(msg.get("ok", 0))
            inc_err = int(msg.get("err", 0))
            inc = inc_ok + inc_err
            if inc <= 0:
                continue

            ok += inc_ok
            err += inc_err
            done += inc
            last_progress_ts = time.time()

            if pbar is not None:
                pbar.update(inc)
                pbar.set_postfix({"ok": ok, "err": err})
    finally:
        if pbar is not None:
            pbar.close()
        try:
            progress_queue.put(None)
        except Exception:
            pass

    return {"done": done, "ok": ok, "err": err, "aborted": aborted, "reason": reason}


def _run_generation_vllm(
    cfg: GenConfig,
    stage_dir: str,
    results_jsonl: str,
    errors_jsonl: str,
    lock_path: str,
    tasks_todo: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run Stage-1 generation with one embedded vLLM engine.

    Unlike the transformers path, vLLM uses tensor parallelism inside a single
    engine. Therefore ``cfg.gpus`` denotes the device set owned by that engine,
    not one data-parallel worker per GPU.
    """
    backend = VLLMMLLMBackend(
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        dtype=cfg.vllm_dtype,
        max_model_len=cfg.vllm_max_model_len,
        limit_mm_per_prompt_image=cfg.vllm_limit_mm_per_prompt_image,
        gpus=cfg.gpus,
    )
    try:
        model, processor = backend.load_model_and_processor(
            model_ckpt=cfg.model_ckpt,
            torch_dtype=cfg.torch_dtype,
            device=None,
            base_model_ckpt=cfg.base_model_ckpt,
            processor_ckpt=cfg.processor_ckpt,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize embedded vLLM MLLM engine. Consider checking model support in vLLM, "
            "--gpus, --vllm-tensor-parallel-size, --vllm-gpu-memory-utilization, and --vllm-max-model-len. "
            f"Original error: {repr(e)}"
        ) from e

    total = len(tasks_todo)
    ok_count = 0
    err_count = 0
    last_print_ts = 0.0
    use_tqdm = bool(_HAS_TQDM and hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
    pbar = tqdm(total=total, desc="Generating(vLLM)", unit="sample", dynamic_ncols=True, file=sys.stderr) if use_tqdm else None
    if pbar is None:
        _eprint(f"[vllm] total={total} (tqdm disabled or non-TTY; printing every {cfg.progress_print_interval_sec}s)")

    def _append_result(rec: Dict[str, Any]) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(results_jsonl, rec)

    def _append_error(err: Dict[str, Any]) -> None:
        with FileLock(lock_path):
            safe_jsonl_append(errors_jsonl, err)

    try:
        for i, t in enumerate(tasks_todo, start=1):
            try:
                with SampleTimeout(cfg.per_sample_timeout_sec):
                    if not os.path.exists(t["image_path"]):
                        raise FileNotFoundError(f"image not found: {t['image_path']}")
                    image = Image.open(t["image_path"]).convert("RGB")
                    t0 = time.time()
                    response, degradation = generate_with_oom_policy(backend, model, processor, image, cfg)
                    t1 = time.time()
                    finish_meta = degradation.get("finish") if isinstance(degradation, dict) else None
                    if not isinstance(finish_meta, dict):
                        finish_meta = {
                            "output_token_count": None,
                            "finish_reason": "unknown",
                            "truncated_by_max_new_tokens": False,
                        }
                    _append_result({
                        "sample_id": t["sample_id"],
                        "domain": t.get("domain"),
                        "generated_at": now_iso(),
                        "model_name": cfg.model_name,
                        "model_ckpt": cfg.model_ckpt,
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
                        "latency_sec": round(float(t1 - t0), 4),
                        "worker": {"worker_id": 0, "gpu_id": cfg.gpus, "engine": "vllm"},
                        "degradation": degradation,
                        "output_token_count": finish_meta.get("output_token_count"),
                        "finish_reason": finish_meta.get("finish_reason"),
                        "truncated_by_max_new_tokens": bool(finish_meta.get("truncated_by_max_new_tokens", False)),
                        "response": response,
                    })
                    ok_count += 1
            except Exception as e:
                err_count += 1
                _append_error({
                    "generated_at": now_iso(),
                    "model_name": cfg.model_name,
                    "model_ckpt": cfg.model_ckpt,
                    "prompt": cfg.prompt,
                    "worker": {"worker_id": 0, "gpu_id": cfg.gpus, "engine": "vllm"},
                    "sample_id": t.get("sample_id"),
                    "domain": t.get("domain"),
                    "source_json": t.get("source_json"),
                    "index_in_source_json": t.get("index_in_source_json"),
                    "image_path": t.get("image_path"),
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=20),
                })
            finally:
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"ok": ok_count, "err": err_count})
                else:
                    now = time.time()
                    if (i == 1) or (i == total) or ((now - last_print_ts) >= cfg.progress_print_interval_sec):
                        _eprint(f"[vllm-progress] {i}/{total} ok={ok_count} err={err_count}")
                        last_print_ts = now
    finally:
        if pbar is not None:
            pbar.close()

    return {"done": total, "ok": ok_count, "err": err_count, "aborted": False, "reason": None}


def run_generation(cfg: GenConfig):
    _best_effort_unbuffered_io()

    # Use explicit spawn context for CUDA safety even when caller script didn't
    # set global multiprocessing start method early enough.
    ctx = mp.get_context("spawn")

    run_dir = os.path.join(cfg.output_root, cfg.model_name, f"stage1_answers-{cfg.run_id}")
    ensure_dir(run_dir)

    stage_dir = os.path.join(run_dir, "stage1_outputs")
    ensure_dir(stage_dir)

    results_jsonl = os.path.join(stage_dir, "results.jsonl")
    errors_jsonl = os.path.join(stage_dir, "errors.jsonl")
    meta_json = os.path.join(stage_dir, "meta.json")
    lock_path = os.path.join(stage_dir, ".write.lock")

    # In resume mode, guard against accidentally changing core params without
    # changing run-id, which would silently mix results from different settings.
    if cfg.mode == "resume":
        check_resume_manifest(meta_json, cfg)

    if not os.path.exists(results_jsonl):
        open(results_jsonl, "a", encoding="utf-8").close()
    if not os.path.exists(errors_jsonl):
        open(errors_jsonl, "a", encoding="utf-8").close()

    meta = RunMeta(
        run_id=cfg.run_id,
        created_at=now_iso(),
        dataset_root=cfg.dataset_root,
        model_name=cfg.model_name,
        model_ckpt=cfg.model_ckpt,
        prompt=cfg.prompt,
        decoding_params={
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        runtime=dict({
            "torch_dtype": cfg.torch_dtype,
            "local_engine": cfg.local_engine,
            "selected_mllm_backend": "vllm_mllm" if cfg.local_engine == "vllm" else getattr(get_backend(cfg.model_name, cfg.model_ckpt), "name", "unknown"),
            "vllm_params": {
                "tensor_parallel_size": cfg.vllm_tensor_parallel_size,
                "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
                "dtype": cfg.vllm_dtype,
                "max_model_len": cfg.vllm_max_model_len,
                "limit_mm_per_prompt_image": cfg.vllm_limit_mm_per_prompt_image,
            } if cfg.local_engine == "vllm" else None,
            "has_qwen_vl_utils": _HAS_QWEN_VL_UTILS,
            "has_qwen3_vl_class": _HAS_QWEN3_VL_CLASS,
            "has_auto_model_for_image_text_to_text": _HAS_AUTO_ITTT,
            "has_auto_model_for_vision2seq": _HAS_AUTO_V2S,
            "qwen3_vl_import_error": _QWEN3_VL_IMPORT_ERROR,
            "auto_vision2seq_import_error": _AUTO_V2S_IMPORT_ERROR,
            "auto_image_text_to_text_import_error": _AUTO_ITTT_IMPORT_ERROR,
            "torch": torch.__version__,
            "has_tqdm": _HAS_TQDM,
        }, **backend_runtime_flags()),
        parallel={
            "gpus": cfg.gpus,
            "num_workers": 1 if cfg.local_engine == "vllm" else len(cfg.gpus),
            "strategy": "single embedded vLLM engine (tensor parallel)" if cfg.local_engine == "vllm" else "1 process per GPU (data parallel)",
            "batch_size_per_worker": cfg.batch_size,
            "batch_sort_by_image_size": cfg.batch_sort_by_image_size,
        },
        resume={
            "mode": cfg.mode,
            "resume_policy": cfg.resume_policy,
            "resume_when_missing_field": cfg.resume_when_missing_field,
            "dedup_after_run": cfg.dedup_after_run,
        },
        oom_policy={
            "oom_max_attempts": cfg.oom_max_attempts,
            "oom_retry_empty_cache": cfg.oom_retry_empty_cache,
            "oom_limit_max_pixels": cfg.oom_limit_max_pixels,
            "oom_reduce_max_new_tokens_factor": cfg.oom_reduce_max_new_tokens_factor,
            "oom_resize_long_edge": cfg.oom_resize_long_edge,
        },
    )
    if not os.path.exists(meta_json):
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    tasks_all = collect_tasks(cfg.dataset_root, cfg.prompt)

    resume_stats: Dict[str, int] = {}
    if cfg.mode == "force":
        tasks_todo = tasks_all
    else:  # resume
        with FileLock(lock_path):
            latest_by_sid = load_latest_results_by_sample_id(results_jsonl)

        tasks_todo = []
        for t in tasks_all:
            sid = t["sample_id"]
            rerun, reason = should_rerun_task(t, latest_by_sid.get(sid), cfg)
            resume_stats[reason] = int(resume_stats.get(reason, 0)) + 1
            if rerun:
                tasks_todo.append(t)

    _eprint(
        f"[run] tasks_total={len(tasks_all)} tasks_todo={len(tasks_todo)} gpus={cfg.gpus} mode={cfg.mode} "
        f"resume_policy={cfg.resume_policy} missing_field={cfg.resume_when_missing_field}"
    )
    if cfg.mode == "resume":
        _eprint(f"[run] resume_stats={resume_stats}")

    if cfg.local_engine == "vllm":
        prog = _run_generation_vllm(
            cfg=cfg,
            stage_dir=stage_dir,
            results_jsonl=results_jsonl,
            errors_jsonl=errors_jsonl,
            lock_path=lock_path,
            tasks_todo=tasks_todo,
        )

        if cfg.dedup_after_run:
            with FileLock(lock_path):
                stats = dedup_jsonl_keep_last_inplace(results_jsonl, key="sample_id")
                safe_jsonl_append(
                    os.path.join(stage_dir, "dedup_log.jsonl"),
                    {"time": now_iso(), "action": "dedup_keep_last", "stats": stats},
                )

        print(
            "Done.\n"
            f"  results: {results_jsonl}\n"
            f"  errors:  {errors_jsonl}\n"
            f"  meta:    {meta_json}\n"
            f"  gpus:    {cfg.gpus}\n"
            f"  mode:    {cfg.mode}\n"
            f"  local_engine: vllm\n"
            f"  vllm_tensor_parallel_size: {cfg.vllm_tensor_parallel_size}\n"
            f"  dedup_after_run: {cfg.dedup_after_run}\n"
            f"  tasks_total: {len(tasks_all)}, tasks_todo: {len(tasks_todo)}\n"
            f"  progress: done={prog.get('done', 0)} ok={prog.get('ok', 0)} err={prog.get('err', 0)}\n"
        )
        return

    num_workers = len(cfg.gpus)
    if num_workers <= 0:
        raise ValueError("No GPUs selected.")

    task_queue = ctx.SimpleQueue()
    # IMPORTANT: avoid deadlock on put() when parent can't consume fast enough
    progress_queue = ctx.SimpleQueue()

    # start workers
    procs = []
    for worker_id, gpu_id in enumerate(cfg.gpus):
        p = ctx.Process(
            target=worker_entry,
            args=(worker_id, gpu_id, cfg, stage_dir, task_queue, progress_queue, lock_path),
            daemon=False,
        )
        p.start()
        procs.append(p)

    dispatch_state = {"dispatched": 0, "done": False, "error": None, "last_dispatch_ts": time.time()}
    dispatch_th = threading.Thread(
        target=_dispatch_tasks,
        args=(tasks_todo, task_queue, num_workers, dispatch_state, cfg, stage_dir),
        daemon=True,
    )
    dispatch_th.start()

    # progress monitoring in parent (runs even while dispatch is ongoing)
    prog = _monitor_progress(
        total=len(tasks_todo),
        progress_queue=progress_queue,
        procs=procs,
        cfg=cfg,
        stage_dir=stage_dir,
        dispatch_state=dispatch_state,
    )

    # ensure workers are not left behind
    if prog.get("aborted", False):
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass

    for p in procs:
        p.join()

    if cfg.dedup_after_run:
        with FileLock(lock_path):
            stats = dedup_jsonl_keep_last_inplace(results_jsonl, key="sample_id")
            safe_jsonl_append(
                os.path.join(stage_dir, "dedup_log.jsonl"),
                {"time": now_iso(), "action": "dedup_keep_last", "stats": stats},
            )

    remaining = max(0, len(tasks_todo) - int(prog.get("done", 0)))

    if prog.get("aborted", False) and remaining > 0:
        _eprint(
            "[aborted] generation did NOT finish.\n"
            f"  reason: {prog.get('reason')}\n"
            f"  tasks_todo: {len(tasks_todo)}\n"
            f"  progress: done={prog.get('done', 0)} ok={prog.get('ok', 0)} err={prog.get('err', 0)} remaining={remaining}\n"
            f"  dispatched: {int(dispatch_state.get('dispatched', 0))}/{len(tasks_todo)} dispatch_done={bool(dispatch_state.get('done', False))} dispatch_error={dispatch_state.get('error')}\n"
            "  You can rerun with --mode resume to continue.\n"
        )
        sys.exit(2)

    print(
        "Done.\n"
        f"  results: {results_jsonl}\n"
        f"  errors:  {errors_jsonl}\n"
        f"  meta:    {meta_json}\n"
        f"  gpus:    {cfg.gpus}\n"
        f"  mode:    {cfg.mode}\n"
        f"  dedup_after_run: {cfg.dedup_after_run}\n"
        f"  tasks_total: {len(tasks_all)}, tasks_todo: {len(tasks_todo)}\n"
        f"  progress: done={prog.get('done', 0)} ok={prog.get('ok', 0)} err={prog.get('err', 0)}\n"
    )


def main():
    args = parse_args()
    cfg = build_config(args)

    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    run_generation(cfg)


if __name__ == "__main__":
    main()
