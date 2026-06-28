#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

# Linux file lock
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    fcntl = None
    _HAS_FCNTL = False


# =========================================================
# 0) Utilities
# =========================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _eprint(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def safe_jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def dedup_keep_last(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    last: Dict[Any, int] = {}
    for i, r in enumerate(records):
        k = r.get(key)
        if k is not None:
            last[k] = i
    keep = set(last.values())
    return [records[i] for i in range(len(records)) if i in keep]


def read_done_ids(per_sample_jsonl: str) -> Set[str]:
    done: Set[str] = set()
    for obj in iter_jsonl(per_sample_jsonl):
        sid = obj.get("sample_id")
        if sid:
            done.add(str(sid))
    return done


def _to_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _normalize_fact_index_list(v: Any, total_gt_fact_count: int) -> List[int]:
    out: List[int] = []
    seen = set()
    if not isinstance(v, list):
        return out
    for x in v:
        idx = _to_int(x, default=-1)
        if 0 <= idx < total_gt_fact_count and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _normalize_bool_optional(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class FileLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fd = None
        self._use_fcntl = bool(_HAS_FCNTL)
        self._use_msvcrt = sys.platform.startswith("win")
        if not self._use_fcntl and not self._use_msvcrt:
            raise RuntimeError("No supported file locking backend on this platform.")

    def __enter__(self):
        ensure_dir(os.path.dirname(self.lock_path))
        self._fd = open(self.lock_path, "a+", encoding="utf-8")
        if self._use_fcntl:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        else:
            import msvcrt  # noqa: PLC0415
            self._fd.seek(0)
            msvcrt.locking(self._fd.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                if self._use_fcntl:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                else:
                    import msvcrt  # noqa: PLC0415
                    self._fd.seek(0)
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                if self._fd is not None:
                    self._fd.close()
            except Exception:
                pass
            self._fd = None


# =========================================================
# 1) Robust JSON parsing helpers
# =========================================================
def _strip_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _iter_balanced_json_objects(text: str):
    start = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _iter_balanced_json_arrays(text: str):
    start = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _json_balance_counts(text: str) -> Dict[str, int]:
    curly = 0
    square = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            curly += 1
        elif ch == "}":
            curly -= 1
        elif ch == "[":
            square += 1
        elif ch == "]":
            square -= 1
    return {"curly": curly, "square": square}


def _repair_json_array_candidate(candidate: str) -> List[Tuple[str, str]]:
    """Return conservative repaired JSON-array candidates.

    The repair is intentionally narrow: remove trailing commas and fill missing
    closing delimiters, especially the common `...}]` vs expected `...}}]` case.
    A repaired candidate is accepted only if json.loads succeeds later.
    """
    s = str(candidate or "").strip()
    if not s:
        return []
    out: List[Tuple[str, str]] = []

    def add(text: str, strategy: str) -> None:
        text = text.strip()
        if text and text != s and all(prev != text for prev, _ in out):
            out.append((text, strategy))

    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", s)
    add(no_trailing_commas, "remove_trailing_commas")

    base_candidates = [s]
    if no_trailing_commas != s:
        base_candidates.append(no_trailing_commas)

    for base in base_candidates:
        counts = _json_balance_counts(base)
        missing_curly = max(0, int(counts.get("curly") or 0))
        missing_square = max(0, int(counts.get("square") or 0))
        if missing_curly == 0 and missing_square == 0:
            continue

        # If the array is already closed but objects are not, insert missing
        # object braces before the final array bracket. This handles outputs like
        # [{"a": {"b": 1}] where the last object misses one or more `}`.
        if missing_curly > 0 and base.rstrip().endswith("]"):
            repaired = base.rstrip()[:-1] + ("}" * missing_curly) + "]" + ("}" * max(0, -int(counts.get("square") or 0)))
            add(repaired, f"insert_{missing_curly}_missing_object_brace_before_final_array_bracket")

        repaired = base + ("}" * missing_curly) + ("]" * missing_square)
        add(repaired, f"append_missing_delimiters_curly_{missing_curly}_square_{missing_square}")

    return out


def parse_json_object(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidates: List[str] = []
    text = raw_text.strip()
    if text:
        candidates.append(text)
    stripped = _strip_code_fence(raw_text)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            last_error = f"expected JSON object, got {type(obj).__name__}"
        except Exception as e:
            last_error = repr(e)

        for block in _iter_balanced_json_objects(candidate):
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj, None
                last_error = f"expected JSON object, got {type(obj).__name__}"
            except Exception as e2:
                last_error = repr(e2)

    return None, last_error or "no JSON object found"


def parse_json_array(raw_text: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    candidates: List[str] = []
    text = raw_text.strip()
    if text:
        candidates.append(text)
    stripped = _strip_code_fence(raw_text)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj, None
            last_error = f"expected JSON array, got {type(obj).__name__}"
        except Exception as e:
            last_error = repr(e)

        for block in _iter_balanced_json_arrays(candidate):
            try:
                obj = json.loads(block)
                if isinstance(obj, list):
                    return obj, None
                last_error = f"expected JSON array, got {type(obj).__name__}"
            except Exception as e2:
                last_error = repr(e2)

        for repaired, strategy in _repair_json_array_candidate(candidate):
            try:
                obj = json.loads(repaired)
                if isinstance(obj, list):
                    return obj, f"repaired_json_array:{strategy}; original_error={last_error}"
                last_error = f"expected JSON array after repair {strategy}, got {type(obj).__name__}"
            except Exception as e3:
                last_error = repr(e3)

    return None, last_error or "no JSON array found"


def _parse_pred_index_key(key: Any) -> int:
    if isinstance(key, int):
        return key if key >= 0 else -1
    s = str(key).strip()
    if s.isdigit():
        return int(s)
    m = re.fullmatch(r"(?i)(?:p|pred(?:_index)?)\s*[-_:]?\s*(\d+)", s)
    if m:
        return int(m.group(1))
    return -1


# =========================================================
# 2) Annotation loading
# =========================================================
_json_cache: Dict[str, Any] = {}


def load_json_cached(path: str) -> Any:
    if path not in _json_cache:
        _json_cache[path] = load_json(path)
    return _json_cache[path]


def resolve_annotation(source_json: str, index_in_source_json: Optional[int]) -> Dict[str, Any]:
    data = load_json_cached(source_json)
    if isinstance(data, list):
        if index_in_source_json is None:
            raise ValueError(f"source_json is a list but index_in_source_json is None: {source_json}")
        if index_in_source_json < 0 or index_in_source_json >= len(data):
            raise IndexError(f"index_in_source_json out of range: {index_in_source_json} for {source_json}")
        ann = data[index_in_source_json]
        if not isinstance(ann, dict):
            raise TypeError(f"annotation is not a dict: {source_json}[{index_in_source_json}]")
        return ann
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected top-level type in source_json: {type(data).__name__}")


def get_gt_object_lists(annotation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    required = annotation.get("object_list")
    optional = annotation.get("optional_object_list")
    if optional is None:
        optional = annotation.get("object_list_optional")

    if not isinstance(required, list):
        required = []
    if not isinstance(optional, list):
        optional = []

    if not required and not optional:
        required = (annotation.get("object_list_fg") or []) + (annotation.get("object_list_normal") or [])

    required = [x for x in required if isinstance(x, dict)]
    optional = [x for x in optional if isinstance(x, dict)]
    return {"required": required, "optional": optional}


# =========================================================
# 3) Text / attribute / number normalization
# =========================================================
PRED_CATEGORY_FIELD = "finest_category"
GT_CATEGORY_FIELD = "multi-granularity categories"


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    text = str(s).lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).split())


def _token_set(s: str) -> Set[str]:
    return set(s.split()) if s else set()


def _value_match_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and a in b:
        return 1.0
    if len(b) >= 4 and b in a:
        return 1.0
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    union = len(ta | tb)
    return inter / max(union, 1)


def _flatten_attr_facts(v: Any, path: str = "") -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    if v is None:
        return facts
    if isinstance(v, dict):
        for k, vv in v.items():
            child_path = f"{path}.{k}" if path else str(k)
            facts.extend(_flatten_attr_facts(vv, child_path))
        return facts
    if isinstance(v, list):
        for i, vv in enumerate(v):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            facts.extend(_flatten_attr_facts(vv, child_path))
        return facts
    value = str(v)
    facts.append({
        "path": path or "value",
        "value": value,
        "normalized_value": _normalize_text(value),
    })
    return facts


def _normalize_attr_name(k: Any) -> Optional[str]:
    if k is None:
        return None
    s = str(k).strip()
    return s if s else None


def _normalize_attr_value(v: Any) -> List[str]:
    out: List[str] = []

    def add_one(x: Any):
        if x is None:
            return
        sx = str(x).strip()
        if sx:
            out.append(sx)

    if v is None:
        return out
    if isinstance(v, list):
        for x in v:
            add_one(x)
    else:
        add_one(v)
    return out


def _parse_name_value_string(s: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"^\s*name\s*:\s*(.*?)\s*,\s*value\s*:\s*(.*)\s*$", s, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def normalize_attributes_structure(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    attrs = obj.get("attributes")
    normalized: Dict[str, List[str]] = {}

    def add_attr(k: Any, v: Any) -> None:
        nk = _normalize_attr_name(k)
        if not nk:
            return
        bucket = normalized.setdefault(nk, [])
        vals = _normalize_attr_value(v)
        for x in vals:
            bucket.append(x)

    if isinstance(attrs, dict):
        for k, v in attrs.items():
            add_attr(k, v)
    elif isinstance(attrs, list):
        for item in attrs:
            if isinstance(item, dict):
                if "name" in item and "value" in item:
                    add_attr(item.get("name"), item.get("value"))
                elif len(item) == 1:
                    k = next(iter(item.keys()))
                    add_attr(k, item.get(k))
                else:
                    for k, v in item.items():
                        add_attr(k, v)
            elif isinstance(item, str):
                nk, nv = _parse_name_value_string(item)
                if nk is not None:
                    add_attr(nk, nv)
                else:
                    add_attr("other_details", item)
            else:
                add_attr("other_details", item)
    elif isinstance(attrs, str):
        nk, nv = _parse_name_value_string(attrs)
        if nk is not None:
            add_attr(nk, nv)
        else:
            add_attr("other_details", attrs)
    elif attrs is not None:
        add_attr("other_details", attrs)

    # legacy top-level
    if "number" in obj:
        add_attr("number", obj.get("number"))
    if "position" in obj:
        add_attr("position", obj.get("position"))
    if "quantity" in obj:
        add_attr("quantity", obj.get("quantity"))
        add_attr("number", obj.get("quantity"))

    # dedup while preserving order
    out: Dict[str, List[str]] = {}
    for k, vals in normalized.items():
        seen = set()
        dst: List[str] = []
        for x in vals:
            if x not in seen:
                seen.add(x)
                dst.append(x)
        out[k] = dst
    return out


_NUMBER_WORDS = {
    "zero": 0,
    "a": 1,          # intentionally conservative; only used for full-token exact match
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "pair": 2,
    "couple": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_UNCERTAIN_PATTERNS = [
    r"\buncertain\b",
    r"\bunknown\b",
    r"\bseveral\b",
    r"\bmultiple\b",
    r"\bmany\b",
    r"\bsome\b",
    r"\bfew\b",
    r"\ba few\b",
    r"\bvarious\b",
    r"\bnumerous\b",
    r"\bmixed quantity\b",
]


def _parse_single_number_string(s: str) -> Tuple[Optional[int], bool]:
    text = str(s).strip().lower()
    if not text:
        return None, False

    for pat in _UNCERTAIN_PATTERNS:
        if re.search(pat, text):
            return None, True

    m = re.search(r"\b(\d+)\b", text)
    if m:
        try:
            return int(m.group(1)), False
        except Exception:
            pass

    tokens = re.findall(r"[a-zA-Z]+", text)
    for tk in tokens:
        if tk in _NUMBER_WORDS:
            return int(_NUMBER_WORDS[tk]), False

    if text in _NUMBER_WORDS:
        return int(_NUMBER_WORDS[text]), False

    return None, False


def _collect_number_raw_vals(attrs: Dict[str, List[str]]) -> List[str]:
    raw_vals: List[str] = []
    for k in ("number", "quantity"):
        if k in attrs and isinstance(attrs[k], list):
            raw_vals.extend(attrs[k])
    return raw_vals


def _contains_gt_uncertain_marker(text: str) -> bool:
    lowered = str(text).strip().lower()
    return bool(re.search(r"\b(?:unknown|uncertain)\b", lowered))


def parse_gt_number_field_from_attrs(attrs: Dict[str, List[str]]) -> Tuple[int, str, str]:
    raw_vals = _collect_number_raw_vals(attrs)

    if not raw_vals:
        return 1, "default_missing", "default_missing"

    numeric_vals: List[int] = []
    saw_uncertain = False
    for x in raw_vals:
        text = str(x).strip()
        lowered = text.lower()
        if not text:
            raise ValueError("GT number field contains empty value")
        if _contains_gt_uncertain_marker(lowered):
            saw_uncertain = True
            continue
        if re.fullmatch(r"\d+", text):
            numeric_vals.append(int(text))
            continue
        raise ValueError(f"GT number field has unsupported value: {x!r}")

    if saw_uncertain:
        return 1, "uncertain", "parsed_attribute"

    uniq_numeric = sorted(set(numeric_vals))
    if len(uniq_numeric) == 1:
        return int(uniq_numeric[0]), "numeric", "parsed_attribute"
    if len(uniq_numeric) >= 2:
        raise ValueError(f"GT number field has conflicting numeric values: {raw_vals!r}")

    raise ValueError(f"GT number field has unsupported value(s): {raw_vals!r}")


def _pred_number_string_to_result(s: str) -> Tuple[Optional[int], bool]:
    text = str(s).strip()
    lowered = text.lower()
    if not text:
        return None, False

    if re.search(r"\b(?:unknown|uncertain)\b", lowered):
        return None, True

    range_or_bound_patterns = [
        r"\b\d+\s*[-~]\s*\d+\b",
        r"\b\d+\s+(?:or|to)\s+\d+\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:or|to)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
        r"\b(?:at\s+least|at\s+most|more\s+than|less\s+than|over|under|approximately|approx(?:\.|imately)?|around|about)\b",
    ]
    for pat in range_or_bound_patterns:
        if re.search(pat, lowered):
            return None, True

    for pat in _UNCERTAIN_PATTERNS:
        if re.search(pat, lowered):
            return None, True

    if re.fullmatch(r"\d+", text):
        return int(text), False

    exact_two_map = {
        "pair": 2,
        "a pair": 2,
        "couple": 2,
        "a couple": 2,
    }
    if lowered in exact_two_map:
        return exact_two_map[lowered], False

    exact_number_words = {
        "a": 1,
        "an": 1,
        "one": 1,
        "single": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    if lowered in exact_number_words:
        return exact_number_words[lowered], False

    return None, False


def parse_pred_number_field_from_attrs(attrs: Dict[str, List[str]]) -> Tuple[int, str, str]:
    raw_vals = _collect_number_raw_vals(attrs)

    if not raw_vals:
        return 1, "default_missing", "default_missing"

    numeric_vals: List[int] = []
    saw_uncertain = False
    for x in raw_vals:
        n, is_uncertain = _pred_number_string_to_result(x)
        if is_uncertain:
            saw_uncertain = True
        if n is not None:
            numeric_vals.append(int(n))

    uniq_numeric = sorted(set(numeric_vals))
    if saw_uncertain:
        return 1, "uncertain", "parsed_attribute"
    if len(uniq_numeric) == 1:
        return int(uniq_numeric[0]), "numeric", "parsed_attribute"
    if len(uniq_numeric) >= 2:
        return 1, "uncertain", "parsed_attribute"

    # raw number field exists but could not be parsed; treat as uncertain, not missing
    return 1, "uncertain", "parsed_attribute"


def parse_number_field_from_attrs(attrs: Dict[str, List[str]]) -> Tuple[int, str, str]:
    """
    Returns:
      number_value: int (always >=1)
      number_type : numeric | uncertain | default_missing
      number_source: parsed_attribute | default_missing
    """
    return parse_pred_number_field_from_attrs(attrs)


def effective_number_value(number_type: str, number_value: Optional[int]) -> Optional[int]:
    if number_type in {"numeric", "default_missing"}:
        if number_value is None:
            return 1
        return max(0, int(number_value))
    return None


def canonicalize_predicted_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    out[PRED_CATEGORY_FIELD] = out.get(PRED_CATEGORY_FIELD)
    attrs = normalize_attributes_structure(out)
    out["attributes"] = attrs
    number_value, number_type, number_source = parse_pred_number_field_from_attrs(attrs)
    out["number_value"] = number_value
    out["number_type"] = number_type
    out["number_source"] = number_source
    return out


def normalize_gt_category_levels(fc: Any) -> List[List[str]]:
    levels: List[List[str]] = []

    def _clean_label(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None

    if isinstance(fc, (list, tuple)):
        for lv in fc:
            if isinstance(lv, (list, tuple)):
                cands: List[str] = []
                seen = set()
                for x in lv:
                    sx = _clean_label(x)
                    if sx and sx not in seen:
                        seen.add(sx)
                        cands.append(sx)
                if cands:
                    levels.append(cands)
            else:
                sx = _clean_label(lv)
                if sx:
                    levels.append([sx])
        return levels

    sx = _clean_label(fc)
    if sx:
        levels.append([sx])
    return levels


def format_gt_categories_display(fc: Any) -> str:
    levels = normalize_gt_category_levels(fc)
    if not levels:
        return str(fc)
    return " > ".join(f"L{i}: {' | '.join(lv)}" for i, lv in enumerate(levels))


def deepest_gt_label(levels: List[List[str]]) -> str:
    if not levels:
        return ""
    return " | ".join(levels[-1])


def normalize_gt_object(raw_gt: Dict[str, Any], gt_index: int, group: str) -> Dict[str, Any]:
    obj = dict(raw_gt)
    attrs = normalize_attributes_structure(obj)
    levels = normalize_gt_category_levels(obj.get(GT_CATEGORY_FIELD))

    if group == "optional":
        number_value, number_type, number_source = 1, "default_missing", "skipped_optional"
    else:
        number_value, number_type, number_source = parse_gt_number_field_from_attrs(attrs)

    return {
        "gt_index": int(gt_index),
        "group": str(group),
        "raw": obj,
        "category_levels": levels,
        "category_display": format_gt_categories_display(obj.get(GT_CATEGORY_FIELD)),
        "deepest_label": deepest_gt_label(levels),
        "attributes": attrs,
        "number_value": number_value,
        "number_type": number_type,
        "number_source": number_source,
    }


def build_attribute_facts_wo_number(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    attrs = obj.get("attributes")
    if not isinstance(attrs, dict):
        return []
    filtered = {k: v for k, v in attrs.items() if str(k).strip().lower() not in {"number", "quantity"}}
    raw_facts = _flatten_attr_facts(filtered)
    raw_facts = [x for x in raw_facts if x.get("normalized_value")]
    out: List[Dict[str, Any]] = []
    for i, fact in enumerate(raw_facts):
        out.append({
            "fact_index": i,
            "path": fact["path"],
            "value": fact["value"],
            "normalized_value": fact["normalized_value"],
        })
    return out


def heuristic_matched_gt_fact_indices(
    pred_facts: List[Dict[str, Any]],
    gt_facts: List[Dict[str, Any]],
) -> List[int]:
    matched: List[int] = []
    for gf in gt_facts:
        best = 0.0
        gp = str(gf.get("path") or "")
        gv = str(gf.get("normalized_value") or "")
        for pf in pred_facts:
            pp = str(pf.get("path") or "")
            pv = str(pf.get("normalized_value") or "")
            score = _value_match_score(gv, pv)
            # mild path bias
            if gp and pp and gp.split(".")[0] == pp.split(".")[0]:
                score = min(1.0, score + 0.1)
            if score > best:
                best = score
            if best >= 0.95:
                break
        if best >= 0.55:
            matched.append(int(gf["fact_index"]))
    return sorted(set(matched))


def heuristic_pair_score(
    pred_category_text: str,
    gt_category_text: str,
    pred_facts: List[Dict[str, Any]],
    gt_facts: List[Dict[str, Any]],
) -> float:
    matched_idx = heuristic_matched_gt_fact_indices(pred_facts, gt_facts)
    fact_score = float(len(matched_idx))
    cat_score = _value_match_score(_normalize_text(pred_category_text), _normalize_text(gt_category_text))
    return fact_score * 10.0 + cat_score


# =========================================================
# 4) GT validator / hierarchy index / optional whitelist
# =========================================================
def _normalize_category_label_for_validator(label: Any) -> str:
    return re.sub(r"\s+", " ", str(label).strip().casefold())


def validate_file_level_category_consistency(required_gt_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    diagnostics: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    parent_signature_by_depth_label: Dict[Tuple[int, str], Tuple[Tuple[str, ...], ...]] = {}
    label_depth_records: Dict[str, Dict[str, Any]] = {}

    for gt in required_gt_objects:
        gt_index = int(gt["gt_index"])
        levels = gt.get("category_levels") or []
        if not isinstance(levels, list) or not levels:
            diagnostics.append({
                "type": "empty_category_levels",
                "gt_index": gt_index,
                "category_display": gt.get("category_display"),
            })
            continue

        prefix_signature: List[Tuple[str, ...]] = []
        for depth, level in enumerate(levels):
            if not isinstance(level, list) or not level:
                diagnostics.append({
                    "type": "empty_level",
                    "gt_index": gt_index,
                    "depth": depth,
                    "category_display": gt.get("category_display"),
                })
                continue

            cleaned: List[Tuple[str, str]] = []
            seen_local: Set[str] = set()
            for lab in level:
                s = str(lab).strip()
                norm = _normalize_category_label_for_validator(s)
                if not norm:
                    diagnostics.append({
                        "type": "blank_label",
                        "gt_index": gt_index,
                        "depth": depth,
                        "raw_level": level,
                    })
                    continue
                if norm in seen_local:
                    diagnostics.append({
                        "type": "duplicate_label_inside_level",
                        "gt_index": gt_index,
                        "depth": depth,
                        "label": s,
                        "normalized_label": norm,
                    })
                    continue
                seen_local.add(norm)
                cleaned.append((s, norm))

            if not cleaned:
                diagnostics.append({
                    "type": "all_blank_level",
                    "gt_index": gt_index,
                    "depth": depth,
                    "raw_level": level,
                })
                continue

            prefix_key = tuple(prefix_signature)
            for label, norm in cleaned:
                record = label_depth_records.setdefault(norm, {
                    "label": label,
                    "depths": set(),
                    "occurrences": [],
                })
                record["depths"].add(depth)
                record["occurrences"].append({
                    "gt_index": gt_index,
                    "depth": depth,
                    "label": label,
                    "category_display": gt.get("category_display"),
                })

                k = (depth, norm)
                if k in parent_signature_by_depth_label:
                    if parent_signature_by_depth_label[k] != prefix_key:
                        warnings.append({
                            "type": "prefix_inconsistency",
                            "gt_index": gt_index,
                            "depth": depth,
                            "label": label,
                            "normalized_label": norm,
                            "existing_prefix": parent_signature_by_depth_label[k],
                            "new_prefix": prefix_key,
                            "category_display": gt.get("category_display"),
                        })
                else:
                    parent_signature_by_depth_label[k] = prefix_key

            prefix_signature.append(tuple(norm for _, norm in cleaned))

    for norm, record in sorted(label_depth_records.items(), key=lambda kv: kv[0]):
        occurrences = record.get("occurrences") or []
        depths = sorted(record["depths"])
        if len(depths) > 1:
            diagnostics.append({
                "type": "label_depth_inconsistency",
                "label": record.get("label"),
                "normalized_label": norm,
                "depths": depths,
                "occurrences": occurrences,
            })

        labels_by_depth: Dict[int, Set[str]] = {}
        for occ in occurrences:
            labels_by_depth.setdefault(int(occ["depth"]), set()).add(str(occ["label"]))
        for depth, labels in sorted(labels_by_depth.items(), key=lambda kv: kv[0]):
            if len(labels) <= 1:
                continue
            diagnostics.append({
                "type": "label_surface_inconsistency",
                "normalized_label": norm,
                "depth": depth,
                "labels": sorted(labels),
                "occurrences": [
                    occ for occ in occurrences
                    if int(occ["depth"]) == depth and str(occ["label"]) in labels
                ],
            })

    ok = len(diagnostics) == 0
    return {
        "ok": ok,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }

def build_required_hierarchy_index(required_gt_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a depth-wise merged hierarchy index.
    Assumes validator has already checked that each normalized label maps to a unique 2D-list node.
    """
    labels_by_depth: Dict[int, List[str]] = {}
    candidate_map: Dict[Tuple[int, str], Set[int]] = {}

    max_depth = 0
    for gt in required_gt_objects:
        gt_index = int(gt["gt_index"])
        levels = gt.get("category_levels") or []
        max_depth = max(max_depth, len(levels))
        for depth, level in enumerate(levels):
            if depth not in labels_by_depth:
                labels_by_depth[depth] = []
            for label in level:
                s = str(label).strip()
                if not s:
                    continue
                if s not in labels_by_depth[depth]:
                    labels_by_depth[depth].append(s)
                candidate_map.setdefault((depth, s), set()).add(gt_index)

    rows: List[List[str]] = []
    coord_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
    coord_meta_view: List[Dict[str, Any]] = []

    for depth in range(max_depth):
        row = labels_by_depth.get(depth, [])
        rows.append(list(row))
        for pos, label in enumerate(row):
            gt_ids = sorted(candidate_map.get((depth, label), set()))
            candidate_leaf_labels = sorted({
                str(required_gt_objects[int(gt_id)].get("deepest_label") or "").strip()
                for gt_id in gt_ids
                if 0 <= int(gt_id) < len(required_gt_objects)
                and str(required_gt_objects[int(gt_id)].get("deepest_label") or "").strip()
            })
            meta = {
                "label": label,
                "candidate_gt_ids": gt_ids,
                "candidate_leaf_labels": candidate_leaf_labels,
                "depth": depth,
            }
            coord_meta[(depth, pos)] = meta
            coord_meta_view.append({
                "coord": [depth, pos],
                "label": label,
                "candidate_gt_ids": gt_ids,
                "candidate_leaf_labels": candidate_leaf_labels,
                "depth": depth,
            })

    return {
        "mode": "hierarchy",
        "rows": rows,
        "coord_meta": coord_meta,       # internal
        "coord_meta_view": coord_meta_view,  # json-friendly
    }


def build_direct_chain_view(required_gt_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    chains = []
    for gt in required_gt_objects:
        chains.append({
            "required_gt_index": int(gt["gt_index"]),
            "levels": gt.get("category_levels") or [],
            "category_display": gt.get("category_display"),
            "candidate_leaf_labels": [str(gt.get("deepest_label") or "").strip()] if str(gt.get("deepest_label") or "").strip() else [],
        })
    return {
        "mode": "direct_chain",
        "chains": chains,
    }


def build_optional_whitelist(optional_gt_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: List[str] = []
    meta: Dict[int, Dict[str, Any]] = {}
    view_items: List[Dict[str, Any]] = []

    for gt in optional_gt_objects:
        gt_index = int(gt["gt_index"])
        levels = gt.get("category_levels") or []
        if levels:
            last_level = levels[-1]
            deepest_labels = [str(x).strip() for x in last_level if str(x).strip()]
            if not deepest_labels:
                deepest_labels = [gt.get("deepest_label") or gt.get("category_display") or f"optional-{gt_index}"]
        else:
            deepest_labels = [gt.get("category_display") or f"optional-{gt_index}"]

        for lab in deepest_labels:
            labels.append(lab)
            idx = len(labels) - 1
            meta[idx] = {
                "label": lab,
                "gt_ids": [gt_index],
            }
            view_items.append({
                "optional_index": idx,
                "label": lab,
                "gt_ids": [gt_index],
            })

    return {
        "labels": labels,
        "meta": meta,
        "view_items": view_items,
    }


# Export policy for `from utils.common import *`:
# include helper names starting with `_` (e.g. `_eprint`, `_safe_int_or_none`) so
# existing refactored modules keep behavior unchanged.
__all__ = [k for k in globals().keys() if not k.startswith("__")]
