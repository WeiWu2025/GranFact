#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Stage-3 evaluator for the Stage-2 extract_and_match output.

Expected invocation:

    python evaluate.py \
      --run-root "${stage2_dir}" \
      --run-id "${stage3_run_id}"

Input convention, per user request:

    {run_root}/per_sample.jsonl

Output convention:

    {run_root}/stage3_eval/{run_id}/

This evaluator consumes the already-materialized accounting fields from
extract_and_match.py. It does NOT redo extraction, candidate selection, or
matching. In particular, normal PRF is computed from pred_flow_accounting and
GT accounting, while granularity-weighted PRF is computed from final real flows.

Important metric semantics:

1. normal precision is prediction-side truthfulness:
       TP_credit / predicted_positive_quantity

   Therefore 1 - normal_precision is the prediction-side hallucination rate.
   Its decomposition contains:
       a) category hallucination quantity;
       b) solver residual quantity hallucination;
       c) uncertain-pred-to-determinate-GT discount hallucination.

2. normal recall is GT coverage:
       recall_covered_quantity / gt_metric_quantity

   Therefore 1 - normal_recall is the miss rate.

3. cat-granularity-weighted and overall-granularity-weighted PRF keep the same
   denominators as normal PRF, but replace the TP numerator with:
       sum(real_flow.tp_credit_quantity * granularity)

   This is equivalent to treating a flow of quantity q as q object-units. If a
   real flow has an effective discount d, q*d units are credited and q*(1-d)
   units are treated as hallucinated units with granularity 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-9


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def mkdirp(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def clamp01(x: Any, default: float = 0.0) -> float:
    v = safe_float(x, default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def safe_div(num: float, den: float) -> Optional[float]:
    num = safe_float(num)
    den = safe_float(den)
    if abs(den) <= EPS:
        return None
    return num / den


def f1_from_pr(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    if p + r <= EPS:
        return 0.0
    return 2.0 * p * r / (p + r)


def metric_triplet(
    *,
    precision_num: float,
    precision_den: float,
    recall_num: Optional[float] = None,
    recall_den: Optional[float] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Return numerator/denominator/P/R/F1 fields with an optional prefix."""
    if recall_num is None:
        recall_num = precision_num
    if recall_den is None:
        recall_den = precision_den

    p = safe_div(precision_num, precision_den)
    r = safe_div(recall_num, recall_den)
    f = f1_from_pr(p, r)

    pre = f"{prefix}_" if prefix else ""
    return {
        f"{pre}precision_numerator": float(precision_num),
        f"{pre}precision_denominator": float(precision_den),
        f"{pre}recall_numerator": float(recall_num),
        f"{pre}recall_denominator": float(recall_den),
        f"{pre}precision": p,
        f"{pre}recall": r,
        f"{pre}f1": f,
    }


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"JSONL line {line_no} in {path} is not an object")
            yield obj


def write_json(path: str | Path, obj: Any) -> None:
    mkdirp(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    mkdirp(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    mkdirp(Path(path).parent)
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    flat_rows = [flatten_for_csv(x) for x in rows]
    fieldnames: List[str] = []
    seen = set()
    for row in flat_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def optional_float_to_hist_value(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def non_null_mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(mean(vals)) if vals else None


def non_null_median(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(median(vals)) if vals else None


CORE_METRIC_RENAME_MAP: List[Tuple[str, str]] = [
    ("normal_micro_f1", "f1"),
    ("normal_micro_precision", "precision"),
    ("normal_micro_recall", "recall"),
    ("overall_granularity_weighted_micro_f1", "overall_gran-f1"),
    ("overall_granularity_weighted_micro_precision", "overall_gran-precision"),
    ("overall_granularity_weighted_micro_recall", "overall_gran-recall"),
    ("cat_granularity_weighted_micro_f1", "cat_gran-f1"),
    ("cat_granularity_weighted_micro_precision", "cat_gran-precision"),
    ("cat_granularity_weighted_micro_recall", "cat_gran-recall"),
    ("avg_matched_granularity_quality", "avg_matched_gran-quality"),
]


def build_core_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    core: Dict[str, Any] = {}
    for key in ("group", "num_samples"):
        if key in metrics:
            core[key] = metrics.get(key)
    for old_key, new_key in CORE_METRIC_RENAME_MAP:
        core[new_key] = metrics.get(old_key)
    return core


def wrap_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "core_metrics": build_core_metrics(metrics),
        "all_metrics": metrics,
    }


def wrap_metrics_list(metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "core_metrics": [build_core_metrics(metrics) for metrics in metrics_list],
        "all_metrics": list(metrics_list),
    }


# -----------------------------------------------------------------------------
# Domain classification
# -----------------------------------------------------------------------------

def normalize_rel_path(path: Any) -> str:
    p = str(path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")

    # The input may be absolute or may contain a dataset prefix. Try to align to
    # the known benchmark roots without requiring exact absolute layout.
    markers = ["maybe-too-easy/", "ready/"]
    positions = [p.find(m) for m in markers if p.find(m) >= 0]
    if positions:
        p = p[min(positions):]
    return p


def classify(rel_path: Any) -> Optional[str]:
    p = normalize_rel_path(rel_path)
    if p.startswith("maybe-too-easy/daily-v0/"):
        return "daily"
    if p.startswith("ready/byx-animal_daily/daily"):
        return "daily"
    if p.startswith("ready/cosmetics/"):
        return "daily"

    if p.startswith("maybe-too-easy/natural-v0/animal"):
        return "animal"
    if p.startswith("ready/byx-animal_daily/animal"):
        return "animal"

    if p.startswith("maybe-too-easy/natural-v0/template"):
        return "plant"

    if p.startswith("ready/building/"):
        return "landmark"
    if p.startswith("ready/cars/"):
        return "car"
    if p.startswith("ready/electronics/"):
        return "electronic"
    if p.startswith("ready/games-genshin/") or p.startswith("ready/games-mc/"):
        return "games"
    return None


def sample_domain(sample: Dict[str, Any]) -> str:
    # Prefer image_path, because the supplied domain function is path-based.
    for key in ("image_path", "source_json"):
        d = classify(sample.get(key))
        if d:
            return d
    return "unknown"


# -----------------------------------------------------------------------------
# Input / output path resolution
# -----------------------------------------------------------------------------

def resolve_input_jsonl(run_root: str, explicit_input: Optional[str] = None) -> str:
    if explicit_input:
        path = Path(explicit_input)
        if not path.exists():
            raise FileNotFoundError(f"--input-jsonl does not exist: {path}")
        return str(path)

    root = Path(run_root)
    candidates = [
        root / "per_sample.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    msg = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not find Stage-2 per-sample JSONL. Tried:\n" + msg
    )


def resolve_output_dir(run_root: str, run_id: str) -> str:
    out_dir = Path(run_root) / "stage3_eval" / str(run_id)
    mkdirp(out_dir)
    return str(out_dir)


# -----------------------------------------------------------------------------
# Granularity helpers
# -----------------------------------------------------------------------------

def category_total_depth(gt_obj: Dict[str, Any]) -> int:
    levels = gt_obj.get("category_levels")
    if isinstance(levels, list) and levels:
        return max(1, len(levels))
    # Fallbacks for future schema variants.
    if gt_obj.get("deepest_label"):
        return 1
    if gt_obj.get("category_display"):
        return 1
    return 1


def predicted_supported_depth(
    pred_index: int,
    gt_index: int,
    *,
    initial_by_pred: Dict[int, Dict[str, Any]],
    per_pred_summary_by_pred: Dict[int, Dict[str, Any]],
    gt_total_depth: int,
) -> float:
    """Return the category credit depth supported by a prediction.

    Primary source is category_credit_depth, which may be fractional when the
    two-step category matcher routes an under-specific prediction to a compatible
    child node. Fallback is the original integer supported_depth.
    """
    sources = [
        initial_by_pred.get(pred_index, {}).get("category_credit_depth"),
        per_pred_summary_by_pred.get(pred_index, {}).get("category_credit_depth"),
        initial_by_pred.get(pred_index, {}).get("supported_depth"),
        per_pred_summary_by_pred.get(pred_index, {}).get("supported_depth"),
    ]
    for x in sources:
        sd = safe_float(x, None)
        if sd is not None and sd > 0:
            return max(1.0, min(float(sd), float(gt_total_depth)))

    # Fallback to matched_level_index + 1.
    sources = [
        initial_by_pred.get(pred_index, {}).get("matched_level_index"),
        per_pred_summary_by_pred.get(pred_index, {}).get("matched_level_index"),
    ]
    for x in sources:
        mi = safe_int(x)
        if mi is not None and mi >= 0:
            return float(max(1, min(int(mi) + 1, int(gt_total_depth))))

    # Last-resort fallback: if the flow exists, give the shallowest non-zero
    # level rather than silently using 0 for a matched object.
    return 1.0


def attribute_score_from_flow(flow: Dict[str, Any]) -> float:
    if "edge_utility" in flow and flow.get("edge_utility") is not None:
        return clamp01(flow.get("edge_utility"), 0.0)

    attr = flow.get("attr_match") or {}
    if not isinstance(attr, dict):
        return 0.0
    gt_cnt = safe_float(attr.get("gt_attribute_fact_count_wo_numberattr"), 0.0)
    matched = safe_float(attr.get("matched_gt_attribute_count_wo_numberattr"), 0.0)
    if gt_cnt <= EPS:
        # No attribute requirement. This follows pair scorer semantics.
        return 1.0
    return clamp01(matched / gt_cnt, 0.0)


def granularity_for_real_flow(
    flow: Dict[str, Any],
    *,
    gt_required_objects: List[Dict[str, Any]],
    initial_by_pred: Dict[int, Dict[str, Any]],
    per_pred_summary_by_pred: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    pi = safe_int(flow.get("pred_index"), -1)
    gj = safe_int(flow.get("gt_index"), -1)
    if pi is None or gj is None or gj < 0 or gj >= len(gt_required_objects):
        return {
            "cat_granularity": 0.0,
            "overall_granularity": 0.0,
            "supported_depth": 0,
            "total_depth": 1,
            "attribute_score": 0.0,
        }

    total_depth = category_total_depth(gt_required_objects[gj])
    supported_depth = predicted_supported_depth(
        int(pi),
        int(gj),
        initial_by_pred=initial_by_pred,
        per_pred_summary_by_pred=per_pred_summary_by_pred,
        gt_total_depth=total_depth,
    )
    attr_score = attribute_score_from_flow(flow)
    cat_g = clamp01(float(supported_depth) / float(total_depth), 0.0)
    overall_g = clamp01((float(supported_depth) + float(attr_score)) / float(total_depth + 1), 0.0)
    return {
        "cat_granularity": cat_g,
        "overall_granularity": overall_g,
        "supported_depth": float(supported_depth),
        "total_depth": int(total_depth),
        "attribute_score": attr_score,
    }


# -----------------------------------------------------------------------------
# Per-sample analysis
# -----------------------------------------------------------------------------

def analyze_sample(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (sample_metrics_row, granularity_unit_rows)."""
    sample_id = str(sample.get("sample_id") or "")
    image_path = sample.get("image_path")
    source_json = sample.get("source_json")
    domain = sample_domain(sample)

    gt_required_objects = list(sample.get("gt_required_objects") or [])
    predicted_objects = list(sample.get("predicted_objects") or [])
    final_flows = list(sample.get("final_flows") or [])
    pred_flow_accounting = list(sample.get("pred_flow_accounting") or [])
    gt_flow_accounting = list(sample.get("gt_flow_accounting") or [])
    initial_match_results = list(sample.get("initial_match_results") or [])
    per_pred_summary = list(sample.get("per_pred_summary") or [])

    initial_by_pred = {
        int(x["pred_index"]): x for x in initial_match_results
        if isinstance(x, dict) and safe_int(x.get("pred_index")) is not None
    }
    per_pred_summary_by_pred = {
        int(x["pred_index"]): x for x in per_pred_summary
        if isinstance(x, dict) and safe_int(x.get("pred_index")) is not None
    }

    # -----------------------------
    # Normal PRF from materialized accounting.
    # -----------------------------
    predicted_positive = sum(safe_float(x.get("predicted_positive_quantity")) for x in pred_flow_accounting if isinstance(x, dict))
    normal_precision_tp = sum(safe_float(x.get("matched_tp_credit_quantity")) for x in pred_flow_accounting if isinstance(x, dict))

    gt_positive = sum(safe_float(x.get("gt_metric_quantity")) for x in gt_flow_accounting if isinstance(x, dict))
    normal_recall_tp = sum(safe_float(x.get("recall_covered_quantity")) for x in gt_flow_accounting if isinstance(x, dict))

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": image_path,
        "source_json": source_json,
        "index_in_source_json": sample.get("index_in_source_json"),
        "domain": domain,
        "num_predicted_objects": len(predicted_objects),
        "num_required_gt_objects": len(gt_required_objects),
        "num_optional_gt_objects": len(sample.get("gt_optional_objects") or []),
        "num_final_flows": len(final_flows),
        "num_real_flows": sum(1 for fl in final_flows if isinstance(fl, dict) and fl.get("flow_kind") == "real"),
        "num_hallucination_flows": sum(1 for fl in final_flows if isinstance(fl, dict) and fl.get("flow_kind") == "hallucination"),
    }
    row.update(metric_triplet(
        prefix="normal",
        precision_num=normal_precision_tp,
        precision_den=predicted_positive,
        recall_num=normal_recall_tp,
        recall_den=gt_positive,
    ))

    # -----------------------------
    # Granularity-weighted TP and granularity unit distributions.
    # -----------------------------
    cat_granularity_tp = 0.0
    overall_granularity_tp = 0.0
    real_flow_tp_from_final = 0.0
    real_flow_assigned_quantity = 0.0
    discount_hallucination_quantity = 0.0
    solver_residual_quantity_from_flows = 0.0

    granularity_unit_rows: List[Dict[str, Any]] = []

    def add_granularity_unit_rows(
        *,
        quantity: float,
        cat_granularity: float,
        overall_granularity: float,
        distribution_scope: str,
        unit_kind: str,
        flow_id: Optional[str] = None,
        pred_index: Optional[int] = None,
        gt_index: Optional[int] = None,
    ) -> None:
        q = safe_float(quantity)
        if q <= EPS:
            return
        granularity_unit_rows.append({
            "sample_id": sample_id,
            "domain": domain,
            "distribution_scope": distribution_scope,
            "unit_kind": unit_kind,
            "quantity": float(q),
            "cat_granularity": clamp01(cat_granularity, 0.0),
            "overall_granularity": clamp01(overall_granularity, 0.0),
            "flow_id": flow_id,
            "pred_index": pred_index,
            "gt_index": gt_index,
        })

    for flow in final_flows:
        if not isinstance(flow, dict):
            continue
        flow_kind = str(flow.get("flow_kind") or "")
        if flow_kind == "real":
            assigned_q = safe_float(flow.get("assigned_quantity"))
            tp_credit_q = safe_float(flow.get("tp_credit_quantity"), assigned_q * safe_float(flow.get("pred_discount"), 1.0))
            tp_credit_q = max(0.0, min(tp_credit_q, assigned_q if assigned_q > EPS else tp_credit_q))
            discount_loss_q = max(0.0, assigned_q - tp_credit_q)

            g = granularity_for_real_flow(
                flow,
                gt_required_objects=gt_required_objects,
                initial_by_pred=initial_by_pred,
                per_pred_summary_by_pred=per_pred_summary_by_pred,
            )
            cat_g = float(g["cat_granularity"])
            overall_g = float(g["overall_granularity"])

            real_flow_tp_from_final += tp_credit_q
            real_flow_assigned_quantity += assigned_q
            discount_hallucination_quantity += discount_loss_q
            cat_granularity_tp += tp_credit_q * cat_g
            overall_granularity_tp += tp_credit_q * overall_g

            pi = safe_int(flow.get("pred_index"))
            gj = safe_int(flow.get("gt_index"))
            fid = str(flow.get("flow_id") or "")

            # 4.1 correct-only distribution: only credited object-units.
            add_granularity_unit_rows(
                quantity=tp_credit_q,
                cat_granularity=cat_g,
                overall_granularity=overall_g,
                distribution_scope="correct_only",
                unit_kind="credited_real_flow",
                flow_id=fid,
                pred_index=pi,
                gt_index=gj,
            )

            # 4.2 all-prediction-units distribution: credited units keep their
            # granularity; discount loss is hallucination with granularity 0.
            add_granularity_unit_rows(
                quantity=tp_credit_q,
                cat_granularity=cat_g,
                overall_granularity=overall_g,
                distribution_scope="all_prediction_units",
                unit_kind="credited_real_flow",
                flow_id=fid,
                pred_index=pi,
                gt_index=gj,
            )
            add_granularity_unit_rows(
                quantity=discount_loss_q,
                cat_granularity=0.0,
                overall_granularity=0.0,
                distribution_scope="all_prediction_units",
                unit_kind="discount_hallucination",
                flow_id=fid,
                pred_index=pi,
                gt_index=gj,
            )

        elif flow_kind == "hallucination" and str(flow.get("hallucination_type") or "") == "quantity":
            q = safe_float(flow.get("assigned_quantity"), safe_float(flow.get("credit_quantity")))
            solver_residual_quantity_from_flows += q
            add_granularity_unit_rows(
                quantity=q,
                cat_granularity=0.0,
                overall_granularity=0.0,
                distribution_scope="all_prediction_units",
                unit_kind="solver_residual_quantity_hallucination",
                flow_id=str(flow.get("flow_id") or ""),
                pred_index=safe_int(flow.get("pred_index")),
                gt_index=None,
            )

    # Category hallucination is not materialized as final_flows. It is in
    # pred_flow_accounting and must be added explicitly.
    category_hallucination_quantity = 0.0
    solver_residual_quantity_from_pred_acc = 0.0
    for item in pred_flow_accounting:
        if not isinstance(item, dict):
            continue
        pi = safe_int(item.get("pred_index"))
        cat_q = safe_float(item.get("stage1_category_hallucination_predicted_positive_quantity"))
        category_hallucination_quantity += cat_q
        add_granularity_unit_rows(
            quantity=cat_q,
            cat_granularity=0.0,
            overall_granularity=0.0,
            distribution_scope="all_prediction_units",
            unit_kind="category_hallucination",
            flow_id=None,
            pred_index=pi,
            gt_index=None,
        )
        solver_residual_quantity_from_pred_acc += safe_float(item.get("residual_hallucination_quantity"))

    row.update(metric_triplet(
        prefix="cat_granularity_weighted",
        precision_num=cat_granularity_tp,
        precision_den=predicted_positive,
        recall_num=cat_granularity_tp,
        recall_den=gt_positive,
    ))
    row.update(metric_triplet(
        prefix="overall_granularity_weighted",
        precision_num=overall_granularity_tp,
        precision_den=predicted_positive,
        recall_num=overall_granularity_tp,
        recall_den=gt_positive,
    ))
    row["avg_matched_granularity_quality"] = safe_div(overall_granularity_tp, normal_precision_tp)

    # -----------------------------
    # Hallucination decomposition and sanity checks.
    # -----------------------------
    solver_residual_quantity = solver_residual_quantity_from_pred_acc
    if abs(solver_residual_quantity_from_pred_acc - solver_residual_quantity_from_flows) > 1e-6:
        # Keep pred accounting as canonical because it is what contributes to PP,
        # but expose the flow-side value and warning.
        solver_residual_quantity = solver_residual_quantity_from_pred_acc

    total_prediction_side_hallucination = (
        category_hallucination_quantity
        + solver_residual_quantity
        + discount_hallucination_quantity
    )
    pred_side_loss = predicted_positive - normal_precision_tp
    all_prediction_unit_quantity = sum(
        safe_float(x.get("quantity"))
        for x in granularity_unit_rows
        if x.get("distribution_scope") == "all_prediction_units"
    )
    correct_only_unit_quantity = sum(
        safe_float(x.get("quantity"))
        for x in granularity_unit_rows
        if x.get("distribution_scope") == "correct_only"
    )

    warnings: List[str] = []
    if abs(normal_precision_tp - real_flow_tp_from_final) > 1e-6:
        warnings.append(
            f"normal TP from pred_flow_accounting ({normal_precision_tp:.12g}) differs "
            f"from sum(final real flow tp_credit_quantity) ({real_flow_tp_from_final:.12g})"
        )
    if abs(normal_recall_tp - normal_precision_tp) > 1e-6:
        warnings.append(
            f"recall covered quantity ({normal_recall_tp:.12g}) differs from precision TP "
            f"({normal_precision_tp:.12g}); PRF keeps pred-side numerator for precision "
            f"and GT-side numerator for recall"
        )
    if abs(solver_residual_quantity_from_pred_acc - solver_residual_quantity_from_flows) > 1e-6:
        warnings.append(
            f"solver residual quantity from pred accounting ({solver_residual_quantity_from_pred_acc:.12g}) "
            f"differs from quantity hallucination flows ({solver_residual_quantity_from_flows:.12g})"
        )
    if abs(pred_side_loss - total_prediction_side_hallucination) > 1e-6:
        warnings.append(
            f"prediction-side loss ({pred_side_loss:.12g}) differs from hallucination decomposition "
            f"({total_prediction_side_hallucination:.12g})"
        )
    if abs(all_prediction_unit_quantity - predicted_positive) > 1e-6:
        warnings.append(
            f"all_prediction_units quantity ({all_prediction_unit_quantity:.12g}) differs from "
            f"predicted_positive ({predicted_positive:.12g})"
        )
    if abs(correct_only_unit_quantity - real_flow_tp_from_final) > 1e-6:
        warnings.append(
            f"correct_only quantity ({correct_only_unit_quantity:.12g}) differs from real-flow TP "
            f"({real_flow_tp_from_final:.12g})"
        )

    row.update({
        "normal_tp_from_final_real_flows": float(real_flow_tp_from_final),
        "real_flow_assigned_quantity": float(real_flow_assigned_quantity),
        "category_hallucination_quantity": float(category_hallucination_quantity),
        "quantity_solver_residual_hallucination_quantity": float(solver_residual_quantity),
        "quantity_solver_residual_hallucination_quantity_from_flows": float(solver_residual_quantity_from_flows),
        "quantity_discount_hallucination_quantity": float(discount_hallucination_quantity),
        "total_prediction_side_hallucination_quantity": float(total_prediction_side_hallucination),
        "prediction_side_loss_quantity": float(pred_side_loss),
        "prediction_side_hallucination_rate_from_decomp": safe_div(total_prediction_side_hallucination, predicted_positive),
        "prediction_side_hallucination_rate_from_precision": (None if row.get("normal_precision") is None else 1.0 - float(row["normal_precision"])),
        "gt_miss_quantity": float(gt_positive - normal_recall_tp),
        "gt_miss_rate_from_recall": (None if row.get("normal_recall") is None else 1.0 - float(row["normal_recall"])),
        "all_prediction_unit_quantity_for_granularity": float(all_prediction_unit_quantity),
        "correct_only_unit_quantity_for_granularity": float(correct_only_unit_quantity),
        "warnings": warnings,
        "num_warnings": len(warnings),
    })

    return row, granularity_unit_rows


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_rows(rows: Sequence[Dict[str, Any]], *, group_name: str) -> Dict[str, Any]:
    rows = list(rows)
    out: Dict[str, Any] = {
        "group": group_name,
        "num_samples": len(rows),
    }

    # Normal micro PRF.
    normal_precision_num = sum(safe_float(r.get("normal_precision_numerator")) for r in rows)
    normal_precision_den = sum(safe_float(r.get("normal_precision_denominator")) for r in rows)
    normal_recall_num = sum(safe_float(r.get("normal_recall_numerator")) for r in rows)
    normal_recall_den = sum(safe_float(r.get("normal_recall_denominator")) for r in rows)
    out.update(metric_triplet(
        prefix="normal_micro",
        precision_num=normal_precision_num,
        precision_den=normal_precision_den,
        recall_num=normal_recall_num,
        recall_den=normal_recall_den,
    ))

    # Granularity-weighted micro PRF.
    cat_num = sum(safe_float(r.get("cat_granularity_weighted_precision_numerator")) for r in rows)
    overall_num = sum(safe_float(r.get("overall_granularity_weighted_precision_numerator")) for r in rows)
    out.update(metric_triplet(
        prefix="cat_granularity_weighted_micro",
        precision_num=cat_num,
        precision_den=normal_precision_den,
        recall_num=cat_num,
        recall_den=normal_recall_den,
    ))
    out.update(metric_triplet(
        prefix="overall_granularity_weighted_micro",
        precision_num=overall_num,
        precision_den=normal_precision_den,
        recall_num=overall_num,
        recall_den=normal_recall_den,
    ))
    out["avg_matched_granularity_quality"] = safe_div(overall_num, normal_precision_num)

    # Scene-wise macro summaries.
    for prefix in ("normal", "cat_granularity_weighted", "overall_granularity_weighted"):
        for metric in ("precision", "recall", "f1"):
            key = f"{prefix}_{metric}"
            vals = [r.get(key) for r in rows]
            out[f"{key}_macro_mean"] = non_null_mean(vals)  # type: ignore[arg-type]
            out[f"{key}_macro_median"] = non_null_median(vals)  # type: ignore[arg-type]

    # Hallucination / miss decomposition.
    pred_pos = normal_precision_den
    gt_pos = normal_recall_den
    cat_hall = sum(safe_float(r.get("category_hallucination_quantity")) for r in rows)
    solver_hall = sum(safe_float(r.get("quantity_solver_residual_hallucination_quantity")) for r in rows)
    discount_hall = sum(safe_float(r.get("quantity_discount_hallucination_quantity")) for r in rows)
    total_hall = cat_hall + solver_hall + discount_hall
    gt_miss = sum(safe_float(r.get("gt_miss_quantity")) for r in rows)

    out.update({
        "predicted_positive_quantity": float(pred_pos),
        "gt_positive_quantity": float(gt_pos),
        "category_hallucination_quantity": float(cat_hall),
        "quantity_solver_residual_hallucination_quantity": float(solver_hall),
        "quantity_discount_hallucination_quantity": float(discount_hall),
        "total_prediction_side_hallucination_quantity": float(total_hall),
        "prediction_side_hallucination_rate_from_decomp": safe_div(total_hall, pred_pos),
        "prediction_side_hallucination_rate_from_precision": (
            None if out.get("normal_micro_precision") is None else 1.0 - float(out["normal_micro_precision"])
        ),
        "gt_miss_quantity": float(gt_miss),
        "gt_miss_rate_from_recall": (
            None if out.get("normal_micro_recall") is None else 1.0 - float(out["normal_micro_recall"])
        ),
        "num_samples_with_warnings": sum(1 for r in rows if int(safe_int(r.get("num_warnings"), 0) or 0) > 0),
        "total_warning_count": sum(int(safe_int(r.get("num_warnings"), 0) or 0) for r in rows),
    })
    return out


def aggregate_by_domain(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)
    return [aggregate_rows(by_domain[d], group_name=d) for d in sorted(by_domain.keys())]


def granularity_distribution_summary(unit_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in unit_rows:
        grouped[(str(r.get("domain") or "unknown"), str(r.get("distribution_scope") or ""))].append(r)
    # Add overall versions.
    for scope in sorted(set(str(r.get("distribution_scope") or "") for r in unit_rows)):
        grouped[("__overall__", scope)] = [r for r in unit_rows if str(r.get("distribution_scope") or "") == scope]

    out: List[Dict[str, Any]] = []
    for (domain, scope), rs in sorted(grouped.items()):
        total_q = sum(safe_float(r.get("quantity")) for r in rs)
        item: Dict[str, Any] = {
            "domain": domain,
            "distribution_scope": scope,
            "total_object_units": float(total_q),
        }
        for score_key in ("cat_granularity", "overall_granularity"):
            weighted_sum = sum(safe_float(r.get("quantity")) * safe_float(r.get(score_key)) for r in rs)
            item[f"{score_key}_object_unit_mean"] = safe_div(weighted_sum, total_q)
            item[f"{score_key}_min"] = min((safe_float(r.get(score_key)) for r in rs), default=None)
            item[f"{score_key}_max"] = max((safe_float(r.get(score_key)) for r in rs), default=None)
        for unit_kind in sorted(set(str(r.get("unit_kind") or "") for r in rs)):
            item[f"unit_kind_quantity__{unit_kind}"] = float(sum(safe_float(r.get("quantity")) for r in rs if str(r.get("unit_kind") or "") == unit_kind))
        out.append(item)
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt, None
    except Exception as e:
        return None, str(e)


def plot_histogram(
    *,
    values: Sequence[Any],
    out_path: str | Path,
    title: str,
    xlabel: str,
    weights: Optional[Sequence[Any]] = None,
    bins: int = 20,
) -> bool:
    plt, err = try_import_matplotlib()
    if plt is None:
        return False

    clean_values: List[float] = []
    clean_weights: Optional[List[float]] = [] if weights is not None else None
    if weights is None:
        for v in values:
            vv = optional_float_to_hist_value(v)
            if vv is None:
                continue
            clean_values.append(vv)
    else:
        for v, w in zip(values, weights):
            vv = optional_float_to_hist_value(v)
            ww = optional_float_to_hist_value(w)
            if vv is None or ww is None or ww <= EPS:
                continue
            clean_values.append(vv)
            assert clean_weights is not None
            clean_weights.append(ww)

    if not clean_values:
        return False

    mkdirp(Path(out_path).parent)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(
        clean_values,
        bins=bins,
        range=(0.0, 1.0),
        weights=clean_weights,
        edgecolor="black",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("object-units" if weights is not None else "samples")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def make_metric_histograms(rows: Sequence[Dict[str, Any]], out_dir: str | Path, *, group_label: str, bins: int = 20) -> List[str]:
    made: List[str] = []
    metric_prefixes = ["normal", "cat_granularity_weighted", "overall_granularity_weighted"]
    metrics = ["precision", "recall", "f1"]
    for prefix in metric_prefixes:
        for metric in metrics:
            key = f"{prefix}_{metric}"
            out_path = Path(out_dir) / f"{key}.png"
            ok = plot_histogram(
                values=[r.get(key) for r in rows],
                out_path=out_path,
                title=f"{group_label}: scene-wise {key}",
                xlabel=key,
                weights=None,
                bins=bins,
            )
            if ok:
                made.append(str(out_path))
    return made


def make_granularity_histograms(unit_rows: Sequence[Dict[str, Any]], out_dir: str | Path, *, group_label: str, bins: int = 20) -> List[str]:
    made: List[str] = []
    scopes = ["correct_only", "all_prediction_units"]
    score_keys = ["cat_granularity", "overall_granularity"]
    for scope in scopes:
        rs = [r for r in unit_rows if str(r.get("distribution_scope") or "") == scope]
        if not rs:
            continue
        for score_key in score_keys:
            out_path = Path(out_dir) / f"{scope}_{score_key}.png"
            ok = plot_histogram(
                values=[r.get(score_key) for r in rs],
                weights=[r.get("quantity") for r in rs],
                out_path=out_path,
                title=f"{group_label}: {scope} {score_key} distribution",
                xlabel=score_key,
                bins=bins,
            )
            if ok:
                made.append(str(out_path))
    return made


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Stage-2 extract_and_match per-sample output.")
    ap.add_argument("--run-root", required=True, help="Stage-2 directory. Input is {run_root}/per_sample.jsonl by default.")
    ap.add_argument("--run-id", required=True, help="Stage-3 eval run id. Output is {run_root}/stage3_eval/{run_id}.")
    ap.add_argument("--input-jsonl", default=None, help="Optional explicit per-sample JSONL path. Overrides default resolution.")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    ap.add_argument("--no-plots", action="store_true", help="Skip histogram generation.")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins over [0, 1]. Default: 20.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_jsonl = resolve_input_jsonl(args.run_root, args.input_jsonl)
    out_dir = resolve_output_dir(args.run_root, args.run_id)

    existing_files = [p for p in Path(out_dir).iterdir()] if Path(out_dir).exists() else []
    if existing_files and not args.overwrite:
        raise RuntimeError(
            f"Output directory already exists and is non-empty: {out_dir}\n"
            "Pass --overwrite to reuse it."
        )

    rows: List[Dict[str, Any]] = []
    all_unit_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, sample in enumerate(iter_jsonl(input_jsonl), start=1):
        try:
            row, unit_rows = analyze_sample(sample)
            row["input_line_index_1based"] = idx
            rows.append(row)
            all_unit_rows.extend(unit_rows)
        except Exception as e:
            failures.append({
                "input_line_index_1based": idx,
                "sample_id": sample.get("sample_id"),
                "error": repr(e),
            })

    if not rows:
        raise RuntimeError(f"No valid samples were evaluated from {input_jsonl}. Failures: {len(failures)}")

    # Aggregates.
    global_metrics = aggregate_rows(rows, group_name="__overall__")
    domain_metrics = aggregate_by_domain(rows)
    gran_summary = granularity_distribution_summary(all_unit_rows)

    # Write tabular outputs.
    write_jsonl(Path(out_dir) / "sample_metrics.jsonl", rows)
    write_csv(Path(out_dir) / "sample_metrics.csv", rows)
    write_json(Path(out_dir) / "global_metrics.json", wrap_metrics(global_metrics))
    write_json(Path(out_dir) / "domain_metrics.json", wrap_metrics_list(domain_metrics))
    write_csv(Path(out_dir) / "domain_metrics.csv", domain_metrics)
    write_jsonl(Path(out_dir) / "granularity_units.jsonl", all_unit_rows)
    write_csv(Path(out_dir) / "granularity_units.csv", all_unit_rows)
    write_json(Path(out_dir) / "granularity_distribution_summary.json", gran_summary)
    write_csv(Path(out_dir) / "granularity_distribution_summary.csv", gran_summary)
    write_json(Path(out_dir) / "failed_samples.json", failures)

    # Convenience decomposition tables.
    hallucination_fields = [
        "sample_id", "domain", "normal_precision", "normal_recall", "normal_f1",
        "normal_precision_denominator", "normal_precision_numerator",
        "category_hallucination_quantity",
        "quantity_solver_residual_hallucination_quantity",
        "quantity_discount_hallucination_quantity",
        "total_prediction_side_hallucination_quantity",
        "prediction_side_hallucination_rate_from_precision",
        "prediction_side_hallucination_rate_from_decomp",
        "gt_miss_quantity", "gt_miss_rate_from_recall",
        "num_warnings", "warnings",
    ]
    hall_rows = [{k: r.get(k) for k in hallucination_fields} for r in rows]
    write_jsonl(Path(out_dir) / "hallucination_decomposition.jsonl", hall_rows)
    write_csv(Path(out_dir) / "hallucination_decomposition.csv", hall_rows)

    # Plots.
    plot_paths: List[str] = []
    if not args.no_plots:
        overall_plot_dir = Path(out_dir) / "histograms" / "overall"
        plot_paths.extend(make_metric_histograms(rows, overall_plot_dir / "scene_prf", group_label="overall", bins=args.bins))
        plot_paths.extend(make_granularity_histograms(all_unit_rows, overall_plot_dir / "granularity", group_label="overall", bins=args.bins))

        by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_domain_units: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_domain[str(r.get("domain") or "unknown")].append(r)
        for u in all_unit_rows:
            by_domain_units[str(u.get("domain") or "unknown")].append(u)
        for domain in sorted(by_domain.keys()):
            d_plot_dir = Path(out_dir) / "histograms" / "by_domain" / domain
            plot_paths.extend(make_metric_histograms(by_domain[domain], d_plot_dir / "scene_prf", group_label=domain, bins=args.bins))
            plot_paths.extend(make_granularity_histograms(by_domain_units.get(domain, []), d_plot_dir / "granularity", group_label=domain, bins=args.bins))

    manifest = {
        "script": "evaluate.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(args.run_root),
        "run_id": str(args.run_id),
        "input_jsonl": str(input_jsonl),
        "output_dir": str(out_dir),
        "num_samples_evaluated": len(rows),
        "num_samples_failed": len(failures),
        "domains": sorted(set(str(r.get("domain") or "unknown") for r in rows)),
        "plot_paths": plot_paths,
        "metric_semantics": {
            "normal_precision": "sum matched_tp_credit_quantity / sum predicted_positive_quantity; 1 - precision is prediction-side hallucination rate",
            "normal_recall": "sum recall_covered_quantity / sum gt_metric_quantity; 1 - recall is GT miss rate",
            "cat_granularity_weighted_prf": "same denominators as normal PRF, numerator=sum(tp_credit_quantity * category_granularity)",
            "overall_granularity_weighted_prf": "same denominators as normal PRF, numerator=sum(tp_credit_quantity * overall_granularity)",
            "category_granularity": "category_credit_depth_or_supported_depth / total_gt_category_depth",
            "overall_granularity": "(category_credit_depth_or_supported_depth + edge_utility) / (total_gt_category_depth + 1)",
            "avg_matched_granularity_quality": "sum(tp_credit_quantity * overall_granularity) / sum(matched_tp_credit_quantity); average overall granularity quality among normally matched object-units",
            "edge_flow_utility": "solver objective utility; currently equal to overall_granularity while edge_utility remains the attribute-only score",
            "all_prediction_units_granularity": "credited real-flow units keep granularity; category, residual, and discount hallucination units have granularity 0",
        },
    }
    write_json(Path(out_dir) / "manifest.json", manifest)

    # Human-readable compact stdout.
    print(json.dumps({
        "input_jsonl": input_jsonl,
        "output_dir": out_dir,
        "num_samples_evaluated": len(rows),
        "num_samples_failed": len(failures),
        "metrics": wrap_metrics(global_metrics),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
