#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_and_match.py

Unified Stage-2 script:
- extract structured objects from Stage-1 answers
- build file-level GT hierarchy index / optional whitelist
- run initial category matching
- build candidate bipartite graph
- detect category hallucination
- score pair-level attribute coverage with LLM
- solve capacitated bipartite matching in code
- materialize matched flows and quantity hallucination
- write final per-sample records for downstream aggregation

IMPORTANT (current version):
- extraction, initial match, and attribute-related matching ALL use the SAME model: judge_model/judge_ckpt
- extractor-model / extractor-ckpt args are kept only for CLI compatibility
  - extractor-model is ignored
  - extractor-ckpt is only used as a fallback for judge-ckpt if judge-ckpt is not provided

Input convention (--run-root points to stage1_dir):
  {run_root}/stage1_outputs/
    meta.json
    results.jsonl
    errors.jsonl

Output convention:
  {run_root}/extract_and_match/{run_id}/
    manifest.json
    per_sample.jsonl
    errors.jsonl
"""

import os
import re
import sys
import json
import time
import math
import argparse
import hashlib
import traceback
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

# Linux file lock
try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    fcntl = None
    _HAS_FCNTL = False

# shared helpers
from utils.common import *

from utils.config import (
    ExtractAndMatchConfig,
    parse_args,
    build_config,
    validate_manifest_params,
)
from utils.io_utils import (
    resolve_stage1_paths,
    extract_and_match_output_dir,
    load_stage1_records,
    reorder_records_by_priority,
    relocate_stage1_records,
)
from utils.extraction_stage import run_extraction_stage
from utils.candidate_stage import run_candidate_selection_stage
from utils.final_matching_stage import run_final_matching_stage
from utils.llm import LocalChatModel, build_chat_model
from utils.metrics import (
    build_per_pred_summary,
    build_per_gt_summary,
)



"""Legacy joint-judge/rule-reduction/postfix helpers removed."""


# =========================================================
# 13) End-to-end sample processing
# =========================================================
def process_one_sample(
    rec: Dict[str, Any],
    *,
    cfg: ExtractAndMatchConfig,
    extractor_model: LocalChatModel,
    judge_model: LocalChatModel,
) -> Dict[str, Any]:
    sample_id = str(rec.get("sample_id") or "")
    image_path = rec.get("image_path")
    source_json = rec.get("source_json")
    index_in_source_json = rec.get("index_in_source_json")
    domain = str(rec.get("domain") or rec.get("type") or "")
    type_ = domain
    response = str(rec.get("response") or "")

    match_error = None

    # -------------------------------------
    # A) GT load + normalize
    # -------------------------------------
    annotation = resolve_annotation(str(source_json), _safe_int_or_none(index_in_source_json))
    gt_lists = get_gt_object_lists(annotation)

    required_gt_objects = [
        normalize_gt_object(x, gt_index=i, group="required")
        for i, x in enumerate(gt_lists["required"])
    ]
    optional_gt_objects = [
        normalize_gt_object(x, gt_index=i, group="optional")
        for i, x in enumerate(gt_lists["optional"])
    ]

    # -------------------------------------
    # B) Validator + required/optional views
    # -------------------------------------
    validator_result = validate_file_level_category_consistency(required_gt_objects)
    if not validator_result["ok"]:
        diag = {
            "sample_id": sample_id,
            "image_path": image_path,
            "source_json": source_json,
            "index_in_source_json": index_in_source_json,
            "validator": validator_result,
        }
        raise RuntimeError(
            "[validator] hierarchy validator failed:\n"
            + json.dumps(diag, ensure_ascii=False, indent=2)
        )

    required_view = build_required_hierarchy_index(required_gt_objects)
    validator_mode = "hierarchy"
    optional_view = build_optional_whitelist(optional_gt_objects)

    if cfg.gt_mode == "atomic_with_contraction":
        raise NotImplementedError(
            "atomic_with_contraction mode is not implemented yet; "
            "atomic GT mode not implemented; "
            "safe rule-based contraction not implemented; "
            "current implementation only supports bucket mode."
        )

    # -------------------------------------
    # C) Extraction stage
    # -------------------------------------
    predicted_objects, extraction_raw, arr_err = run_extraction_stage(
        response=response,
        type_=type_,
        cfg=cfg,
        extractor_model=extractor_model,
    )

    initial_match_results: List[Dict[str, Any]] = []
    initial_match_debug: Dict[str, Any] = {}
    candidate_graph: Dict[str, Any] = {
        "pred_to_candidate_gts": {},
        "gt_to_candidate_preds": {},
        "edges": [],
        "required_pred_indices": [],
        "optional_pred_indices": [],
        "category_hallucination_list": [],
    }
    pair_attribute_scoring_payloads: List[Dict[str, Any]] = []
    pair_attribute_scores: List[Dict[str, Any]] = []
    pair_attribute_score_debug: List[Dict[str, Any]] = []
    solver_input: Dict[str, Any] = {"pred_nodes": [], "gt_nodes": [], "edges": []}
    solver_output: Dict[str, Any] = {
        "objective": {"total_matched_quantity": 0, "total_cost": 0},
        "activated_edges": [],
        "assigned_quantities": [],
        "pred_solution": [],
        "gt_solution": [],
    }
    constraint_violations: List[Dict[str, Any]] = []
    quantity_hallucination_accounting: Dict[str, Any] = {"items": [], "total_residual_quantity": 0.0}
    resolved_pairs: List[Dict[str, Any]] = []
    final_flows: List[Dict[str, Any]] = []
    pred_flow_accounting: List[Dict[str, Any]] = []
    gt_flow_accounting: List[Dict[str, Any]] = []

    if predicted_objects is not None:
        # -------------------------------------
        # D) Candidate selection stage
        # -------------------------------------
        initial_match_results, initial_match_debug, candidate_graph, _ = run_candidate_selection_stage(
            cfg=cfg,
            judge_model=judge_model,
            predicted_objects=predicted_objects,
            required_view=required_view,
            optional_view=optional_view,
            required_gt_objects=required_gt_objects,
            optional_gt_objects=optional_gt_objects,
            type_=type_,
        )

        # -------------------------------------
        # E/F/G) Final matching stage
        # -------------------------------------
        final_stage_result = run_final_matching_stage(
            cfg=cfg,
            judge_model=judge_model,
            predicted_objects=predicted_objects,
            required_gt_objects=required_gt_objects,
            initial_match_results=initial_match_results,
            candidate_graph=candidate_graph,
        )
        pair_attribute_scoring_payloads = final_stage_result["pair_attribute_scoring_payloads"]
        pair_attribute_scores = final_stage_result["pair_attribute_scores"]
        pair_attribute_score_debug = final_stage_result["pair_attribute_score_debug"]
        solver_input = final_stage_result["solver_input"]
        solver_output = final_stage_result["solver_output"]
        constraint_violations = final_stage_result["constraint_violations"]
        resolved_pairs = final_stage_result["resolved_pairs"]
        final_flows = final_stage_result["final_flows"]
        pred_flow_accounting = final_stage_result["pred_flow_accounting"]
        gt_flow_accounting = final_stage_result["gt_flow_accounting"]
        quantity_hallucination_accounting = final_stage_result["quantity_hallucination_accounting"]

    # -------------------------------------
    # H) Summaries
    # -------------------------------------
    per_pred_summary = build_per_pred_summary(
        pred_objects=predicted_objects or [],
        initial_match_results=initial_match_results,
        final_flows=final_flows,
        pred_flow_accounting=pred_flow_accounting,
    )
    per_gt_summary = build_per_gt_summary(
        required_gt_objects=required_gt_objects,
        optional_gt_objects=optional_gt_objects,
        final_flows=final_flows,
        solver_output=solver_output,
        gt_flow_accounting=gt_flow_accounting,
    )

    sample_record: Dict[str, Any] = {
        "sample_id": sample_id,
        "domain": domain,
        "image_path": image_path,
        "source_json": source_json,
        "index_in_source_json": index_in_source_json,
        "input": {
            "response": response,
            "domain": domain,
            "type": type_,
            "stage1_generated_at": rec.get("generated_at"),
            "model_name": rec.get("model_name"),
            "model_ckpt": rec.get("model_ckpt"),
            "prompt": rec.get("prompt"),
        },
        "gt_required_objects": required_gt_objects,
        "gt_optional_objects": optional_gt_objects,
        "predicted_objects": predicted_objects,

        "required_hierarchy_view": {
            "mode": required_view.get("mode"),
            "rows": required_view.get("rows"),
            "coord_meta": required_view.get("coord_meta_view"),
            "chains": required_view.get("chains"),
        },
        "optional_whitelist_view": {
            "labels": optional_view.get("labels"),
            "items": optional_view.get("view_items"),
        },

        "initial_match_results": initial_match_results,
        "candidate_graph": candidate_graph,
        "category_hallucination_list": candidate_graph.get("category_hallucination_list"),
        "pair_attribute_scoring_payloads": pair_attribute_scoring_payloads,
        "pair_attribute_scores": pair_attribute_scores,
        "pair_attribute_score_debug": pair_attribute_score_debug,
        "solver_input": solver_input,
        "solver_output": solver_output,
        "quantity_hallucination_accounting": quantity_hallucination_accounting,
        "constraint_violations": constraint_violations,

        "resolved_pairs": resolved_pairs,
        "final_flows": final_flows,
        "pred_flow_accounting": pred_flow_accounting,
        "gt_flow_accounting": gt_flow_accounting,

        "per_pred_summary": per_pred_summary,
        "per_gt_summary": per_gt_summary,

        "debug": {
            "matching_policy": "stage1_candidate_graph_plus_pair_scorer_plus_capacitated_b_matching",
            "precision_policy": "quantity_weighted_with_pairwise_uncertain_pred_discount",
            "flow_score_policy": "overall_granularity",
            "recall_policy": "solver_repair_accounting_with_stage1_gt_imputation",
            "validator_mode": validator_mode,
            "validator": validator_result,
            "extraction": {
                "raw_text": extraction_raw,
                "parse_error": arr_err.get("parse_error") if isinstance(arr_err, dict) else arr_err,
                "guided_json": arr_err.get("guided_json") if isinstance(arr_err, dict) else None,
                "model_used": cfg.judge_model,
                "ckpt_used": cfg.judge_ckpt,
            },
            "initial_match": initial_match_debug,
            "solver_repair": solver_output.get("repair"),
        },
        "parse_error": None,
        "match_error": match_error,
    }
    return sample_record


# =========================================================
# 14) Single-process runner
# =========================================================
def run_single_process(
    cfg: ExtractAndMatchConfig,
    tasks: List[Dict[str, Any]],
    out_dir: str,
    lock_path: str,
) -> Dict[str, int]:
    per_sample_jsonl = os.path.join(out_dir, "per_sample.jsonl")
    errors_jsonl = os.path.join(out_dir, "errors.jsonl")

    # One unified model for all stages.
    unified_model = build_chat_model(cfg)

    total = len(tasks)
    ok_cnt = 0
    err_cnt = 0
    last_print = 0.0

    for idx, rec in enumerate(tasks, start=1):
        now = time.time()
        if idx == 1 or (now - last_print) >= cfg.progress_interval_sec:
            _eprint(f"[progress] {idx - 1}/{total} ok={ok_cnt} err={err_cnt}")
            last_print = now

        sid = rec.get("sample_id")
        t0 = time.time()
        try:
            sample_record = process_one_sample(
                rec,
                cfg=cfg,
                extractor_model=unified_model,
                judge_model=unified_model,
            )
            sample_record["time"] = now_iso()
            sample_record["latency_sec"] = round(time.time() - t0, 4)

            with FileLock(lock_path):
                safe_jsonl_append(per_sample_jsonl, sample_record)
            ok_cnt += 1

        except Exception as e:
            err = {
                "time": now_iso(),
                "sample_id": sid,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=50),
                "image_path": rec.get("image_path"),
                "source_json": rec.get("source_json"),
                "index_in_source_json": rec.get("index_in_source_json"),
                "latency_sec": round(time.time() - t0, 4),
            }
            with FileLock(lock_path):
                safe_jsonl_append(errors_jsonl, err)
            err_cnt += 1

    _eprint(f"[progress] done {total}/{total} ok={ok_cnt} err={err_cnt}")
    return {"done": total, "ok": ok_cnt, "err": err_cnt}


# =========================================================
# 15) Main run
# =========================================================
def run_extract_and_match(cfg: ExtractAndMatchConfig) -> None:
    paths = resolve_stage1_paths(cfg.run_root)
    if not os.path.exists(paths["results_jsonl"]):
        raise FileNotFoundError(f"stage1 results.jsonl not found: {paths['results_jsonl']}")

    out_dir = extract_and_match_output_dir(cfg.run_root, cfg.run_id)
    ensure_dir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    per_sample_jsonl = os.path.join(out_dir, "per_sample.jsonl")
    errors_jsonl = os.path.join(out_dir, "errors.jsonl")
    lock_path = os.path.join(out_dir, ".write.lock")

    if not os.path.exists(per_sample_jsonl):
        open(per_sample_jsonl, "a", encoding="utf-8").close()
    if not os.path.exists(errors_jsonl):
        open(errors_jsonl, "a", encoding="utf-8").close()

    if cfg.mode == "resume" and os.path.exists(manifest_path):
        validate_manifest_params(manifest_path, cfg)

    stage1_meta = None
    if os.path.exists(paths["meta_json"]):
        try:
            stage1_meta = load_json(paths["meta_json"])
        except Exception:
            stage1_meta = None

    recs = load_stage1_records(paths["results_jsonl"])
    recs, relocation_debug = relocate_stage1_records(
        recs,
        dataset_root_override=cfg.dataset_root_override,
    )
    if cfg.mode == "resume":
        done_ids = read_done_ids(per_sample_jsonl)
        recs_todo = [r for r in recs if str(r.get("sample_id")) not in done_ids]
    else:
        recs_todo = recs

    recs_todo, priority_debug = reorder_records_by_priority(
        recs_todo,
        priority_image_paths=cfg.priority_image_paths,
        only_priority=cfg.only_priority,
    )

    _eprint(
        f"[run] stage1_records={len(recs)} todo={len(recs_todo)} "
        f"mode={cfg.mode} unified_model={cfg.judge_model}"
    )

    manifest = {
        "run_id": cfg.run_id,
        "created_at": now_iso(),
        "run_root": os.path.abspath(cfg.run_root),
        "stage1_results_jsonl": os.path.abspath(paths["results_jsonl"]),
        "dataset_root_override": cfg.dataset_root_override,

        # extractor_* retained for compatibility, but now identical to judge_*
        "extractor_model": cfg.extractor_model,
        "extractor_ckpt": cfg.extractor_ckpt,

        "judge_model": cfg.judge_model,
        "judge_ckpt": cfg.judge_ckpt,

        "extractor_params": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_new_tokens": cfg.max_new_tokens,
        },
        "judge_params": {
            "temperature": cfg.judge_temperature,
            "top_p": cfg.judge_top_p,
            "max_new_tokens": cfg.judge_max_new_tokens,
        },
        "gt_mode": cfg.gt_mode,
        "candidate_strategy": cfg.candidate_strategy,
        "model_sharing": {
            "all_stages_use_judge_model": True,
        },
        "llm_backend": cfg.llm_backend,
        "vllm_params": {
            "tensor_parallel_size": cfg.vllm_tensor_parallel_size,
            "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
            "dtype": cfg.vllm_dtype,
            "max_model_len": cfg.vllm_max_model_len,
        },
        "guided_json": {
            "mode": cfg.guided_json_mode,
            "applied_phases": [
                "extraction",
                "candidate_step1",
                "candidate_step2",
                "pair_attribute_scoring",
            ],
            "extraction_schema_note": "attributes uses open additionalProperties with string-array values",
        },
        "runtime": {
            "python": sys.version,
            "platform": sys.platform,
            "execution_mode": "single_process",
        },
        "resume": {
            "mode": cfg.mode,
            "skipped_done": (len(recs) - len(recs_todo)) if cfg.mode == "resume" else 0,
        },
        "parallel": {
            "num_workers": cfg.num_workers,
        },
        "io": {
            "output_dir": os.path.abspath(out_dir),
            "input_stage1_meta_present": bool(stage1_meta is not None),
            "priority_image_paths_count": len(cfg.priority_image_paths),
            "priority_image_paths_file": cfg.priority_image_paths_file,
            "only_priority": bool(cfg.only_priority),
            "priority_debug": priority_debug,
            "relocation_debug": relocation_debug,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not recs_todo:
        _eprint("[run] nothing to do (already complete or filtered by only-priority).")
        return

    prog = run_single_process(
        cfg=cfg,
        tasks=recs_todo,
        out_dir=out_dir,
        lock_path=lock_path,
    )

    _eprint(f"[run] summary: done={prog['done']} ok={prog['ok']} err={prog['err']}")
    _eprint(f"[run] outputs: {out_dir}")


def main():
    args = parse_args()
    cfg = build_config(args)
    run_extract_and_match(cfg)


if __name__ == "__main__":
    main()
