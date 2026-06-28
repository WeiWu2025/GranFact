#!/usr/bin/env python3
# -*- coding: utf-8 -*>

"""
Pairwise candidate selection strategy (expanded level-wise).

For each (pred, gt) pair, check each level from deepest to shallowest:
- For each level, call LLM to check if pred is-a any of the level candidates
- Stop when first match is found (preferring more specific matches)

Cache mechanism: Level match results are cached at granularity of (pred_category, level_candidate),
allowing cross-GT reuse. For example, checking "iPhone 15 Pro Max is-a smartphone" can be reused
across all GTs that have "smartphone" at their L1.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from utils.common import PRED_CATEGORY_FIELD, parse_json_object
from utils.guided_json import (
    candidate_pairwise_level_match_schema,
    generate_parse_with_guided_json_policy,
)
from utils.matching import build_candidate_graph
from utils.prompts import build_candidate_pairwise_level_match_messages

if TYPE_CHECKING:
    from utils.config import ExtractAndMatchConfig
    from utils.llm import LocalChatModel


# =========================================================
# Parser
# =========================================================

def parse_level_match_result(
    raw_text: str,
    pred_category: str,
    target_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse three-way level matching result."""
    parsed, parse_error = parse_json_object(raw_text)
    debug = {
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error,
        "pred_category": pred_category,
        "target_label": target_label,
    }

    if not isinstance(parsed, dict):
        raise ValueError(
            "[level-match] failed to parse JSON:\n"
            + json.dumps(debug, ensure_ascii=False, indent=2)
        )

    decision = parsed.get("decision")
    valid_decisions = ["is_a", "cannot_refer_to", "may_refer_to"]
    if decision not in valid_decisions:
        decision = "may_refer_to"  # Default to continuing on parse error

    return {
        "decision": decision,
        "reason": str(parsed.get("reason") or ""),
    }, debug



# =========================================================
# Cache utilities
# =========================================================

def _compute_level_cache_key(pred_category: str, target_label: str) -> str:
    """Compute cache key for level match: pred_category|target_label."""
    return f"{pred_category}|{target_label}"


# =========================================================
# Level-wise matching with cache
# =========================================================


def check_level_match(
    *,
    cfg: "ExtractAndMatchConfig",
    judge_model: "LocalChatModel",
    pred_category: str,
    target_label: str,
    type_: str,
    level_match_cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """
    Three-way level matching: is_a, cannot_refer_to, or may_refer_to.

    Returns:
        (result_dict with decision/reason, from_cache, debug_info)
    """
    cache_key = _compute_level_cache_key(pred_category, target_label)

    # Check cache first
    if cache_key in level_match_cache:
        return level_match_cache[cache_key], True, {"from_cache": True, "cache_key": cache_key}

    # Call LLM
    messages = build_candidate_pairwise_level_match_messages(
        pred_category=pred_category,
        target_label=target_label,
        type_=type_,
    )

    def _parse(raw_text: str):
        return parse_level_match_result(raw_text, pred_category, target_label)

    try:
        (result, parse_debug), _ = generate_parse_with_guided_json_policy(
            cfg=cfg,
            model=judge_model,
            messages=messages,
            schema=candidate_pairwise_level_match_schema(),
            schema_name="level_match",
            parse_fn=_parse,
            max_new_tokens=cfg.judge_max_new_tokens,
            temperature=cfg.judge_temperature,
            top_p=cfg.judge_top_p,
        )
        result["decision"] = result.get("decision", "may_refer_to")
    except Exception as e:
        result = {"decision": "may_refer_to", "reason": repr(e)}
        parse_debug = {"error": repr(e)}

    # Cache the result
    level_match_cache[cache_key] = result

    return result, False, parse_debug


def run_pairwise_match_for_pred(
    *,
    cfg: "ExtractAndMatchConfig",
    judge_model: "LocalChatModel",
    pred_category: str,
    pred_index: int,
    gt_objects: List[Dict[str, Any]],
    gt_start_index: int,
    match_type: str,
    type_: str = "",
    level_match_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run pairwise matching for one prediction against all GTs.

    Args:
        gt_objects: List of GT objects to match against
        gt_start_index: Starting index for GT objects (for optional, starts from len(required))
        match_type: "required" or "optional" - determines how results are categorized
        type_: Domain type
        level_match_cache: Shared cache for level match results
    """
    results = []

    for offset, gt_obj in enumerate(gt_objects):
        gt_index = gt_start_index + offset
        chain = gt_obj.get("category_levels", [])
        if not chain:
            results.append({
                "pred_index": pred_index,
                "gt_index": gt_index,
                "status": "none",
                "matched_level_index": None,
                "matched_label": None,
                "level_match_details": [],
            })
            continue

        matched_level_index = None
        matched_label = None
        level_match_details = []
        all_from_cache = True

        # Iterate from deepest level to L0
        for level_idx, level_candidates in reversed(list(enumerate(chain))):
            level_detail = {
                "level_index": level_idx,
                "candidates": level_candidates,
                "checked": False,
                "decision": None,
                "reason": None,
                "matched_candidate": None,
                "from_cache": False,
                "candidate_details": [],  # Track each candidate's result
            }

            # Check each candidate in this level
            level_decisions = []  # Collect decisions for all candidates
            for candidate in level_candidates:
                result, from_cache, _debug_info = check_level_match(
                    cfg=cfg,
                    judge_model=judge_model,
                    pred_category=pred_category,
                    target_label=candidate,
                    type_=type_,
                    level_match_cache=level_match_cache,
                )

                if not from_cache:
                    all_from_cache = False

                decision = result.get("decision", "may_refer_to")
                level_detail["candidate_details"].append({
                    "candidate": candidate,
                    "decision": decision,
                    "reason": result.get("reason", ""),
                })
                level_decisions.append(decision)

            # After checking ALL candidates, determine level decision:
            # 1. Any is_a → level is is_a (stop GT)
            # 2. All cannot_refer_to → level is cannot_refer_to (stop GT)
            # 3. Any may_refer_to (or mix) → level is may_refer_to (continue to shallower)
            if "is_a" in level_decisions:
                level_detail["checked"] = True
                level_detail["decision"] = "is_a"
                # Find the matching candidate
                for cd in level_detail["candidate_details"]:
                    if cd["decision"] == "is_a":
                        level_detail["matched_candidate"] = cd["candidate"]
                        matched_level_index = level_idx
                        matched_label = cd["candidate"]
                        break
                level_match_details.append(level_detail)
                break  # Matched, stop this GT

            elif all(d == "cannot_refer_to" for d in level_decisions):
                level_detail["checked"] = True
                level_detail["decision"] = "cannot_refer_to"
                level_detail["reason"] = f"All {len(level_candidates)} candidates returned cannot_refer_to"
                level_match_details.append(level_detail)
                break  # Cannot match, stop this GT

            else:
                # Mix of may_refer_to and/or some cannot_refer_to
                level_detail["checked"] = True
                level_detail["decision"] = "may_refer_to"
                level_match_details.append(level_detail)
                # Continue to shallower level

        # Determine final status
        if matched_level_index is not None:
            results.append({
                "pred_index": pred_index,
                "gt_index": gt_index,
                "match_type": match_type,
                "status": "matched",
                "matched_level_index": matched_level_index,
                "matched_label": matched_label,
                "from_cache": all_from_cache,
                "level_match_details": level_match_details,
            })
        else:
            results.append({
                "pred_index": pred_index,
                "gt_index": gt_index,
                "match_type": match_type,
                "status": "none",
                "matched_level_index": None,
                "matched_label": None,
                "from_cache": all_from_cache,
                "level_match_details": level_match_details,
            })

    return results


def convert_pairwise_results_to_initial_match(
    pairwise_results: List[Dict[str, Any]],
    pred_index: int,
    pred_category: str,
    required_gt_objects: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Convert pairwise results to initial_match_results format.

    Priority: required GTs > optional GTs
    - If matched to required GT, status = "required"
    - If only matched to optional GT, status = "optional"
    - If no required or optional match exists, status = "none" (hallucination)

    Returns:
        initial_match_row: Match result for this prediction (aggregated across all GTs)
        hallucination_entry: Hallucination entry if status is "none", else None
    """
    # Separate required and optional results
    required_results = [r for r in pairwise_results if r.get("match_type") == "required"]
    optional_results = [r for r in pairwise_results if r.get("match_type") == "optional"]

    # Find best required match
    required_matched = [r for r in required_results if r["status"] == "matched"]
    optional_matched = [r for r in optional_results if r["status"] == "matched"]

    matched_level_index = None
    candidate_gt_ids = []
    required_label = None
    total_levels_hint = None
    status = "none"
    reason = "pairwise_no_match_in_any_gt_chain"

    # Priority 1: Check required GT matches
    if required_matched:
        # Find deepest required match
        for result in required_matched:
            if matched_level_index is None or result["matched_level_index"] > matched_level_index:
                matched_level_index = result["matched_level_index"]
                required_label = result["matched_label"]
            candidate_gt_ids.append(result["gt_index"])

        status = "required"

    # Priority 2: Check optional GT matches (only if no valid required match)
    if status == "none" and optional_matched:
        matched_level_index = None
        candidate_gt_ids = []

        # Find deepest optional match
        for result in optional_matched:
            if matched_level_index is None or result["matched_level_index"] > matched_level_index:
                matched_level_index = result["matched_level_index"]
                required_label = result["matched_label"]
            candidate_gt_ids.append(result["gt_index"])

        status = "optional"

    # Remove duplicate GT IDs while preserving order
    seen = set()
    unique_candidate_gt_ids = []
    for gid in candidate_gt_ids:
        if gid not in seen:
            seen.add(gid)
            unique_candidate_gt_ids.append(gid)

    # Calculate total levels from any matched GT
    if matched_level_index is not None and unique_candidate_gt_ids:
        # Determine which objects list to use based on first matched GT
        first_gt_id = unique_candidate_gt_ids[0]
        if first_gt_id < len(required_gt_objects):
            gt_chain = required_gt_objects[first_gt_id].get("category_levels", [])
            if gt_chain:
                total_levels_hint = len(gt_chain)

    if status != "none":
        return {
            "pred_index": pred_index,
            "status": status,
            "required_coord": None,
            "optional_index": None,
            "candidate_gt_ids": unique_candidate_gt_ids,
            "supported_depth": matched_level_index + 1 if matched_level_index is not None else None,
            "category_credit_depth": float(matched_level_index + 1) if matched_level_index is not None else None,
            "matched_level_index": matched_level_index,
            "total_levels_hint": total_levels_hint,
            "required_label": required_label,
            "fallback_required_gt_index": unique_candidate_gt_ids[0] if unique_candidate_gt_ids else None,
        }, None
    else:
        # No match or conflicts with all siblings - hallucination
        return {
            "pred_index": pred_index,
            "status": "none",
            "required_coord": None,
            "optional_index": None,
            "candidate_gt_ids": [],
            "supported_depth": None,
            "category_credit_depth": None,
            "matched_level_index": None,
            "total_levels_hint": None,
            "required_label": None,
            "fallback_required_gt_index": None,
        }, {
            "pred_index": pred_index,
            "pred_category_text": pred_category,
            "reason": reason,
            "matched_level_index": matched_level_index,
        }


# =========================================================
# Main entry point
# =========================================================

def run_candidate_selection_pairwise(
    *,
    cfg: "ExtractAndMatchConfig",
    judge_model: "LocalChatModel",
    predicted_objects: List[Dict[str, Any]],
    required_view: Dict[str, Any],
    optional_view: Dict[str, Any],
    required_gt_objects: List[Dict[str, Any]],
    optional_gt_objects: List[Dict[str, Any]],
    type_: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], str]:
    """
    Run pairwise candidate selection (expanded level-wise mode).

    Strategy:
    1. First try to match against optional GTs
    2. If matched to optional, status = "optional"
    3. If not matched to optional, try required GTs
    4. If matched to required, status = "required"
    5. If neither, status = "none" (hallucination)

    Returns:
        initial_match_results: List of match results for each prediction
        debug: Debug information
        candidate_graph: Candidate bipartite graph
        raw_output: Raw LLM output for first call (compatibility)
    """
    initial_match_results: List[Dict[str, Any]] = []
    all_debug: Dict[str, Any] = {"strategy": "pairwise", "predictions": [], "cache_stats": {}}
    hallucination_list: List[Dict[str, Any]] = []

    # Caches shared across predictions within the same sample
    # level_match_cache: "pred_category|target_label" -> {decision, reason}
    level_match_cache: Dict[str, Dict[str, Any]] = {}

    cache_hits = 0
    cache_misses = 0

    # Total GT count for indexing
    num_required = len(required_gt_objects)
    num_optional = len(optional_gt_objects)

    for pi, pred_obj in enumerate(predicted_objects):
        pred_category = pred_obj.get(PRED_CATEGORY_FIELD, "")
        pred_debug: Dict[str, Any] = {
            "pred_index": pi,
            "pred_category": pred_category,
        }

        # Step 1: Try to match against optional GTs first
        optional_pairwise_results = []
        if num_optional > 0:
            optional_pairwise_results = run_pairwise_match_for_pred(
                cfg=cfg,
                judge_model=judge_model,
                pred_category=pred_category,
                pred_index=pi,
                gt_objects=optional_gt_objects,
                gt_start_index=num_required,  # optional GTs continue indexing from required
                match_type="optional",
                type_=type_,
                level_match_cache=level_match_cache,
            )

            for r in optional_pairwise_results:
                if r.get("from_cache"):
                    cache_hits += 1
                else:
                    cache_misses += 1

        # Step 2: Try to match against required GTs
        required_pairwise_results = run_pairwise_match_for_pred(
            cfg=cfg,
            judge_model=judge_model,
            pred_category=pred_category,
            pred_index=pi,
            gt_objects=required_gt_objects,
            gt_start_index=0,
            match_type="required",
            type_=type_,
            level_match_cache=level_match_cache,
        )

        for r in required_pairwise_results:
            if r.get("from_cache"):
                cache_hits += 1
            else:
                cache_misses += 1

        # Combine results: required results + optional results
        pairwise_results = required_pairwise_results + optional_pairwise_results

        # Keep level_match_details for debug
        pred_debug["pairwise_results"] = list(pairwise_results)

        # Convert to initial_match format
        initial_match_row, hallucination_entry = convert_pairwise_results_to_initial_match(
            pairwise_results=pairwise_results,
            pred_index=pi,
            pred_category=pred_category,
            required_gt_objects=required_gt_objects,
        )

        initial_match_results.append(initial_match_row)
        if hallucination_entry:
            hallucination_list.append(hallucination_entry)

        pred_debug["initial_match"] = initial_match_row
        all_debug["predictions"].append(pred_debug)

    # Build candidate graph
    candidate_graph = build_candidate_graph(
        initial_match_results=initial_match_results,
        predicted_objects=predicted_objects,
        required_gt_objects=required_gt_objects,
    )
    candidate_graph["category_hallucination_list"] = hallucination_list

    # Cache stats
    all_debug["cache_stats"] = {
        "hits": cache_hits,
        "misses": cache_misses,
        "total": cache_hits + cache_misses,
        "unique_level_candidates_cached": len(level_match_cache),
    }

    # Extract raw output for compatibility
    first_raw = None
    for pred_debug in all_debug["predictions"]:
        for r in pred_debug.get("pairwise_results", []):
            if "parse_debug" in r:
                first_raw = r["parse_debug"].get("raw_text")
                break
        if first_raw:
            break

    return initial_match_results, all_debug, candidate_graph, first_raw
