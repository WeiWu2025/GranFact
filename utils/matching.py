#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional, Tuple

from utils.common import *
from utils.guided_json import generate_parse_with_guided_json_policy, pair_attribute_scoring_schema
from utils.prompts import build_pair_attribute_scoring_messages

def build_candidate_graph(
    *,
    initial_match_results: List[Dict[str, Any]],
    predicted_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pred_to_candidate_gts: Dict[str, List[int]] = {}
    gt_to_candidate_preds: Dict[str, List[int]] = {}
    edges: List[Dict[str, Any]] = []
    category_hallucination_list: List[Dict[str, Any]] = []
    optional_pred_indices: List[int] = []
    required_pred_indices: List[int] = []

    for row in initial_match_results:
        pi = int(row["pred_index"])
        status = str(row.get("status") or "none")
        candidate_gt_ids = sorted(set(int(x) for x in (row.get("candidate_gt_ids") or [])))

        if status == "required" and candidate_gt_ids:
            required_pred_indices.append(pi)
            pred_to_candidate_gts[str(pi)] = candidate_gt_ids
            for gj in candidate_gt_ids:
                gt_to_candidate_preds.setdefault(str(gj), []).append(pi)
                edges.append({
                    "edge_id": f"p{pi}->g{gj}",
                    "pred_index": pi,
                    "gt_index": gj,
                    "pred_category_text": predicted_objects[pi].get(PRED_CATEGORY_FIELD),
                    "gt_category_text": required_gt_objects[gj].get("deepest_label"),
                    "anchor_required_coord": row.get("required_coord"),
                    "anchor_depth": row.get("supported_depth"),
                    "category_credit_depth": row.get("category_credit_depth"),
                    "category_match_type": row.get("category_match_type"),
                    "compatible_required_coord": row.get("compatible_required_coord"),
                    "compatible_required_label": row.get("compatible_required_label"),
                    "anchor_label": row.get("required_label"),
                })
        elif status == "optional":
            optional_pred_indices.append(pi)
        elif status == "none":
            category_hallucination_list.append({
                "pred_index": pi,
                "pred_category_text": predicted_objects[pi].get(PRED_CATEGORY_FIELD),
                "reason": row.get("category_rejection_reason") or "no_legal_required_or_optional_candidate",
                "category_match_type": row.get("category_match_type"),
                "category_match_reason": row.get("category_match_reason"),
                "rejected_required_coord": row.get("rejected_required_coord"),
                "rejected_required_label": row.get("rejected_required_label"),
            })

    for gj, pred_indices in list(gt_to_candidate_preds.items()):
        gt_to_candidate_preds[gj] = sorted(set(pred_indices))

    return {
        "pred_to_candidate_gts": pred_to_candidate_gts,
        "gt_to_candidate_preds": gt_to_candidate_preds,
        "edges": edges,
        "required_pred_indices": sorted(set(required_pred_indices)),
        "optional_pred_indices": sorted(set(optional_pred_indices)),
        "category_hallucination_list": category_hallucination_list,
    }


def build_pair_attribute_scoring_payload(
    *,
    pred_index: int,
    pred_obj: Dict[str, Any],
    candidate_gt_ids: List[int],
    initial_match_row: Dict[str, Any],
    required_gt_objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pred_facts = build_attribute_facts_wo_number(pred_obj)
    candidates: List[Dict[str, Any]] = []
    for gj in candidate_gt_ids:
        if int(gj) < 0 or int(gj) >= len(required_gt_objects):
            raise ValueError(
                f"pair attribute scoring received non-required GT index: {gj}; "
                f"num_required_gt={len(required_gt_objects)}"
            )
        gt_obj = required_gt_objects[gj]
        gt_facts = build_attribute_facts_wo_number(gt_obj)
        candidates.append({
            "candidate_id": f"g{gj}",
            "gt_index": gj,
            "gt_attr_facts_wo_number": [
                {"fact_index": f["fact_index"], "path": f["path"], "value": f["value"]}
                for f in gt_facts
            ],
        })

    return {
        "pred_index": pred_index,
        "pred": {
            "attr_facts_wo_number": [
                {"fact_index": f["fact_index"], "path": f["path"], "value": f["value"]}
                for f in pred_facts
            ],
        },
        "candidates": candidates,
    }


def parse_pair_attribute_scoring_result(
    raw_text: str,
    *,
    payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    parsed, parse_error = parse_json_object(raw_text)
    debug: Dict[str, Any] = {
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error,
        "schema_errors": [],
    }

    if not isinstance(parsed, dict):
        raise ValueError(
            "[pair-scorer] failed to parse scorer JSON object:\n"
            + json.dumps(debug, ensure_ascii=False, indent=2)
        )

    parsed_candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(parsed_candidates, dict):
        raise ValueError(
            "[pair-scorer] missing candidates field:\n"
            + json.dumps(debug, ensure_ascii=False, indent=2)
        )

    results: List[Dict[str, Any]] = []
    for cand in payload.get("candidates") or []:
        candidate_id = str(cand["candidate_id"])
        gt_index = int(cand["gt_index"])
        gt_fact_count = len(cand.get("gt_attr_facts_wo_number") or [])
        item = parsed_candidates.get(candidate_id) if isinstance(parsed_candidates, dict) else None

        if not isinstance(item, dict):
            debug["schema_errors"].append({
                "candidate_id": candidate_id,
                "error": "missing candidate result",
            })
            raise ValueError(
                "[pair-scorer] missing candidate result:\n"
                + json.dumps(debug, ensure_ascii=False, indent=2)
            )

        pred_fact_count = len(payload.get("pred", {}).get("attr_facts_wo_number") or [])

        raw_correct_pred = item.get("correct_pred_fact_indices") if isinstance(item, dict) else None
        raw_wrong_pred = item.get("wrong_pred_fact_indices") if isinstance(item, dict) else None
        raw_recalled_gt = item.get("recalled_gt_fact_indices") if isinstance(item, dict) else None

        # Backward compatibility: old model output used matched_gt_fact_indices
        # and has_contradiction only. Treat matched_gt as recalled_gt, and leave
        # pred-side correct/wrong empty unless the new fields are present.
        if raw_recalled_gt is None:
            raw_recalled_gt = item.get("matched_gt_fact_indices") if isinstance(item, dict) else None

        if raw_correct_pred is None:
            raw_correct_pred = []
        if raw_wrong_pred is None:
            raw_wrong_pred = []

        if not isinstance(raw_correct_pred, list):
            debug["schema_errors"].append({
                "candidate_id": candidate_id,
                "item": item,
                "error": "correct_pred_fact_indices must be a list",
            })
            raise ValueError(
                "[pair-scorer] invalid correct_pred_fact_indices:\n"
                + json.dumps(debug, ensure_ascii=False, indent=2)
            )
        if not isinstance(raw_wrong_pred, list):
            debug["schema_errors"].append({
                "candidate_id": candidate_id,
                "item": item,
                "error": "wrong_pred_fact_indices must be a list",
            })
            raise ValueError(
                "[pair-scorer] invalid wrong_pred_fact_indices:\n"
                + json.dumps(debug, ensure_ascii=False, indent=2)
            )
        if not isinstance(raw_recalled_gt, list):
            debug["schema_errors"].append({
                "candidate_id": candidate_id,
                "item": item,
                "error": "recalled_gt_fact_indices must be a list",
            })
            raise ValueError(
                "[pair-scorer] invalid recalled_gt_fact_indices:\n"
                + json.dumps(debug, ensure_ascii=False, indent=2)
            )

        correct_pred_fact_indices = _normalize_fact_index_list(raw_correct_pred, pred_fact_count)
        wrong_pred_fact_indices = _normalize_fact_index_list(raw_wrong_pred, pred_fact_count)

        # Enforce disjoint pred-side labels. If the LLM put a pred fact in both
        # lists, prefer "wrong" because explicit contradiction should be visible.
        wrong_set = set(wrong_pred_fact_indices)
        correct_pred_fact_indices = [idx for idx in correct_pred_fact_indices if idx not in wrong_set]

        recalled_gt_fact_indices = _normalize_fact_index_list(raw_recalled_gt, gt_fact_count)
        uncertain_pred_fact_indices = [
            idx for idx in range(pred_fact_count)
            if idx not in set(correct_pred_fact_indices) and idx not in wrong_set
        ]
        unrecalled_gt_fact_indices = [
            idx for idx in range(gt_fact_count)
            if idx not in set(recalled_gt_fact_indices)
        ]

        pred_judged_count = len(correct_pred_fact_indices) + len(wrong_pred_fact_indices)
        attribute_precision = 1.0 if pred_judged_count <= 0 else len(correct_pred_fact_indices) / float(pred_judged_count)
        attribute_recall = 1.0 if gt_fact_count <= 0 else len(recalled_gt_fact_indices) / float(gt_fact_count)
        if attribute_precision + attribute_recall <= 1e-12:
            attribute_f1 = 0.0
        else:
            attribute_f1 = 2.0 * attribute_precision * attribute_recall / (attribute_precision + attribute_recall)
        edge_utility = _clamp01(float(attribute_f1))
        contradiction = len(wrong_pred_fact_indices) > 0

        results.append({
            "pred_index": int(payload["pred_index"]),
            "gt_index": gt_index,
            "candidate_id": candidate_id,
            "correct_pred_fact_indices_wo_numberattr": correct_pred_fact_indices,
            "wrong_pred_fact_indices_wo_numberattr": wrong_pred_fact_indices,
            "uncertain_pred_fact_indices_wo_numberattr": uncertain_pred_fact_indices,
            "recalled_gt_fact_indices_wo_numberattr": recalled_gt_fact_indices,
            "unrecalled_gt_fact_indices_wo_numberattr": unrecalled_gt_fact_indices,
            "matched_gt_fact_indices_wo_numberattr": recalled_gt_fact_indices,
            "matched_gt_attribute_count_wo_numberattr": len(recalled_gt_fact_indices),
            "gt_attribute_fact_count_wo_numberattr": gt_fact_count,
            "pred_attribute_fact_count_wo_numberattr": pred_fact_count,
            "correct_pred_attribute_count_wo_numberattr": len(correct_pred_fact_indices),
            "wrong_pred_attribute_count_wo_numberattr": len(wrong_pred_fact_indices),
            "uncertain_pred_attribute_count_wo_numberattr": len(uncertain_pred_fact_indices),
            "recalled_gt_attribute_count_wo_numberattr": len(recalled_gt_fact_indices),
            "unrecalled_gt_attribute_count_wo_numberattr": len(unrecalled_gt_fact_indices),
            "attribute_precision": _clamp01(float(attribute_precision)),
            "attribute_recall": _clamp01(float(attribute_recall)),
            "attribute_f1": _clamp01(float(attribute_f1)),
            "has_contradiction": bool(contradiction),
            "edge_utility": edge_utility,
            "reason": str(item.get("reason") or "") if isinstance(item, dict) else "",
        })

    return results, debug


def run_pair_attribute_scoring(
    *,
    cfg: "ExtractAndMatchConfig",
    judge_model: "LocalChatModel",
    predicted_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
    initial_match_results: List[Dict[str, Any]],
    candidate_graph: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    initial_map = {int(x["pred_index"]): x for x in initial_match_results}
    payloads: List[Dict[str, Any]] = []
    pair_attribute_scores: List[Dict[str, Any]] = []
    pair_attribute_score_debug: List[Dict[str, Any]] = []

    for pi in candidate_graph.get("required_pred_indices") or []:
        init_row = initial_map.get(int(pi), {})
        candidate_gt_ids = list(candidate_graph.get("pred_to_candidate_gts", {}).get(str(pi)) or [])
        if not candidate_gt_ids:
            continue

        payload = build_pair_attribute_scoring_payload(
            pred_index=int(pi),
            pred_obj=predicted_objects[int(pi)],
            candidate_gt_ids=candidate_gt_ids,
            initial_match_row=init_row,
            required_gt_objects=required_gt_objects,
        )
        payloads.append(payload)
        messages = build_pair_attribute_scoring_messages(payload)

        def _parse_pair(raw_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            return parse_pair_attribute_scoring_result(
                raw_text,
                payload=payload,
            )

        (parsed_scores, debug), guided_debug = generate_parse_with_guided_json_policy(
            cfg=cfg,
            model=judge_model,
            messages=messages,
            schema=pair_attribute_scoring_schema(),
            schema_name="pair_attribute_scoring_v1",
            parse_fn=_parse_pair,
            max_new_tokens=cfg.judge_max_new_tokens,
            temperature=cfg.judge_temperature,
            top_p=cfg.judge_top_p,
        )
        raw_text = guided_debug.get("fallback_raw_text") or guided_debug.get("first_raw_text")
        debug["guided_json"] = guided_debug
        pair_attribute_scores.extend(parsed_scores)
        pair_attribute_score_debug.append({
            "pred_index": int(pi),
            "payload": payload,
            "raw_text": raw_text,
            "parsed": debug.get("parsed"),
            "parse_error": debug.get("parse_error"),
            "schema_errors": debug.get("schema_errors"),
            "guided_json": debug.get("guided_json"),
        })

    return payloads, pair_attribute_scores, pair_attribute_score_debug
