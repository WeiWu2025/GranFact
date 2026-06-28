#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

from utils.common import *

def pred_total_quantity_for_precision(pred_obj: Dict[str, Any]) -> int:
    eff = effective_number_value(str(pred_obj.get("number_type") or ""), pred_obj.get("number_value"))
    if eff is None:
        return 1
    return max(1, int(eff))


def compute_pred_metric_quantities_for_precision(
    pred_objects: List[Dict[str, Any]],
    initial_match_results: List[Dict[str, Any]],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    initial_map = {int(x["pred_index"]): x for x in initial_match_results}

    definite_vals: List[float] = []
    for pi, pred_obj in enumerate(pred_objects):
        init = initial_map.get(pi, {})
        init_status = str(init.get("status") or "none")
        if init_status == "optional":
            continue

        number_type = str(pred_obj.get("number_type") or "")
        number_value = pred_obj.get("number_value")
        if number_type == "uncertain":
            continue

        eff = effective_number_value(number_type, number_value)
        if eff is None:
            eff = 1
        definite_vals.append(float(max(1, int(eff))))

    uncertain_fill_value = (sum(definite_vals) / len(definite_vals)) if definite_vals else 1.0

    metric_quantities: Dict[int, float] = {}
    debug_items: List[Dict[str, Any]] = []
    for pi, pred_obj in enumerate(pred_objects):
        init = initial_map.get(pi, {})
        init_status = str(init.get("status") or "none")
        number_type = str(pred_obj.get("number_type") or "")
        number_value = pred_obj.get("number_value")

        if init_status == "optional":
            metric_quantity = 0.0
            source = "optional_neutral"
        elif number_type == "uncertain":
            metric_quantity = float(uncertain_fill_value)
            source = "uncertain_fill_mean" if definite_vals else "uncertain_fill_default_1"
        else:
            eff = effective_number_value(number_type, number_value)
            if eff is None:
                eff = 1
            metric_quantity = float(max(1, int(eff)))
            source = "effective_number_value"

        metric_quantities[pi] = metric_quantity
        debug_items.append({
            "pred_index": pi,
            "initial_status": init_status,
            "number_type": number_type,
            "number_value": number_value,
            "pred_metric_quantity_for_precision": metric_quantity,
            "pred_metric_quantity_source": source,
        })

    return metric_quantities, {
        "definite_pred_number_values_for_precision": definite_vals,
        "uncertain_fill_value_for_precision": uncertain_fill_value,
        "items": debug_items,
    }


def gt_definite_quantity_for_exhaustion(gt_obj: Dict[str, Any]) -> Optional[int]:
    number_type = str(gt_obj.get("number_type") or "")
    if number_type == "uncertain":
        return None
    eff = effective_number_value(number_type, gt_obj.get("number_value"))
    if eff is None:
        return None
    return max(0, int(eff))


def compute_required_gt_metric_quantities(required_gt_objects: List[Dict[str, Any]]) -> Tuple[Dict[int, float], Dict[str, Any]]:
    definite_vals: List[float] = []
    for gt in required_gt_objects:
        number_type = str(gt.get("number_type") or "")
        if number_type == "uncertain":
            continue
        eff = effective_number_value(number_type, gt.get("number_value"))
        if eff is None:
            eff = 1
        definite_vals.append(float(max(1, int(eff))))

    uncertain_fill_value = (sum(definite_vals) / len(definite_vals)) if definite_vals else 1.0

    metric_quantities: Dict[int, float] = {}
    debug_items: List[Dict[str, Any]] = []
    for gt in required_gt_objects:
        gj = int(gt["gt_index"])
        number_type = str(gt.get("number_type") or "")
        number_value = gt.get("number_value")

        if number_type == "uncertain":
            metric_quantity = float(uncertain_fill_value)
            source = "uncertain_fill_mean" if definite_vals else "uncertain_fill_default_1"
        else:
            eff = effective_number_value(number_type, number_value)
            if eff is None:
                eff = 1
            metric_quantity = float(max(1, int(eff)))
            source = "effective_number_value"

        metric_quantities[gj] = metric_quantity
        debug_items.append({
            "gt_index": gj,
            "number_type": number_type,
            "number_value": number_value,
            "gt_metric_quantity": metric_quantity,
            "gt_metric_quantity_source": source,
        })

    return metric_quantities, {
        "definite_gt_number_values": definite_vals,
        "uncertain_fill_value": uncertain_fill_value,
        "items": debug_items,
    }


def build_per_pred_summary(
    *,
    pred_objects: List[Dict[str, Any]],
    initial_match_results: List[Dict[str, Any]],
    final_flows: List[Dict[str, Any]],
    pred_flow_accounting: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    initial_map = {int(x["pred_index"]): x for x in initial_match_results}
    accounting_map = {int(x["pred_index"]): x for x in pred_flow_accounting}
    flows_by_pred: Dict[int, List[Dict[str, Any]]] = {}
    for fl in final_flows:
        flows_by_pred.setdefault(int(fl["pred_index"]), []).append(fl)

    out: List[Dict[str, Any]] = []
    for pi, po in enumerate(pred_objects):
        init = initial_map.get(pi, {})
        acc = accounting_map.get(pi, {})
        flows = flows_by_pred.get(pi, [])
        real_flows = [fl for fl in flows if fl.get("flow_kind") == "real"]
        hall_flows = [fl for fl in flows if fl.get("flow_kind") == "hallucination"]
        init_status = str(init.get("status") or "none")

        if init_status == "optional":
            status = "optional_matched"
        elif init_status == "none":
            status = "category_hallucination"
        elif real_flows:
            status = "required_matched"
        else:
            status = "required_unmatched"

        pred_fact_wo = len(build_attribute_facts_wo_number(po))
        matched_wo_sum = sum(int((fl.get("attr_match") or {}).get("matched_gt_attribute_count_wo_numberattr") or 0) for fl in real_flows)
        matched_with_sum = sum(int((fl.get("attr_match") or {}).get("matched_gt_attribute_count_with_numberattr") or 0) for fl in real_flows)

        out.append({
            "pred_index": pi,
            "status": status,
            "initial_status": init_status,
            "required_coord": init.get("required_coord"),
            "optional_index": init.get("optional_index"),
            "candidate_gt_ids": init.get("candidate_gt_ids") or [],
            "supported_depth": init.get("supported_depth"),
            "category_credit_depth": init.get("category_credit_depth"),
            "category_match_type": init.get("category_match_type"),
            "compatible_required_coord": init.get("compatible_required_coord"),
            "compatible_required_label": init.get("compatible_required_label"),
            "matched_level_index": init.get("matched_level_index"),
            "flow_ids": [fl["flow_id"] for fl in flows],
            "real_flow_ids": [fl["flow_id"] for fl in real_flows],
            "hallucination_flow_ids": [fl["flow_id"] for fl in hall_flows],
            "matched_gt_indices": sorted(set(int(fl["gt_index"]) for fl in real_flows if fl.get("gt_index") is not None)),
            "pred_attribute_fact_count_wo_numberattr": pred_fact_wo,
            "pred_attribute_fact_count_with_numberattr": pred_fact_wo,
            "matched_gt_attribute_count_wo_numberattr_sum_over_real_flows": matched_wo_sum,
            "matched_gt_attribute_count_with_numberattr_sum_over_real_flows": matched_with_sum,
            "pred_total_quantity": acc.get("pred_total_quantity", 0),
            "predicted_positive_quantity": acc.get("predicted_positive_quantity", 0),
            "stage2_predicted_positive_quantity": acc.get("stage2_predicted_positive_quantity", 0),
            "stage1_category_hallucination_predicted_positive_quantity": acc.get(
                "stage1_category_hallucination_predicted_positive_quantity", 0
            ),
            "stage1_category_hallucination_predicted_positive_source": acc.get(
                "stage1_category_hallucination_predicted_positive_source"
            ),
            "matched_tp_credit_quantity": acc.get("matched_tp_credit_quantity", acc.get("matched_credit_quantity", 0)),
            "matched_pp_credit_quantity": acc.get("matched_pp_credit_quantity", acc.get("matched_raw_quantity", 0)),
            "matched_credit_quantity": acc.get("matched_credit_quantity", 0),
            "matched_raw_quantity": acc.get("matched_raw_quantity", 0),
            "residual_hallucination_quantity": acc.get("residual_hallucination_quantity", 0),
            "quantity_state": acc.get("quantity_state"),
            "metric_eligible": acc.get("metric_eligible", init_status == "required"),
            "solver_pred_solution": acc.get("solver_pred_solution"),
        })
    return out


def build_per_gt_summary(
    *,
    required_gt_objects: List[Dict[str, Any]],
    optional_gt_objects: List[Dict[str, Any]],
    final_flows: List[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]] = None,
    gt_flow_accounting: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    incoming_by_gt: Dict[tuple, List[Dict[str, Any]]] = {}
    for fl in final_flows:
        if fl.get("flow_kind") != "real":
            continue
        gt_index = fl.get("gt_index")
        if gt_index is None:
            continue
        incoming_by_gt.setdefault(("required", int(gt_index)), []).append(fl)

    gt_solution_map = {
        int(x["gt_index"]): x
        for x in ((solver_output or {}).get("gt_solution") or [])
        if isinstance(x, dict) and x.get("gt_index") is not None
    }
    gt_accounting_map = {
        int(x["gt_index"]): x
        for x in (gt_flow_accounting or [])
        if isinstance(x, dict) and x.get("gt_index") is not None
    }

    out: List[Dict[str, Any]] = []
    for gt in required_gt_objects:
        gj = int(gt["gt_index"])
        incoming = incoming_by_gt.get(("required", gj), [])
        gt_solution = gt_solution_map.get(gj, {})
        gt_acc = gt_accounting_map.get(gj, {})
        gt_metric_quantity = float(gt_acc.get("gt_metric_quantity") or gt_solution.get("capacity_chosen") or 0.0)
        incoming_credit_quantity_sum = sum(float(fl.get("tp_credit_quantity") or fl.get("credit_quantity") or 0.0) for fl in incoming)
        incoming_raw_quantity_sum = sum(float(fl.get("assigned_quantity") or 0.0) for fl in incoming)
        recall_covered_quantity = float(gt_acc.get("recall_covered_quantity") or min(gt_metric_quantity, incoming_credit_quantity_sum))
        out.append({
            "gt_index": gj,
            "group": "required",
            "gt_metric_quantity": gt_metric_quantity,
            "gt_metric_quantity_debug": gt_acc if gt_acc else gt_solution,
            "is_hit_by_any_flow": incoming_credit_quantity_sum > 0,
            "incoming_flow_ids": [fl["flow_id"] for fl in incoming],
            "incoming_pred_indices": sorted(set(int(fl["pred_index"]) for fl in incoming)),
            "incoming_flow_count": len(incoming),
            "incoming_credit_quantity_sum": incoming_credit_quantity_sum,
            "incoming_raw_quantity_sum": incoming_raw_quantity_sum,
            "recall_covered_quantity": recall_covered_quantity,
            "has_candidate_pred": bool(gt_acc.get("has_candidate_pred", True)),
            "gt_metric_quantity_source": gt_acc.get("gt_metric_quantity_source"),
            "gt_owner_violation": False,
        })

    for gt in optional_gt_objects:
        gj = int(gt["gt_index"])
        out.append({
            "gt_index": gj,
            "group": "optional",
            "gt_metric_quantity": 0.0,
            "gt_metric_quantity_debug": None,
            "is_hit_by_any_flow": False,
            "incoming_flow_ids": [],
            "incoming_pred_indices": [],
            "incoming_flow_count": 0,
            "incoming_credit_quantity_sum": 0.0,
            "incoming_raw_quantity_sum": 0.0,
            "recall_covered_quantity": 0.0,
            "gt_owner_violation": False,
        })
    return out
