#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import copy
from typing import Any, Dict, List, Optional

from utils.common import *

def _pred_discount(pred_obj: Dict[str, Any]) -> float:
    return 0.5 if str(pred_obj.get("number_type") or "") == "uncertain" else 1.0


def _flow_pred_discount(pred_obj: Dict[str, Any], gt_obj: Optional[Dict[str, Any]]) -> float:
    if str(pred_obj.get("number_type") or "") != "uncertain":
        return 1.0
    if gt_obj is not None and str(gt_obj.get("number_type") or "") == "uncertain":
        return 1.0
    return _pred_discount(pred_obj)


def _category_total_depth_for_flow_score(gt_obj: Dict[str, Any]) -> int:
    levels = gt_obj.get("category_levels")
    if isinstance(levels, list) and levels:
        return max(1, len(levels))
    if gt_obj.get("deepest_label"):
        return 1
    if gt_obj.get("category_display"):
        return 1
    return 1


def _supported_depth_for_flow_score(edge: Dict[str, Any], gt_obj: Dict[str, Any]) -> float:
    total_depth = _category_total_depth_for_flow_score(gt_obj)
    credit_depth = _to_float(edge.get("category_credit_depth"), None)
    if credit_depth is not None and credit_depth > 0:
        return max(1.0, min(float(credit_depth), float(total_depth)))
    sd = _safe_int_or_none(edge.get("anchor_depth"))
    if sd is not None and sd > 0:
        return float(max(1, min(int(sd), int(total_depth))))
    coord = edge.get("anchor_required_coord")
    if isinstance(coord, list) and coord:
        d = _safe_int_or_none(coord[0])
        if d is not None and d >= 0:
            return float(max(1, min(int(d) + 1, int(total_depth))))
    return 1.0


def _overall_granularity_flow_utility(
    *,
    edge: Dict[str, Any],
    score_item: Dict[str, Any],
    gt_obj: Dict[str, Any],
) -> float:
    total_depth = _category_total_depth_for_flow_score(gt_obj)
    supported_depth = _supported_depth_for_flow_score(edge, gt_obj)
    attr_score = _clamp01(float(score_item.get("edge_utility") or 0.0))
    return _clamp01((float(supported_depth) + float(attr_score)) / float(total_depth + 1))


def _edge_flow_utility(edge: Dict[str, Any]) -> float:
    return _clamp01(float(edge.get("edge_flow_utility", edge.get("edge_utility", 0.0)) or 0.0))


def build_solver_input(
    *,
    candidate_graph: Dict[str, Any],
    pair_attribute_scores: List[Dict[str, Any]],
    pred_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pred_to_candidate_gts = {
        int(k): sorted(set(int(x) for x in v))
        for k, v in (candidate_graph.get("pred_to_candidate_gts") or {}).items()
    }
    gt_to_candidate_preds = {
        int(k): sorted(set(int(x) for x in v))
        for k, v in (candidate_graph.get("gt_to_candidate_preds") or {}).items()
    }

    score_map: Dict[tuple, Dict[str, Any]] = {}
    for item in pair_attribute_scores:
        score_map[(int(item["pred_index"]), int(item["gt_index"]))] = item

    pred_supply_fixed: Dict[int, int] = {}
    for pi in candidate_graph.get("required_pred_indices") or []:
        po = pred_objects[int(pi)]
        eff = effective_number_value(str(po.get("number_type") or ""), po.get("number_value"))
        pred_supply_fixed[int(pi)] = max(1, int(eff)) if eff is not None else 1

    gt_capacity_fixed: Dict[int, int] = {}
    for gj in range(len(required_gt_objects)):
        gt_obj = required_gt_objects[gj]
        eff = effective_number_value(str(gt_obj.get("number_type") or ""), gt_obj.get("number_value"))
        gt_capacity_fixed[gj] = max(1, int(eff)) if eff is not None else 1

    pred_nodes: List[Dict[str, Any]] = []
    for pi in candidate_graph.get("required_pred_indices") or []:
        po = pred_objects[int(pi)]
        fixed = int(pred_supply_fixed[int(pi)])
        pred_nodes.append({
            "pred_index": int(pi),
            "number_type": str(po.get("number_type") or ""),
            "number_value": po.get("number_value"),
            "supply_fixed": fixed,
            "supply_lower_bound": fixed,
            "supply_upper_bound": fixed,
            "discount": _pred_discount(po),
        })

    gt_nodes: List[Dict[str, Any]] = []
    for gj, gt_obj in enumerate(required_gt_objects):
        fixed = int(gt_capacity_fixed[gj])
        gt_nodes.append({
            "gt_index": int(gj),
            "number_type": str(gt_obj.get("number_type") or ""),
            "number_value": gt_obj.get("number_value"),
            "capacity_fixed": fixed,
            "capacity_lower_bound": fixed,
            "capacity_upper_bound": fixed,
            "incoming_pred_indices": list(gt_to_candidate_preds.get(gj, [])),
        })

    edges: List[Dict[str, Any]] = []
    attr_debug_fields = [
        "correct_pred_fact_indices_wo_numberattr",
        "wrong_pred_fact_indices_wo_numberattr",
        "uncertain_pred_fact_indices_wo_numberattr",
        "recalled_gt_fact_indices_wo_numberattr",
        "unrecalled_gt_fact_indices_wo_numberattr",
        "correct_pred_attribute_count_wo_numberattr",
        "wrong_pred_attribute_count_wo_numberattr",
        "uncertain_pred_attribute_count_wo_numberattr",
        "recalled_gt_attribute_count_wo_numberattr",
        "unrecalled_gt_attribute_count_wo_numberattr",
        "attribute_precision",
        "attribute_recall",
        "attribute_f1",
        "reason",
    ]
    for edge in candidate_graph.get("edges") or []:
        pi = int(edge["pred_index"])
        gj = int(edge["gt_index"])
        score_item = score_map.get((pi, gj))
        if score_item is None:
            raise ValueError(f"missing pair attribute score for edge ({pi}, {gj})")
        edge_flow_utility = _overall_granularity_flow_utility(
            edge=edge,
            score_item=score_item,
            gt_obj=required_gt_objects[gj],
        )
        edge_record = {
            "edge_id": edge["edge_id"],
            "pred_index": pi,
            "gt_index": gj,
            "edge_upper_bound": int(max(0, min(int(pred_supply_fixed[pi]), int(gt_capacity_fixed[gj])))),
            "edge_utility": float(score_item.get("edge_utility") or 0.0),
            "edge_flow_utility": float(edge_flow_utility),
            "flow_score_policy": "overall_granularity",
            "has_contradiction": bool(score_item.get("has_contradiction")),
            "matched_gt_fact_indices_wo_numberattr": list(score_item.get("matched_gt_fact_indices_wo_numberattr") or []),
            "matched_gt_attribute_count_wo_numberattr": int(score_item.get("matched_gt_attribute_count_wo_numberattr") or 0),
            "pred_attribute_fact_count_wo_numberattr": int(score_item.get("pred_attribute_fact_count_wo_numberattr") or 0),
            "gt_attribute_fact_count_wo_numberattr": int(score_item.get("gt_attribute_fact_count_wo_numberattr") or 0),
            "anchor_required_coord": edge.get("anchor_required_coord"),
            "anchor_depth": edge.get("anchor_depth"),
            "category_credit_depth": edge.get("category_credit_depth"),
            "category_match_type": edge.get("category_match_type"),
            "compatible_required_coord": edge.get("compatible_required_coord"),
            "compatible_required_label": edge.get("compatible_required_label"),
            "anchor_label": edge.get("anchor_label"),
        }
        for field in attr_debug_fields:
            if field in score_item:
                edge_record[field] = score_item.get(field)
        edges.append(edge_record)

    return {
        "pred_nodes": pred_nodes,
        "gt_nodes": gt_nodes,
        "edges": edges,
        "pred_to_candidate_gts": {str(k): v for k, v in pred_to_candidate_gts.items()},
        "gt_to_candidate_preds": {str(k): v for k, v in gt_to_candidate_preds.items()},
    }


class _MCFEdge:
    __slots__ = ("to", "rev", "cap", "cost", "orig_cap", "meta")

    def __init__(self, to: int, rev: int, cap: int, cost: int, meta: Optional[Dict[str, Any]] = None):
        self.to = to
        self.rev = rev
        self.cap = int(cap)
        self.cost = int(cost)
        self.orig_cap = int(cap)
        self.meta = meta


def _mcf_add_edge(graph: List[List[_MCFEdge]], frm: int, to: int, cap: int, cost: int, meta: Optional[Dict[str, Any]] = None) -> _MCFEdge:
    fwd = _MCFEdge(to=to, rev=len(graph[to]), cap=cap, cost=cost, meta=meta)
    rev = _MCFEdge(to=frm, rev=len(graph[frm]), cap=0, cost=-cost, meta=None)
    graph[frm].append(fwd)
    graph[to].append(rev)
    return fwd


def _mcf_shortest_path(graph: List[List[_MCFEdge]], source: int, sink: int):
    n = len(graph)
    inf = 10 ** 18
    dist = [inf] * n
    prev_v = [-1] * n
    prev_e = [-1] * n
    in_queue = [False] * n
    q = deque([source])
    dist[source] = 0
    in_queue[source] = True

    while q:
        v = q.popleft()
        in_queue[v] = False
        for ei, e in enumerate(graph[v]):
            if e.cap <= 0:
                continue
            nd = dist[v] + e.cost
            if nd < dist[e.to]:
                dist[e.to] = nd
                prev_v[e.to] = v
                prev_e[e.to] = ei
                if not in_queue[e.to]:
                    q.append(e.to)
                    in_queue[e.to] = True

    return dist, prev_v, prev_e


def solve_capacitated_b_matching(solver_input: Dict[str, Any]) -> Dict[str, Any]:
    pred_nodes = solver_input.get("pred_nodes") or []
    gt_nodes = solver_input.get("gt_nodes") or []
    edge_items = [x for x in (solver_input.get("edges") or []) if int(x.get("edge_upper_bound") or 0) > 0]

    pred_id_to_node = {int(x["pred_index"]): idx for idx, x in enumerate(pred_nodes)}
    gt_id_to_node = {int(x["gt_index"]): idx for idx, x in enumerate(gt_nodes)}

    source = 0
    pred_offset = 1
    gt_offset = pred_offset + len(pred_nodes)
    sink = gt_offset + len(gt_nodes)
    graph: List[List[_MCFEdge]] = [[] for _ in range(sink + 1)]
    edge_refs: Dict[tuple, _MCFEdge] = {}

    total_possible_flow = sum(int(x.get("supply_upper_bound") or 0) for x in pred_nodes)
    utility_scale = 1000
    big_m = (utility_scale + 1) * (total_possible_flow + 1)

    for idx, pred_node in enumerate(pred_nodes):
        cap = int(pred_node.get("supply_upper_bound") or 0)
        if cap > 0:
            _mcf_add_edge(graph, source, pred_offset + idx, cap, 0, meta={"kind": "source_pred", "pred_index": int(pred_node["pred_index"])})

    for idx, gt_node in enumerate(gt_nodes):
        cap = int(gt_node.get("capacity_upper_bound") or 0)
        if cap > 0:
            _mcf_add_edge(graph, gt_offset + idx, sink, cap, 0, meta={"kind": "gt_sink", "gt_index": int(gt_node["gt_index"])})

    for edge in edge_items:
        pi = int(edge["pred_index"])
        gj = int(edge["gt_index"])
        pred_node_idx = pred_id_to_node.get(pi)
        gt_node_idx = gt_id_to_node.get(gj)
        if pred_node_idx is None or gt_node_idx is None:
            continue
        utility_int = int(round(_edge_flow_utility(edge) * utility_scale))
        ref = _mcf_add_edge(
            graph,
            pred_offset + pred_node_idx,
            gt_offset + gt_node_idx,
            int(edge.get("edge_upper_bound") or 0),
            -(big_m + utility_int),
            meta={
                "kind": "pred_gt",
                "edge_id": edge.get("edge_id"),
                "pred_index": pi,
                "gt_index": gj,
                "edge_utility": float(edge.get("edge_utility") or 0.0),
                "edge_flow_utility": _edge_flow_utility(edge),
            },
        )
        edge_refs[(pi, gj)] = ref

    total_flow = 0
    total_cost = 0
    while True:
        dist, prev_v, prev_e = _mcf_shortest_path(graph, source, sink)
        if prev_v[sink] < 0 or dist[sink] >= 0:
            break

        addf = 10 ** 18
        v = sink
        while v != source:
            pv = prev_v[v]
            pe = prev_e[v]
            addf = min(addf, graph[pv][pe].cap)
            v = pv

        v = sink
        while v != source:
            pv = prev_v[v]
            pe = prev_e[v]
            e = graph[pv][pe]
            e.cap -= addf
            graph[v][e.rev].cap += addf
            v = pv

        total_flow += int(addf)
        total_cost += int(addf) * int(dist[sink])

    assigned_quantities: List[Dict[str, Any]] = []
    pred_assigned_quantity: Dict[int, int] = {int(x["pred_index"]): 0 for x in pred_nodes}
    gt_assigned_quantity: Dict[int, int] = {int(x["gt_index"]): 0 for x in gt_nodes}
    activated_edges: List[Dict[str, Any]] = []
    edge_item_map = {(int(x["pred_index"]), int(x["gt_index"])): x for x in edge_items}

    for (pi, gj), ref in edge_refs.items():
        q = int(ref.orig_cap - ref.cap)
        if q <= 0:
            continue
        pred_assigned_quantity[pi] = pred_assigned_quantity.get(pi, 0) + q
        gt_assigned_quantity[gj] = gt_assigned_quantity.get(gj, 0) + q
        edge_item = edge_item_map.get((pi, gj), {})
        assigned_quantities.append({
            "pred_index": pi,
            "gt_index": gj,
            "assigned_quantity": q,
            "edge_id": edge_item.get("edge_id"),
            "edge_utility": edge_item.get("edge_utility"),
            "edge_flow_utility": edge_item.get("edge_flow_utility"),
        })
        activated_edges.append({
            "pred_index": pi,
            "gt_index": gj,
            "edge_id": edge_item.get("edge_id"),
            "assigned_quantity": q,
            "edge_utility": edge_item.get("edge_utility"),
            "edge_flow_utility": edge_item.get("edge_flow_utility"),
        })

    pred_solution: List[Dict[str, Any]] = []
    for pred_node in pred_nodes:
        pi = int(pred_node["pred_index"])
        assigned = int(pred_assigned_quantity.get(pi, 0))
        supply_fixed = pred_node.get("supply_fixed")
        if supply_fixed is not None:
            supply_chosen = int(supply_fixed)
        else:
            supply_chosen = max(int(pred_node.get("supply_lower_bound") or 1), assigned)
        pred_solution.append({
            "pred_index": pi,
            "assigned_quantity": assigned,
            "supply_fixed": supply_fixed,
            "supply_upper_bound": int(pred_node.get("supply_upper_bound") or 0),
            "supply_chosen": int(supply_chosen),
            "quantity_hallucination_residual": max(0, int(supply_chosen) - assigned),
            "discount": float(pred_node.get("discount") or 1.0),
            "number_type": pred_node.get("number_type"),
            "number_value": pred_node.get("number_value"),
        })

    gt_solution: List[Dict[str, Any]] = []
    for gt_node in gt_nodes:
        gj = int(gt_node["gt_index"])
        assigned = int(gt_assigned_quantity.get(gj, 0))
        capacity_fixed = gt_node.get("capacity_fixed")
        if capacity_fixed is not None:
            capacity_chosen = int(capacity_fixed)
        else:
            capacity_chosen = max(int(gt_node.get("capacity_lower_bound") or 1), assigned)
        gt_solution.append({
            "gt_index": gj,
            "assigned_quantity": assigned,
            "capacity_fixed": capacity_fixed,
            "capacity_upper_bound": int(gt_node.get("capacity_upper_bound") or 0),
            "capacity_chosen": int(capacity_chosen),
            "number_type": gt_node.get("number_type"),
            "number_value": gt_node.get("number_value"),
        })

    return {
        "objective": {
            "total_matched_quantity": int(total_flow),
            "total_cost": int(total_cost),
        },
        "activated_edges": activated_edges,
        "assigned_quantities": assigned_quantities,
        "pred_solution": pred_solution,
        "gt_solution": gt_solution,
    }


def _rank_key_for_fp_pred_to_gt(edge: Dict[str, Any]) -> tuple:
    return (
        -_edge_flow_utility(edge),
        -int(edge.get("matched_gt_attribute_count_wo_numberattr") or 0),
        int(edge.get("gt_index") or 0),
    )


def _rank_key_for_fn_gt_to_pred(edge: Dict[str, Any]) -> tuple:
    return (
        -_edge_flow_utility(edge),
        -int(edge.get("matched_gt_attribute_count_wo_numberattr") or 0),
        int(edge.get("pred_index") or 0),
    )


def repair_uncertain_quantities(
    *,
    solver_input: Dict[str, Any],
    solver_output: Dict[str, Any],
    predicted_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    repaired_output = copy.deepcopy(solver_output)
    edge_items = [x for x in (solver_input.get("edges") or []) if int(x.get("edge_upper_bound") or 0) > 0]
    edge_map = {(int(x["pred_index"]), int(x["gt_index"])): x for x in edge_items}

    assigned_map: Dict[tuple, int] = {(pi, gj): 0 for (pi, gj) in edge_map.keys()}
    for x in repaired_output.get("assigned_quantities") or []:
        pi = int(x["pred_index"])
        gj = int(x["gt_index"])
        if (pi, gj) in assigned_map:
            assigned_map[(pi, gj)] = int(x.get("assigned_quantity") or 0)

    pred_solution_map = {int(x["pred_index"]): x for x in (repaired_output.get("pred_solution") or [])}
    gt_solution_map = {int(x["gt_index"]): x for x in (repaired_output.get("gt_solution") or [])}

    pred_is_uncertain = {
        int(x["pred_index"]): str(x.get("number_type") or "") == "uncertain"
        for x in (repaired_output.get("pred_solution") or [])
    }
    gt_is_uncertain = {
        int(x["gt_index"]): str(x.get("number_type") or "") == "uncertain"
        for x in (repaired_output.get("gt_solution") or [])
    }

    pred_to_edges: Dict[int, List[Dict[str, Any]]] = {}
    gt_to_edges: Dict[int, List[Dict[str, Any]]] = {}
    for e in edge_items:
        pi = int(e["pred_index"])
        gj = int(e["gt_index"])
        pred_to_edges.setdefault(pi, []).append(e)
        gt_to_edges.setdefault(gj, []).append(e)

    pred_assigned: Dict[int, int] = {pi: 0 for pi in pred_solution_map.keys()}
    gt_assigned: Dict[int, int] = {gj: 0 for gj in gt_solution_map.keys()}
    for (pi, gj), q in assigned_map.items():
        if q <= 0:
            continue
        pred_assigned[pi] = pred_assigned.get(pi, 0) + q
        gt_assigned[gj] = gt_assigned.get(gj, 0) + q

    pred_supply_chosen: Dict[int, int] = {
        pi: int(sol.get("supply_chosen") or 0)
        for pi, sol in pred_solution_map.items()
    }
    gt_capacity_chosen: Dict[int, int] = {
        gj: int(sol.get("capacity_chosen") or 0)
        for gj, sol in gt_solution_map.items()
    }

    repair_events: List[Dict[str, Any]] = []

    # Pass-1: repair FP residual by expanding uncertain GT capacities.
    for pi in sorted(pred_solution_map.keys()):
        residual = int(pred_supply_chosen.get(pi, 0)) - int(pred_assigned.get(pi, 0))
        if residual <= 0:
            continue
        candidates = [
            e for e in (pred_to_edges.get(pi) or [])
            if gt_is_uncertain.get(int(e.get("gt_index") or -1), False)
        ]
        if not candidates:
            continue
        best = sorted(candidates, key=_rank_key_for_fp_pred_to_gt)[0]
        gj = int(best["gt_index"])
        delta = int(residual)
        if delta <= 0:
            continue

        assigned_map[(pi, gj)] = int(assigned_map.get((pi, gj), 0)) + delta
        pred_assigned[pi] = int(pred_assigned.get(pi, 0)) + delta
        gt_assigned[gj] = int(gt_assigned.get(gj, 0)) + delta
        gt_capacity_chosen[gj] = int(gt_capacity_chosen.get(gj, 0)) + delta
        repair_events.append({
            "pass": "fp_residual_absorption",
            "pred_index": pi,
            "gt_index": gj,
            "delta": delta,
            "reason": "expand_uncertain_gt_to_absorb_pred_residual",
            "edge_utility": float(best.get("edge_utility") or 0.0),
            "edge_flow_utility": _edge_flow_utility(best),
            "matched_gt_attribute_count_wo_numberattr": int(best.get("matched_gt_attribute_count_wo_numberattr") or 0),
        })

    # Pass-2: repair FN residual by expanding uncertain pred supplies.
    for gj in sorted(gt_solution_map.keys()):
        residual = int(gt_capacity_chosen.get(gj, 0)) - int(gt_assigned.get(gj, 0))
        if residual <= 0:
            continue
        candidates = [
            e for e in (gt_to_edges.get(gj) or [])
            if pred_is_uncertain.get(int(e.get("pred_index") or -1), False)
        ]
        if not candidates:
            continue
        best = sorted(candidates, key=_rank_key_for_fn_gt_to_pred)[0]
        pi = int(best["pred_index"])
        delta = int(residual)
        if delta <= 0:
            continue

        assigned_map[(pi, gj)] = int(assigned_map.get((pi, gj), 0)) + delta
        pred_assigned[pi] = int(pred_assigned.get(pi, 0)) + delta
        gt_assigned[gj] = int(gt_assigned.get(gj, 0)) + delta
        pred_supply_chosen[pi] = int(pred_supply_chosen.get(pi, 0)) + delta
        repair_events.append({
            "pass": "fn_residual_coverage",
            "pred_index": pi,
            "gt_index": gj,
            "delta": delta,
            "reason": "expand_uncertain_pred_to_cover_gt_residual",
            "edge_utility": float(best.get("edge_utility") or 0.0),
            "edge_flow_utility": _edge_flow_utility(best),
            "matched_gt_attribute_count_wo_numberattr": int(best.get("matched_gt_attribute_count_wo_numberattr") or 0),
        })

    assigned_quantities: List[Dict[str, Any]] = []
    activated_edges: List[Dict[str, Any]] = []
    for (pi, gj), q in sorted(assigned_map.items()):
        if int(q) <= 0:
            continue
        edge_item = edge_map.get((pi, gj), {})
        assigned_quantities.append({
            "pred_index": int(pi),
            "gt_index": int(gj),
            "assigned_quantity": int(q),
            "edge_id": edge_item.get("edge_id"),
            "edge_utility": edge_item.get("edge_utility"),
            "edge_flow_utility": edge_item.get("edge_flow_utility"),
        })
        activated_edges.append({
            "pred_index": int(pi),
            "gt_index": int(gj),
            "edge_id": edge_item.get("edge_id"),
            "assigned_quantity": int(q),
            "edge_utility": edge_item.get("edge_utility"),
            "edge_flow_utility": edge_item.get("edge_flow_utility"),
        })

    pred_solution: List[Dict[str, Any]] = []
    for pi in sorted(pred_solution_map.keys()):
        old = pred_solution_map[pi]
        assigned = int(pred_assigned.get(pi, 0))
        supply_chosen = int(pred_supply_chosen.get(pi, 0))
        supply_upper_bound = max(int(old.get("supply_upper_bound") or 0), supply_chosen)
        pred_solution.append({
            **old,
            "assigned_quantity": assigned,
            "supply_chosen": supply_chosen,
            "supply_upper_bound": supply_upper_bound,
            "quantity_hallucination_residual": max(0, supply_chosen - assigned),
        })

    gt_solution: List[Dict[str, Any]] = []
    for gj in sorted(gt_solution_map.keys()):
        old = gt_solution_map[gj]
        assigned = int(gt_assigned.get(gj, 0))
        capacity_chosen = int(gt_capacity_chosen.get(gj, 0))
        capacity_upper_bound = max(int(old.get("capacity_upper_bound") or 0), capacity_chosen)
        gt_solution.append({
            **old,
            "assigned_quantity": assigned,
            "capacity_chosen": capacity_chosen,
            "capacity_upper_bound": capacity_upper_bound,
        })

    repaired_output["assigned_quantities"] = assigned_quantities
    repaired_output["activated_edges"] = activated_edges
    repaired_output["pred_solution"] = pred_solution
    repaired_output["gt_solution"] = gt_solution
    repaired_output["objective"] = {
        "total_matched_quantity": int(sum(int(x.get("assigned_quantity") or 0) for x in assigned_quantities)),
        "total_cost": int((solver_output.get("objective") or {}).get("total_cost") or 0),
    }
    repaired_output["repair"] = {
        "policy": "augment_only_minimal_increment_no_resolve",
        "events": repair_events,
        "pred_residual_ledger": {
            str(pi): max(0, int(pred_supply_chosen.get(pi, 0)) - int(pred_assigned.get(pi, 0)))
            for pi in sorted(pred_solution_map.keys())
        },
        "gt_residual_ledger": {
            str(gj): max(0, int(gt_capacity_chosen.get(gj, 0)) - int(gt_assigned.get(gj, 0)))
            for gj in sorted(gt_solution_map.keys())
        },
    }
    return repaired_output


def validate_solver_output(
    *,
    solver_input: Dict[str, Any],
    solver_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    edge_upper = {
        (int(x["pred_index"]), int(x["gt_index"])): int(x.get("edge_upper_bound") or 0)
        for x in (solver_input.get("edges") or [])
    }
    pred_meta = {int(x["pred_index"]): x for x in (solver_input.get("pred_nodes") or [])}
    gt_meta = {int(x["gt_index"]): x for x in (solver_input.get("gt_nodes") or [])}
    pred_assigned: Dict[int, int] = {}
    gt_assigned: Dict[int, int] = {}

    for item in solver_output.get("assigned_quantities") or []:
        pi = int(item["pred_index"])
        gj = int(item["gt_index"])
        q = int(item.get("assigned_quantity") or 0)
        if (pi, gj) not in edge_upper:
            violations.append({
                "type": "illegal_edge_flow",
                "pred_index": pi,
                "gt_index": gj,
                "assigned_quantity": q,
            })
            continue
        if q < 0 or q > edge_upper[(pi, gj)]:
            violations.append({
                "type": "edge_capacity_violation",
                "pred_index": pi,
                "gt_index": gj,
                "assigned_quantity": q,
                "edge_upper_bound": edge_upper[(pi, gj)],
            })
        pred_assigned[pi] = pred_assigned.get(pi, 0) + q
        gt_assigned[gj] = gt_assigned.get(gj, 0) + q

    for pi, meta in pred_meta.items():
        assigned = pred_assigned.get(pi, 0)
        if assigned > int(meta.get("supply_upper_bound") or 0):
            violations.append({
                "type": "pred_supply_upper_violation",
                "pred_index": pi,
                "assigned_quantity": assigned,
                "supply_upper_bound": int(meta.get("supply_upper_bound") or 0),
            })
        fixed = meta.get("supply_fixed")
        if fixed is not None and assigned > int(fixed):
            violations.append({
                "type": "pred_fixed_supply_violation",
                "pred_index": pi,
                "assigned_quantity": assigned,
                "supply_fixed": int(fixed),
            })

    for gj, meta in gt_meta.items():
        assigned = gt_assigned.get(gj, 0)
        if assigned > int(meta.get("capacity_upper_bound") or 0):
            violations.append({
                "type": "gt_capacity_upper_violation",
                "gt_index": gj,
                "assigned_quantity": assigned,
                "capacity_upper_bound": int(meta.get("capacity_upper_bound") or 0),
            })
        fixed = meta.get("capacity_fixed")
        if fixed is not None and assigned > int(fixed):
            violations.append({
                "type": "gt_fixed_capacity_violation",
                "gt_index": gj,
                "assigned_quantity": assigned,
                "capacity_fixed": int(fixed),
            })

    return violations


def materialize_solver_results(
    *,
    solver_input: Dict[str, Any],
    solver_output: Dict[str, Any],
    predicted_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
    initial_match_results: List[Dict[str, Any]],
):
    initial_map = {int(x["pred_index"]): x for x in initial_match_results}
    edge_meta = {
        (int(x["pred_index"]), int(x["gt_index"])): x
        for x in (solver_input.get("edges") or [])
    }
    pred_solution_map = {int(x["pred_index"]): x for x in (solver_output.get("pred_solution") or [])}
    gt_solution_map = {int(x["gt_index"]): x for x in (solver_output.get("gt_solution") or [])}
    assigned_map = {
        (int(x["pred_index"]), int(x["gt_index"])): int(x.get("assigned_quantity") or 0)
        for x in (solver_output.get("assigned_quantities") or [])
    }

    resolved_pairs: List[Dict[str, Any]] = []
    final_flows: List[Dict[str, Any]] = []
    pred_flow_accounting: List[Dict[str, Any]] = []
    gt_flow_accounting: List[Dict[str, Any]] = []
    quantity_hallucination_items: List[Dict[str, Any]] = []
    hall_counter_by_pred: Dict[int, int] = {}
    attr_debug_fields = [
        "correct_pred_fact_indices_wo_numberattr",
        "wrong_pred_fact_indices_wo_numberattr",
        "uncertain_pred_fact_indices_wo_numberattr",
        "recalled_gt_fact_indices_wo_numberattr",
        "unrecalled_gt_fact_indices_wo_numberattr",
        "correct_pred_attribute_count_wo_numberattr",
        "wrong_pred_attribute_count_wo_numberattr",
        "uncertain_pred_attribute_count_wo_numberattr",
        "recalled_gt_attribute_count_wo_numberattr",
        "unrecalled_gt_attribute_count_wo_numberattr",
        "attribute_precision",
        "attribute_recall",
        "attribute_f1",
        "reason",
    ]

    for (pi, gj), assigned_quantity in sorted(assigned_map.items()):
        if assigned_quantity <= 0:
            continue
        meta = edge_meta[(pi, gj)]
        pred_solution = pred_solution_map.get(pi, {})
        pred_discount = _flow_pred_discount(predicted_objects[pi], required_gt_objects[gj])
        flow_id = f"pred{pi}_gt{gj}_0"
        tp_credit_quantity = float(pred_discount) * float(assigned_quantity)
        pp_credit_quantity = float(assigned_quantity)
        attr_debug_payload = {field: meta.get(field) for field in attr_debug_fields if field in meta}
        resolved_pair = {
            "pair_id": meta.get("edge_id") or flow_id,
            "pred_index": pi,
            "gt_index": gj,
            "resolved_by": "solver",
            "group_id": None,
            "assigned_quantity": int(assigned_quantity),
            "pred_category_text": predicted_objects[pi].get(PRED_CATEGORY_FIELD),
            "gt_category_text": required_gt_objects[gj].get("deepest_label"),
            "edge_utility": float(meta.get("edge_utility") or 0.0),
            "edge_flow_utility": _edge_flow_utility(meta),
            "matched_gt_fact_indices_wo_numberattr": list(meta.get("matched_gt_fact_indices_wo_numberattr") or []),
            "matched_gt_attribute_count_wo_numberattr": int(meta.get("matched_gt_attribute_count_wo_numberattr") or 0),
            "pred_attribute_fact_count_wo_numberattr": int(meta.get("pred_attribute_fact_count_wo_numberattr") or 0),
            "gt_attribute_fact_count_wo_numberattr": int(meta.get("gt_attribute_fact_count_wo_numberattr") or 0),
            "has_contradiction": bool(meta.get("has_contradiction")),
            "category_credit_depth": meta.get("category_credit_depth"),
            "category_match_type": meta.get("category_match_type"),
            "compatible_required_coord": meta.get("compatible_required_coord"),
            "compatible_required_label": meta.get("compatible_required_label"),
        }
        resolved_pair.update(attr_debug_payload)
        resolved_pairs.append(resolved_pair)
        attr_match = {
            "matched_gt_fact_indices_wo_numberattr": list(meta.get("matched_gt_fact_indices_wo_numberattr") or []),
            "matched_gt_attribute_count_wo_numberattr": int(meta.get("matched_gt_attribute_count_wo_numberattr") or 0),
            "pred_attribute_fact_count_wo_numberattr": int(meta.get("pred_attribute_fact_count_wo_numberattr") or 0),
            "gt_attribute_fact_count_wo_numberattr": int(meta.get("gt_attribute_fact_count_wo_numberattr") or 0),
            "matched_gt_attribute_count_with_numberattr": int(meta.get("matched_gt_attribute_count_wo_numberattr") or 0),
            "pred_attribute_fact_count_with_numberattr": int(meta.get("pred_attribute_fact_count_wo_numberattr") or 0),
            "gt_attribute_fact_count_with_numberattr": int(meta.get("gt_attribute_fact_count_wo_numberattr") or 0),
            "number_attr_tp": 0,
            "number_attr_pred_positive": 0,
            "number_attr_gt_positive": 0,
        }
        attr_match.update(attr_debug_payload)
        final_flows.append({
            "flow_id": flow_id,
            "flow_kind": "real",
            "pred_index": pi,
            "gt_index": gj,
            "resolved_by": "solver",
            "group_id": None,
            "assigned_quantity": int(assigned_quantity),
            # TP uses pair-level pred discount for uncertain-pred/numeric-GT; PP keeps raw assigned quantity.
            "tp_credit_quantity": float(tp_credit_quantity),
            "pp_credit_quantity": float(pp_credit_quantity),
            "credit_quantity": float(tp_credit_quantity),
            "pred_total_quantity": float(pred_solution.get("supply_chosen") or 0),
            "gt_metric_quantity": float(gt_solution_map.get(gj, {}).get("capacity_chosen") or 0),
            "pred_category_text": predicted_objects[pi].get(PRED_CATEGORY_FIELD),
            "gt_category_text": required_gt_objects[gj].get("deepest_label"),
            "attr_match": attr_match,
            "edge_utility": float(meta.get("edge_utility") or 0.0),
            "edge_flow_utility": _edge_flow_utility(meta),
            "category_credit_depth": meta.get("category_credit_depth"),
            "category_match_type": meta.get("category_match_type"),
            "compatible_required_coord": meta.get("compatible_required_coord"),
            "compatible_required_label": meta.get("compatible_required_label"),
            "pred_discount": float(pred_discount),
        })

    for pi, pred_obj in enumerate(predicted_objects):
        init_status = str(initial_map.get(pi, {}).get("status") or "none")
        flows = [fl for fl in final_flows if int(fl["pred_index"]) == pi and fl.get("flow_kind") == "real"]
        pred_solution = pred_solution_map.get(pi)
        metric_eligible = init_status == "required"

        if init_status == "optional":
            pred_total_quantity = 0.0
            matched_tp_credit_quantity = 0.0
            matched_pp_credit_quantity = 0.0
            matched_raw_quantity = 0.0
            residual_quantity = 0.0
            quantity_state = "optional_neutral"
        elif init_status == "none":
            pred_total_quantity = 0.0
            matched_tp_credit_quantity = 0.0
            matched_pp_credit_quantity = 0.0
            matched_raw_quantity = 0.0
            residual_quantity = 0.0
            quantity_state = "category_hallucination"
        else:
            pred_total_quantity = float((pred_solution or {}).get("supply_chosen") or 0.0)
            matched_tp_credit_quantity = sum(float(fl.get("tp_credit_quantity") or 0.0) for fl in flows)
            matched_pp_credit_quantity = sum(float(fl.get("pp_credit_quantity") or 0.0) for fl in flows)
            matched_raw_quantity = sum(float(fl.get("assigned_quantity") or 0.0) for fl in flows)
            residual_quantity = float((pred_solution or {}).get("quantity_hallucination_residual") or 0.0)
            if matched_raw_quantity <= 0 and residual_quantity > 0:
                quantity_state = "none"
            elif residual_quantity > 0:
                quantity_state = "under"
            else:
                quantity_state = "full"

        hallucination_flow_ids: List[str] = []
        if metric_eligible and residual_quantity > 0:
            seq = hall_counter_by_pred.get(pi, 0)
            hall_counter_by_pred[pi] = seq + 1
            hall_flow_id = f"pred{pi}_hall_{seq}"
            hallucination_flow_ids.append(hall_flow_id)
            quantity_hallucination_items.append({
                "pred_index": pi,
                "pred_category_text": pred_obj.get(PRED_CATEGORY_FIELD),
                "residual_quantity": float(residual_quantity),
                "number_type": str(pred_obj.get("number_type") or ""),
                "number_value": pred_obj.get("number_value"),
            })
            final_flows.append({
                "flow_id": hall_flow_id,
                "flow_kind": "hallucination",
                "hallucination_type": "quantity",
                "pred_index": pi,
                "gt_index": None,
                "resolved_by": "solver_residual",
                "group_id": None,
                "assigned_quantity": float(residual_quantity),
                "tp_credit_quantity": float(residual_quantity),
                "pp_credit_quantity": float(residual_quantity),
                "credit_quantity": float(residual_quantity),
                "pred_total_quantity": float(pred_total_quantity),
                "pred_category_text": pred_obj.get(PRED_CATEGORY_FIELD),
                "gt_category_text": None,
                "hallucination_reason": "solver_residual_quantity",
                "attr_match": {
                    "matched_gt_fact_indices_wo_numberattr": [],
                    "matched_gt_attribute_count_wo_numberattr": 0,
                    "pred_attribute_fact_count_wo_numberattr": 0,
                    "gt_attribute_fact_count_wo_numberattr": 0,
                    "matched_gt_attribute_count_with_numberattr": 0,
                    "pred_attribute_fact_count_with_numberattr": 0,
                    "gt_attribute_fact_count_with_numberattr": 0,
                    "number_attr_tp": 0,
                    "number_attr_pred_positive": 0,
                    "number_attr_gt_positive": 0,
                },
                "pred_discount": float((pred_solution or {}).get("discount") or _pred_discount(pred_obj)),
            })

        pred_flow_accounting.append({
            "pred_index": pi,
            "metric_eligible": metric_eligible,
            "initial_status": init_status,
            "pred_number_type": str(pred_obj.get("number_type") or ""),
            "pred_number_value": pred_obj.get("number_value"),
            "pred_total_quantity": float(pred_total_quantity),
            "matched_gt_indices": sorted(set(int(fl["gt_index"]) for fl in flows if fl.get("gt_index") is not None)),
            "real_flow_ids": [fl["flow_id"] for fl in flows],
            "matched_tp_credit_quantity": float(matched_tp_credit_quantity),
            "matched_pp_credit_quantity": float(matched_pp_credit_quantity),
            "matched_credit_quantity": float(matched_tp_credit_quantity),
            "matched_raw_quantity": float(matched_raw_quantity),
            "residual_hallucination_quantity": float(residual_quantity),
            "hallucination_flow_ids": hallucination_flow_ids,
            "quantity_state": quantity_state,
            "solver_pred_solution": pred_solution,
        })

    # Stage-1 category hallucination preds should still contribute PP.
    stage2_pred_indices = [
        int(item["pred_index"]) for item in pred_flow_accounting
        if str(item.get("initial_status") or "") == "required"
    ]
    stage2_pp_total = sum(
        float((pred_solution_map.get(pi, {}) or {}).get("supply_chosen") or 0.0)
        for pi in stage2_pred_indices
    )

    stage1_numeric_pred_items: List[Dict[str, Any]] = []
    for pi, pred_obj in enumerate(predicted_objects):
        init_status = str(initial_map.get(pi, {}).get("status") or "none")
        if init_status != "none":
            continue
        number_type = str(pred_obj.get("number_type") or "")
        if number_type == "uncertain":
            continue
        eff = effective_number_value(number_type, pred_obj.get("number_value"))
        stage1_numeric_pred_items.append({
            "pred_index": pi,
            "quantity": float(max(1, int(eff)) if eff is not None else 1),
        })

    stage1_numeric_pred_sum = sum(float(x.get("quantity") or 0.0) for x in stage1_numeric_pred_items)
    stage1_uncertain_pred_fill_den = len(stage2_pred_indices) + len(stage1_numeric_pred_items)
    stage1_uncertain_pred_fill = (
        (stage2_pp_total + stage1_numeric_pred_sum) / float(stage1_uncertain_pred_fill_den)
        if stage1_uncertain_pred_fill_den > 0
        else 1.0
    )

    for item in pred_flow_accounting:
        pi = int(item["pred_index"])
        init_status = str(item.get("initial_status") or "none")
        pred_obj = predicted_objects[pi]
        number_type = str(pred_obj.get("number_type") or "")
        stage1_pp = 0.0
        stage1_pp_source = "not_stage1_category_hallucination"
        if init_status == "none":
            if number_type == "uncertain":
                stage1_pp = float(stage1_uncertain_pred_fill)
                stage1_pp_source = "stage1_uncertain_pred_imputed_mean"
            else:
                eff = effective_number_value(number_type, pred_obj.get("number_value"))
                stage1_pp = float(max(1, int(eff)) if eff is not None else 1)
                stage1_pp_source = "stage1_numeric_pred_direct"

        if init_status == "required":
            stage2_pp = float(item.get("pred_total_quantity") or 0.0)
        else:
            stage2_pp = 0.0
        item["stage2_predicted_positive_quantity"] = float(stage2_pp)
        item["stage1_category_hallucination_predicted_positive_quantity"] = float(stage1_pp)
        item["stage1_category_hallucination_predicted_positive_source"] = stage1_pp_source
        item["predicted_positive_quantity"] = float(stage2_pp + stage1_pp)

    incoming_real_by_gt: Dict[int, List[Dict[str, Any]]] = {}
    for fl in final_flows:
        if fl.get("flow_kind") != "real":
            continue
        gj = fl.get("gt_index")
        if gj is None:
            continue
        incoming_real_by_gt.setdefault(int(gj), []).append(fl)

    gt_to_candidate_preds = {
        int(k): [int(x) for x in v]
        for k, v in (solver_input.get("gt_to_candidate_preds") or {}).items()
    }
    stage1_no_candidate_gt_indices = [
        int(gj) for gj in range(len(required_gt_objects))
        if len(gt_to_candidate_preds.get(int(gj), [])) == 0
    ]

    gt_known_items: List[float] = []
    for gj in range(len(required_gt_objects)):
        gt_obj = required_gt_objects[gj]
        if gj in stage1_no_candidate_gt_indices:
            number_type = str(gt_obj.get("number_type") or "")
            if number_type == "uncertain":
                continue
            eff = effective_number_value(number_type, gt_obj.get("number_value"))
            gt_known_items.append(float(max(1, int(eff)) if eff is not None else 1))
        else:
            gt_known_items.append(float((gt_solution_map.get(gj, {}) or {}).get("capacity_chosen") or 0.0))

    gt_uncertain_fill = (sum(gt_known_items) / float(len(gt_known_items))) if gt_known_items else 1.0

    for gj, gt_obj in enumerate(required_gt_objects):
        incoming = incoming_real_by_gt.get(int(gj), [])
        incoming_tp = sum(float(fl.get("tp_credit_quantity") or 0.0) for fl in incoming)
        incoming_raw = sum(float(fl.get("assigned_quantity") or 0.0) for fl in incoming)
        has_candidate_pred = len(gt_to_candidate_preds.get(int(gj), [])) > 0

        if has_candidate_pred:
            gt_metric_quantity = float((gt_solution_map.get(gj, {}) or {}).get("capacity_chosen") or 0.0)
            gt_metric_source = "solver_repair_capacity_chosen"
        else:
            number_type = str(gt_obj.get("number_type") or "")
            if number_type == "uncertain":
                gt_metric_quantity = float(gt_uncertain_fill)
                gt_metric_source = "stage1_no_candidate_uncertain_gt_imputed_mean"
            else:
                eff = effective_number_value(number_type, gt_obj.get("number_value"))
                gt_metric_quantity = float(max(1, int(eff)) if eff is not None else 1)
                gt_metric_source = "stage1_no_candidate_numeric_gt_direct"

        gt_flow_accounting.append({
            "gt_index": int(gj),
            "has_candidate_pred": bool(has_candidate_pred),
            "incoming_tp_credit_quantity": float(incoming_tp),
            "incoming_raw_quantity": float(incoming_raw),
            "gt_metric_quantity": float(gt_metric_quantity),
            "gt_metric_quantity_source": gt_metric_source,
            "recall_covered_quantity": float(min(gt_metric_quantity, incoming_tp)),
            "solver_gt_solution": gt_solution_map.get(gj),
        })

    final_flows.sort(key=lambda x: (int(x["pred_index"]), 0 if x.get("flow_kind") == "real" else 1, str(x["flow_id"])))
    quantity_hallucination_accounting = {
        "items": quantity_hallucination_items,
        "total_residual_quantity": float(sum(float(x.get("residual_quantity") or 0.0) for x in quantity_hallucination_items)),
        "stage1_uncertain_pred_imputation": {
            "fill_value": float(stage1_uncertain_pred_fill),
            "stage2_pp_total": float(stage2_pp_total),
            "stage1_numeric_pred_sum": float(stage1_numeric_pred_sum),
            "denominator": int(stage1_uncertain_pred_fill_den),
        },
        "stage1_uncertain_gt_imputation": {
            "fill_value": float(gt_uncertain_fill),
            "known_items_count": int(len(gt_known_items)),
        },
    }
    return resolved_pairs, final_flows, pred_flow_accounting, gt_flow_accounting, quantity_hallucination_accounting
