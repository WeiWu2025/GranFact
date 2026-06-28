#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from utils.matching import run_pair_attribute_scoring
from utils.solver import (
    build_solver_input,
    solve_capacitated_b_matching,
    repair_uncertain_quantities,
    validate_solver_output,
    materialize_solver_results,
)

if TYPE_CHECKING:
    from utils.config import ExtractAndMatchConfig
    from utils.llm import LocalChatModel


def run_final_matching_stage(
    *,
    cfg: ExtractAndMatchConfig,
    judge_model: LocalChatModel,
    predicted_objects: List[Dict[str, Any]],
    required_gt_objects: List[Dict[str, Any]],
    initial_match_results: List[Dict[str, Any]],
    candidate_graph: Dict[str, Any],
) -> Dict[str, Any]:
    pair_attribute_scoring_payloads, pair_attribute_scores, pair_attribute_score_debug = run_pair_attribute_scoring(
        cfg=cfg,
        judge_model=judge_model,
        predicted_objects=predicted_objects,
        required_gt_objects=required_gt_objects,
        initial_match_results=initial_match_results,
        candidate_graph=candidate_graph,
    )

    solver_input = build_solver_input(
        candidate_graph=candidate_graph,
        pair_attribute_scores=pair_attribute_scores,
        pred_objects=predicted_objects,
        required_gt_objects=required_gt_objects,
    )
    baseline_solver_output = solve_capacitated_b_matching(solver_input)
    constraint_violations = validate_solver_output(
        solver_input=solver_input,
        solver_output=baseline_solver_output,
    )
    if constraint_violations:
        raise RuntimeError(
            "[solver] constraint violations detected:\n"
            + json.dumps(constraint_violations, ensure_ascii=False, indent=2)
        )

    solver_output = repair_uncertain_quantities(
        solver_input=solver_input,
        solver_output=baseline_solver_output,
        predicted_objects=predicted_objects,
        required_gt_objects=required_gt_objects,
    )

    resolved_pairs, final_flows, pred_flow_accounting, gt_flow_accounting, quantity_hallucination_accounting = materialize_solver_results(
        solver_input=solver_input,
        solver_output=solver_output,
        predicted_objects=predicted_objects,
        required_gt_objects=required_gt_objects,
        initial_match_results=initial_match_results,
    )

    return {
        "pair_attribute_scoring_payloads": pair_attribute_scoring_payloads,
        "pair_attribute_scores": pair_attribute_scores,
        "pair_attribute_score_debug": pair_attribute_score_debug,
        "solver_input": solver_input,
        "solver_output": solver_output,
        "constraint_violations": constraint_violations,
        "resolved_pairs": resolved_pairs,
        "final_flows": final_flows,
        "pred_flow_accounting": pred_flow_accounting,
        "gt_flow_accounting": gt_flow_accounting,
        "quantity_hallucination_accounting": quantity_hallucination_accounting,
    }
