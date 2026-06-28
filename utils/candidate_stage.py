#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from utils.candidate_stage_pairwise import run_candidate_selection_pairwise

if TYPE_CHECKING:
    from utils.config import ExtractAndMatchConfig
    from utils.llm import LocalChatModel


DEFAULT_CANDIDATE_STRATEGY = "pairwise"


def run_candidate_selection_stage(
    *,
    cfg: ExtractAndMatchConfig,
    judge_model: LocalChatModel,
    predicted_objects: List[Dict[str, Any]],
    required_view: Dict[str, Any],
    optional_view: Dict[str, Any],
    required_gt_objects: List[Dict[str, Any]],
    optional_gt_objects: List[Dict[str, Any]],
    type_: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], str]:
    strategy_key = str(getattr(cfg, "candidate_strategy", None) or DEFAULT_CANDIDATE_STRATEGY).strip().lower()
    if strategy_key != "pairwise":
        raise ValueError(
            f"unknown candidate selection strategy: {strategy_key!r}. "
            "This release only supports 'pairwise'."
        )

    return run_candidate_selection_pairwise(
        cfg=cfg,
        judge_model=judge_model,
        predicted_objects=predicted_objects,
        required_view=required_view,
        optional_view=optional_view,
        required_gt_objects=required_gt_objects,
        optional_gt_objects=optional_gt_objects,
        type_=type_,
    )
