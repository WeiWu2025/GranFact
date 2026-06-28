#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")


def guided_json_enabled_for_first_attempt(cfg: Any) -> bool:
    return str(getattr(cfg, "guided_json_mode", "fallback") or "fallback").strip().lower() == "on"


def guided_json_enabled_for_fallback(cfg: Any) -> bool:
    return str(getattr(cfg, "guided_json_mode", "fallback") or "fallback").strip().lower() == "fallback"


def extraction_objects_schema() -> Dict[str, Any]:
    """Schema for extraction outputs.

    Attribute names are intentionally open-ended via additionalProperties. The
    value schema follows the prompt contract: every attribute value should be a
    list of strings, including number=["1"] or number=["uncertain"]. Extra
    top-level fields are allowed so canonicalization can ignore/preserve them
    without making the constrained decoder overfit a closed object schema.
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "finest_category": {"type": "string"},
                "attributes": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "required": ["finest_category", "attributes"],
            "additionalProperties": True,
        },
    }


def candidate_pairwise_level_match_schema() -> Dict[str, Any]:
    """Schema for pairwise candidate level matching results."""
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["is_a", "cannot_refer_to", "may_refer_to"],
                "description": (
                    "One of three decisions:\n"
                    "  - is_a: Prediction is-a (same type, subtype, or encompasses) the target\n"
                    "  - cannot_refer_to: Prediction definitely cannot refer to this target (wrong domain, brand, type, etc.)\n"
                    "  - may_refer_to: Prediction might refer to target, but not certain (too generic, or uncertain)"
                )
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this decision was made."
            }
        },
        "required": ["decision", "reason"],
        "additionalProperties": True,
    }



def pair_attribute_scoring_schema() -> Dict[str, Any]:
    candidate_result = {
        "type": "object",
        "properties": {
            "correct_pred_fact_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "wrong_pred_fact_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "recalled_gt_fact_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "reason": {"type": "string"},

            # Legacy fields accepted for backward compatibility with cached or
            # unconstrained model outputs. The parser maps matched_gt_* to
            # recalled_gt_* when the new field is absent.
            "matched_gt_fact_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "has_contradiction": {"type": "boolean"},
            "edge_utility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["correct_pred_fact_indices", "wrong_pred_fact_indices", "recalled_gt_fact_indices"],
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "object",
                "additionalProperties": candidate_result,
            }
        },
        "required": ["candidates"],
        "additionalProperties": True,
    }


def generate_parse_with_guided_json_policy(
    *,
    cfg: Any,
    model: Any,
    messages: Any,
    schema: Dict[str, Any],
    schema_name: str,
    parse_fn: Callable[[str], T],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[T, Dict[str, Any]]:
    """Generate and parse with off/fallback/on guided JSON policy.

    parse_fn should raise on parse/schema errors and return the parsed result on
    success. The returned debug records both attempts without changing existing
    parser-specific debug payloads.
    """
    mode = str(getattr(cfg, "guided_json_mode", "fallback") or "fallback").strip().lower()
    if mode not in {"off", "fallback", "on"}:
        mode = "fallback"

    first_guided = mode == "on"
    first_raw = model.generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        guided_json=(schema if first_guided else None),
    )
    first_generation_info = getattr(model, "last_generation_info", None)
    try:
        parsed = parse_fn(first_raw)
        return parsed, {
            "mode": mode,
            "schema_name": schema_name,
            "enabled_first_attempt": bool(first_guided),
            "fallback_triggered": False,
            "fallback_success": None,
            "first_raw_text": first_raw,
            "first_error": None,
            "first_generation_info": first_generation_info,
        }
    except Exception as first_error:
        if mode != "fallback":
            raise

        fallback_raw = model.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            guided_json=schema,
        )
        fallback_generation_info = getattr(model, "last_generation_info", None)
        try:
            parsed = parse_fn(fallback_raw)
            return parsed, {
                "mode": mode,
                "schema_name": schema_name,
                "enabled_first_attempt": False,
                "fallback_triggered": True,
                "fallback_success": True,
                "first_raw_text": first_raw,
                "first_error": repr(first_error),
                "fallback_raw_text": fallback_raw,
                "fallback_error": None,
                "first_generation_info": first_generation_info,
                "fallback_generation_info": fallback_generation_info,
            }
        except Exception as fallback_error:
            raise RuntimeError(
                f"[guided-json:{schema_name}] normal generation failed and guided fallback failed:\n"
                f"normal_error={first_error!r}\n"
                f"guided_error={fallback_error!r}\n"
                f"normal_raw={first_raw}\n"
                f"guided_raw={fallback_raw}"
            ) from fallback_error
