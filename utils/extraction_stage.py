#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from utils.common import parse_json_array, canonicalize_predicted_object
from utils.guided_json import extraction_objects_schema, generate_parse_with_guided_json_policy
from utils.prompts import build_extraction_messages

if TYPE_CHECKING:
    from utils.config import ExtractAndMatchConfig
    from utils.llm import LocalChatModel


def run_extraction_stage(
    *,
    response: str,
    type_: str,
    cfg: ExtractAndMatchConfig,
    extractor_model: LocalChatModel,
) -> Tuple[List[Dict[str, Any]], str, Any]:
    extraction_messages = build_extraction_messages(response, type_=type_)

    def _parse_extraction(raw_text: str) -> Tuple[List[Dict[str, Any]], str, Any]:
        parsed_array, arr_err = parse_json_array(raw_text)
        if not isinstance(parsed_array, list):
            raise ValueError(
                "[extraction] parse failed:\n"
                + json.dumps(
                    {
                        "error": arr_err or "extract parse failed",
                        "raw_text": raw_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        predicted_objects: List[Dict[str, Any]] = [
            canonicalize_predicted_object(x) for x in parsed_array if isinstance(x, dict)
        ]
        return predicted_objects, raw_text, arr_err

    (predicted_objects, extraction_raw, arr_err), guided_debug = generate_parse_with_guided_json_policy(
        cfg=cfg,
        model=extractor_model,
        messages=extraction_messages,
        schema=extraction_objects_schema(),
        schema_name="extraction_objects_v1",
        parse_fn=_parse_extraction,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    extraction_raw = guided_debug.get("fallback_raw_text") or guided_debug.get("first_raw_text") or extraction_raw
    return predicted_objects, extraction_raw, {
        "parse_error": arr_err,
        "guided_json": guided_debug,
    }
