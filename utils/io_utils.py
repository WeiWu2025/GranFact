#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.common import iter_jsonl, dedup_keep_last

def _path_key_variants(p: Any) -> List[str]:
    if p is None:
        return []
    s = str(p).strip()
    if not s:
        return []

    out: List[str] = []

    def add(x: str) -> None:
        x = x.strip()
        if not x:
            return
        x = x.replace("\\", "/")
        x = re.sub(r"/+", "/", x)
        x = x.lstrip("./")
        if x and x not in out:
            out.append(x)

    add(os.path.normpath(s))
    add(s)

    slash = s.replace("\\", "/")
    # Support both absolute and relative paths anchored under benchmark dataset
    # roots such as:
    #   /.../testset/ready/electronics/image/1.jpg
    #   /.../testset_0515_todo/ready/electronics/image/1.jpg
    # Priority files can then use the dataset-relative tail:
    #   ready/electronics/image/1.jpg
    # Intentionally do not special-case domain directories (e.g. ready/) so that
    # only explicit testset* anchors define dataset-relative paths.
    m = re.search(r"(^|/)testset[^/]*/(.+)$", slash)
    if m:
        add(m.group(2))

    return out


def reorder_records_by_priority(
    records: List[Dict[str, Any]],
    priority_image_paths: List[str],
    only_priority: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not priority_image_paths:
        return records, {"enabled": False, "matched_count": 0}

    priority_rank: Dict[str, int] = {}
    priority_original_by_rank: Dict[int, str] = {}
    for rank, p in enumerate(priority_image_paths):
        priority_original_by_rank[rank] = str(p)
        for k in _path_key_variants(p):
            if k not in priority_rank:
                priority_rank[k] = rank

    ranked_items: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    matched_count = 0
    matched_ranks = set()
    matched_examples: List[Dict[str, Any]] = []

    for orig_idx, rec in enumerate(records):
        rec_keys = _path_key_variants(rec.get("image_path"))
        ranks = [priority_rank[k] for k in rec_keys if k in priority_rank]
        if ranks:
            best_rank = min(ranks)
            matched_ranks.add(best_rank)
            matched_count += 1
            if len(matched_examples) < 10:
                matched_examples.append({
                    "priority_rank": int(best_rank),
                    "priority_path": priority_original_by_rank.get(best_rank),
                    "sample_id": rec.get("sample_id"),
                    "image_path": rec.get("image_path"),
                    "matched_record_keys": rec_keys[:8],
                })
            sort_key = (0, best_rank, orig_idx)
        else:
            if only_priority:
                continue
            sort_key = (1, 10**9, orig_idx)
        ranked_items.append((sort_key, rec))

    ranked_items.sort(key=lambda x: x[0])
    out = [x[1] for x in ranked_items]
    unmatched_ranks = [
        rank for rank in range(len(priority_image_paths))
        if rank not in matched_ranks
    ]
    return out, {
        "enabled": True,
        "configured_count": len(priority_image_paths),
        "matched_count": matched_count,
        "unmatched_priority_count": len(unmatched_ranks),
        "unmatched_priority_examples": [
            {
                "priority_rank": int(rank),
                "priority_path": priority_original_by_rank.get(rank),
                "priority_keys": _path_key_variants(priority_original_by_rank.get(rank)),
            }
            for rank in unmatched_ranks[:10]
        ],
        "matched_examples": matched_examples,
        "only_priority": bool(only_priority),
    }


def resolve_stage1_paths(run_root: str) -> Dict[str, str]:
    stage1_out = os.path.join(run_root, "stage1_outputs")
    return {
        "stage1_out": stage1_out,
        "results_jsonl": os.path.join(stage1_out, "results.jsonl"),
        "meta_json": os.path.join(stage1_out, "meta.json"),
        "errors_jsonl": os.path.join(stage1_out, "errors.jsonl"),
    }


def extract_and_match_output_dir(run_root: str, run_id: str) -> str:
    return os.path.join(run_root, "stage2-ext_match", run_id)


def load_stage1_records(results_jsonl: str) -> List[Dict[str, Any]]:
    recs = list(iter_jsonl(results_jsonl))
    recs = dedup_keep_last(recs, key="sample_id")
    return recs


def _relative_to_testset(path: Any) -> Optional[str]:
    if path is None:
        return None
    s = str(path).strip()
    if not s:
        return None
    slash = s.replace("\\", "/")
    m = re.search(r"(^|/)testset[^/]*/(.+)$", slash)
    if not m:
        return None
    rel = m.group(2).strip("/")
    return rel or None


def _relocate_path_to_dataset_root(path: Any, dataset_root_override: Optional[str]) -> Optional[str]:
    if not dataset_root_override:
        return None
    rel = _relative_to_testset(path)
    if not rel:
        return None
    return os.path.normpath(os.path.join(str(dataset_root_override), rel))


def relocate_stage1_records(
    records: List[Dict[str, Any]],
    dataset_root_override: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Relocate Stage-1 records generated on another machine.

    Stage-1 stores absolute source_json/image_path. If those paths contain a
    stable 'testset*' anchor, preserve the path after that anchor and prepend
    the current-machine dataset_root_override.
    """
    if not dataset_root_override:
        return records, {"enabled": False}

    out: List[Dict[str, Any]] = []
    source_json_relocated = 0
    image_path_relocated = 0
    source_json_unresolved = 0
    image_path_unresolved = 0

    for rec in records:
        nr = copy.deepcopy(rec)

        new_source_json = _relocate_path_to_dataset_root(nr.get("source_json"), dataset_root_override)
        if new_source_json:
            if "source_json_original" not in nr:
                nr["source_json_original"] = nr.get("source_json")
            nr["source_json"] = new_source_json
            source_json_relocated += 1
        else:
            source_json_unresolved += 1

        new_image_path = _relocate_path_to_dataset_root(nr.get("image_path"), dataset_root_override)
        if new_image_path:
            if "image_path_original" not in nr:
                nr["image_path_original"] = nr.get("image_path")
            nr["image_path"] = new_image_path
            image_path_relocated += 1
        else:
            image_path_unresolved += 1

        out.append(nr)

    return out, {
        "enabled": True,
        "dataset_root_override": os.path.abspath(str(dataset_root_override)),
        "total": len(records),
        "source_json_relocated": source_json_relocated,
        "source_json_unresolved": source_json_unresolved,
        "image_path_relocated": image_path_relocated,
        "image_path_unresolved": image_path_unresolved,
    }
