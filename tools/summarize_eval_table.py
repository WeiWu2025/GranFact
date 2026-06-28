#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize one or more Stage-3 evaluation results into compact Markdown/CSV/PNG
summary tables.

Each experiment points to a Stage-3 evaluation directory containing
``global_metrics.json``. Provide experiments from the CLI rather than editing
machine-specific paths into this release script.

Examples:

  python tools/summarize_eval_table.py \
    --experiment qwen3-vl-8b,aggressive,./model_outputs/qwen3-vl-8b/stage1_answers-.../stage2-ext_match/.../stage3_eval/... \
    --experiment internvl,neutral,/path/to/another/stage3_eval

  python tools/summarize_eval_table.py \
    --experiments-json ./experiments.json \
    --output-dir ./eval_summary_tables

``experiments.json`` may be either:

  [
    {"model": "your_model", "prompt": "aggressive", "eval_dir": "/path/to/stage3_eval"}
  ]

or:

  {"experiments": [...same list...]}

Outputs by default:
  eval_summary_tables/eval_summary_table.md
  eval_summary_tables/eval_summary_table.csv
  eval_summary_tables/eval_summary_table.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

OUTPUT_DIR = "./eval_summary_tables"
OUTPUT_STEM = "eval_summary_table"
DECIMALS = 4

# Full metric names in global_metrics.json["all_metrics"]. Fallback short names
# are from global_metrics.json["core_metrics"].
METRIC_SPECS = [
    {"column": "Prec", "all_key": "normal_micro_precision", "core_key": "precision"},
    {"column": "Recall", "all_key": "normal_micro_recall", "core_key": "recall"},
    {"column": "F1", "all_key": "normal_micro_f1", "core_key": "f1"},
    {
        "column": "Overall-Gran Prec",
        "all_key": "overall_granularity_weighted_micro_precision",
        "core_key": "overall_gran-precision",
    },
    {
        "column": "Overall-Gran Recall",
        "all_key": "overall_granularity_weighted_micro_recall",
        "core_key": "overall_gran-recall",
    },
    {
        "column": "Overall-Gran F1",
        "all_key": "overall_granularity_weighted_micro_f1",
        "core_key": "overall_gran-f1",
    },
    {
        "column": "Avg Matched Gran Quality",
        "all_key": "avg_matched_granularity_quality",
        "core_key": "avg_matched_gran-quality",
    },
]

BASE_COLUMNS = ["Model", "Prompt"]
COLUMNS = BASE_COLUMNS + [x["column"] for x in METRIC_SPECS] + ["Eval Samples"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def format_metric(x: Any, decimals: int = DECIMALS) -> str:
    v = safe_float(x)
    if v is None:
        return "NA"
    return f"{v:.{decimals}f}"


def metric_value(global_metrics: Dict[str, Any], all_key: str, core_key: str) -> Optional[float]:
    all_metrics = global_metrics.get("all_metrics")
    if isinstance(all_metrics, dict):
        v = safe_float(all_metrics.get(all_key))
        if v is not None:
            return v

    core_metrics = global_metrics.get("core_metrics")
    if isinstance(core_metrics, dict):
        v = safe_float(core_metrics.get(core_key))
        if v is not None:
            return v

    # In case the file is an unwrapped metrics dict.
    v = safe_float(global_metrics.get(all_key))
    if v is not None:
        return v
    return safe_float(global_metrics.get(core_key))


def eval_sample_count(global_metrics: Dict[str, Any]) -> str:
    """Return evaluated sample count from global_metrics.json when available."""
    candidates: List[Any] = []

    all_metrics = global_metrics.get("all_metrics")
    if isinstance(all_metrics, dict):
        candidates.append(all_metrics.get("num_samples"))

    core_metrics = global_metrics.get("core_metrics")
    if isinstance(core_metrics, dict):
        candidates.append(core_metrics.get("num_samples"))

    # Fallbacks for possible future/unwrapped variants.
    candidates.append(global_metrics.get("num_samples"))
    candidates.append(global_metrics.get("num_samples_evaluated"))

    for x in candidates:
        v = safe_float(x)
        if v is None:
            continue
        return str(int(v)) if abs(v - int(v)) < 1e-9 else str(v)
    return "NA"


def build_row(exp: Dict[str, str]) -> Dict[str, str]:
    model = str(exp.get("model") or "").strip() or "UNKNOWN_MODEL"
    prompt = str(exp.get("prompt") or "").strip() or "UNKNOWN_PROMPT"
    eval_dir = Path(str(exp.get("eval_dir") or "").strip())
    metrics_path = eval_dir / "global_metrics.json"

    row: Dict[str, str] = {"Model": model, "Prompt": prompt}

    if not metrics_path.exists():
        for spec in METRIC_SPECS:
            row[spec["column"]] = "NA"
        row["Eval Samples"] = "NA"
        row["_warning"] = f"Missing global_metrics.json: {metrics_path}"
        return row

    global_metrics = load_json(metrics_path)
    if not isinstance(global_metrics, dict):
        raise RuntimeError(f"global_metrics.json is not an object: {metrics_path}")

    for spec in METRIC_SPECS:
        v = metric_value(global_metrics, all_key=spec["all_key"], core_key=spec["core_key"])
        row[spec["column"]] = format_metric(v)
    row["Eval Samples"] = eval_sample_count(global_metrics)
    return row


def markdown_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", "<br>")


def write_markdown_table(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(markdown_escape(c) for c in COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row.get(c, "")) for c in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})


def wrap_cell_text(text: str, width: int) -> str:
    s = str(text)
    if len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=True))


def write_png_table(path: Path, rows: Sequence[Dict[str, str]], *, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to write the PNG table. Install matplotlib or pass --no-png."
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)

    display_columns = [
        "Model",
        "Prompt",
        "Prec",
        "Recall",
        "F1",
        "Overall-Gran\nPrec",
        "Overall-Gran\nRecall",
        "Overall-Gran\nF1",
        "Avg Matched\nGran Quality",
        "Eval\nSamples",
    ]
    cell_text: List[List[str]] = []
    for row in rows:
        cell_text.append([
            wrap_cell_text(row.get("Model", ""), 24),
            wrap_cell_text(row.get("Prompt", ""), 14),
            row.get("Prec", ""),
            row.get("Recall", ""),
            row.get("F1", ""),
            row.get("Overall-Gran Prec", ""),
            row.get("Overall-Gran Recall", ""),
            row.get("Overall-Gran F1", ""),
            row.get("Avg Matched Gran Quality", ""),
            row.get("Eval Samples", ""),
        ])

    n_rows = max(1, len(rows))
    fig_w = 15.5
    fig_h = max(2.4, 0.55 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    col_widths = [0.25, 0.11, 0.07, 0.07, 0.07, 0.105, 0.105, 0.105, 0.12, 0.065]
    table = ax.table(
        cellText=cell_text,
        colLabels=display_columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F7F7")
        else:
            cell.set_facecolor("#FFFFFF")
        if c in {0, 1} and r > 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_experiment_spec(spec: str) -> Dict[str, str]:
    """Parse 'model,prompt,eval_dir'. eval_dir may contain commas if quoted via JSON instead."""
    parts = [x.strip() for x in str(spec).split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--experiment must be 'model,prompt,eval_dir', e.g. "
            "qwen3-vl-8b,aggressive,./model_outputs/.../stage3_eval/..."
        )
    return {"model": parts[0], "prompt": parts[1], "eval_dir": parts[2]}


def load_experiments_json(path: Path) -> List[Dict[str, str]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        obj = obj.get("experiments")
    if not isinstance(obj, list):
        raise RuntimeError(f"experiments JSON must be a list or object with an 'experiments' list: {path}")

    out: List[Dict[str, str]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise RuntimeError(f"experiments[{i}] is not an object in {path}")
        out.append({
            "model": str(item.get("model") or "").strip(),
            "prompt": str(item.get("prompt") or "").strip(),
            "eval_dir": str(item.get("eval_dir") or "").strip(),
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize Stage-3 global_metrics.json files into Markdown/CSV/PNG tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/summarize_eval_table.py \
    --experiment qwen3-vl-8b,aggressive,./model_outputs/qwen3-vl-8b/.../stage3_eval/qwen3.5-27b-evaluation

  python tools/summarize_eval_table.py \
    --experiments-json ./experiments.json \
    --no-png
""",
    )
    p.add_argument(
        "--experiment",
        action="append",
        default=[],
        type=parse_experiment_spec,
        help="Experiment spec: model,prompt,eval_dir. Can be passed multiple times.",
    )
    p.add_argument(
        "--experiments-json",
        type=str,
        default=None,
        help="Optional JSON file with experiments list. Combined with --experiment entries.",
    )
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--output-stem", type=str, default=OUTPUT_STEM)
    p.add_argument("--title", type=str, default="Benchmark Evaluation Summary")
    p.add_argument("--no-png", action="store_true", help="Skip PNG generation; useful if matplotlib is unavailable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments: List[Dict[str, str]] = []
    if args.experiments_json:
        experiments.extend(load_experiments_json(Path(args.experiments_json)))
    experiments.extend(args.experiment)

    if not experiments:
        raise SystemExit(
            "No experiments provided. Use --experiment model,prompt,/path/to/stage3_eval "
            "or --experiments-json /path/to/experiments.json."
        )

    out_dir = Path(args.output_dir)
    rows = [build_row(exp) for exp in experiments]

    warnings = [row.get("_warning") for row in rows if row.get("_warning")]
    for warning in warnings:
        print(f"[warn] {warning}")

    md_path = out_dir / f"{args.output_stem}.md"
    csv_path = out_dir / f"{args.output_stem}.csv"
    png_path = out_dir / f"{args.output_stem}.png"

    write_markdown_table(md_path, rows)
    write_csv_table(csv_path, rows)
    if not args.no_png:
        write_png_table(png_path, rows, title=args.title)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")
    if not args.no_png:
        print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
