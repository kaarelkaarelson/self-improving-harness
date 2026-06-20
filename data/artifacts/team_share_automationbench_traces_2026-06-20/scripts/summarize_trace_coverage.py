#!/usr/bin/env python3
"""Summarize AutomationBench trace coverage for data collection planning."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                role = message.get("role") or "message"
                content = message.get("content") or ""
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    if prompt is None:
        return ""
    return str(prompt)


def task_info(record: dict[str, Any]) -> tuple[str, str, str, str]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    name = task.get("task") or task.get("task_id") or record.get("task")
    name = str(name or "unknown")
    category = name.split(".", 1)[0] if "." in name else "unknown"
    example_id = task.get("example_id") or record.get("example_id") or ""
    prompt = prompt_text(task.get("prompt") or record.get("prompt"))
    return category, name, str(example_id), prompt


def score(record: dict[str, Any]) -> tuple[float, bool]:
    raw_score = record.get("score") if isinstance(record.get("score"), dict) else {}
    partial = as_float(
        raw_score.get("partial_credit")
        if raw_score
        else record.get("partial_credit", record.get("reward")),
        0.0,
    )
    strict_raw = (
        raw_score.get("task_completed_correctly")
        if raw_score
        else record.get("task_completed_correctly")
    )
    correct_raw = raw_score.get("correct") if raw_score else record.get("correct")
    strict = bool(strict_raw) or bool(correct_raw) or partial >= 1.0
    return partial, strict


def model_name(record: dict[str, Any]) -> str:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    agent = record.get("agent") if isinstance(record.get("agent"), dict) else {}
    return str(record.get("model") or evaluation.get("model") or agent.get("name") or "unknown")


def short_prompt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def gap_reason(row: dict[str, Any], *, good_threshold: float, diversity_target: int) -> str:
    reasons: list[str] = []
    if row["good_trace_count"] == 0:
        reasons.append(f"no_trace_ge_{good_threshold:g}")
    if row["strict_trace_count"] == 0:
        reasons.append("no_strict_trace")
    if 0 < row["good_model_count"] < diversity_target:
        reasons.append(f"low_model_diversity_lt_{diversity_target}")
    if row["good_trace_count"] < diversity_target:
        reasons.append(f"low_trace_diversity_lt_{diversity_target}")
    return ";".join(reasons) if reasons else "covered"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gap_markdown(path: Path, task_rows: list[dict[str, Any]], category_rows: list[dict[str, Any]]) -> None:
    needs = [row for row in task_rows if row["gap_reason"] != "covered"]
    no_good = [row for row in task_rows if row["good_trace_count"] == 0]
    no_strict = [row for row in task_rows if row["strict_trace_count"] == 0]
    low_diversity = [
        row
        for row in task_rows
        if row["good_trace_count"] > 0 and "low_" in row["gap_reason"]
    ]

    lines = [
        "# AutomationBench Trace Coverage Gaps",
        "",
        "Use `task_coverage.csv` as the detailed handoff sheet. This file highlights the tasks most likely to benefit from new frontier/native rollouts.",
        "",
        "## Category Summary",
        "",
        "| Category | Tasks | Good tasks | Strict tasks | No good | No strict | Good traces | Strict traces |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_rows:
        lines.append(
            "| {category} | {task_count} | {tasks_with_good} | {tasks_with_strict} | "
            "{tasks_without_good} | {tasks_without_strict} | {good_trace_count} | {strict_trace_count} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            f"Tasks needing attention: `{len(needs)}`",
            f"Tasks with no good trace: `{len(no_good)}`",
            f"Tasks with no strict trace: `{len(no_strict)}`",
            "",
            "## Highest Priority",
            "",
            "| Category | Task | Best partial | Good traces | Strict traces | Gap reason | Prompt |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    priority = sorted(
        needs,
        key=lambda row: (
            row["good_trace_count"] > 0,
            row["strict_trace_count"] > 0,
            -row["best_partial_credit"],
            row["category"],
            row["task"],
        ),
    )[:100]
    for row in priority:
        prompt = str(row["prompt_short"]).replace("|", "\\|")
        lines.append(
            f"| {row['category']} | `{row['task']}` | {row['best_partial_credit']:.3f} | "
            f"{row['good_trace_count']} | {row['strict_trace_count']} | {row['gap_reason']} | {prompt} |"
        )

    if low_diversity:
        lines.extend(
            [
                "",
                "## Covered But Thin",
                "",
                "These tasks have at least one good trace, but fewer than the target number of good traces/models.",
                "",
                "| Category | Task | Best partial | Good traces | Good models | Best models |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(low_diversity, key=lambda item: (item["good_trace_count"], item["category"], item["task"]))[:100]:
            lines.append(
                f"| {row['category']} | `{row['task']}` | {row['best_partial_credit']:.3f} | "
                f"{row['good_trace_count']} | {row['good_model_count']} | {row['best_models']} |"
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shareable AutomationBench trace coverage reports")
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--good-threshold", type=float, default=0.5)
    parser.add_argument("--diversity-target", type=int, default=5)
    parser.add_argument("--prompt-limit", type=int, default=220)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input_jsonl]
    output_dir = Path(args.output_dir)
    by_task: dict[str, dict[str, Any]] = {}
    model_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "model": "",
            "trace_count": 0,
            "good_trace_count": 0,
            "strict_trace_count": 0,
            "partial_scores": [],
            "tasks_with_good": set(),
            "tasks_with_strict": set(),
        }
    )

    for record in iter_jsonl(input_paths):
        category, task, example_id, prompt = task_info(record)
        partial, strict = score(record)
        model = model_name(record)
        row = by_task.setdefault(
            task,
            {
                "category": category,
                "task": task,
                "example_id": example_id,
                "prompt": prompt,
                "total_trace_count": 0,
                "model_count": set(),
                "good_trace_count": 0,
                "good_model_count": set(),
                "strict_trace_count": 0,
                "strict_model_count": set(),
                "partial_scores": [],
                "best_by_model": {},
            },
        )
        row["total_trace_count"] += 1
        row["model_count"].add(model)
        row["partial_scores"].append(partial)
        if partial >= args.good_threshold:
            row["good_trace_count"] += 1
            row["good_model_count"].add(model)
        if strict:
            row["strict_trace_count"] += 1
            row["strict_model_count"].add(model)
        row["best_by_model"][model] = max(partial, row["best_by_model"].get(model, 0.0))

        model_row = model_rows[model]
        model_row["model"] = model
        model_row["trace_count"] += 1
        model_row["partial_scores"].append(partial)
        if partial >= args.good_threshold:
            model_row["good_trace_count"] += 1
            model_row["tasks_with_good"].add(task)
        if strict:
            model_row["strict_trace_count"] += 1
            model_row["tasks_with_strict"].add(task)

    task_rows: list[dict[str, Any]] = []
    for row in by_task.values():
        scores = row["partial_scores"]
        best_models = sorted(row["best_by_model"].items(), key=lambda item: (-item[1], item[0]))[:5]
        flat = {
            "category": row["category"],
            "task": row["task"],
            "example_id": row["example_id"],
            "prompt": row["prompt"],
            "prompt_short": short_prompt(row["prompt"], args.prompt_limit),
            "total_trace_count": row["total_trace_count"],
            "model_count": len(row["model_count"]),
            "best_partial_credit": max(scores) if scores else 0.0,
            "mean_partial_credit": statistics.fmean(scores) if scores else 0.0,
            "good_trace_count": row["good_trace_count"],
            "good_model_count": len(row["good_model_count"]),
            "strict_trace_count": row["strict_trace_count"],
            "strict_model_count": len(row["strict_model_count"]),
            "best_models": ";".join(f"{model}:{score:.3f}" for model, score in best_models),
        }
        flat["gap_reason"] = gap_reason(
            flat,
            good_threshold=args.good_threshold,
            diversity_target=args.diversity_target,
        )
        task_rows.append(flat)

    task_rows.sort(
        key=lambda row: (
            row["gap_reason"] == "covered",
            row["good_trace_count"],
            row["strict_trace_count"],
            -row["best_partial_credit"],
            row["category"],
            row["task"],
        )
    )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        by_category[row["category"]].append(row)
    category_rows: list[dict[str, Any]] = []
    for category, rows in sorted(by_category.items()):
        category_rows.append(
            {
                "category": category,
                "task_count": len(rows),
                "tasks_with_good": sum(row["good_trace_count"] > 0 for row in rows),
                "tasks_with_strict": sum(row["strict_trace_count"] > 0 for row in rows),
                "tasks_without_good": sum(row["good_trace_count"] == 0 for row in rows),
                "tasks_without_strict": sum(row["strict_trace_count"] == 0 for row in rows),
                "good_trace_count": sum(row["good_trace_count"] for row in rows),
                "strict_trace_count": sum(row["strict_trace_count"] for row in rows),
                "avg_best_partial_credit": statistics.fmean(row["best_partial_credit"] for row in rows),
            }
        )

    model_summary_rows: list[dict[str, Any]] = []
    for row in model_rows.values():
        scores = row["partial_scores"]
        model_summary_rows.append(
            {
                "model": row["model"],
                "trace_count": row["trace_count"],
                "good_trace_count": row["good_trace_count"],
                "strict_trace_count": row["strict_trace_count"],
                "tasks_with_good": len(row["tasks_with_good"]),
                "tasks_with_strict": len(row["tasks_with_strict"]),
                "mean_partial_credit": statistics.fmean(scores) if scores else 0.0,
            }
        )
    model_summary_rows.sort(key=lambda row: (-row["tasks_with_strict"], -row["tasks_with_good"], row["model"]))

    write_csv(
        output_dir / "task_coverage.csv",
        task_rows,
        [
            "category",
            "task",
            "example_id",
            "gap_reason",
            "best_partial_credit",
            "mean_partial_credit",
            "good_trace_count",
            "good_model_count",
            "strict_trace_count",
            "strict_model_count",
            "total_trace_count",
            "model_count",
            "best_models",
            "prompt_short",
            "prompt",
        ],
    )
    need_help_rows = [row for row in task_rows if row["gap_reason"] != "covered"]
    write_csv(
        output_dir / "tasks_need_help.csv",
        need_help_rows,
        [
            "category",
            "task",
            "example_id",
            "gap_reason",
            "best_partial_credit",
            "good_trace_count",
            "good_model_count",
            "strict_trace_count",
            "strict_model_count",
            "best_models",
            "prompt_short",
            "prompt",
        ],
    )
    write_csv(
        output_dir / "category_summary.csv",
        category_rows,
        [
            "category",
            "task_count",
            "tasks_with_good",
            "tasks_with_strict",
            "tasks_without_good",
            "tasks_without_strict",
            "good_trace_count",
            "strict_trace_count",
            "avg_best_partial_credit",
        ],
    )
    write_csv(
        output_dir / "model_summary.csv",
        model_summary_rows,
        [
            "model",
            "trace_count",
            "good_trace_count",
            "strict_trace_count",
            "tasks_with_good",
            "tasks_with_strict",
            "mean_partial_credit",
        ],
    )
    write_gap_markdown(output_dir / "gaps.md", task_rows, category_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_jsonl": [str(path) for path in input_paths],
                "good_threshold": args.good_threshold,
                "diversity_target": args.diversity_target,
                "tasks": len(task_rows),
                "tasks_needing_attention": sum(row["gap_reason"] != "covered" for row in task_rows),
                "tasks_without_good": sum(row["good_trace_count"] == 0 for row in task_rows),
                "tasks_without_strict": sum(row["strict_trace_count"] == 0 for row in task_rows),
            },
            indent=2,
        )
    )
    print(f"Wrote {output_dir}")
    print(f"Tasks: {len(task_rows)}")
    print(f"Need attention: {sum(row['gap_reason'] != 'covered' for row in task_rows)}")
    print(f"No good trace: {sum(row['good_trace_count'] == 0 for row in task_rows)}")
    print(f"No strict trace: {sum(row['strict_trace_count'] == 0 for row in task_rows)}")


if __name__ == "__main__":
    main()
