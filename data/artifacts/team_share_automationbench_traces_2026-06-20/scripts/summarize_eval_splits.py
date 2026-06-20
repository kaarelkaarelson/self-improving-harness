#!/usr/bin/env python3
"""Summarize an AutomationBench eval JSON overall and by task split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_tasks_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    tasks: set[str] = set()
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.add(line)
    return tasks


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(tasks)
    passed = sum(1 for task in tasks if task.get("passed"))
    score_sum = sum(float(task.get("score") or 0.0) for task in tasks)
    input_tokens = sum(int(task.get("input_tokens") or 0) for task in tasks)
    output_tokens = sum(int(task.get("output_tokens") or 0) for task in tasks)
    zero_output = sum(1 for task in tasks if int(task.get("output_tokens") or 0) == 0)
    return {
        "task_count": count,
        "avg_score": score_sum / count if count else 0.0,
        "pass_rate": passed / count if count else 0.0,
        "passed_count": passed,
        "failed_count": count - passed,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "tasks_with_zero_output": zero_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AutomationBench eval JSON by task split")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--train-tasks-file")
    parser.add_argument("--heldout-tasks-file")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.eval_json).read_text())
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"Eval JSON has no tasks list: {args.eval_json}")

    train_names = read_tasks_file(args.train_tasks_file)
    heldout_names = read_tasks_file(args.heldout_tasks_file)
    summaries: dict[str, Any] = {"all": summarize(tasks)}

    if train_names is not None:
        train_tasks = [task for task in tasks if task.get("name") in train_names]
        summaries["train_tasks"] = summarize(train_tasks)
        summaries["train_tasks"]["requested_count"] = len(train_names)
        summaries["train_tasks"]["missing_count"] = len(train_names) - len(train_tasks)

    if heldout_names is not None:
        heldout_tasks = [task for task in tasks if task.get("name") in heldout_names]
        summaries["heldout_tasks"] = summarize(heldout_tasks)
        summaries["heldout_tasks"]["requested_count"] = len(heldout_names)
        summaries["heldout_tasks"]["missing_count"] = len(heldout_names) - len(heldout_tasks)

    output = {
        "eval_json": str(Path(args.eval_json)),
        "train_tasks_file": args.train_tasks_file,
        "heldout_tasks_file": args.heldout_tasks_file,
        "summaries": summaries,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
