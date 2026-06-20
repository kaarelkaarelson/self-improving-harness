#!/usr/bin/env python3
"""Run AutomationBench evaluations through either supported harness path."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def default_verifiers_output_path(model: str, toolset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    return ROOT / "runs" / "verifiers_eval" / f"{stamp}_{safe_model}_{toolset}.json"


def add_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None, dry_run: bool) -> int:
    print("Command:", flush=True)
    print(" ".join(command), flush=True)
    print(f"CWD: {cwd}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode


def run_verifiers(args: argparse.Namespace, extra_args: list[str]) -> int:
    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    if not benchmarks_dir.exists():
        raise SystemExit(f"Benchmarks directory does not exist: {benchmarks_dir}")

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else default_verifiers_output_path(args.model, args.toolset)
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "run",
        "auto-bench",
        "--model",
        args.model,
        "--domains",
        args.domains,
        "--num-examples",
        str(args.num_examples),
        "--max-steps",
        str(args.max_steps),
        "--toolset",
        args.toolset,
        "--api",
        args.api,
        "--max-concurrent",
        str(args.max_concurrent),
        "--export-json",
        str(output_json),
    ]
    add_optional(command, "--base-url", args.base_url)
    if args.api_key:
        command += ["--api-key", args.api_key]
    else:
        command += ["--api-key-var", args.api_key_var]
    add_optional(command, "--tasks", args.tasks)
    add_optional(command, "--reasoning-effort", args.reasoning_effort)
    add_optional(command, "--search-top-k", args.search_top_k)
    command += extra_args

    env = os.environ.copy()
    if not args.api_key and is_local_base_url(args.base_url) and not env.get(args.api_key_var):
        env[args.api_key_var] = "dummy"

    print(f"Output: {output_json}", flush=True)
    return run_command(command, cwd=benchmarks_dir, env=env, dry_run=args.dry_run)


def run_native(args: argparse.Namespace, extra_args: list[str]) -> int:
    harness = Path(args.harness).resolve()
    if not harness.exists():
        raise SystemExit(f"Native harness does not exist: {harness}")

    command = [
        sys.executable,
        str(harness),
        "--agent",
        args.agent,
        "--domains",
        args.domains,
        "--start-index",
        str(args.start_index),
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    add_optional(command, "--tasks", args.tasks)
    add_optional(command, "--num-tasks", args.num_tasks)
    add_optional(command, "--run-dir", args.run_dir)
    add_optional(command, "--output", args.output)
    add_optional(command, "--agent-config", args.agent_config)
    if args.embed_trace:
        command.append("--embed-trace")
    if args.stop_on_error:
        command.append("--stop-on-error")
    if args.list_tasks:
        command.append("--list-tasks")
    command += extra_args

    return run_command(command, cwd=ROOT, env=None, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AutomationBench evals")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    verifiers = subparsers.add_parser(
        "verifiers",
        aliases=["built-in", "builtin", "automationbench"],
        help="Run AutomationBench's built-in OpenAI-compatible runner.",
    )
    verifiers.add_argument("--model", required=True, help="Model name passed to the API")
    verifiers.add_argument("--base-url", help="OpenAI-compatible base URL")
    verifiers.add_argument("--api-key", help="Explicit API key passed to auto-bench")
    verifiers.add_argument("--api-key-var", default="OPENAI_API_KEY")
    verifiers.add_argument(
        "--api",
        default="auto",
        choices=["auto", "anthropic", "chat_completions", "responses"],
    )
    verifiers.add_argument("--domains", default="all")
    verifiers.add_argument("--tasks", help="Comma-separated AutomationBench task ids")
    verifiers.add_argument("--num-examples", type=int, default=-1)
    verifiers.add_argument("--max-steps", type=int, default=50)
    verifiers.add_argument("--toolset", default="api", choices=["api", "zapier", "limited_zapier"])
    verifiers.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    verifiers.add_argument("--max-concurrent", type=int, default=16)
    verifiers.add_argument("--search-top-k", type=int)
    verifiers.add_argument("--output-json", help="Where to export AutomationBench results JSON")
    verifiers.add_argument("--benchmarks-dir", default=str(ROOT / "benchmarks"))
    verifiers.add_argument("--dry-run", action="store_true")
    verifiers.set_defaults(func=run_verifiers)

    native = subparsers.add_parser(
        "native",
        help="Run Codex, Claude Code, Devin, or noop through the MCP bridge.",
    )
    native.add_argument("--agent", choices=["codex", "claude", "devin", "noop"], required=True)
    native.add_argument("--domains", default="finance")
    native.add_argument("--tasks")
    native.add_argument("--num-tasks", type=int)
    native.add_argument("--start-index", type=int, default=0)
    native.add_argument("--run-dir")
    native.add_argument("--output")
    native.add_argument("--agent-config")
    native.add_argument("--timeout-sec", type=int, default=1800)
    native.add_argument("--embed-trace", action="store_true")
    native.add_argument("--stop-on-error", action="store_true")
    native.add_argument("--list-tasks", action="store_true")
    native.add_argument("--harness", default=str(ROOT / "harness" / "native_eval.py"))
    native.add_argument("--dry-run", action="store_true")
    native.set_defaults(func=run_native)

    return parser


def main() -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    raise SystemExit(args.func(args, extra_args))


if __name__ == "__main__":
    main()
