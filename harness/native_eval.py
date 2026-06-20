#!/usr/bin/env python3
"""Native-agent AutomationBench harness.

Loads AutomationBench tasks locally, exposes the task world through a per-task
stdio MCP server, runs a native agent headlessly, then grades the final world
with AutomationBench's deterministic rubric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from automationbench.domains import DEFAULT_DOMAINS, DOMAIN_ALIASES, get_combined_dataset
from automationbench.rubric import partial_credit, task_completed_correctly
from automationbench.runner import strip_none_values
from automationbench.schema.world import WorldState


AGENT_TASK_PREAMBLE = """You are running an AutomationBench task in a local simulator.

Use the automationbench MCP tools to inspect and mutate the simulated business APIs:
- api_search discovers endpoint URLs and schemas.
- api_fetch calls those endpoints and changes the simulated world.
- base64_encode encodes text when an API requires base64url fields.

Do not ask clarifying questions. Complete the workflow as far as possible using the provided task messages. Do not call real external services."""


@dataclass(frozen=True)
class TaskPaths:
    task_dir: Path
    prompt_path: Path
    initial_state_path: Path
    final_state_path: Path
    tool_log_path: Path
    stdout_path: Path
    stderr_path: Path
    last_message_path: Path
    mcp_config_path: Path
    agent_export_path: Path


@dataclass
class AgentRun:
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None
    command: list[str]

    @property
    def success(self) -> bool:
        return self.returncode == 0 and self.error is None


def _slug(value: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (slug or "task")[:max_len]


def _load_jsonish(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    return value


def normalize_task(row: dict[str, Any]) -> dict[str, Any]:
    info = _load_jsonish(row.get("info", {}), "info") or {}
    info = strip_none_values(info)
    if "assertions" in info:
        info["assertions"] = [strip_none_values(assertion) for assertion in info["assertions"]]

    return {
        **row,
        "info": info,
        "prompt": row.get("prompt", ""),
    }


def format_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        body = prompt
    elif isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict):
                parts.append(str(message))
                continue
            role = str(message.get("role", "message")).upper()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content)
            parts.append(f"{role} MESSAGE:\n{content}")
        body = "\n\n".join(parts)
    else:
        body = json.dumps(prompt)

    return f"{AGENT_TASK_PREAMBLE}\n\n{body}".strip()


def read_json(path: Path) -> Any | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_task_paths(run_dir: Path, index: int, task_id: str) -> TaskPaths:
    task_dir = run_dir / f"{index:04d}_{_slug(task_id)}"
    return TaskPaths(
        task_dir=task_dir,
        prompt_path=task_dir / "prompt.txt",
        initial_state_path=task_dir / "initial_state.json",
        final_state_path=task_dir / "final_state.json",
        tool_log_path=task_dir / "mcp_tool_calls.jsonl",
        stdout_path=task_dir / "agent_stdout.jsonl",
        stderr_path=task_dir / "agent_stderr.log",
        last_message_path=task_dir / "last_message.txt",
        mcp_config_path=task_dir / "mcp_config.json",
        agent_export_path=task_dir / "agent_export.json",
    )


def server_command(paths: TaskPaths) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "harness" / "mcp_bridge_server.py"),
        "--initial-state",
        str(paths.initial_state_path),
        "--final-state",
        str(paths.final_state_path),
        "--tool-log",
        str(paths.tool_log_path),
        "--index-file",
        str(paths.task_dir / "api_search_index.txt"),
    ]


def toml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def toml_array(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(str(value)) for value in values) + "]"


def build_mcp_config(paths: TaskPaths) -> dict[str, Any]:
    cmd = server_command(paths)
    return {
        "mcpServers": {
            "automationbench": {
                "command": cmd[0],
                "args": cmd[1:],
                "cwd": str(paths.task_dir),
            }
        }
    }


def command_for_record(command: list[str]) -> list[str]:
    return [part if len(part) < 500 else part[:497] + "..." for part in command]


def run_process(
    command: list[str],
    *,
    cwd: Path,
    prompt: str,
    timeout_sec: int,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> AgentRun:
    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update(env)

    try:
        completed = subprocess.run(
            command,
            input=prompt,
            cwd=cwd,
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        write_text(stdout_path, stdout)
        write_text(stderr_path, stderr)
        return AgentRun(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
            command=command_for_record(command),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        write_text(stdout_path, stdout)
        write_text(stderr_path, stderr + f"\nTimed out after {timeout_sec}s\n")
        return AgentRun(
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            error=f"timed out after {timeout_sec}s",
            command=command_for_record(command),
        )
    except FileNotFoundError as exc:
        write_text(stdout_path, "")
        write_text(stderr_path, str(exc))
        return AgentRun(
            returncode=None,
            stdout="",
            stderr=str(exc),
            error=str(exc),
            command=command_for_record(command),
        )


class AgentAdapter:
    name = "agent"

    def __init__(self, config: dict[str, Any], timeout_sec: int):
        self.config = config
        self.timeout_sec = timeout_sec

    async def run_task(self, prompt: str, paths: TaskPaths) -> AgentRun:
        raise NotImplementedError


class NoopAdapter(AgentAdapter):
    name = "noop"

    async def run_task(self, prompt: str, paths: TaskPaths) -> AgentRun:
        paths.final_state_path.write_text(paths.initial_state_path.read_text())
        write_text(paths.stdout_path, "")
        write_text(paths.stderr_path, "")
        return AgentRun(returncode=0, stdout="", stderr="", error=None, command=["noop"])


class CodexAdapter(AgentAdapter):
    name = "codex"

    async def run_task(self, prompt: str, paths: TaskPaths) -> AgentRun:
        cmd = server_command(paths)
        tool_timeout = int(self.config.get("tool_timeout_sec", 120))
        startup_timeout = int(self.config.get("startup_timeout_sec", 20))

        config_overrides = [
            f"approval_policy={toml_string(self.config.get('approval_policy', 'never'))}",
            f"mcp_servers.automationbench.command={toml_string(cmd[0])}",
            f"mcp_servers.automationbench.args={toml_array(cmd[1:])}",
            f"mcp_servers.automationbench.cwd={toml_string(paths.task_dir)}",
            "mcp_servers.automationbench.enabled=true",
            "mcp_servers.automationbench.required=true",
            f"mcp_servers.automationbench.startup_timeout_sec={startup_timeout}",
            f"mcp_servers.automationbench.tool_timeout_sec={tool_timeout}",
            'mcp_servers.automationbench.default_tools_approval_mode="approve"',
        ]

        command = [self.config.get("binary", "codex"), "exec", "--json", "--ephemeral"]
        command += ["--skip-git-repo-check", "-C", str(paths.task_dir)]
        command += ["--sandbox", self.config.get("sandbox", "workspace-write")]
        command += ["--output-last-message", str(paths.last_message_path)]

        if self.config.get("ignore_rules", True):
            command.append("--ignore-rules")
        if self.config.get("ignore_user_config", False):
            command.append("--ignore-user-config")
        if model := self.config.get("model"):
            command += ["--model", str(model)]
        if profile := self.config.get("profile"):
            command += ["--profile", str(profile)]
        for override in config_overrides + list(self.config.get("config", [])):
            command += ["-c", override]
        command += list(self.config.get("extra_args", []))
        command.append("-")

        return run_process(
            command,
            cwd=paths.task_dir,
            prompt=prompt,
            timeout_sec=self.timeout_sec,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
        )


class ClaudeCodeAdapter(AgentAdapter):
    name = "claude"

    async def run_task(self, prompt: str, paths: TaskPaths) -> AgentRun:
        write_json(paths.mcp_config_path, build_mcp_config(paths))
        output_format = self.config.get("output_format", "stream-json")
        command = [
            self.config.get("binary", "claude"),
            "--print",
            "--output-format",
            output_format,
            "--strict-mcp-config",
            "--mcp-config",
            str(paths.mcp_config_path),
            "--permission-mode",
            self.config.get("permission_mode", "bypassPermissions"),
        ]
        if output_format == "stream-json":
            command.append("--verbose")
        if model := self.config.get("model"):
            command += ["--model", str(model)]
        command += list(self.config.get("extra_args", []))

        return run_process(
            command,
            cwd=paths.task_dir,
            prompt=prompt,
            timeout_sec=self.timeout_sec,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
        )


class DevinAdapter(AgentAdapter):
    name = "devin"

    async def run_task(self, prompt: str, paths: TaskPaths) -> AgentRun:
        cmd = server_command(paths)
        server_name = str(self.config.get("server_name", "automationbench"))
        local_config_path = paths.task_dir / ".devin" / "config.local.json"
        write_json(
            local_config_path,
            {
                "mcpServers": {
                    server_name: {
                        "command": cmd[0],
                        "args": cmd[1:],
                        "transport": "stdio",
                        "env": {"PYTHONUNBUFFERED": "1"},
                    }
                }
            },
        )
        write_json(paths.mcp_config_path, {"devin_local_config": str(local_config_path)})

        custom_command = self.config.get("command")
        template = custom_command
        if template is None:
            template = [
                self.config.get("binary", "devin"),
                "--print",
                "--prompt-file",
                "{prompt_path}",
                "--permission-mode",
                self.config.get("permission_mode", "dangerous"),
                "--export",
                "{agent_export}",
            ]
            if self.config.get("respect_workspace_trust") is not None:
                template += [
                    "--respect-workspace-trust",
                    str(self.config["respect_workspace_trust"]).lower(),
                ]
            if model := self.config.get("model"):
                template += ["--model", str(model)]
            if agent_config := self.config.get("devin_agent_config"):
                template += ["--agent-config", str(agent_config)]
        if isinstance(template, str):
            raise ValueError("Devin adapter config 'command' must be an argv list, not a shell string")

        replacements = {
            "mcp_config": str(paths.mcp_config_path),
            "run_dir": str(paths.task_dir),
            "prompt_path": str(paths.prompt_path),
            "agent_export": str(paths.agent_export_path),
        }
        command = [str(part).format(**replacements) for part in template]
        command += list(self.config.get("extra_args", []))

        return run_process(
            command,
            cwd=paths.task_dir,
            prompt=prompt if custom_command is not None else "",
            timeout_sec=self.timeout_sec,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
        )


def get_agent_adapter(agent_name: str, config: dict[str, Any], timeout_sec: int) -> AgentAdapter:
    adapters: dict[str, type[AgentAdapter]] = {
        "codex": CodexAdapter,
        "claude": ClaudeCodeAdapter,
        "devin": DevinAdapter,
        "noop": NoopAdapter,
    }
    return adapters[agent_name](config, timeout_sec)


async def run_single_task(
    row: dict[str, Any],
    *,
    index: int,
    run_dir: Path,
    agent: AgentAdapter,
    embed_trace: bool,
) -> dict[str, Any]:
    task = normalize_task(row)
    task_id = str(task.get("task") or task.get("example_id") or f"task_{index}")
    example_id = task.get("example_id")
    paths = make_task_paths(run_dir, index, task_id)
    paths.task_dir.mkdir(parents=True, exist_ok=True)

    prompt = format_prompt(task["prompt"])
    info = task["info"]
    initial_state = strip_none_values(info.get("initial_state", {}))
    write_text(paths.prompt_path, prompt)
    write_json(paths.initial_state_path, initial_state)

    print(f"[{index}] {task_id}: running {agent.name}", flush=True)
    started = time.time()

    try:
        run = await agent.run_task(prompt, paths)
    except Exception as exc:
        run = AgentRun(
            returncode=None,
            stdout="",
            stderr=str(exc),
            error=str(exc),
            command=[agent.name],
        )
        write_text(paths.stdout_path, "")
        write_text(paths.stderr_path, str(exc))

    final_state = read_json(paths.final_state_path) or initial_state
    final_world = WorldState(**strip_none_values(final_state))
    state_for_grading = {
        "world": final_world,
        "initial_state": initial_state,
        "info": info,
    }
    partial_score = partial_credit(state_for_grading)
    strict_score = task_completed_correctly(state_for_grading)
    elapsed = time.time() - started

    print(
        f"[{index}] {task_id}: partial_credit={partial_score:.3f} "
        f"strict={strict_score:.0f} agent_success={run.success}",
        flush=True,
    )

    result = {
        "task_id": task_id,
        "example_id": example_id,
        "agent": agent.name,
        "agent_success": run.success,
        "returncode": run.returncode,
        "error": run.error,
        "partial_credit": partial_score,
        "task_completed_correctly": strict_score,
        "elapsed_sec": round(elapsed, 3),
        "paths": {
            "task_dir": str(paths.task_dir),
            "prompt": str(paths.prompt_path),
            "initial_state": str(paths.initial_state_path),
            "final_state": str(paths.final_state_path),
            "tool_log": str(paths.tool_log_path),
            "stdout": str(paths.stdout_path),
            "stderr": str(paths.stderr_path),
            "last_message": str(paths.last_message_path),
            "agent_export": str(paths.agent_export_path),
        },
        "command": run.command,
        "assertion_results": state_for_grading.get("_assertion_results", []),
    }
    if embed_trace:
        result["trace"] = run.stdout
        result["stderr"] = run.stderr
    return result


def expand_domains(domains_arg: str) -> list[str]:
    raw = [d.strip() for d in domains_arg.split(",") if d.strip()]
    domains: list[str] = []
    for domain in raw:
        if domain == "all":
            domains.extend(DEFAULT_DOMAINS)
        elif domain in DOMAIN_ALIASES:
            domains.extend(DOMAIN_ALIASES[domain])
        else:
            domains.append(domain)
    return domains


def make_run_dir(agent: str, requested: str | None) -> Path:
    if requested:
        run_dir = Path(requested)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "runs" / f"{stamp}_{agent}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_agent_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def save_results(path: Path, results: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"results": results}
    if summary is not None:
        payload["summary"] = summary
    write_json(path, payload)


async def main() -> None:
    parser = argparse.ArgumentParser(description="AutomationBench native-agent harness")
    parser.add_argument("--agent", choices=["codex", "claude", "devin", "noop"], required=True)
    parser.add_argument("--domains", default="finance", help="Comma-separated domains, aliases, or all")
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", default=None, help="Defaults to <run-dir>/results.json")
    parser.add_argument("--agent-config", default=None, help="JSON config for the selected adapter")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--embed-trace", action="store_true", help="Embed stdout/stderr in results JSON")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    domains = expand_domains(args.domains)
    print(f"Loading AutomationBench domains: {', '.join(domains)}", flush=True)
    dataset = get_combined_dataset(domains)

    rows = list(dataset)
    if args.tasks:
        selected = {task.strip() for task in args.tasks.split(",") if task.strip()}
        rows = [row for row in rows if row.get("task") in selected]
    if args.start_index:
        rows = rows[args.start_index :]
    if args.num_tasks is not None:
        rows = rows[: args.num_tasks]

    if args.list_tasks:
        for row in rows:
            print(row.get("task"))
        return

    run_dir = make_run_dir(args.agent, args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "results.json"
    agent_config = load_agent_config(args.agent_config)
    agent = get_agent_adapter(args.agent, agent_config, args.timeout_sec)

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Tasks: {len(rows)}", flush=True)

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        result = await run_single_task(
            row,
            index=args.start_index + i,
            run_dir=run_dir,
            agent=agent,
            embed_trace=args.embed_trace,
        )
        results.append(result)
        save_results(output_path, results)
        if args.stop_on_error and not result["agent_success"]:
            break

    if results:
        summary = {
            "total_tasks": len(results),
            "agent_successes": sum(1 for r in results if r["agent_success"]),
            "avg_partial_credit": sum(r["partial_credit"] for r in results) / len(results),
            "avg_task_completed_correctly": sum(r["task_completed_correctly"] for r in results)
            / len(results),
            "run_dir": str(run_dir),
        }
    else:
        summary = {
            "total_tasks": 0,
            "agent_successes": 0,
            "avg_partial_credit": 0.0,
            "avg_task_completed_correctly": 0.0,
            "run_dir": str(run_dir),
        }
    save_results(output_path, results, summary)

    print("\nSummary", flush=True)
    print(f"  tasks: {summary['total_tasks']}", flush=True)
    print(f"  agent_successes: {summary['agent_successes']}", flush=True)
    print(f"  avg_partial_credit: {summary['avg_partial_credit']:.3f}", flush=True)
    print(f"  avg_strict: {summary['avg_task_completed_correctly']:.3f}", flush=True)
    print(f"  results: {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
