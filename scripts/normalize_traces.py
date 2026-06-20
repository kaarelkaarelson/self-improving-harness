#!/usr/bin/env python3
"""Normalize native AutomationBench traces into SFT-ready JSONL records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "automationbench-native-sft-v1"


def read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="replace")


def read_json(path: str | Path | None) -> Any | None:
    text = read_text(path)
    if not text:
        return None
    return json.loads(text)


def iter_jsonl(path: str | Path | None) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    with p.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"type": "unparsed_line", "raw": line}
            if isinstance(payload, dict):
                yield line_no, payload
            else:
                yield line_no, {"type": "raw_json_value", "value": payload}


def json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") == "tool_reference":
                parts.append(item.get("tool_name", ""))
            else:
                parts.append(json_text(item))
        return "\n".join(part for part in parts if part)
    return json_text(content)


def codex_result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        return content_to_text(result["content"])
    return json_text(result)


def normalize_tool_log(path: str | Path | None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for _, payload in iter_jsonl(path):
        calls.append(
            {
                "index": payload.get("call_index"),
                "ts": payload.get("ts"),
                "tool": payload.get("tool"),
                "arguments": payload.get("arguments"),
                "result": payload.get("result"),
                "error": payload.get("error"),
            }
        )
    return calls


def normalize_codex_trace(path: str | Path | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, payload in iter_jsonl(path):
        event_type = payload.get("type")
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        item_type = item.get("type")
        base = {
            "source": "native_trace",
            "agent": "codex",
            "line": line_no,
            "native_type": event_type,
        }

        if event_type == "item.completed" and item_type == "agent_message":
            text = item.get("text", "")
            if text:
                events.append({**base, "type": "assistant_message", "content": text})
        elif event_type == "item.completed" and item_type == "mcp_tool_call":
            events.append(
                {
                    **base,
                    "type": "tool_call",
                    "tool_call_id": item.get("id"),
                    "server": item.get("server"),
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "result": codex_result_text(item.get("result")),
                    "error": item.get("error"),
                    "status": item.get("status"),
                }
            )
        elif event_type in {"thread.started", "turn.started", "turn.completed"}:
            event: dict[str, Any] = {**base, "type": "lifecycle"}
            if usage := payload.get("usage"):
                event["usage"] = usage
            if thread_id := payload.get("thread_id"):
                event["thread_id"] = thread_id
            events.append(event)
        elif event_type == "unparsed_line":
            events.append({**base, "type": "raw_event", "raw": payload.get("raw")})
    return events


def normalize_claude_trace(path: str | Path | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, payload in iter_jsonl(path):
        event_type = payload.get("type")
        base = {
            "source": "native_trace",
            "agent": "claude",
            "line": line_no,
            "native_type": event_type,
        }

        if event_type == "assistant":
            message = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
            for block in message.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and block.get("text"):
                    events.append(
                        {
                            **base,
                            "type": "assistant_message",
                            "message_id": message.get("id"),
                            "content": block["text"],
                        }
                    )
                elif block_type == "tool_use":
                    events.append(
                        {
                            **base,
                            "type": "tool_call",
                            "message_id": message.get("id"),
                            "tool_call_id": block.get("id"),
                            "tool": block.get("name"),
                            "arguments": block.get("input"),
                        }
                    )
        elif event_type == "user":
            message = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
            for block in message.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    events.append(
                        {
                            **base,
                            "type": "tool_result",
                            "tool_call_id": block.get("tool_use_id"),
                            "result": content_to_text(block.get("content")),
                        }
                    )
        elif event_type == "result":
            events.append(
                {
                    **base,
                    "type": "run_summary",
                    "result": payload.get("result"),
                    "is_error": payload.get("is_error"),
                    "duration_ms": payload.get("duration_ms"),
                    "num_turns": payload.get("num_turns"),
                    "usage": payload.get("usage"),
                    "cost_usd": payload.get("total_cost_usd"),
                }
            )
        elif event_type == "system" and payload.get("subtype") == "init":
            events.append(
                {
                    **base,
                    "type": "system",
                    "subtype": "init",
                    "session_id": payload.get("session_id"),
                    "mcp_servers": payload.get("mcp_servers"),
                    "model": payload.get("model"),
                    "permission_mode": payload.get("permissionMode"),
                }
            )
        elif event_type == "unparsed_line":
            events.append({**base, "type": "raw_event", "raw": payload.get("raw")})
    return events


def normalize_devin_trace(path: str | Path | None) -> list[dict[str, Any]]:
    p = Path(path) if path else None
    if not p or not p.exists() or p.stat().st_size == 0:
        return []

    text = p.read_text(errors="replace")
    events: list[dict[str, Any]] = []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        parsed_lines = list(iter_jsonl(p))
        if parsed_lines:
            for line_no, line_payload in parsed_lines:
                events.append(
                    {
                        "source": "native_trace",
                        "agent": "devin",
                        "line": line_no,
                        "type": "raw_event",
                        "event": line_payload,
                    }
                )
            return events
        return [{"source": "native_trace", "agent": "devin", "type": "raw_text", "content": text}]

    if isinstance(payload, dict):
        if isinstance(payload.get("steps"), list):
            events.extend(normalize_devin_steps(payload))
            return events
        if isinstance(payload.get("messages"), list):
            iterable = payload["messages"]
        elif isinstance(payload.get("events"), list):
            iterable = payload["events"]
        else:
            iterable = [payload]
    elif isinstance(payload, list):
        iterable = payload
    else:
        iterable = [{"value": payload}]

    for index, event in enumerate(iterable):
        if not isinstance(event, dict):
            events.append(
                {
                    "source": "native_trace",
                    "agent": "devin",
                    "index": index,
                    "type": "raw_event",
                    "event": event,
                }
            )
            continue

        role = event.get("role") or event.get("type")
        content = event.get("content") or event.get("text") or event.get("message")
        if role in {"assistant", "agent"} and content:
            events.append(
                {
                    "source": "native_trace",
                    "agent": "devin",
                    "index": index,
                    "type": "assistant_message",
                    "content": content_to_text(content),
                }
            )
        elif role in {"tool", "tool_result"}:
            events.append(
                {
                    "source": "native_trace",
                    "agent": "devin",
                    "index": index,
                    "type": "tool_result",
                    "tool": event.get("name") or event.get("tool_name"),
                    "result": content_to_text(content),
                    "event": event,
                }
            )
        else:
            events.append(
                {
                    "source": "native_trace",
                    "agent": "devin",
                    "index": index,
                    "type": "raw_event",
                    "event": event,
                }
            )
    return events


def devin_tool_name_and_args(tool_call: dict[str, Any]) -> tuple[str | None, Any]:
    function_name = tool_call.get("function_name")
    arguments = tool_call.get("arguments")
    if function_name == "mcp_call_tool" and isinstance(arguments, dict):
        return arguments.get("tool_name"), arguments.get("arguments")
    return function_name, arguments


def normalize_devin_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        source = step.get("source")
        base = {
            "source": "native_trace",
            "agent": "devin",
            "index": step.get("step_id"),
            "timestamp": step.get("timestamp"),
            "native_source": source,
        }

        if source == "agent":
            if step.get("reasoning_content"):
                events.append(
                    {
                        **base,
                        "type": "reasoning",
                        "content": step.get("reasoning_content"),
                    }
                )
            if step.get("message"):
                events.append(
                    {
                        **base,
                        "type": "assistant_message",
                        "content": step.get("message"),
                    }
                )
            for tool_call in step.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_name, arguments = devin_tool_name_and_args(tool_call)
                events.append(
                    {
                        **base,
                        "type": "tool_call",
                        "tool_call_id": tool_call.get("tool_call_id"),
                        "adapter_tool": tool_call.get("function_name"),
                        "tool": tool_name,
                        "arguments": arguments,
                    }
                )
            observation = step.get("observation")
            if isinstance(observation, dict):
                for result in observation.get("results", []) or []:
                    if not isinstance(result, dict):
                        continue
                    events.append(
                        {
                            **base,
                            "type": "tool_result",
                            "tool_call_id": result.get("source_call_id"),
                            "result": content_to_text(result.get("content")),
                        }
                    )
            if step.get("metrics"):
                events.append({**base, "type": "usage", "usage": step.get("metrics")})
        elif source == "user":
            events.append({**base, "type": "user_message", "content": step.get("message")})

    if payload.get("final_metrics"):
        events.append(
            {
                "source": "native_trace",
                "agent": "devin",
                "type": "run_summary",
                "usage": payload.get("final_metrics"),
                "schema_version": payload.get("schema_version"),
                "session_id": payload.get("session_id"),
                "model": (payload.get("agent") or {}).get("model_name")
                if isinstance(payload.get("agent"), dict)
                else None,
            }
        )
    return events


def normalize_native_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    paths = result.get("paths", {}) if isinstance(result.get("paths"), dict) else {}
    agent = result.get("agent")
    if agent == "codex":
        return normalize_codex_trace(paths.get("stdout"))
    if agent == "claude":
        return normalize_claude_trace(paths.get("stdout"))
    if agent == "devin":
        return normalize_devin_trace(paths.get("agent_export") or paths.get("stdout"))
    return []


def tool_marker(name: str, payload: Any) -> str:
    return f"<{name}>\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n</{name}>"


def build_training_messages(
    prompt: str,
    native_events: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    for event in native_events:
        event_type = event.get("type")
        if event_type == "assistant_message" and event.get("content"):
            messages.append({"role": "assistant", "content": event["content"]})
        elif event_type == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": tool_marker(
                        "tool_call",
                        {
                            "id": event.get("tool_call_id"),
                            "tool": event.get("tool"),
                            "arguments": event.get("arguments"),
                        },
                    ),
                }
            )
            if event.get("result") or event.get("error"):
                messages.append(
                    {
                        "role": "tool",
                        "name": str(event.get("tool") or ""),
                        "content": tool_marker(
                            "tool_result",
                            {"result": event.get("result"), "error": event.get("error")},
                        ),
                    }
                )
        elif event_type == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "name": str(event.get("tool") or event.get("tool_call_id") or ""),
                    "content": event.get("result", ""),
                }
            )

    if len(messages) == 1:
        for call in tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": tool_marker(
                        "tool_call",
                        {
                            "id": call.get("index"),
                            "tool": call.get("tool"),
                            "arguments": call.get("arguments"),
                        },
                    ),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": str(call.get("tool") or ""),
                    "content": tool_marker(
                        "tool_result",
                        {"result": call.get("result"), "error": call.get("error")},
                    ),
                }
            )

    return messages


def result_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif (path / "results.json").exists():
            files.append(path / "results.json")
        else:
            raise FileNotFoundError(f"No results.json found for {path}")
    return files


def iter_results(inputs: list[str]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for results_path in result_files(inputs):
        payload = read_json(results_path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"{results_path} does not look like a harness results file")
        for result in payload["results"]:
            if isinstance(result, dict):
                yield results_path, result


def make_record(
    results_path: Path,
    result: dict[str, Any],
    *,
    include_final_state: bool,
    include_initial_state: bool,
) -> dict[str, Any]:
    paths = result.get("paths", {}) if isinstance(result.get("paths"), dict) else {}
    prompt = read_text(paths.get("prompt"))
    tool_calls = normalize_tool_log(paths.get("tool_log"))
    native_events = normalize_native_events(result)

    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "task_id": result.get("task_id"),
            "example_id": result.get("example_id"),
            "prompt": prompt,
        },
        "agent": {
            "name": result.get("agent"),
            "success": result.get("agent_success"),
            "returncode": result.get("returncode"),
            "error": result.get("error"),
            "command": result.get("command"),
        },
        "score": {
            "partial_credit": result.get("partial_credit"),
            "task_completed_correctly": result.get("task_completed_correctly"),
            "elapsed_sec": result.get("elapsed_sec"),
        },
        "assertion_results": result.get("assertion_results", []),
        "artifacts": {
            "results": str(results_path),
            **paths,
        },
        "tool_calls": tool_calls,
        "native_events": native_events,
        "training_messages": build_training_messages(prompt, native_events, tool_calls),
    }

    if include_initial_state:
        record["initial_state"] = read_json(paths.get("initial_state"))
    if include_final_state:
        record["final_state"] = read_json(paths.get("final_state"))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize native harness traces for SFT")
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs or results.json files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--min-partial-credit", type=float, default=0.0)
    parser.add_argument("--require-agent-success", action="store_true")
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--include-initial-state", action="store_true")
    parser.add_argument("--include-final-state", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    written = 0
    skipped = 0
    by_agent: dict[str, int] = {}
    with output_path.open("w") as out:
        for results_path, result in iter_results(args.runs):
            seen += 1
            partial = float(result.get("partial_credit") or 0.0)
            strict = float(result.get("task_completed_correctly") or 0.0)
            if partial < args.min_partial_credit:
                skipped += 1
                continue
            if args.require_agent_success and not result.get("agent_success"):
                skipped += 1
                continue
            if args.require_strict and strict < 1.0:
                skipped += 1
                continue

            record = make_record(
                results_path,
                result,
                include_initial_state=args.include_initial_state,
                include_final_state=args.include_final_state,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            agent = str(result.get("agent") or "unknown")
            by_agent[agent] = by_agent.get(agent, 0) + 1

    print(f"Read results: {seen}")
    print(f"Wrote records: {written}")
    print(f"Skipped records: {skipped}")
    print(f"By agent: {json.dumps(by_agent, sort_keys=True)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
