#!/usr/bin/env python3
"""Select AutomationBench traces and export a Prime-RL SFT dataset.

Input records can come from:
- scripts/download_prime_traces.py, schema automationbench-prime-sft-v1
- scripts/normalize_traces.py, schema automationbench-native-sft-v1

The output directory is a local Hugging Face dataset directory:
- train.jsonl
- optional validation.jsonl
- selected_traces.jsonl, with audit metadata
- manifest.json
- sft.toml, a Prime-RL config template
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "automationbench-prime-rl-sft-v1"


API_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "api_search",
            "description": (
                "Search available AutomationBench API endpoints by keyword before "
                "calling api_fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_fetch",
            "description": (
                "Call a discovered AutomationBench API endpoint. This mutates the "
                "task-local simulated world."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "url": {"type": "string"},
                    "params": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": True},
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "body": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": True},
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["method", "url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "base64_encode",
            "description": "Encode plaintext to base64url for API fields that require it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "data": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def iter_jsonl(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield path, line_no, payload


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def as_str_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalize_arguments(arguments: Any) -> str:
    if arguments is None or arguments == "":
        return "{}"
    if isinstance(arguments, str):
        # Prime-RL will deserialize this before rendering. Keep non-JSON strings
        # as JSON strings so downstream json.loads succeeds.
        try:
            json.loads(arguments)
            return arguments
        except json.JSONDecodeError:
            return json.dumps(arguments, ensure_ascii=False)
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def normalize_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        messages: list[dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role") or "user")
                if role not in {"system", "user", "assistant", "tool"}:
                    role = "user"
                message: dict[str, Any] = {
                    "role": role,
                    "content": as_str_or_json(item.get("content", "")),
                }
                if role == "tool" and item.get("tool_call_id") is not None:
                    message["tool_call_id"] = str(item["tool_call_id"])
                messages.append(message)
            else:
                messages.append({"role": "user", "content": as_str_or_json(item)})
        return messages
    return [{"role": "user", "content": as_str_or_json(prompt)}]


def normalize_tool_call(raw_call: Any, fallback_index: int) -> dict[str, Any] | None:
    call = parse_jsonish(raw_call)
    if not isinstance(call, dict):
        return None

    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or call.get("tool") or call.get("tool_name") or function.get("name")
    arguments = (
        call.get("arguments")
        if "arguments" in call
        else call.get("input")
        if "input" in call
        else function.get("arguments")
    )
    if not name:
        return None

    return {
        "id": str(call.get("id") or call.get("tool_call_id") or f"call_{fallback_index}"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": normalize_arguments(arguments),
        },
    }


def messages_from_prime(record: dict[str, Any]) -> list[dict[str, Any]]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    messages = normalize_prompt_messages(task.get("prompt"))
    completion = trace.get("completion")
    if not isinstance(completion, list):
        return messages

    call_index = 0
    for event in completion:
        if not isinstance(event, dict):
            continue
        role = event.get("role")
        if role == "assistant":
            tool_calls: list[dict[str, Any]] = []
            for raw_call in event.get("tool_calls") or []:
                call_index += 1
                tool_call = normalize_tool_call(raw_call, call_index)
                if tool_call:
                    tool_calls.append(tool_call)

            content = event.get("content")
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if isinstance(content, str) and content else None,
                        "tool_calls": tool_calls,
                    }
                )
            elif content is not None:
                messages.append({"role": "assistant", "content": as_str_or_json(content)})
        elif role == "tool":
            tool_call_id = event.get("tool_call_id") or event.get("id") or f"call_{call_index}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id),
                    "content": as_str_or_json(event.get("content", "")),
                }
            )
    return messages


def messages_from_native(record: dict[str, Any]) -> list[dict[str, Any]]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    messages = [{"role": "user", "content": as_str_or_json(task.get("prompt", ""))}]
    native_events = record.get("native_events")
    tool_calls = record.get("tool_calls")
    call_index = 0
    last_tool_call_id: str | None = None

    if isinstance(native_events, list):
        for event in native_events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "assistant_message" and event.get("content"):
                messages.append({"role": "assistant", "content": as_str_or_json(event["content"])})
            elif event_type == "tool_call":
                call_index += 1
                raw_call = {
                    "id": event.get("tool_call_id") or f"call_{call_index}",
                    "name": event.get("tool"),
                    "arguments": event.get("arguments"),
                }
                tool_call = normalize_tool_call(raw_call, call_index)
                if not tool_call:
                    continue
                last_tool_call_id = tool_call["id"]
                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                if event.get("result") is not None or event.get("error") is not None:
                    result = {"result": event.get("result"), "error": event.get("error")}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": last_tool_call_id,
                            "content": json_dumps(result),
                        }
                    )
            elif event_type == "tool_result":
                tool_call_id = event.get("tool_call_id") or last_tool_call_id or f"call_{call_index}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call_id),
                        "content": as_str_or_json(event.get("result", "")),
                    }
                )

    if len(messages) == 1 and isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_index += 1
            raw_call = {
                "id": call.get("index") or f"call_{call_index}",
                "name": call.get("tool"),
                "arguments": call.get("arguments"),
            }
            tool_call = normalize_tool_call(raw_call, call_index)
            if not tool_call:
                continue
            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            result = {"result": call.get("result"), "error": call.get("error")}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json_dumps(result),
                }
            )

    return messages


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    schema = record.get("schema_version")
    if schema == "automationbench-prime-sft-v1":
        return messages_from_prime(record)
    if schema == "automationbench-native-sft-v1":
        return messages_from_native(record)

    if isinstance(record.get("messages"), list):
        return record["messages"]
    if isinstance(record.get("training_messages"), list):
        return record["training_messages"]
    raise ValueError(f"Unsupported trace schema: {schema!r}")


def score(record: dict[str, Any]) -> tuple[float, float]:
    raw_score = record.get("score") if isinstance(record.get("score"), dict) else {}
    partial = as_float(raw_score.get("partial_credit"), 0.0)
    strict = as_float(raw_score.get("task_completed_correctly"), 0.0)
    return partial, strict


def model_name(record: dict[str, Any]) -> str:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    agent = record.get("agent") if isinstance(record.get("agent"), dict) else {}
    return str(evaluation.get("model") or agent.get("name") or "unknown")


def evaluation_id(record: dict[str, Any]) -> str | None:
    evaluation = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    value = evaluation.get("id")
    return str(value) if value else None


def trace_id(record: dict[str, Any]) -> str | None:
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    value = trace.get("trace_id")
    return str(value) if value else None


def task_key(record: dict[str, Any]) -> str:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    for key in ("example_id", "task_id", "task"):
        value = task.get(key)
        if value:
            return str(value)
    prompt = task.get("prompt")
    return hashlib.sha256(as_str_or_json(prompt).encode("utf-8")).hexdigest()


def task_name(record: dict[str, Any]) -> str | None:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    value = task.get("task") or task.get("task_id")
    return str(value) if value else None


def message_trace_size(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def has_assistant_signal(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            if function.get("name"):
                names.add(str(function["name"]))
    return names


def normalize_tool_def(raw_tool: dict[str, Any]) -> dict[str, Any] | None:
    if raw_tool.get("type") == "function" and isinstance(raw_tool.get("function"), dict):
        return raw_tool
    name = raw_tool.get("name")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": raw_tool.get("description", ""),
            "parameters": raw_tool.get("parameters") or raw_tool.get("inputSchema") or {"type": "object"},
        },
    }


def explicit_tools(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = [record.get("tools"), record.get("tool_defs")]
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    candidates.extend([info.get("tools"), info.get("tool_defs")])

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = parse_jsonish(candidate)
        if not isinstance(candidate, list):
            continue
        for raw_tool in candidate:
            if not isinstance(raw_tool, dict):
                continue
            tool = normalize_tool_def(raw_tool)
            if not tool:
                continue
            name = tool.get("function", {}).get("name")
            if name and name not in seen:
                tools.append(tool)
                seen.add(str(name))
    return tools


def tools_for_record(record: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = explicit_tools(record)
    seen = {
        str(tool.get("function", {}).get("name"))
        for tool in tools
        if isinstance(tool.get("function"), dict) and tool.get("function", {}).get("name")
    }
    names = tool_names(messages)
    for tool in API_TOOL_DEFS:
        name = str(tool["function"]["name"])
        if name in names and name not in seen:
            tools.append(tool)
            seen.add(name)
    return tools


class TokenCounter:
    def __init__(self, tokenizer_name: str, *, offline: bool):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=offline,
        )

    def count(self, record: dict[str, Any], messages: list[dict[str, Any]]) -> int:
        tools = tools_for_record(record, messages)
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": False}
        if tools:
            kwargs["tools"] = tools
        try:
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            token_ids = self.tokenizer.encode(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        return len(token_ids)


_WORKER_CONFIG: dict[str, Any] = {}
_WORKER_TOKEN_COUNTER: TokenCounter | None = None


def init_record_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_TOKEN_COUNTER
    _WORKER_CONFIG = config
    _WORKER_TOKEN_COUNTER = None
    if config.get("max_rendered_tokens") is not None:
        _WORKER_TOKEN_COUNTER = TokenCounter(
            str(config["tokenizer"]),
            offline=bool(config["tokenizer_offline"]),
        )


def process_record_for_selection(
    item: tuple[Path, int, dict[str, Any]],
) -> dict[str, Any]:
    path, line_no, record = item
    partial, strict = score(record)
    if partial < float(_WORKER_CONFIG["min_partial_credit"]):
        return {"status": "skip", "reason": "score"}
    if bool(_WORKER_CONFIG["require_strict"]) and strict < 1.0:
        return {"status": "skip", "reason": "score"}
    try:
        messages = build_messages(record)
    except Exception:
        return {"status": "skip", "reason": "parse"}
    if not has_assistant_signal(messages):
        return {"status": "skip", "reason": "messages"}

    rendered_tokens = None
    if _WORKER_TOKEN_COUNTER is not None:
        try:
            rendered_tokens = _WORKER_TOKEN_COUNTER.count(record, messages)
        except Exception:
            return {"status": "skip", "reason": "parse"}
        if rendered_tokens > int(_WORKER_CONFIG["max_rendered_tokens"]):
            return {"status": "skip", "reason": "tokens"}
        record = {**record, "rendered_tokens": rendered_tokens}

    return {
        "status": "candidate",
        "candidate": {
            "record": record,
            "path": str(path),
            "path_obj": path,
            "line_no": line_no,
            "messages": messages,
            "partial": partial,
            "strict": strict,
            "task_key": task_key(record),
            "model": model_name(record),
            "message_size": message_trace_size(messages),
            "rendered_tokens": rendered_tokens,
        },
    }


def collect_candidates(
    input_paths: list[Path],
    *,
    config: dict[str, Any],
    workers: int,
    chunksize: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter({"score": 0, "messages": 0, "parse": 0, "tokens": 0})

    def consume(result: dict[str, Any]) -> None:
        if result.get("status") == "candidate":
            candidates.append(result["candidate"])
        else:
            skipped[str(result.get("reason") or "parse")] += 1

    processed = 0
    if workers <= 1:
        init_record_worker(config)
        for item in iter_jsonl(input_paths):
            processed += 1
            consume(process_record_for_selection(item))
            if progress_every and processed % progress_every == 0:
                print(
                    f"Processed {processed} traces; candidates={len(candidates)}; "
                    f"skipped={json.dumps(dict(skipped), sort_keys=True)}",
                    flush=True,
                )
        return candidates, skipped

    with mp.Pool(
        processes=workers,
        initializer=init_record_worker,
        initargs=(config,),
    ) as pool:
        results = pool.imap_unordered(
            process_record_for_selection,
            iter_jsonl(input_paths),
            chunksize=chunksize,
        )
        for result in results:
            processed += 1
            consume(result)
            if progress_every and processed % progress_every == 0:
                print(
                    f"Processed {processed} traces; candidates={len(candidates)}; "
                    f"skipped={json.dumps(dict(skipped), sort_keys=True)}",
                    flush=True,
                )

    return candidates, skipped


def selected_source(record: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    return {
        "input_path": str(path),
        "input_line": line_no,
        "schema_version": record.get("schema_version"),
        "source": record.get("source"),
        "evaluation_id": evaluation_id(record),
        "trace_id": trace_id(record),
        "model": model_name(record),
    }


def build_sft_row(
    record: dict[str, Any],
    *,
    path: Path,
    line_no: int,
    rank_for_task: int,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    partial, strict = score(record)
    row: dict[str, Any] = {
        "messages": messages,
        "source": selected_source(record, path, line_no),
        "task_key": task_key(record),
        "task": task_name(record),
        "example_id": (record.get("task") or {}).get("example_id")
        if isinstance(record.get("task"), dict)
        else None,
        "model": model_name(record),
        "evaluation_id": evaluation_id(record),
        "trace_id": trace_id(record),
        "rank_for_task": rank_for_task,
        "partial_credit": partial,
        "task_completed_correctly": strict,
        "reward": partial,
    }
    if "rendered_tokens" in record:
        row["rendered_tokens"] = record["rendered_tokens"]
    tools = tools_for_record(record, messages)
    if tools:
        # Keep this JSON-encoded. Prime-RL accepts strings here, and this avoids
        # Arrow schema conflicts when different rows carry different tool schemas.
        row["tools"] = json.dumps(tools, ensure_ascii=False)
    return row


def selection_sort_key(item: dict[str, Any], prefer_shorter_trace: bool) -> tuple[Any, ...]:
    partial, strict = item["partial"], item["strict"]
    length = item["message_size"] if prefer_shorter_trace else -item["message_size"]
    return (-strict, -partial, length, item["model"], item["path"], item["line_no"])


def select_records(
    records: list[dict[str, Any]],
    *,
    top_k_per_task: int,
    prefer_shorter_trace: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["task_key"], []).append(item)

    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: selection_sort_key(item, prefer_shorter_trace))
        for rank, item in enumerate(ordered[:top_k_per_task], start=1):
            selected.append({**item, "rank_for_task": rank})

    return sorted(
        selected,
        key=lambda item: (
            item["task_key"],
            item["rank_for_task"],
            item["model"],
            item["path"],
            item["line_no"],
        ),
    )


def split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_fraction <= 0:
        return rows, []
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(rows) * validation_fraction)))
    val_indices = set(indices[:val_count])
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    validation = [row for index, row in enumerate(rows) if index in val_indices]
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def toml_quote(value: str | Path) -> str:
    return json.dumps(str(value))


def path_from_prime_rl_cwd(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return Path("..") / resolved.relative_to(ROOT)
    except ValueError:
        return resolved


def is_qwen35_model(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", ".")
    return "qwen3.5" in normalized


def write_sft_config(
    path: Path,
    *,
    output_dir: Path,
    dataset_dir: Path,
    base_model: str,
    seq_len: int,
    batch_size: int,
    max_steps: int,
    lr: float,
    validation: bool,
    num_gpus: int,
    gpus_per_node: int,
    micro_batch_size: int | None,
    attn: str | None,
    cp: int,
    cp_style: str | None,
    model_impl: str | None,
    optimization_dtype: str | None,
    reduce_dtype: str | None,
    loss_impl: str | None,
    activation_checkpoint_freq: int | None,
    pack_function: str | None,
) -> None:
    lines = [
        f"output_dir = {toml_quote(output_dir)}",
        f"max_steps = {max_steps}",
    ]
    if loss_impl is not None:
        lines.append(f"loss_impl = {toml_quote(loss_impl)}")
    lines.append("")
    if num_gpus > 1:
        lines.extend(
            [
                "[deployment]",
                'type = "single_node"',
                f"num_gpus = {num_gpus}",
                f"gpus_per_node = {gpus_per_node}",
                "",
            ]
        )
    lines.extend(
        [
            "[ckpt]",
            "",
        ]
    )
    lines.extend(
        [
            "[model]",
            f"name = {toml_quote(base_model)}",
            f"seq_len = {seq_len}",
        ]
    )
    if attn is not None:
        lines.append(f"attn = {toml_quote(attn)}")
    if cp > 1:
        lines.append(f"cp = {cp}")
    if cp_style is not None:
        lines.append(f"cp_style = {toml_quote(cp_style)}")
    if model_impl is not None:
        lines.append(f"impl = {toml_quote(model_impl)}")
    if optimization_dtype is not None:
        lines.append(f"optimization_dtype = {toml_quote(optimization_dtype)}")
    if reduce_dtype is not None:
        lines.append(f"reduce_dtype = {toml_quote(reduce_dtype)}")
    lines.append("")
    if activation_checkpoint_freq is not None:
        lines.extend(
            [
                "[model.compile]",
                "",
                "[model.ac]",
                f"freq = {activation_checkpoint_freq}",
                "",
            ]
        )
    lines.extend(
        [
            "[data]",
            'type = "sft"',
            f"name = {toml_quote(dataset_dir)}",
            'splits = ["train"]',
            f"seq_len = {seq_len}",
            f"batch_size = {batch_size}",
        ]
    )
    if micro_batch_size is not None:
        lines.append(f"micro_batch_size = {micro_batch_size}")
    if pack_function is not None:
        lines.append(f"pack_function = {toml_quote(pack_function)}")
    lines.extend(
        [
            "",
            "[optim]",
            f"lr = {lr}",
            "",
        ]
    )
    if validation:
        lines.extend(
            [
                "[val]",
                "interval = 50",
                "",
                "[val.data]",
                f"name = {toml_quote(dataset_dir)}",
                'splits = ["validation"]',
                f"seq_len = {seq_len}",
                f"batch_size = {batch_size}",
            ]
        )
        if micro_batch_size is not None:
            lines.append(f"micro_batch_size = {micro_batch_size}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Prime-RL SFT dataset from AutomationBench traces")
    parser.add_argument("--input-jsonl", nargs="+", required=True, help="Prime/native normalized JSONL files")
    parser.add_argument("--output-dir", required=True, help="Dataset directory to write")
    parser.add_argument("--min-partial-credit", type=float, default=0.5)
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--top-k-per-task", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-rendered-tokens",
        type=int,
        help="Drop traces whose rendered chat-template length exceeds this token count.",
    )
    parser.add_argument(
        "--tokenizer",
        default="PrimeIntellect/Qwen3-0.6B",
        help="Tokenizer used with --max-rendered-tokens.",
    )
    parser.add_argument(
        "--tokenizer-online",
        action="store_true",
        help="Allow tokenizer downloads while applying --max-rendered-tokens. Default is offline/cache-only.",
    )
    parser.add_argument(
        "--workers",
        "--num-workers",
        dest="workers",
        type=int,
        default=1,
        help="Parallel worker processes for per-trace normalization and token counting.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Number of input traces sent to each multiprocessing worker batch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many traces; use 0 to disable.",
    )
    parser.add_argument(
        "--prefer-longer-trace",
        action="store_true",
        help="Use longer traces as the tie-breaker. Default prefers shorter traces at equal score.",
    )
    parser.add_argument("--base-model", default="PrimeIntellect/Qwen3-0.6B")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--attn")
    parser.add_argument("--cp", type=int)
    parser.add_argument("--cp-style", choices=["ring", "ulysses"])
    parser.add_argument("--model-impl", choices=["hf", "custom", "auto"])
    parser.add_argument("--optimization-dtype", choices=["bfloat16", "float32"])
    parser.add_argument("--reduce-dtype", choices=["bfloat16", "float32"])
    parser.add_argument("--loss-impl")
    parser.add_argument("--activation-checkpoint-freq", type=int)
    parser.add_argument("--pack-function")
    parser.add_argument("--prime-rl-output-dir", default="outputs/automationbench-sft")
    args = parser.parse_args()

    qwen35_defaults = is_qwen35_model(args.base_model)
    num_gpus = args.num_gpus if args.num_gpus is not None else (8 if qwen35_defaults else 1)
    gpus_per_node = args.gpus_per_node if args.gpus_per_node is not None else num_gpus
    cp = args.cp if args.cp is not None else (4 if qwen35_defaults and num_gpus >= 4 else 1)
    micro_batch_size = args.micro_batch_size
    if micro_batch_size is None and qwen35_defaults:
        micro_batch_size = 1
    cp_style = args.cp_style
    if cp_style is None and cp > 1:
        cp_style = "ulysses"
    attn = args.attn
    if attn is None and qwen35_defaults:
        attn = "flash_attention_3"
    model_impl = args.model_impl
    if model_impl is None and qwen35_defaults:
        model_impl = "custom"
    optimization_dtype = args.optimization_dtype
    if optimization_dtype is None and qwen35_defaults:
        optimization_dtype = "bfloat16"
    reduce_dtype = args.reduce_dtype
    if reduce_dtype is None and qwen35_defaults:
        reduce_dtype = "bfloat16"
    loss_impl = args.loss_impl
    if loss_impl is None and qwen35_defaults:
        loss_impl = "liger_fused"
    activation_checkpoint_freq = args.activation_checkpoint_freq
    if activation_checkpoint_freq is None and qwen35_defaults:
        activation_checkpoint_freq = 1
    pack_function = args.pack_function
    if pack_function is None and qwen35_defaults:
        pack_function = "cat"

    input_paths = [Path(path) for path in args.input_jsonl]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)
    worker_config = {
        "min_partial_credit": args.min_partial_credit,
        "require_strict": args.require_strict,
        "max_rendered_tokens": args.max_rendered_tokens,
        "tokenizer": args.tokenizer,
        "tokenizer_offline": not args.tokenizer_online,
    }
    candidates, skipped = collect_candidates(
        input_paths,
        config=worker_config,
        workers=workers,
        chunksize=chunksize,
        progress_every=max(0, args.progress_every),
    )

    selected = select_records(
        candidates,
        top_k_per_task=args.top_k_per_task,
        prefer_shorter_trace=not args.prefer_longer_trace,
    )
    if args.max_records is not None:
        selected = selected[: args.max_records]

    rows = [
        build_sft_row(
            item["record"],
            path=item["path_obj"],
            line_no=item["line_no"],
            rank_for_task=item["rank_for_task"],
            messages=item["messages"],
        )
        for item in selected
    ]
    train_rows, validation_rows = split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    write_jsonl(output_dir / "train.jsonl", train_rows)
    if validation_rows:
        write_jsonl(output_dir / "validation.jsonl", validation_rows)
    else:
        validation_path = output_dir / "validation.jsonl"
        if validation_path.exists():
            validation_path.unlink()

    write_jsonl(output_dir / "selected_traces.jsonl", rows)

    by_model: dict[str, int] = {}
    for row in rows:
        by_model[row["model"]] = by_model.get(row["model"], 0) + 1
    prime_rl_config_path = path_from_prime_rl_cwd(output_dir / "sft.toml")
    prime_rl_dataset_dir = path_from_prime_rl_cwd(output_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": [str(path) for path in input_paths],
        "output_dir": str(output_dir),
        "filters": {
            "min_partial_credit": args.min_partial_credit,
            "require_strict": args.require_strict,
            "top_k_per_task": args.top_k_per_task,
            "max_records": args.max_records,
            "prefer_shorter_trace": not args.prefer_longer_trace,
            "validation_fraction": args.validation_fraction,
            "max_rendered_tokens": args.max_rendered_tokens,
            "tokenizer": args.tokenizer if args.max_rendered_tokens is not None else None,
            "workers": workers,
            "chunksize": chunksize,
        },
        "counts": {
            "candidates": len(candidates),
            "selected": len(rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "skipped": dict(skipped),
            "by_model": by_model,
        },
        "prime_rl": {
            "config": str(prime_rl_config_path),
            "dataset_name": str(prime_rl_dataset_dir),
            "output_dir": args.prime_rl_output_dir,
            "command": f"cd prime-rl && uv run sft @ {prime_rl_config_path} --ckpt",
            "training_config": {
                "base_model": args.base_model,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "micro_batch_size": micro_batch_size,
                "num_gpus": num_gpus,
                "gpus_per_node": gpus_per_node,
                "attn": attn,
                "cp": cp,
                "cp_style": cp_style,
                "model_impl": model_impl,
                "optimization_dtype": optimization_dtype,
                "reduce_dtype": reduce_dtype,
                "loss_impl": loss_impl,
                "activation_checkpoint_freq": activation_checkpoint_freq,
                "pack_function": pack_function,
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    write_sft_config(
        output_dir / "sft.toml",
        output_dir=Path(args.prime_rl_output_dir),
        dataset_dir=prime_rl_dataset_dir,
        base_model=args.base_model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        validation=bool(validation_rows),
        num_gpus=num_gpus,
        gpus_per_node=gpus_per_node,
        micro_batch_size=micro_batch_size,
        attn=attn,
        cp=cp,
        cp_style=cp_style,
        model_impl=model_impl,
        optimization_dtype=optimization_dtype,
        reduce_dtype=reduce_dtype,
        loss_impl=loss_impl,
        activation_checkpoint_freq=activation_checkpoint_freq,
        pack_function=pack_function,
    )

    print(f"Candidates: {len(candidates)}")
    print(f"Selected: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(validation_rows)}")
    print(f"Skipped: {json.dumps(dict(skipped), sort_keys=True)}")
    print(f"By model: {json.dumps(by_model, sort_keys=True)}")
    print(f"Dataset dir: {output_dir}")
    print(f"Prime-RL config: {output_dir / 'sft.toml'}")


if __name__ == "__main__":
    main()
