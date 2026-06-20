#!/usr/bin/env python3
"""Per-task AutomationBench MCP server.

The server exposes AutomationBench's API toolset over stdio MCP and binds every
tool call to a single in-memory WorldState. The current world is written after
startup and after every tool call so the harness can grade even if the client
terminates the server process instead of sending an orderly shutdown.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Keep simulated AutomationBench tools offline. Codex itself should use
# CODEX_API_KEY or saved Codex auth; this process must not inherit OpenAI API
# credentials because some simulated ChatGPT/Salesforce helpers check this var.
os.environ.pop("OPENAI_API_KEY", None)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from automationbench.schema.world import WorldState
from automationbench.tools.api.encode import base64_encode as ab_base64_encode
from automationbench.tools.api.fetch import api_fetch as ab_api_fetch
from automationbench.tools.api.search import api_search as ab_api_search
from automationbench.tools.api import search as api_search_module


INSTRUCTIONS = (
    "Use these tools to complete the AutomationBench workflow. Search for API "
    "endpoints with api_search, call them with api_fetch, and use base64_encode "
    "when an API requires base64url encoded content. All changes are applied to "
    "the task-local simulated world; do not use external network services."
)


def _json_default(value: Any) -> str:
    return repr(value)


def _coerce_jsonish(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=_json_default)


class ServerState:
    def __init__(self, world: WorldState, final_state_path: Path, tool_log_path: Path | None):
        self.world = world
        self.final_state_path = final_state_path
        self.tool_log_path = tool_log_path
        self.call_index = 0
        self.dump_world()

    def dump_world(self) -> None:
        self.final_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.final_state_path.with_suffix(self.final_state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.world.model_dump(mode="json"), indent=2))
        tmp_path.replace(self.final_state_path)

    def log_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.tool_log_path is None:
            return

        self.tool_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "call_index": self.call_index,
            "tool": name,
            "arguments": arguments,
            "result": result,
            "error": error,
        }
        with self.tool_log_path.open("a") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")

    def call(self, name: str, arguments: dict[str, Any], fn) -> str:
        self.call_index += 1
        try:
            result = fn()
        except Exception as exc:
            self.dump_world()
            self.log_tool(name, arguments, error=str(exc))
            raise

        self.dump_world()
        self.log_tool(name, arguments, result=result)
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutomationBench stdio MCP server")
    parser.add_argument("positional", nargs="*", help="Compatibility: <initial_state> <final_state>")
    parser.add_argument("--initial-state", dest="initial_state")
    parser.add_argument("--final-state", dest="final_state")
    parser.add_argument("--tool-log", dest="tool_log")
    parser.add_argument("--index-file", dest="index_file")
    args = parser.parse_args()

    if args.positional:
        if len(args.positional) != 2:
            parser.error("positional form is: <initial_state> <final_state>")
        if args.initial_state or args.final_state:
            parser.error("use either positional paths or --initial-state/--final-state, not both")
        args.initial_state, args.final_state = args.positional

    if not args.initial_state or not args.final_state:
        parser.error("--initial-state and --final-state are required")

    return args


def tool_schemas() -> list[dict[str, Any]]:
    jsonish_schema = {
        "anyOf": [
            {"type": "object", "additionalProperties": True},
            {"type": "string"},
            {"type": "null"},
        ]
    }
    return [
        {
            "name": "api_search",
            "description": (
                "Search available API endpoints by keyword. Use this before api_fetch "
                "to discover exact endpoint URLs, methods, params, and request bodies."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Space-separated API-native search terms.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of endpoints to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "api_fetch",
            "description": (
                "Call an API endpoint by full URL. params and body may be JSON objects "
                "or JSON strings. This mutates the task-local AutomationBench world."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method, for example GET, POST, PUT, PATCH, DELETE.",
                    },
                    "url": {"type": "string", "description": "Full endpoint URL from api_search."},
                    "params": jsonish_schema,
                    "body": jsonish_schema,
                },
                "required": ["method", "url"],
                "additionalProperties": False,
            },
        },
        {
            "name": "base64_encode",
            "description": (
                "Encode plaintext to base64url for API fields such as Gmail raw "
                "message bodies. Prefer the text argument; data is accepted as an alias."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Plaintext to encode."},
                    "data": {"type": "string", "description": "Alias for text."},
                },
                "additionalProperties": False,
            },
        },
    ]


class JsonRpcServer:
    def __init__(self, state: ServerState):
        self.state = state

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            return self.result(
                request_id,
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "automationbench", "version": "0.1.0"},
                    "instructions": INSTRUCTIONS,
                },
            )

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return self.result(request_id, {})

        if method == "tools/list":
            return self.result(request_id, {"tools": tool_schemas()})

        if method == "tools/call":
            params = request.get("params") or {}
            try:
                text = self.call_tool(
                    str(params.get("name", "")),
                    params.get("arguments") or {},
                )
                return self.result(
                    request_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
            except Exception as exc:
                return self.result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )

        if method == "shutdown":
            self.state.dump_world()
            return self.result(request_id, None)

        if request_id is None:
            return None
        return self.error(request_id, -32601, f"Method not found: {method}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "api_search":
            query = str(arguments.get("query", ""))
            top_k = int(arguments.get("top_k", 5))
            return self.state.call(
                "api_search",
                {"query": query, "top_k": top_k},
                lambda: ab_api_search(query=query, top_k=top_k),
            )

        if name == "api_fetch":
            method = str(arguments.get("method", ""))
            url = str(arguments.get("url", ""))
            params = arguments.get("params")
            body = arguments.get("body")
            return self.state.call(
                "api_fetch",
                {"method": method, "url": url, "params": params, "body": body},
                lambda: ab_api_fetch(
                    world=self.state.world,
                    method=method.upper(),
                    url=url,
                    params=_coerce_jsonish(params),
                    body=_coerce_jsonish(body),
                ),
            )

        if name == "base64_encode":
            value = arguments.get("text")
            if value is None:
                value = arguments.get("data")
            if value is None:
                raise ValueError("base64_encode requires text")
            return self.state.call(
                "base64_encode",
                {"text": arguments.get("text"), "data": arguments.get("data")},
                lambda: ab_base64_encode(text=str(value)),
            )

        raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def serve(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
            except Exception as exc:
                response = self.error(None, -32603, str(exc))
            if response is not None:
                print(json.dumps(response), flush=True)


def main() -> None:
    args = _parse_args()
    initial_state_path = Path(args.initial_state)
    final_state_path = Path(args.final_state)
    tool_log_path = Path(args.tool_log) if args.tool_log else None

    if args.index_file:
        api_search_module.INDEX_FILE = Path(args.index_file)
    else:
        api_search_module.INDEX_FILE = final_state_path.parent / "api_search_index.txt"

    with initial_state_path.open() as f:
        initial_state = json.load(f)

    state = ServerState(WorldState(**initial_state), final_state_path, tool_log_path)

    def dump_and_exit(signum: int | None = None, _frame: Any = None) -> None:
        state.dump_world()
        if signum is not None:
            raise SystemExit(0)

    atexit.register(dump_and_exit)
    signal.signal(signal.SIGTERM, dump_and_exit)
    signal.signal(signal.SIGINT, dump_and_exit)

    JsonRpcServer(state).serve()
    state.dump_world()


if __name__ == "__main__":
    main()
