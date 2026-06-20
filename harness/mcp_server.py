#!/usr/bin/env python3
"""
AutomationBench MCP Server

Exposes AutomationBench tools (api_search, api_fetch, base64_encode) and scoring
as MCP tools that any MCP-compatible agent (Devin CLI, Claude Code) can use.

Usage:
    # Start with a specific task:
    python harness/mcp_server.py --domain finance --task-index 0

    # Or specify a task by name:
    python harness/mcp_server.py --domain sales --task-name sales.multi_hop_lookup

    # List available tasks in a domain:
    python harness/mcp_server.py --domain finance --list-tasks
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
from pathlib import Path

# Add benchmarks to path so we can import automationbench
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from automationbench.domains import get_combined_dataset, get_available_domains
from automationbench.runner import strip_none_values
from automationbench.schema.world import WorldState
from automationbench.tools.api import api_search, api_fetch, base64_encode
from automationbench.rubric import partial_credit
from automationbench.rubric.registry import AssertionRegistry
import automationbench.rubric.assertions  # noqa: F401 - triggers handler registration

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# --- Global task state ---
_world: WorldState | None = None
_initial_state: dict | None = None
_task_info: dict | None = None
_task_prompt: str | None = None
_tool_call_log: list[dict] = []


def load_task(domain: str, task_index: int = 0, task_name: str | None = None) -> dict:
    """Load a task from the dataset and initialize world state."""
    global _world, _initial_state, _task_info, _task_prompt, _tool_call_log

    dataset = get_combined_dataset([domain])
    logger.info(f"Loaded {len(dataset)} tasks from domain '{domain}'")

    if task_name:
        # Find by name
        for i, row in enumerate(dataset):
            if row["task"] == task_name:
                task_index = i
                break
        else:
            available = [row["task"] for row in dataset]
            raise ValueError(f"Task '{task_name}' not found. Available: {available}")

    if task_index >= len(dataset):
        raise ValueError(f"Task index {task_index} out of range (max {len(dataset) - 1})")

    task = dataset[task_index]
    logger.info(f"Loading task: {task['task']} (index {task_index})")

    # Parse info
    info = task.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)

    # Initialize world state
    initial_state_dict = strip_none_values(info.get("initial_state", {}))
    _world = WorldState(**initial_state_dict)
    _initial_state = copy.deepcopy(initial_state_dict)
    _task_info = info

    # Extract prompt
    prompt = task.get("prompt", [])
    if isinstance(prompt, list):
        # Combine system + user messages into a readable prompt
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"[System]\n{content}")
                elif role == "user":
                    parts.append(f"[Task]\n{content}")
            elif isinstance(msg, str):
                parts.append(msg)
        _task_prompt = "\n\n".join(parts)
    else:
        _task_prompt = str(prompt)

    _tool_call_log = []

    return {
        "task_name": task["task"],
        "task_index": task_index,
        "domain": domain,
        "prompt": _task_prompt,
        "num_assertions": len(info.get("assertions", [])),
    }


# --- MCP Server ---
server = Server("automationbench")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    tools = [
        Tool(
            name="get_task",
            description=(
                "Get the current task prompt and instructions. "
                "Call this first to understand what you need to do."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="api_search",
            description=(
                "Search available API endpoints by keyword. "
                "Use this to discover which endpoint to call before using api_fetch. "
                "Ranks results using BM25 over endpoint descriptions. "
                "Returns full endpoint details including the URL to pass to api_fetch."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Space-separated keywords (e.g. 'inbox messages', 'trash', 'send label'). "
                            "Uses API-native terms: 'messages' not 'emails', 'trash' not 'delete'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="api_fetch",
            description=(
                "Call an API endpoint by its full URL, routing to the appropriate service. "
                "Use api_search first to discover the correct URL and parameters for an endpoint."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, PATCH, DELETE).",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "Full API URL from api_search results "
                            "(e.g. 'https://api.hubapi.com/crm/v3/objects/contacts')."
                        ),
                    },
                    "params": {
                        "type": "string",
                        "description": "Query parameters as a JSON string (e.g. '{\"labelIds\": \"INBOX\"}').",
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body as a JSON string.",
                    },
                },
                "required": ["method", "url"],
            },
        ),
        Tool(
            name="base64_encode",
            description=(
                "Encode text to base64url — the format Gmail API body fields require. "
                "Use this before passing email body content to api_fetch for Gmail endpoints."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Plaintext to encode (email body or full RFC 2822 message).",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="score",
            description=(
                "Check your current score — how many assertions pass against the current world state. "
                "Call this when you think you've completed the task to see your result."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    global _world, _tool_call_log

    if _world is None and name != "get_task":
        return [TextContent(type="text", text="Error: No task loaded. The server was not started with a task.")]

    if name == "get_task":
        if _task_prompt is None:
            return [TextContent(type="text", text="Error: No task loaded.")]
        return [TextContent(type="text", text=_task_prompt)]

    elif name == "api_search":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        result = api_search(query=query, top_k=top_k)
        _tool_call_log.append({"tool": "api_search", "args": {"query": query, "top_k": top_k}})
        return [TextContent(type="text", text=result)]

    elif name == "api_fetch":
        method = arguments.get("method", "GET")
        url = arguments.get("url", "")
        params = arguments.get("params")
        body = arguments.get("body")
        result = api_fetch(world=_world, method=method, url=url, params=params, body=body)
        _tool_call_log.append({"tool": "api_fetch", "args": {"method": method, "url": url, "params": params, "body": body}})
        return [TextContent(type="text", text=result)]

    elif name == "base64_encode":
        text = arguments.get("text", "")
        result = base64_encode(text=text)
        _tool_call_log.append({"tool": "base64_encode", "args": {"text": text[:50] + "..."}})
        return [TextContent(type="text", text=result)]

    elif name == "score":
        state = {
            "world": _world,
            "info": _task_info,
            "initial_state": _initial_state,
        }
        score = partial_credit(state)
        assertion_results = state.get("_assertion_results", [])

        # Format results
        lines = [f"Score: {score:.2%}"]
        lines.append(f"Tool calls made: {len(_tool_call_log)}")
        lines.append("")
        lines.append("Assertion results:")
        for i, ar in enumerate(assertion_results, 1):
            status = "EXCLUDED" if ar.get("excluded") else ("PASS" if ar["passed"] else "FAIL")
            lines.append(f"  {i}. [{status}] {ar['type']} {ar.get('params', {})}")

        passed = sum(1 for ar in assertion_results if ar["passed"] and not ar.get("excluded"))
        total = sum(1 for ar in assertion_results if not ar.get("excluded"))
        lines.append(f"\nPassed: {passed}/{total}")

        if score == 1.0:
            lines.append("\n*** TASK COMPLETED SUCCESSFULLY ***")

        return [TextContent(type="text", text="\n".join(lines))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    parser = argparse.ArgumentParser(description="AutomationBench MCP Server")
    parser.add_argument("--domain", type=str, default=None, help="Task domain")
    parser.add_argument("--task-index", type=int, default=None, help="Task index within domain")
    parser.add_argument("--task-name", type=str, default=None, help="Task name (overrides --task-index)")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--list-domains", action="store_true", help="List available domains and exit")

    args = parser.parse_args()

    # Env vars override CLI args (for batch runner)
    args.domain = os.environ.get("AB_DOMAIN", args.domain) or "finance"
    if args.task_index is None:
        args.task_index = int(os.environ.get("AB_TASK_INDEX", "0"))
    args.task_name = os.environ.get("AB_TASK_NAME", args.task_name)

    if args.list_domains:
        domains = get_available_domains()
        print("Available domains:")
        for d in domains:
            print(f"  - {d}")
        return

    if args.list_tasks:
        dataset = get_combined_dataset([args.domain])
        print(f"Tasks in '{args.domain}' ({len(dataset)} total):")
        for i, row in enumerate(dataset):
            print(f"  {i}: {row['task']}")
        return

    # Load the task
    task_info = load_task(
        domain=args.domain,
        task_index=args.task_index,
        task_name=args.task_name,
    )
    logger.info(f"Task loaded: {task_info['task_name']}")
    logger.info(f"Assertions: {task_info['num_assertions']}")

    # Run MCP server over stdio
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
