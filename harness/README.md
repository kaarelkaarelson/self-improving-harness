# AutomationBench Agent Harness

Run AutomationBench tasks with any MCP-compatible coding agent (Devin CLI, Claude Code).

## How it works

```
┌─────────────────────────────────┐
│  Coding Agent (Devin/Claude)    │
│  - Calls tools naturally        │
│  - Full reasoning traces saved  │
└────────────┬────────────────────┘
             │ MCP protocol (stdio)
             ▼
┌─────────────────────────────────┐
│  MCP Server (this harness)      │
│  Tools: get_task, api_search,   │
│  api_fetch, base64_encode,      │
│  score                          │
└────────────┬────────────────────┘
             │ Direct Python calls
             ▼
┌─────────────────────────────────┐
│  AutomationBench WorldState     │
│  (in-memory, mutated by tools)  │
└─────────────────────────────────┘
```

The agent sees 5 tools:
- **get_task** — Returns the task prompt/instructions
- **api_search** — Search available API endpoints by keyword
- **api_fetch** — Call an API endpoint (mutates world state)
- **base64_encode** — Encode text for Gmail API
- **score** — Check assertions against current world state

## Setup

```bash
# From the repo root, install benchmarks deps:
cd benchmarks && uv sync

# Verify it works:
uv run python ../harness/mcp_server.py --list-domains
uv run python ../harness/mcp_server.py --domain finance --list-tasks
```

## Usage with Devin CLI

Add to your project config (`.devin/config.json`):

```json
{
  "mcpServers": {
    "automationbench": {
      "command": "uv",
      "args": ["run", "--directory", "./benchmarks", "python", "../harness/mcp_server.py", "--domain", "finance", "--task-index", "0"],
      "cwd": "."
    }
  }
}
```

Then start Devin and tell it:

> Use the automationbench tools. Call `get_task` first to see the task, then solve it using `api_search` and `api_fetch`. When done, call `score` to check your result.

## Usage with Claude Code

Add to `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "automationbench": {
      "command": "uv",
      "args": ["run", "--directory", "./benchmarks", "python", "../harness/mcp_server.py", "--domain", "finance", "--task-index", "0"]
    }
  }
}
```

## Changing the task

Edit the `--domain` and `--task-index` (or `--task-name`) args:

```bash
# By index:
--domain sales --task-index 0

# By name:
--domain sales --task-name sales.multi_hop_lookup

# List what's available:
uv run python ../harness/mcp_server.py --domain sales --list-tasks
```

## Available domains

| Domain | Tasks |
|--------|-------|
| sales | 100 |
| marketing | 100 |
| operations | 100 |
| support | 100 |
| finance | 100 |
| hr | 100 |

## What the agent trace shows

Since the agent uses MCP tools naturally, its full session trace includes:
- Every reasoning step
- Every tool call (api_search, api_fetch) with arguments and results
- The final score

In Devin: `devin --export` gives you the full ATIF trace.
In Claude Code: The JSONL session file has everything.

## Scoring

The `score` tool returns:
- Partial credit (0.0–1.0): fraction of assertions passed
- Per-assertion breakdown (PASS/FAIL/EXCLUDED)
- Total tool calls made

A score of 1.0 means the task was completed perfectly.
