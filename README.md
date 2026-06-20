# Self-Improving Harness

Harness for evaluating and improving AI coding agents on realistic business workflows.

## Setup

```bash
git clone --recurse-submodules https://github.com/kaarelkaarelson/self-improving-harness.git
cd self-improving-harness
cp .env.example .env               # fill in your PRIME_API_KEY
uv tool install -U prime
prime login                        # select "self-improving harness" team
```

## Running Evals

```bash
source .env && export PRIME_API_KEY
prime eval run zapier/AutomationBench -m openai/gpt-5.5 --num-examples 1 --rollouts-per-example 1 --hosted --env-args '{"domains": "finance"}'
```

## Running Evals (Local, no Prime account needed)

```bash
cd benchmarks && uv sync
export OPENAI_API_KEY=sk-...
uv run auto-bench --model gpt-5-mini --domains finance --tasks finance.invoice_email_extract
```

## Running with a Coding Agent (Devin / Claude Code)

Instead of hitting a model API directly, you can run tasks through a coding agent via MCP. The agent sees the tools natively and its full reasoning trace is captured in the session history.

```bash
cd benchmarks && uv sync   # one-time setup
```

**With Devin CLI** (config already in `.devin/config.json`):

```bash
devin
# Then: "Use the automationbench tools. Call get_task first to see the task,
#        solve it using api_search and api_fetch, then call score."
```

**With Claude Code** (add to `.claude/mcp.json`):

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

**Change the task** by editing the `--domain` and `--task-index` (or `--task-name`) in the MCP config:

```bash
# List available tasks:
cd benchmarks && uv run python ../harness/mcp_server.py --domain sales --list-tasks

# Example: run sales.multi_hop_lookup
# Set args to: [..., "--domain", "sales", "--task-name", "sales.multi_hop_lookup"]
```

The agent gets 5 tools: `get_task`, `api_search`, `api_fetch`, `base64_encode`, `score`. After solving, `score` reports pass/fail per assertion. Full traces live in the agent's session history (`devin --export` or Claude Code JSONL).

## Gotchas

- **"Insufficient balance"** -- add funds at [Prime billing](https://app.primeintellect.ai/dashboard/billing) under the `self-improving harness` team (not personal).
- **Local `prime eval run` crashes** with `'NoneType' object has no attribute 'get'` -- known ZMQ bug in verifiers 0.1.14. Use `--hosted` or run via the repo directly (`uv run auto-bench`).
- **`uv` version** -- needs 0.11+ for Prime CLI. Run `curl -LsSf https://astral.sh/uv/install.sh | sh` to update.

## Benchmarks

Uses [Zapier AutomationBench](https://github.com/zapier/AutomationBench) (MIT, submodule in `benchmarks/`). 600 tasks across 6 domains (Sales, Marketing, Operations, Support, Finance, HR) with 47 simulated SaaS tools.
