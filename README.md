# Self-Improving Harness

Harness for evaluating and improving AI coding agents on realistic business workflows.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/kaarelkaarelson/self-improving-harness.git
cd self-improving-harness/benchmarks
uv sync
export OPENAI_API_KEY=sk-...
uv run auto-bench --model gpt-5-mini --domains finance --tasks finance.invoice_email_extract
```

## Prime Intellect

Run via [Prime Intellect](https://app.primeintellect.ai) under the `self-improving harness` team:

```bash
uv tool install -U prime
prime login                        # select "self-improving harness" team

# Eval (smoke test, 1 task)
prime eval run zapier/AutomationBench --num-examples 1 --env-args '{"domains": "finance"}'

# Full eval
prime eval run zapier/AutomationBench

# Hosted eval (Prime manages infra)
prime eval run zapier/AutomationBench --hosted
```

## Benchmarks

Uses [Zapier AutomationBench](https://github.com/zapier/AutomationBench) (MIT, submodule in `benchmarks/`). 600 tasks across 6 domains (Sales, Marketing, Operations, Support, Finance, HR) with 47 simulated SaaS tools.
