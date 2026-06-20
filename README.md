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

## Running Evals (Prime Intellect)

Always use `--hosted` -- local runs crash on a known verifiers ZMQ bug.

```bash
source .env && export PRIME_API_KEY

# Single finance task, 1 rollout
prime eval run zapier/AutomationBench -m openai/gpt-5.5 --num-examples 1 --rollouts-per-example 1 --hosted --env-args '{"domains": "finance"}'

# 5 finance tasks
prime eval run zapier/AutomationBench -m openai/gpt-5.5 --num-examples 5 --rollouts-per-example 1 --hosted --env-args '{"domains": "finance"}'

# Full eval, all domains (600 tasks)
prime eval run zapier/AutomationBench -m openai/gpt-5.5 --rollouts-per-example 1 --hosted

# Follow logs
prime eval logs <EVAL_ID> -f
```

Results appear at `https://app.primeintellect.ai/dashboard/evaluations/<EVAL_ID>`.

## Running Evals (Local, no Prime account needed)

```bash
cd benchmarks && uv sync
export OPENAI_API_KEY=sk-...
uv run auto-bench --model gpt-5-mini --domains finance --tasks finance.invoice_email_extract
```

## Gotchas

- **"Insufficient balance"** -- add funds at [Prime billing](https://app.primeintellect.ai/dashboard/billing) under the `self-improving harness` team (not personal).
- **Local `prime eval run` crashes** with `'NoneType' object has no attribute 'get'` -- known ZMQ bug in verifiers 0.1.14. Use `--hosted` or run via the repo directly (`uv run auto-bench`).
- **`uv` version** -- needs 0.11+ for Prime CLI. Run `curl -LsSf https://astral.sh/uv/install.sh | sh` to update.

## Benchmarks

Uses [Zapier AutomationBench](https://github.com/zapier/AutomationBench) (MIT, submodule in `benchmarks/`). 600 tasks across 6 domains (Sales, Marketing, Operations, Support, Finance, HR) with 47 simulated SaaS tools.
