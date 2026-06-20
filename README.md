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

## Benchmarks

Uses [Zapier AutomationBench](https://github.com/zapier/AutomationBench) (MIT, submodule in `benchmarks/`). 600 tasks across 6 domains (Sales, Marketing, Operations, Support, Finance, HR) with 47 simulated SaaS tools.
