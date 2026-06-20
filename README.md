# Self-Improving Harness

Harness for evaluating and improving agents on AutomationBench workflows.

The repo supports two eval paths:

- **Built-in AutomationBench runner**: `auto-bench` drives a model through the benchmark's OpenAI/Anthropic-compatible tool loop.
- **Native coding-agent runner**: Codex, Claude Code, or Devin run their own loop and reach AutomationBench through a local stdio MCP bridge.

Both paths mutate only AutomationBench's simulated API world. Generated traces are normalized into one SFT format for Prime-RL.

## Setup

```bash
git clone --recurse-submodules https://github.com/kaarelkaarelson/self-improving-harness.git
cd self-improving-harness
cp .env.example .env               # fill in PRIME_API_KEY when using Prime APIs
uv tool install -U prime
prime login                        # select the "self-improving harness" team
```

Install AutomationBench dependencies:

```bash
cd benchmarks
uv sync
```

Prime-RL is expected as the `prime-rl/` submodule:

```bash
git submodule update --init --recursive prime-rl
cd prime-rl
uv sync
```

## Running Evals

Run the built-in AutomationBench/verifiers runner against a hosted or local OpenAI-compatible model:

```bash
python scripts/run_eval.py verifiers \
  --model gpt-5-mini \
  --domains finance \
  --tasks finance.invoice_email_extract \
  --num-examples 1 \
  --toolset api
```

For an SFT checkpoint served from Prime-RL:

```bash
python scripts/run_eval.py verifiers \
  --model automationbench-sft \
  --base-url http://localhost:8000/v1 \
  --domains finance \
  --num-examples 5 \
  --toolset api
```

Run a native coding agent through the MCP bridge:

```bash
python scripts/run_eval.py native \
  --agent codex \
  --domains finance \
  --tasks finance.invoice_email_extract
```

Supported native agents are `codex`, `claude`, `devin`, and `noop`. Each native run writes `runs/<timestamp>_<agent>/results.json` plus per-task prompt, initial/final world state, MCP tool log, stdout/stderr, and agent-native export when available.

## MCP Servers

There are two MCP servers:

- `harness/mcp_server.py` is interactive: it loads a task itself and exposes `get_task`, `api_search`, `api_fetch`, `base64_encode`, and `score`.
- `harness/mcp_bridge_server.py` is the native eval bridge: it receives a per-task `initial_state.json`, exposes only AutomationBench API tools, dumps `final_state.json` after every call, and keeps `OPENAI_API_KEY` out of the server process.

Use the interactive server directly when experimenting with an agent:

```bash
cd benchmarks
uv run --with mcp python ../harness/mcp_server.py --domain finance --list-tasks
uv run --with mcp python ../harness/mcp_server.py --domain finance --task-index 0
```

The native eval runner uses `harness/mcp_bridge_server.py` automatically.

## Trace Collection

Download public Prime leaderboard traces:

```bash
set -a
source ~/.env
set +a

python scripts/download_prime_traces.py \
  --output-dir runs/prime_leaderboard \
  --output-jsonl runs/prime_leaderboard_all.jsonl
```

Normalize local native runs:

```bash
python scripts/normalize_traces.py \
  --runs runs/<codex-run> runs/<claude-run> runs/<devin-run> \
  --output runs/native_sft.jsonl \
  --min-partial-credit 0.5 \
  --require-agent-success
```

Leaderboard and native traces are normalized to the same tool-call/message shape for SFT.

## Preparing SFT Data

Export the best traces into a local Hugging Face dataset directory consumable by Prime-RL:

```bash
python scripts/prepare_prime_rl_sft.py \
  --input-jsonl runs/prime_leaderboard_all.jsonl runs/native_sft.jsonl \
  --output-dir data/automationbench_sft_qwen35_64k \
  --min-partial-credit 0.5 \
  --top-k-per-task 1 \
  --validation-fraction 0.05 \
  --max-rendered-tokens 65536 \
  --workers 8 \
  --base-model Qwen/Qwen3.5-9B \
  --seq-len 65536 \
  --batch-size 2
```

The output includes:

- `train.jsonl` and optional `validation.jsonl`.
- `selected_traces.jsonl` for auditability.
- `manifest.json` with filter counts and source metadata.
- `sft.toml`, a Prime-RL training config pointing at the local dataset.

Run SFT:

```bash
python scripts/run_prime_rl_sft.py \
  --prime-rl-dir prime-rl \
  --config data/automationbench_sft_qwen35_64k/sft.toml \
  --ckpt
```

Serve a checkpoint for re-eval:

```bash
cd prime-rl
uv run inference --model.name outputs/automationbench-sft/weights/step_<N> --server.port 8000
```

## Transfer To Cluster

Package downloaded raw traces and prepared datasets:

```bash
python scripts/package_artifacts.py \
  --inputs runs/prime_leaderboard_full runs/prime_leaderboard_all.jsonl data/automationbench_sft_qwen3_65536 \
  --output artifacts/automationbench_traces_64k.tar.zst
```

Copy to Voltage Park and clone the repo:

```bash
ssh vp 'mkdir -p /data/alex/dev'
ssh vp 'cd /data/alex/dev && git clone --recurse-submodules https://github.com/kaarelkaarelson/self-improving-harness.git'
scp artifacts/automationbench_traces_64k.tar.zst vp:/data/alex/dev/
```

For cluster training, keep W&B credentials out of git. Put the personal key in a
private env file on the cluster:

```bash
mkdir -p ~/.config/self-improving-harness
cat > ~/.config/self-improving-harness/sft.env <<'EOF'
WANDB_API_KEY=...
WANDB_ENTITY=alexandonian
WANDB_PROJECT=automationbench-sft
EOF
chmod 600 ~/.config/self-improving-harness/sft.env
```

Submit the generic `sft` Slurm job:

```bash
cd /data/alex/dev/self-improving-harness
sbatch scripts/slurm_sft.sh
```

## Gotchas

- **Local `prime eval run` crashes** with `'NoneType' object has no attribute 'get'` in verifiers 0.1.14. Use hosted Prime evals or run AutomationBench locally through `scripts/run_eval.py verifiers`.
- **Native MCP server env**: `harness/mcp_bridge_server.py` removes `OPENAI_API_KEY` from its own process so simulated helper paths stay offline. Use `CODEX_API_KEY` or saved agent auth for the coding agent itself.
- **API search index**: native tasks write the BM25 index inside each task run directory, not inside the AutomationBench submodule.
- **Grading**: native evals pass `initial_state` into the rubric so AutomationBench's free-assertion exclusion is preserved.

## Benchmarks

Uses [Zapier AutomationBench](https://github.com/zapier/AutomationBench) as the `benchmarks/` submodule. AutomationBench has 600 tasks across sales, marketing, operations, support, finance, and HR.
