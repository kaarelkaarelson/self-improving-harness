# AutomationBench Trace Coverage Share

This bundle is the small, Discord-friendly subset of the VP repo that is useful for finding AutomationBench tasks where we need better traces. It intentionally excludes raw trace dumps, model checkpoints, `prime-rl`, caches, and env/API key files.

## Open These First

- `artifacts/trace_coverage_2026-06-20/tasks_need_help.csv`
  - Best file for teammates to inspect.
  - One row per task that needs more or better data.
  - Tasks are included when they have no good trace, no strict-success trace, or fewer than the diversity target of good traces/models.
- `artifacts/trace_coverage_2026-06-20/gaps.md`
  - Human-readable summary of the biggest coverage gaps.
- `artifacts/trace_coverage_2026-06-20/task_coverage.csv`
  - Full per-task coverage table for all 600 AutomationBench tasks.
- `artifacts/trace_coverage_2026-06-20/category_summary.csv`
  - Coverage by AutomationBench category.
- `artifacts/trace_coverage_2026-06-20/model_summary.csv`
  - Coverage by source model/eval.
- `artifacts/trace_coverage_2026-06-20/manifest.json`
  - Generation settings and top-level counts.

## Current Coverage Settings

- Source input: `runs/prime_leaderboard_all.jsonl`
- Good trace threshold: `partial_credit >= 0.5`
- Diversity target: 5 good traces/models per task
- Total tasks: 600
- Tasks needing attention: 416
- Tasks with no good trace: 45
- Tasks with no strict-success trace: 373

## Included Scripts

- `scripts/download_prime_traces.py`
  - Pulls leaderboard/eval traces from Prime Intellect when `PRIME_API_KEY` is available.
- `scripts/normalize_traces.py`
  - Normalizes downloaded traces into the common trace JSONL format.
- `scripts/summarize_trace_coverage.py`
  - Produces `tasks_need_help.csv`, `task_coverage.csv`, summaries, and `gaps.md`.
- `scripts/prepare_prime_rl_sft.py`
  - Filters normalized traces into Prime-RL SFT train/validation data.
- `scripts/summarize_eval_splits.py`
  - Computes full/train/heldout summaries from one eval JSON and split task files.
- `scripts/run_eval.py` and `scripts/deploy_checkpoint_eval.py`
  - Eval/deployment helpers for AutomationBench model runs.

## Regenerate Coverage

From the repo root, after the leaderboard traces are available:

```bash
python3 scripts/summarize_trace_coverage.py \
  --input-jsonl runs/prime_leaderboard_all.jsonl \
  --output-dir artifacts/trace_coverage_2026-06-20 \
  --good-threshold 0.5 \
  --diversity-target 5
```

## What Is Not Included

- Raw traces and normalized SFT datasets, because they can be large.
- Checkpoints and training outputs.
- `.env`, W&B config, API keys, SSH config, or Hugging Face cache.
- The `prime-rl` repo.
