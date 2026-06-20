# AutomationBench SFT Update - 2026-06-20

## Summary

We completed an end-to-end Qwen3.5 9B SFT run on the AutomationBench trace corpus and evaluated the base model and the trained checkpoint with the same local verifier path.

On the fixed 60-task evaluation slice, average partial credit improved from `0.1716` to `0.3351` (+`0.1635`, `1.95x`). Strict pass rate moved from `0/60` to `7/60` (+`11.7` percentage points).

This should be treated as a loop-validating result, not a frontier benchmark result: the evaluation command used `--domains all --num-examples 60`, which currently selects the first 60 combined tasks, all from `sales.*`. The SFT corpus is also derived from AutomationBench traces, so the next result should use a held-out or stratified split before we make generalization claims.

## Before/After Eval

| Model | Checkpoint | Avg partial | Strict pass rate | Passed | Empty responses | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B base | `Qwen/Qwen3.5-9B` | `0.1716` | `0.0%` | `0/60` | `0` | `0` |
| Qwen3.5 9B SFT | `step_200` | `0.3351` | `11.7%` | `7/60` | `0` | `0` |

Artifacts on the cluster:

- Baseline JSON: `/data/alex/dev/self-improving-harness/runs/qwen35_9b_base_all60.json`
- SFT JSON: `/data/alex/dev/self-improving-harness/runs/qwen35_9b_sft64k200_all60.json`
- Final checkpoint: `/data/alex/dev/self-improving-harness/prime-rl/outputs/automationbench-sft-qwen35-64k-200step-full-cp1-lr2e5/weights/step_200`

Eval settings:

- `--domains all`
- `--num-examples 60`
- `--toolset api`
- `--max-steps 50`
- `--max-concurrent 4`
- `--max-model-len 65536`
- Qwen3.5 processor/parser with Prime-RL/vLLM serving and Triton GDN prefill backend

## Training Run

The stable full-data run used the 64k-token prepared dataset:

- Dataset: `data/automationbench_sft_qwen3_65536`
- Selected traces: `526`
- Train/validation split: `500` / `26`
- Skipped for length: `160` traces over 64k tokens
- Base model: `Qwen/Qwen3.5-9B`
- Sequence length: `65536`
- Hardware: `8x H100`
- Context parallelism: `1`
- Global batch size: `8`
- Micro batch size: `1`
- Learning rate: `2e-5`
- Steps: `200`
- Loss implementation: `liger_fused`
- Attention: `flash_attention_3`
- W&B run: `https://wandb.ai/alexandonian/automationbench-sft/runs/mkhjj1ge`

The job completed successfully:

- Slurm job: `14987`
- Exit code: `0:0`
- Elapsed: `00:38:22`
- Final `loss/mean`: `0.00296`
- `loss/nan_count`: `0`
- Final `optim/grad_norm`: `0.80859`
- Peak memory: `50.4 GiB`
- Training log: `/data/alex/dev/self-improving-harness/logs/sft-14987.out`

## 131k Context Note

We also attempted a 131k-token full selected run. The CP=1 path failed on the first backward pass with a Triton CUDA OOM in the FLA gated delta rule backward. Earlier context-parallel variants showed NaNs, so the stable production run for this update used the 64k CP=1 path.

## Next Steps

1. Run a held-out or task-stratified eval split to separate training-trace memorization from generalization.
2. Run a broader AutomationBench sweep across domains after the fixed-slice path is validated.
3. Evaluate intermediate checkpoints (`step_50`, `step_100`, `step_150`) because the final loss is already very low at `step_200`.
4. Revisit 131k training after the context-parallel label/loss path is hardened.
