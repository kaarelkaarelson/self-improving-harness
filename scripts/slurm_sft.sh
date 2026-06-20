#!/usr/bin/env bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/alex/dev/self-improving-harness}"
CONFIG="${CONFIG:-data/automationbench_sft_qwen3_65536/sft.toml}"
PRIVATE_ENV="${PRIVATE_ENV:-$HOME/.config/self-improving-harness/sft.env}"

cd "$REPO_DIR"
mkdir -p logs

if [[ -f "$PRIVATE_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$PRIVATE_ENV"
  set +a
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Put it in $PRIVATE_ENV or export it before sbatch." >&2
  exit 2
fi

if [[ -f "$REPO_DIR/prime-rl/.venv/bin/activate" ]]; then
  # Ensure trainer subprocesses resolve python/torchrun from Prime-RL's env.
  # The sft entrypoint launches torchrun internally.
  source "$REPO_DIR/prime-rl/.venv/bin/activate"
fi

python3 scripts/run_prime_rl_sft.py \
  --prime-rl-dir prime-rl \
  --config "$CONFIG" \
  --ckpt \
  --wandb
