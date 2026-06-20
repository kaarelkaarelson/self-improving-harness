#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/alex/dev/self-improving-harness}"
CHECKPOINT="${1:?Usage: sbatch scripts/slurm_checkpoint_eval.sh <checkpoint-path> [deploy_checkpoint_eval.py args...]}"
shift

cd "$REPO_DIR"
mkdir -p logs

if [[ -f "$REPO_DIR/prime-rl/.venv/bin/activate" ]]; then
  source "$REPO_DIR/prime-rl/.venv/bin/activate"
fi

export PYTHONPATH="$REPO_DIR/scripts/prime_rl_compat${PYTHONPATH:+:$PYTHONPATH}"

python3 scripts/deploy_checkpoint_eval.py \
  --prime-rl-dir prime-rl \
  --checkpoint "$CHECKPOINT" \
  "$@"
