#!/usr/bin/env python3
"""Launch Prime-RL SFT with a generated AutomationBench config."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prime-RL SFT from a generated sft.toml")
    parser.add_argument("--config", required=True, help="Path to sft.toml from prepare_prime_rl_sft.py")
    parser.add_argument("--prime-rl-dir", default=str(ROOT / "prime-rl"))
    parser.add_argument("--ckpt", action="store_true", help="Enable Prime-RL checkpoint/weights output")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()

    config = Path(args.config).resolve()
    prime_rl_dir = Path(args.prime_rl_dir).resolve()
    if not config.exists():
        raise SystemExit(f"SFT config does not exist: {config}")
    if not prime_rl_dir.exists():
        raise SystemExit(f"Prime-RL directory does not exist: {prime_rl_dir}")

    sft_entrypoint = prime_rl_dir / ".venv" / "bin" / "sft"
    if sft_entrypoint.exists():
        command = [str(sft_entrypoint), "@", str(config)]
    else:
        command = ["uv", "run", "sft", "@", str(config)]
    if args.ckpt:
        command.append("--ckpt")
    if args.wandb:
        command.append("--wandb")
    command += extra_args

    print("Command:", flush=True)
    print(" ".join(command), flush=True)
    print(f"CWD: {prime_rl_dir}", flush=True)
    if args.dry_run:
        return

    completed = subprocess.run(command, cwd=prime_rl_dir, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
