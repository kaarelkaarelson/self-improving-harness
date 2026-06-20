#!/usr/bin/env python3
"""Serve a Prime-RL checkpoint and run AutomationBench against it."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def default_run_dir(checkpoint: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = checkpoint.rstrip("/").replace("/", "_").replace(":", "_")
    return ROOT / "runs" / "checkpoint_eval" / f"{stamp}_{safe_name}"


def checkpoint_path(args: argparse.Namespace) -> str:
    return str(Path(args.checkpoint).resolve() if args.resolve_checkpoint else args.checkpoint)


def maybe_copy_processor_assets(args: argparse.Namespace) -> None:
    if not args.processor_source:
        return
    target = Path(checkpoint_path(args))
    if not target.exists():
        return
    if (target / "preprocessor_config.json").exists() or (target / "processor_config.json").exists():
        return

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_source, trust_remote_code=args.trust_remote_code)
    processor.save_pretrained(target)


def inference_command(args: argparse.Namespace) -> list[str]:
    prime_rl_dir = Path(args.prime_rl_dir).resolve()
    inference_entrypoint = prime_rl_dir / ".venv" / "bin" / "inference"
    if inference_entrypoint.exists():
        command = [str(inference_entrypoint)]
    else:
        command = ["uv", "run", "inference"]

    command += [
        "--model.name",
        checkpoint_path(args),
        "--server.host",
        args.bind_host,
        "--server.port",
        str(args.port),
        "--parallel.tp",
        str(args.tensor_parallel_size),
        "--model.dtype",
        args.dtype,
    ]
    if args.max_model_len is not None:
        command += ["--model.max-model-len", str(args.max_model_len)]
    if args.enforce_eager:
        command.append("--model.enforce-eager")
    command += args.inference_args
    return command


def wait_for_models(base_url: str, timeout_sec: int, process: subprocess.Popen[str]) -> list[str]:
    deadline = time.monotonic() + timeout_sec
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Inference server exited before readiness with code {process.returncode}: {last_error}")
        try:
            with urlopen(f"{base_url}/models", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(models, list):
                ids = [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
                if ids:
                    return ids
                last_error = f"{base_url}/models returned no model ids"
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    raise TimeoutError(f"Inference server did not become ready within {timeout_sec}s: {last_error}")


def eval_command(args: argparse.Namespace, *, base_url: str, model: str, output_json: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_eval.py"),
        "verifiers",
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key",
        args.api_key,
        "--api",
        args.api,
        "--domains",
        args.domains,
        "--num-examples",
        str(args.num_examples),
        "--max-steps",
        str(args.max_steps),
        "--toolset",
        args.toolset,
        "--max-concurrent",
        str(args.max_concurrent),
        "--output-json",
        str(output_json),
        "--benchmarks-dir",
        str(Path(args.benchmarks_dir).resolve()),
    ]
    if args.tasks:
        command += ["--tasks", args.tasks]
    if args.search_top_k is not None:
        command += ["--search-top-k", str(args.search_top_k)]
    command += args.eval_args
    return command


def terminate_process(process: subprocess.Popen[str], timeout_sec: int = 30) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=timeout_sec)


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path | None, dry_run: bool) -> int:
    print("Command:", flush=True)
    print(" ".join(command), flush=True)
    print(f"CWD: {cwd}", flush=True)
    if dry_run:
        return 0
    if log_path is None:
        completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a Prime-RL checkpoint and run AutomationBench eval")
    parser.add_argument("--checkpoint", required=True, help="HF/Prime-RL weight checkpoint directory to serve")
    parser.add_argument("--prime-rl-dir", default=str(ROOT / "prime-rl"))
    parser.add_argument("--run-dir", help="Directory for server logs and eval output")
    parser.add_argument("--bind-host", default="0.0.0.0", help="Host passed to Prime-RL inference")
    parser.add_argument("--host", default="127.0.0.1", help="Host used by the eval client")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wait-timeout-sec", type=int, default=900)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--processor-source",
        help="Optional source model/path to copy processor assets into a local checkpoint before serving.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-resolve-checkpoint",
        action="store_false",
        dest="resolve_checkpoint",
        help="Pass --checkpoint through exactly instead of resolving to an absolute path.",
    )
    parser.set_defaults(resolve_checkpoint=True)
    parser.add_argument("--eval-model", help="Model id sent to AutomationBench. Defaults to first /v1/models id.")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--tasks")
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--toolset", default="api", choices=["api", "zapier", "limited_zapier"])
    parser.add_argument("--api", default="chat_completions", choices=["auto", "anthropic", "chat_completions", "responses"])
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--search-top-k", type=int)
    parser.add_argument("--benchmarks-dir", default=str(ROOT / "benchmarks"))
    parser.add_argument("--output-json", help="Eval JSON path. Defaults under --run-dir.")
    parser.add_argument("--serve-only", action="store_true", help="Start server and wait for readiness, then block.")
    parser.add_argument("--keep-server", action="store_true", help="Leave inference server running after eval.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--inference-arg",
        action="append",
        default=[],
        dest="inference_args",
        help="Extra argument forwarded to Prime-RL inference. Repeat for flag and value.",
    )
    parser.add_argument(
        "--eval-arg",
        action="append",
        default=[],
        dest="eval_args",
        help="Extra argument forwarded to scripts/run_eval.py verifiers. Repeat for flag and value.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir(args.checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json).resolve() if args.output_json else run_dir / "automationbench_eval.json"
    base_url = f"http://{args.host}:{args.port}/v1"
    prime_rl_dir = Path(args.prime_rl_dir).resolve()
    env = os.environ.copy()
    compat_path = str(ROOT / "scripts" / "prime_rl_compat")
    env["PYTHONPATH"] = f"{compat_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else compat_path

    command = inference_command(args)
    print("Inference command:", flush=True)
    print(" ".join(command), flush=True)
    print(f"Inference log: {run_dir / 'inference.log'}", flush=True)
    if args.dry_run:
        model = args.eval_model or args.checkpoint
        eval_cmd = eval_command(args, base_url=base_url, model=model, output_json=output_json)
        return run_command(eval_cmd, cwd=ROOT, env=env, log_path=None, dry_run=True)

    maybe_copy_processor_assets(args)

    log = (run_dir / "inference.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=prime_rl_dir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; stopping inference server.", flush=True)
        terminate_process(process)
        log.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        models = wait_for_models(base_url, args.wait_timeout_sec, process)
        model = args.eval_model or models[0]
        print(f"Inference ready at {base_url}; model={model}", flush=True)
        if args.serve_only:
            print("Serve-only mode; press Ctrl-C to stop.", flush=True)
            while process.poll() is None:
                time.sleep(30)
            return process.returncode or 0

        eval_cmd = eval_command(args, base_url=base_url, model=model, output_json=output_json)
        eval_log = run_dir / "eval.log"
        rc = run_command(eval_cmd, cwd=ROOT, env=env, log_path=eval_log, dry_run=False)
        print(f"Eval log: {eval_log}", flush=True)
        print(f"Eval output: {output_json}", flush=True)
        return rc
    finally:
        if args.keep_server:
            print(f"Leaving inference server running with pid {process.pid}", flush=True)
        else:
            terminate_process(process)
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
