#!/usr/bin/env python3
"""Check GPU training dependencies before launching Prime-RL SFT."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys


def module_status(name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    return {"ok": True, "version": str(version) if version is not None else "unknown"}


def main() -> None:
    report: dict[str, object] = {"python": sys.version}

    try:
        import torch

        report["torch"] = {
            "ok": True,
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                    "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        report["torch"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    for name in [
        "triton",
        "flash_attn",
        "fla",
        "fla.ops",
        "tilelang",
        "transformers",
    ]:
        report[name] = module_status(name)

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        report["nvidia_smi"] = {
            "ok": completed.returncode == 0,
            "stdout": completed.stdout.strip().splitlines(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        report["nvidia_smi"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(report, indent=2, sort_keys=True))
    required = ["torch", "triton", "flash_attn", "fla", "tilelang"]
    failed = [name for name in required if not isinstance(report.get(name), dict) or not report[name].get("ok")]
    if failed:
        raise SystemExit(f"Missing or broken required modules: {', '.join(failed)}")


if __name__ == "__main__":
    main()
