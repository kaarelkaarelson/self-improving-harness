#!/usr/bin/env python3
"""Package trace/data artifacts for transfer to a training node."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def relative_inputs(inputs: list[str], root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.exists():
            raise SystemExit(f"Input does not exist: {path}")
        try:
            paths.append(path.relative_to(root))
        except ValueError as exc:
            raise SystemExit(f"Input must be inside {root}: {path}") from exc
    return paths


def write_manifest(path: Path, inputs: list[Path], root: Path) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "inputs": [str(path) for path in inputs],
    }
    path.write_text(json.dumps(manifest, indent=2))


def make_tar_gz(output: Path, inputs: list[Path], root: Path, manifest: Path) -> None:
    with tarfile.open(output, "w:gz") as tar:
        for path in inputs:
            tar.add(root / path, arcname=str(path))
        tar.add(manifest, arcname="artifact_manifest.json")


def make_zip(output: Path, inputs: list[Path], root: Path, manifest: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in inputs:
            full_path = root / path
            if full_path.is_dir():
                for child in full_path.rglob("*"):
                    if child.is_file():
                        archive.write(child, child.relative_to(root))
            else:
                archive.write(full_path, path)
        archive.write(manifest, "artifact_manifest.json")


def make_tar_zst(output: Path, inputs: list[Path], root: Path, manifest: Path) -> None:
    if shutil.which("tar") is None:
        raise SystemExit("Creating .tar.zst archives requires tar on PATH")
    command = ["tar", "--zstd", "-cf", str(output), "-C", str(root)]
    command.extend(str(path) for path in inputs)
    command.extend(["-C", str(manifest.parent), manifest.name])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package AutomationBench artifacts")
    parser.add_argument("--inputs", nargs="+", required=True, help="Files/directories under --root")
    parser.add_argument("--output", required=True, help=".tar.gz, .tgz, .tar.zst, or .zip")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    inputs = relative_inputs(args.inputs, root)
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = Path(tmpdir) / "artifact_manifest.json"
        write_manifest(manifest, inputs, root)
        name = output.name
        if name.endswith((".tar.gz", ".tgz")):
            make_tar_gz(output, inputs, root, manifest)
        elif name.endswith(".zip"):
            make_zip(output, inputs, root, manifest)
        elif name.endswith(".tar.zst"):
            make_tar_zst(output, inputs, root, manifest)
        else:
            raise SystemExit("Output must end with .tar.gz, .tgz, .tar.zst, or .zip")

    print(output)


if __name__ == "__main__":
    main()
