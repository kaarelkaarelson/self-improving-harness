#!/usr/bin/env python3
"""Download public Prime AutomationBench leaderboard traces.

The Prime web UI exposes the public leaderboard through a tRPC endpoint. The
sample traces themselves are available through the same Prime eval samples API
used by `prime eval samples`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "automationbench-prime-sft-v1"
DEFAULT_APP_BASE_URL = "https://app.primeintellect.ai"
DEFAULT_API_BASE_URL = "https://api.primeintellect.ai"
DEFAULT_ENVIRONMENT_ID = "fe1t1r1he3154ngqngodxlhw"


def slug(value: str, max_len: int = 120) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (result or "value")[:max_len]


def strip_api_v1(url: str) -> str:
    return url.rstrip("/").removesuffix("/api/v1")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout_sec: int = 60,
    retries: int = 4,
) -> Any:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "self-improving-harness-prime-trace-downloader/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt == retries - 1:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise RuntimeError(f"GET {url} failed: {exc}") from exc
        time.sleep(min(2**attempt, 20))

    raise RuntimeError(f"GET {url} failed: {last_error}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def list_public_evals(
    *,
    app_base_url: str,
    environment_id: str,
    limit: int,
    skip: int = 0,
) -> dict[str, Any]:
    input_payload = {
        "0": {
            "json": {
                "environmentId": environment_id,
                "skip": skip,
                "limit": limit,
            }
        }
    }
    query = urllib.parse.urlencode(
        {
            "batch": "1",
            "input": json.dumps(input_payload, separators=(",", ":")),
        }
    )
    url = f"{app_base_url.rstrip('/')}/api/trpc/environments.getEvaluationsFromMongoDB?{query}"
    payload = request_json(url)
    try:
        return payload[0]["result"]["data"]["json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected tRPC leaderboard response shape: {payload!r}") from exc


def get_eval(
    evaluation_id: str,
    *,
    api_base_url: str,
    api_key: str,
) -> dict[str, Any]:
    url = f"{strip_api_v1(api_base_url)}/api/v1/evaluations/{evaluation_id}"
    payload = request_json(url, api_key=api_key)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected eval response for {evaluation_id}: {payload!r}")
    return payload


def get_samples_page(
    evaluation_id: str,
    *,
    api_base_url: str,
    api_key: str,
    page: int,
    limit: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"page": page, "limit": limit})
    url = f"{strip_api_v1(api_base_url)}/api/v1/evaluations/{evaluation_id}/samples?{query}"
    payload = request_json(url, api_key=api_key)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise RuntimeError(f"Unexpected samples response for {evaluation_id} page {page}: {payload!r}")
    return payload


def public_eval_id(evaluation: dict[str, Any]) -> str:
    value = evaluation.get("_id") or evaluation.get("id") or evaluation.get("evaluation_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Evaluation has no id: {evaluation!r}")
    return value


def eval_model(evaluation: dict[str, Any]) -> str:
    metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
    return str(evaluation.get("model_name") or metadata.get("model") or "unknown-model")


def eval_total_samples(evaluation: dict[str, Any]) -> int | None:
    for key in ("total_samples", "totalSamples"):
        value = evaluation.get(key)
        if isinstance(value, int):
            return value
    statistics = evaluation.get("statistics") if isinstance(evaluation.get("statistics"), dict) else {}
    value = statistics.get("totalResults")
    return value if isinstance(value, int) else None


def eval_dir_name(evaluation: dict[str, Any]) -> str:
    evaluation_id = public_eval_id(evaluation)
    return f"{slug(eval_model(evaluation), 80)}__{evaluation_id}"


def score_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    info = sample.get("info") if isinstance(sample.get("info"), dict) else {}
    metrics = info.get("metrics") if isinstance(info.get("metrics"), dict) else {}
    return {
        "reward": sample.get("reward"),
        "score": sample.get("score"),
        "correct": sample.get("correct"),
        "partial_credit": metrics.get("partial_credit", sample.get("reward")),
        "task_completed_correctly": metrics.get("task_completed_correctly"),
    }


def prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        messages = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role") or "user")
                messages.append({"role": role, "content": item.get("content", "")})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": json_dumps(prompt)}]


def parse_tool_call(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {"raw": value}


def tool_marker(name: str, payload: Any) -> str:
    return f"<{name}>\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n</{name}>"


def build_training_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = prompt_messages(sample.get("prompt"))
    completion = sample.get("completion")
    if not isinstance(completion, list):
        return messages

    for event in completion:
        if not isinstance(event, dict):
            continue
        role = event.get("role")
        if role == "assistant":
            content = event.get("content")
            if isinstance(content, str) and content:
                messages.append({"role": "assistant", "content": content})
            for raw_call in event.get("tool_calls") or []:
                call = parse_tool_call(raw_call)
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                tool_name = call.get("name") or function.get("name")
                arguments = call.get("arguments") or function.get("arguments")
                messages.append(
                    {
                        "role": "assistant",
                        "content": tool_marker(
                            "tool_call",
                            {
                                "id": call.get("id"),
                                "tool": tool_name,
                                "arguments": arguments,
                            },
                        ),
                    }
                )
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "name": str(event.get("name") or event.get("tool_call_id") or ""),
                    "content": event.get("content", ""),
                }
            )
    return messages


def normalize_sample(
    sample: dict[str, Any],
    *,
    evaluation: dict[str, Any],
    include_initial_state: bool,
    include_full_info: bool,
) -> dict[str, Any]:
    evaluation_id = public_eval_id(evaluation)
    metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
    info = sample.get("info") if isinstance(sample.get("info"), dict) else {}

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "prime_leaderboard",
        "evaluation": {
            "id": evaluation_id,
            "name": evaluation.get("name"),
            "model": eval_model(evaluation),
            "framework": evaluation.get("framework") or metadata.get("framework"),
            "dataset": evaluation.get("dataset"),
            "source_evaluation_id": metadata.get("source_evaluation_id"),
            "source_file": metadata.get("source_file"),
        },
        "task": {
            "example_id": sample.get("example_id"),
            "task": sample.get("task"),
            "prompt": sample.get("prompt"),
        },
        "trace": {
            "trace_id": sample.get("trace_id"),
            "rollout_number": sample.get("rollout_number"),
            "completion": sample.get("completion"),
        },
        "score": score_from_sample(sample),
        "assertion_results": info.get("assertions", []),
        "info": {
            "metrics": info.get("metrics"),
            "source_evaluation_id": info.get("source_evaluation_id"),
            "source_sample_id": info.get("source_sample_id"),
            "source_file": info.get("source_file"),
        },
        "training_messages": build_training_messages(sample),
    }

    if include_initial_state:
        record["initial_state"] = info.get("initial_state")
    if include_full_info:
        record["info"] = info
    return record


def iter_downloaded_samples(eval_dir: Path) -> Iterable[dict[str, Any]]:
    pages_dir = eval_dir / "pages"
    for page_path in sorted(pages_dir.glob("page_*.json")):
        payload = read_json(page_path)
        for sample in payload.get("samples", []):
            if isinstance(sample, dict):
                yield sample


def downloaded_sample_count(eval_dir: Path) -> int:
    count = 0
    for page_path in sorted((eval_dir / "pages").glob("page_*.json")):
        payload = read_json(page_path)
        samples = payload.get("samples")
        if isinstance(samples, list):
            count += len(samples)
    return count


def download_eval_samples(
    evaluation: dict[str, Any],
    *,
    output_dir: Path,
    api_base_url: str,
    api_key: str,
    page_size: int,
    max_samples: int | None,
    force: bool,
) -> int:
    evaluation_id = public_eval_id(evaluation)
    eval_dir = output_dir / eval_dir_name(evaluation)
    pages_dir = eval_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "eval.json", evaluation)

    expected_total = eval_total_samples(evaluation)
    if max_samples is not None:
        expected_total = min(expected_total or max_samples, max_samples)
    if expected_total is None:
        expected_total = page_size

    page = 1
    downloaded = 0
    while downloaded < expected_total:
        page_path = pages_dir / f"page_{page:06d}.json"
        desired_limit = page_size
        if page_path.exists() and not force:
            payload = read_json(page_path)
            if payload.get("limit") != desired_limit:
                payload = get_samples_page(
                    evaluation_id,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    page=page,
                    limit=desired_limit,
                )
                write_json(page_path, payload)
        else:
            payload = get_samples_page(
                evaluation_id,
                api_base_url=api_base_url,
                api_key=api_key,
                page=page,
                limit=desired_limit,
            )
            write_json(page_path, payload)

        samples = payload.get("samples", [])
        if not samples:
            break
        downloaded += len(samples)

        total = payload.get("total")
        total_pages = payload.get("total_pages")
        if isinstance(total, int):
            effective_total = total
            if max_samples is not None:
                effective_total = min(effective_total, max_samples)
            expected_total = min(expected_total, effective_total)
        if isinstance(total_pages, int) and page >= total_pages:
            break
        page += 1

    return downloaded_sample_count(eval_dir)


def write_normalized_jsonl(
    evaluations: list[dict[str, Any]],
    *,
    output_dir: Path,
    output_jsonl: Path,
    max_samples_per_eval: int | None,
    min_partial_credit: float,
    require_strict: bool,
    include_initial_state: bool,
    include_full_info: bool,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w") as out:
        for evaluation in evaluations:
            eval_dir = output_dir / eval_dir_name(evaluation)
            seen_for_eval = 0
            for sample in iter_downloaded_samples(eval_dir):
                if max_samples_per_eval is not None and seen_for_eval >= max_samples_per_eval:
                    break
                seen_for_eval += 1
                score = score_from_sample(sample)
                partial = float(score.get("partial_credit") or 0.0)
                strict = float(score.get("task_completed_correctly") or 0.0)
                if partial < min_partial_credit:
                    continue
                if require_strict and strict < 1.0:
                    continue
                record = normalize_sample(
                    sample,
                    evaluation=evaluation,
                    include_initial_state=include_initial_state,
                    include_full_info=include_full_info,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    return written


def load_evaluations_from_manifest(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(evaluations, list):
        raise ValueError(f"{path} does not contain an evaluations list")
    return [item for item in evaluations if isinstance(item, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Prime AutomationBench leaderboard traces")
    parser.add_argument("--output-dir", default="runs/prime_leaderboard")
    parser.add_argument("--output-jsonl", help="Optional normalized SFT JSONL output path")
    parser.add_argument("--eval-id", action="append", help="Download one eval id; repeatable")
    parser.add_argument("--manifest", help="Use an existing manifest instead of listing the UI endpoint")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--limit-evals", type=int)
    parser.add_argument("--max-samples-per-eval", type=int)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--min-partial-credit", type=float, default=0.0)
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--include-initial-state", action="store_true")
    parser.add_argument("--include-full-info", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download existing page files")
    parser.add_argument("--app-base-url", default=os.getenv("PRIME_APP_BASE_URL", DEFAULT_APP_BASE_URL))
    parser.add_argument("--api-base-url", default=os.getenv("PRIME_API_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--environment-id", default=DEFAULT_ENVIRONMENT_ID)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        evaluations = load_evaluations_from_manifest(Path(args.manifest))
    elif args.eval_id:
        api_key = os.getenv("PRIME_API_KEY")
        if not api_key:
            raise SystemExit("PRIME_API_KEY is required when using --eval-id")
        evaluations = [
            get_eval(eval_id, api_base_url=args.api_base_url, api_key=api_key)
            for eval_id in args.eval_id
        ]
    else:
        listed = list_public_evals(
            app_base_url=args.app_base_url,
            environment_id=args.environment_id,
            limit=100,
        )
        evaluations = listed.get("evaluations", [])
        if not isinstance(evaluations, list):
            raise SystemExit("Leaderboard response did not contain an evaluations list")
        evaluations = [item for item in evaluations if isinstance(item, dict)]
        manifest_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "prime_public_trpc",
            "environment_id": args.environment_id,
            "pagination": listed.get("pagination"),
            "statistics": listed.get("statistics"),
            "evaluations": evaluations,
        }
        write_json(output_dir / "manifest.json", manifest_payload)

    if args.limit_evals is not None:
        evaluations = evaluations[: args.limit_evals]

    print(f"Evaluations: {len(evaluations)}")
    for evaluation in evaluations:
        print(
            "\t".join(
                [
                    public_eval_id(evaluation),
                    eval_model(evaluation),
                    str(eval_total_samples(evaluation) or ""),
                    str(
                        (evaluation.get("metadata") or {}).get("avg_partial_credit")
                        if isinstance(evaluation.get("metadata"), dict)
                        else evaluation.get("avg_score", "")
                    ),
                ]
            )
        )

    if args.list_only:
        return

    api_key = os.getenv("PRIME_API_KEY")
    if not api_key:
        raise SystemExit("PRIME_API_KEY is required for sample downloads")

    total_downloaded = 0
    for evaluation in evaluations:
        count = download_eval_samples(
            evaluation,
            output_dir=output_dir,
            api_base_url=args.api_base_url,
            api_key=api_key,
            page_size=args.page_size,
            max_samples=args.max_samples_per_eval,
            force=args.force,
        )
        total_downloaded += count
        print(f"Downloaded {count} samples for {public_eval_id(evaluation)}")

    print(f"Downloaded samples: {total_downloaded}")

    if args.output_jsonl:
        written = write_normalized_jsonl(
            evaluations,
            output_dir=output_dir,
            output_jsonl=Path(args.output_jsonl),
            max_samples_per_eval=args.max_samples_per_eval,
            min_partial_credit=args.min_partial_credit,
            require_strict=args.require_strict,
            include_initial_state=args.include_initial_state,
            include_full_info=args.include_full_info,
        )
        print(f"Wrote normalized records: {written}")
        print(f"Output JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
