"""Measure sequential streaming latency for a set of prompts."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import httpx


DEFAULT_PROMPTS = Path("benchmarks/phase2_prompts.jsonl")


def percentile(values: Iterable[float], percentile_rank: float) -> float | None:
    if not 0 <= percentile_rank <= 100:
        raise ValueError("percentile_rank must be between 0 and 100")
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile_rank / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, object]]]:
    event = "message"
    data_lines: list[str] = []

    def parse_event() -> tuple[str, dict[str, object]] | None:
        if not data_lines:
            return None
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise ValueError("SSE data must be a JSON object")
        return event, payload

    for line in lines:
        if not line:
            parsed = parse_event()
            if parsed is not None:
                yield parsed
            event = "message"
            data_lines = []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    parsed = parse_event()
    if parsed is not None:
        yield parsed


def load_prompts(path: Path) -> list[dict[str, object]]:
    prompt_cases: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as prompt_file:
        for line_number, line in enumerate(prompt_file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            if not isinstance(case, dict) or not isinstance(case.get("prompt"), str) or not case["prompt"]:
                raise ValueError(f"Line {line_number} must be an object with a non-empty prompt")
            token_limit = case.get("max_new_tokens", 32)
            if isinstance(token_limit, bool) or not isinstance(token_limit, int) or not 1 <= token_limit <= 256:
                raise ValueError(f"Line {line_number} max_new_tokens must be an integer from 1 to 256")
            prompt_cases.append(
                {
                    "id": str(case.get("id", line_number)),
                    "prompt": case["prompt"],
                    "max_new_tokens": token_limit,
                }
            )
    if not prompt_cases:
        raise ValueError(f"No prompts found in {path}")
    return prompt_cases


def _request(
    client: httpx.Client,
    prompt_case: dict[str, object],
) -> dict[str, object]:
    request_started = time.perf_counter()
    first_token_at: float | None = None
    token_arrivals: list[float] = []
    final_metrics: dict[str, object] | None = None
    token_count = 0

    with client.stream(
        "POST",
        "/generate",
        json={"prompt": prompt_case["prompt"], "max_new_tokens": prompt_case["max_new_tokens"]},
    ) as response:
        response.raise_for_status()
        for event, payload in iter_sse_events(response.iter_lines()):
            if event == "token":
                arrived_at = time.perf_counter()
                first_token_at = first_token_at or arrived_at
                token_arrivals.append(arrived_at)
                token_count += 1
            elif event == "done":
                final_metrics = payload
            elif event == "error":
                raise RuntimeError(f"Generation failed for prompt {prompt_case['id']}")

    client_total_seconds = time.perf_counter() - request_started
    if final_metrics is None:
        raise RuntimeError(f"No done event received for prompt {prompt_case['id']}")

    client_inter_token_ms = [
        (later - earlier) * 1000 for earlier, later in zip(token_arrivals, token_arrivals[1:])
    ]
    return {
        "prompt_id": prompt_case["id"],
        "prompt_chars": len(str(prompt_case["prompt"])),
        "prompt_tokens": final_metrics.get("prompt_tokens"),
        "requested_new_tokens": prompt_case["max_new_tokens"],
        "generated_tokens": final_metrics.get("generated_tokens", token_count),
        "server_ttft_seconds": final_metrics.get("ttft_seconds"),
        "client_ttft_seconds": first_token_at - request_started if first_token_at is not None else None,
        "server_mean_inter_token_latency_ms": final_metrics.get("mean_inter_token_latency_ms"),
        "client_mean_inter_token_latency_ms": (
            sum(client_inter_token_ms) / len(client_inter_token_ms) if client_inter_token_ms else None
        ),
        "server_total_seconds": final_metrics.get("total_seconds"),
        "client_total_seconds": client_total_seconds,
        "server_tokens_per_second": final_metrics.get("tokens_per_second"),
    }


def _metric_summary(rows: list[dict[str, object]], field: str) -> dict[str, float | None]:
    values = [value for row in rows if isinstance((value := row.get(field)), (int, float))]
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "server_ttft_seconds",
        "client_ttft_seconds",
        "server_mean_inter_token_latency_ms",
        "client_mean_inter_token_latency_ms",
        "server_total_seconds",
        "client_total_seconds",
        "server_tokens_per_second",
    )
    groups: dict[tuple[int | None, int | None], list[dict[str, object]]] = {}
    for row in rows:
        prompt_tokens = row.get("prompt_tokens")
        requested_new_tokens = row.get("requested_new_tokens")
        prompt_length = prompt_tokens if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) else None
        output_length = (
            requested_new_tokens
            if isinstance(requested_new_tokens, int) and not isinstance(requested_new_tokens, bool)
            else None
        )
        groups.setdefault((prompt_length, output_length), []).append(row)

    def describe(group_rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "requests": len(group_rows),
            "metrics": {metric: _metric_summary(group_rows, metric) for metric in metrics},
        }

    by_workload = {}
    for (prompt_tokens, requested_new_tokens), group_rows in sorted(
        groups.items(), key=lambda item: (item[0][0] or -1, item[0][1] or -1)
    ):
        prompt_label = "unknown" if prompt_tokens is None else str(prompt_tokens)
        output_label = "unknown" if requested_new_tokens is None else str(requested_new_tokens)
        by_workload[f"{prompt_label} prompt tokens / {output_label} new tokens"] = {
            "prompt_tokens": prompt_tokens,
            "requested_new_tokens": requested_new_tokens,
            **describe(group_rows),
        }

    return {
        "requests": len(rows),
        "overall": describe(rows),
        "by_workload": by_workload,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL for the inference server")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="JSONL file with id, prompt, and optional max_new_tokens")
    parser.add_argument("--repeat", type=int, default=3, help="Measured rounds over the prompt set")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="Unrecorded request for every prompt before measurement")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup_rounds < 0 or args.timeout <= 0:
        parser.error("repeat and timeout must be positive; warmup-rounds cannot be negative")
    return args


def main() -> None:
    args = _parse_args()
    prompt_cases = load_prompts(args.prompts)
    timeout = httpx.Timeout(args.timeout)
    rows: list[dict[str, object]] = []

    with httpx.Client(base_url=args.url.rstrip("/"), timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        server_info = health.json()
        print(f"server: model={server_info['model']} device={server_info['device']}")

        for _ in range(args.warmup_rounds):
            for prompt_case in prompt_cases:
                _request(client, prompt_case)
        if args.warmup_rounds:
            print(f"warm-up requests: {len(prompt_cases) * args.warmup_rounds}")

        for repeat_index in range(args.repeat):
            for prompt_case in prompt_cases:
                result = _request(client, prompt_case)
                result["round"] = repeat_index + 1
                rows.append(result)
                print(
                    f"round={repeat_index + 1} prompt={result['prompt_id']} "
                    f"input={result['prompt_tokens']} tokens "
                    f"TTFT={result['server_ttft_seconds']:.4f}s"
                )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "server": server_info,
        "prompt_file": str(args.prompts),
        "warmup_rounds": args.warmup_rounds,
        "repeat": args.repeat,
        "request_metrics": summarize(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.out_dir / "phase2_server_requests.jsonl"
    summary_path = args.out_dir / "phase2_server_summary.json"
    with requests_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"request metrics: {requests_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
