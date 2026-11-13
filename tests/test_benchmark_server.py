import json
from pathlib import Path

import pytest

from benchmark_server import iter_sse_events, load_prompts, percentile, summarize


def test_percentile_interpolates_and_handles_empty_values() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == 3.85
    assert percentile([], 95) is None


def test_sse_parser_reads_token_and_done_events() -> None:
    lines = [
        "event: token",
        'data: {"index":1,"text":"hello"}',
        "",
        ": keep-alive",
        "",
        "event: done",
        'data: {"generated_tokens":1}',
    ]

    assert list(iter_sse_events(lines)) == [
        ("token", {"index": 1, "text": "hello"}),
        ("done", {"generated_tokens": 1}),
    ]


def test_prompt_corpus_and_summary_group_by_prompt_length() -> None:
    prompts = load_prompts(Path("benchmarks/phase2_prompts.jsonl"))
    assert [prompt["id"] for prompt in prompts] == [
        "short_explanation",
        "medium_service_design",
        "long_benchmark_review",
    ]

    rows = [
        {
            "prompt_tokens": 20,
            "requested_new_tokens": 32,
            "server_ttft_seconds": 0.1,
            "server_tokens_per_second": 10.0,
        },
        {
            "prompt_tokens": 256,
            "requested_new_tokens": 64,
            "server_ttft_seconds": 0.2,
            "server_tokens_per_second": 8.0,
        },
    ]
    summary = summarize(rows)
    assert summary["requests"] == 2
    assert summary["by_workload"]["20 prompt tokens / 32 new tokens"]["requests"] == 1
    assert summary["by_workload"]["256 prompt tokens / 64 new tokens"]["requests"] == 1
    assert summary["overall"]["metrics"]["server_ttft_seconds"]["p50"] == pytest.approx(0.15)
    assert summary["overall"]["metrics"]["server_ttft_seconds"]["p95"] == pytest.approx(0.195)
