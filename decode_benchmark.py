"""Compare causal-LM decoding with and without a KV cache.

This is deliberately explicit code. It is not intended to replace a production
runtime; it exposes the two different computational paths so their behavior can
be measured and understood before using vLLM or TensorRT-LLM.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DecodeResult:
    text: str
    seconds: float
    generated_tokens: int
    peak_memory_mb: float | None

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode_without_cache(model, tokenizer, prompt: str, new_tokens: int, device: torch.device) -> DecodeResult:
    """Generate by recomputing all previous tokens at every decoding step."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()

    for _ in range(new_tokens):
        outputs = model(input_ids=generated, use_cache=False)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    _synchronize(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return DecodeResult(tokenizer.decode(generated[0], skip_special_tokens=True), seconds, new_tokens, peak)


@torch.inference_mode()
def decode_with_cache(model, tokenizer, prompt: str, new_tokens: int, device: torch.device) -> DecodeResult:
    """Generate with past_key_values so only the newest token is decoded per step."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = prompt_ids.clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()

    outputs = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated = torch.cat((generated, next_token), dim=1)

    for _ in range(new_tokens - 1):
        # The cache already contains the prompt and all earlier generated tokens.
        # Passing the full sequence here would defeat the purpose of caching.
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True, return_dict=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated = torch.cat((generated, next_token), dim=1)

    _synchronize(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return DecodeResult(tokenizer.decode(generated[0], skip_special_tokens=True), seconds, new_tokens, peak)


def print_result(label: str, result: DecodeResult) -> None:
    print(f"\n[{label}]")
    print(f"text: {result.text}")
    print(f"seconds: {result.seconds:.4f}")
    print(f"tokens/sec: {result.tokens_per_second:.2f}")
    if result.peak_memory_mb is not None:
        print(f"peak GPU memory (MB): {result.peak_memory_mb:.2f}")


def save_results(
    path: Path,
    model_name: str,
    device: torch.device,
    prompt_tokens: int,
    uncached: DecodeResult,
    cached: DecodeResult,
    outputs_match: bool,
) -> None:
    fields = (
        "timestamp_utc",
        "model",
        "device",
        "prompt_tokens",
        "generated_tokens",
        "mode",
        "seconds",
        "tokens_per_second",
        "peak_memory_mb",
        "outputs_match",
        "speedup_vs_uncached",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    timestamp = datetime.now(timezone.utc).isoformat()
    speedup = uncached.seconds / cached.seconds if cached.seconds else 0.0
    rows = []
    for mode, result in (("uncached", uncached), ("cached", cached)):
        rows.append(
            {
                "timestamp_utc": timestamp,
                "model": model_name,
                "device": str(device),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "mode": mode,
                "seconds": f"{result.seconds:.6f}",
                "tokens_per_second": f"{result.tokens_per_second:.4f}",
                "peak_memory_mb": "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.2f}",
                "outputs_match": outputs_match,
                "speedup_vs_uncached": f"{speedup:.4f}" if mode == "cached" else "1.0000",
            }
        )

    with path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, uncached: DecodeResult, cached: DecodeResult, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("Without cache", "With cache")
    results = (uncached, cached)
    colors = ("#c46b48", "#327a78")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    for axis, values, title, ylabel in (
        (axes[0], [result.seconds for result in results], "Decode time", "Seconds (lower is better)"),
        (axes[1], [result.tokens_per_second for result in results], "Throughput", "Tokens / second (higher is better)"),
    ):
        bars = axis.bar(labels, values, color=colors, width=0.58)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.bar_label(bars, fmt="%.2f")
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(f"Greedy decoding comparison: {model_name}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="The future of efficient machine learning is")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/phase1_results.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results/phase1_comparison.png"))
    parser.add_argument("--no-plot", dest="plot", action="store_const", const=None)
    args = parser.parse_args()

    if args.new_tokens < 1:
        raise ValueError("--new-tokens must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    print(f"model: {args.model}")
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    uncached = decode_without_cache(model, tokenizer, args.prompt, args.new_tokens, device)
    cached = decode_with_cache(model, tokenizer, args.prompt, args.new_tokens, device)
    print_result("without KV cache", uncached)
    print_result("with KV cache", cached)
    outputs_match = uncached.text == cached.text
    speedup = uncached.seconds / cached.seconds if cached.seconds else 0.0
    prompt_tokens = tokenizer(args.prompt, return_tensors="pt").input_ids.shape[1]
    save_results(args.results_csv, args.model, device, prompt_tokens, uncached, cached, outputs_match)
    if args.plot is not None:
        save_plot(args.plot, uncached, cached, args.model)

    print(f"\noutputs match: {outputs_match}")
    print(f"speedup: {speedup:.2f}x")
    print(f"results CSV: {args.results_csv}")
    if args.plot is not None:
        print(f"comparison plot: {args.plot}")


if __name__ == "__main__":
    main()
