"""Compare causal-LM decoding with and without a KV cache.

This is deliberately explicit code. It is not intended to replace a production
runtime; it exposes the two different computational paths so their behavior can
be measured and understood before using vLLM or TensorRT-LLM.
"""

from __future__ import annotations

import argparse
import csv
import statistics
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
def decode_without_cache(model, tokenizer, prompt: str | torch.Tensor, new_tokens: int, device: torch.device) -> DecodeResult:
    """Generate by recomputing all previous tokens at every decoding step."""
    input_ids = _prompt_ids(tokenizer, prompt, device)
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
def decode_with_cache(model, tokenizer, prompt: str | torch.Tensor, new_tokens: int, device: torch.device) -> DecodeResult:
    """Generate with past_key_values so only the newest token is decoded per step."""
    prompt_ids = _prompt_ids(tokenizer, prompt, device)
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


def _prompt_ids(tokenizer, prompt: str | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(prompt, torch.Tensor):
        return prompt.to(device)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def make_prompt_ids(tokenizer, prompt: str, target_tokens: int | None) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if target_tokens is None:
        return prompt_ids.unsqueeze(0)
    if prompt_ids.numel() == 0:
        raise ValueError("The prompt must encode to at least one token")
    repeats = (target_tokens + prompt_ids.numel() - 1) // prompt_ids.numel()
    return prompt_ids.repeat(repeats)[:target_tokens].unsqueeze(0)


def parse_int_list(value: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{option} values must be positive")
    return values


def save_results(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "timestamp_utc",
        "model",
        "device",
        "prompt_tokens",
        "generated_tokens",
        "warmup_runs",
        "trial",
        "order",
        "mode",
        "seconds",
        "tokens_per_second",
        "peak_memory_mb",
        "outputs_match",
        "speedup_vs_uncached",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as results_file:
            existing_fields = tuple(next(csv.reader(results_file), ()))
        if existing_fields != fields:
            raise ValueError(f"Existing CSV schema does not match: {path}")

    with path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, summaries: list[dict[str, float | int]], model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"P{item['prompt_tokens']}/G{item['generated_tokens']}" for item in summaries]
    positions = list(range(len(summaries)))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(summaries) * 1.4), 4))

    axes[0].bar(
        [position - width / 2 for position in positions],
        [item["uncached_seconds"] for item in summaries],
        width,
        label="Without cache",
        color="#c46b48",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        [item["cached_seconds"] for item in summaries],
        width,
        label="With cache",
        color="#327a78",
    )
    axes[1].bar(
        [position - width / 2 for position in positions],
        [item["uncached_tokens_per_second"] for item in summaries],
        width,
        label="Without cache",
        color="#c46b48",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [item["cached_tokens_per_second"] for item in summaries],
        width,
        label="With cache",
        color="#327a78",
    )

    for axis, title, ylabel in (
        (axes[0], "Median decode time", "Seconds (lower is better)"),
        (axes[1], "Median throughput", "Tokens / second (higher is better)"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.spines[["top", "right"]].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    figure.suptitle(f"Greedy decoding sweep: {model_name}", y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="The future of efficient machine learning is")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--prompt-lengths", help="Comma-separated exact prompt token lengths")
    parser.add_argument("--token-counts", help="Comma-separated generated-token counts")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/phase1_repeated_results.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results/phase1_sweep.png"))
    parser.add_argument("--no-plot", dest="plot", action="store_const", const=None)
    args = parser.parse_args()

    if args.new_tokens < 1 or args.warmup_runs < 0 or args.repeat_runs < 1:
        raise ValueError("token count and repeat runs must be positive; warmup runs cannot be negative")
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

    prompt_lengths = parse_int_list(args.prompt_lengths, "--prompt-lengths") if args.prompt_lengths else [None]
    token_counts = parse_int_list(args.token_counts, "--token-counts") if args.token_counts else [args.new_tokens]
    context_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
    decode_functions = {"uncached": decode_without_cache, "cached": decode_with_cache}
    rows: list[dict[str, str]] = []
    summaries: list[dict[str, float | int]] = []

    for config_index, target_prompt_tokens in enumerate(prompt_lengths):
        prompt_ids = make_prompt_ids(tokenizer, args.prompt, target_prompt_tokens)
        prompt_tokens = prompt_ids.shape[1]
        for generated_tokens in token_counts:
            if context_limit and prompt_tokens + generated_tokens > context_limit:
                raise ValueError(
                    f"prompt ({prompt_tokens}) plus generation ({generated_tokens}) exceeds "
                    f"model context limit ({context_limit})"
                )

            for warmup in range(args.warmup_runs):
                warmup_modes = ("uncached", "cached") if warmup % 2 == 0 else ("cached", "uncached")
                for mode in warmup_modes:
                    decode_functions[mode](model, tokenizer, prompt_ids, generated_tokens, device)

            uncached_results = []
            cached_results = []
            outputs_match_all = True
            for trial in range(1, args.repeat_runs + 1):
                modes = ("uncached", "cached") if (trial + config_index) % 2 == 1 else ("cached", "uncached")
                trial_results = {}
                for mode in modes:
                    trial_results[mode] = decode_functions[mode](
                        model, tokenizer, prompt_ids, generated_tokens, device
                    )

                uncached = trial_results["uncached"]
                cached = trial_results["cached"]
                outputs_match = uncached.text == cached.text
                outputs_match_all = outputs_match_all and outputs_match
                speedup = uncached.seconds / cached.seconds if cached.seconds else 0.0
                timestamp = datetime.now(timezone.utc).isoformat()
                order = ",".join(modes)
                for mode, result in (("uncached", uncached), ("cached", cached)):
                    rows.append(
                        {
                            "timestamp_utc": timestamp,
                            "model": args.model,
                            "device": str(device),
                            "prompt_tokens": str(prompt_tokens),
                            "generated_tokens": str(generated_tokens),
                            "warmup_runs": str(args.warmup_runs),
                            "trial": str(trial),
                            "order": order,
                            "mode": mode,
                            "seconds": f"{result.seconds:.6f}",
                            "tokens_per_second": f"{result.tokens_per_second:.4f}",
                            "peak_memory_mb": "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.2f}",
                            "outputs_match": str(outputs_match),
                            "speedup_vs_uncached": f"{speedup:.4f}" if mode == "cached" else "1.0000",
                        }
                    )
                uncached_results.append(uncached)
                cached_results.append(cached)

            summary = {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "uncached_seconds": statistics.median(result.seconds for result in uncached_results),
                "cached_seconds": statistics.median(result.seconds for result in cached_results),
                "uncached_tokens_per_second": statistics.median(result.tokens_per_second for result in uncached_results),
                "cached_tokens_per_second": statistics.median(result.tokens_per_second for result in cached_results),
                "speedup": statistics.median(
                    uncached.seconds / cached.seconds
                    for uncached, cached in zip(uncached_results, cached_results)
                    if cached.seconds
                ),
            }
            summaries.append(summary)
            print(
                f"prompt={prompt_tokens} generated={generated_tokens} "
                f"median uncached={summary['uncached_seconds']:.4f}s "
                f"cached={summary['cached_seconds']:.4f}s "
                f"speedup={summary['speedup']:.2f}x outputs_match={outputs_match_all}"
            )

    save_results(args.results_csv, rows)
    if args.plot is not None:
        save_plot(args.plot, summaries, args.model)

    print(f"results CSV: {args.results_csv}")
    if args.plot is not None:
        print(f"comparison plot: {args.plot}")


if __name__ == "__main__":
    main()
