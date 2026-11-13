# Phase 1: What the first experiment is proving

## Autoregressive decoding

At each step, a causal language model produces a probability distribution for
the next token. Greedy decoding selects the token with the largest logit. The
new token is appended to the context and the process repeats.

## Why the uncached version is wasteful

Without a cache, step `t` runs the model over the prompt plus all `t` generated
tokens again. The same historical key and value tensors are repeatedly
recomputed.

## What the cache changes

The first forward pass computes the prompt and returns `past_key_values`. Each
later pass receives only the newest token plus those cached keys and values.
The model still attends to the complete history, but it does not reconstruct
the historical K/V tensors from scratch.

## Questions to answer after running

1. Do both paths produce identical text?
2. How does the speed difference change as `--new-tokens` increases?
3. What happens when the prompt becomes much longer?
4. Why is a cache a memory tradeoff rather than a free optimization?
5. Why will many concurrent requests require a more sophisticated cache manager?

Do not draw performance conclusions from the tiny development model. The first
goal is correctness and understanding. Later phases will repeat the experiment
with a larger model, CUDA, concurrency, and production serving engines.

## First local run

On 2026-11-12, the default 32-token run completed on CPU with Python 3.12.11,
PyTorch 2.15.0, Transformers 5.22.2, and `sshleifer/tiny-gpt2`. The uncached
path took 0.0584 seconds (547.89 tokens/sec); the cached path took 0.0133
seconds (2398.77 tokens/sec), a measured 4.38x speedup. Both paths produced
identical text.

This is one short run on a tiny model, without warmup or repeated trials, so
timing noise and launch overhead are significant. The result supports that the
cache path is wired correctly and can reduce repeated computation; it does not
predict throughput or memory behavior for larger models or GPU serving. The
benchmark now appends raw per-mode metrics to `results/phase1_results.csv` and
creates `results/phase1_comparison.png` for each run.

## Colab T4 follow-up

The same script was run three times on a Colab Tesla T4 with `gpt2`, a 7-token
prompt, and 128 generated tokens. All three cached and uncached runs produced
identical text. Median uncached time was 1.5046 seconds (85.07 tokens/sec); the
cached median was 0.9863 seconds (129.77 tokens/sec), or a 1.53x median
speedup. Individual speedups ranged from 1.12x to 2.35x, showing substantial
run-to-run variation in this single-pass benchmark.

The GPU measurements are recorded in `results/phase1_t4.csv`; the companion
chart, `results/phase1_t4_comparison.png`, summarizes the median times and
throughput. The output shows why caching helps: later decoding steps pass only
the newest token and reuse prior keys and values instead of recomputing the
whole prefix. The CUDA run is much faster in absolute time than the tiny-model
CPU run, but their speedup ratios are not directly comparable because the model
and token count differ.

The benchmark has no warmup and alternates neither run order nor multiple
samples within a process. Treat the median as a useful learning measurement,
not a stable performance claim. The peak allocated GPU memory was 536.61 MB
uncached and 494.24 MB cached in these runs; this does not mean KV caching
generally saves memory, since cached key/value tensors grow with context length
and the tiny model's weights and transient allocations dominate this sample.
