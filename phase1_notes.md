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
