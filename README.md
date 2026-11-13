# LLM Inference Systems Lab

A from-first-principles study and benchmark suite for production LLM inference.

## Current milestone: autoregressive decoding and KV caching

This milestone compares two decoding strategies:

1. **Without KV caching:** recompute the transformer over the complete sequence for every generated token.
2. **With KV caching:** compute the prompt once, retain previous keys and values, and run the model only on the newest token during decoding.

The development default is `sshleifer/tiny-gpt2` so the code can be validated quickly. It is not used as evidence for final performance claims.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the first experiment

```bash
python decode_benchmark.py \
  --prompt "The future of efficient machine learning is" \
  --new-tokens 32
```

For a CUDA machine:

```bash
python decode_benchmark.py \
  --model sshleifer/tiny-gpt2 \
  --device cuda \
  --new-tokens 64
```

The script reports generated text, execution time, tokens per second, and peak GPU memory when CUDA is available. It appends per-mode measurements to `results/phase1_results.csv` and writes a comparison plot to `results/phase1_comparison.png`. Override either path with `--results-csv` or `--plot`; pass `--no-plot` to skip plotting.

Each run compares greedy decoding with and without a KV cache. Repeat with different `--new-tokens` and prompt lengths to see how the tradeoff changes; treat single-run timings as directional, not production benchmarks.

## Repository roadmap

- `phase1/`: model execution, manual decoding, KV-cache behavior
- `phase2/`: HTTP serving and streaming responses
- `phase3/`: batching and scheduling
- `phase4/`: vLLM benchmarking and workload generation
- `phase5/`: quantization and profiling
- `results/`: reproducible raw measurements and plots
- `reports/`: technical explanations and experiment conclusions

Every optimization will be supported by a baseline, a measurement, and an explanation of the tradeoff.
