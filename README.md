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

For each configuration, the script reports median decode time, median throughput, speedup, and output agreement. On CUDA it also records peak allocated GPU memory. Raw per-mode trials are appended to `results/phase1_repeated_results.csv`, and the median sweep plot is written to `results/phase1_sweep.png`. Override these paths with `--results-csv` and `--plot`; pass `--no-plot` to skip plotting.

Each configuration uses one warm-up per mode and five measured trials by default; measured order alternates to reduce order bias.

Sweep prompt and generation lengths on a CUDA machine:

```bash
python decode_benchmark.py \
  --model gpt2 \
  --device cuda \
  --prompt-lengths 7,64,256 \
  --token-counts 64,128 \
  --warmup-runs 1 \
  --repeat-runs 5
```

`--prompt-lengths` constructs exact token lengths by repeating or truncating the tokenized `--prompt`; this controls context size but does not represent natural long prompts. For realistic prompt distributions, use representative text inputs. `--new-tokens` remains the single generation-length option when `--token-counts` is omitted.

## Repository roadmap

- `phase1/`: model execution, manual decoding, KV-cache behavior
- `phase2/`: HTTP serving and streaming responses
- `phase3/`: batching and scheduling
- `phase4/`: vLLM benchmarking and workload generation
- `phase5/`: quantization and profiling
- `results/`: reproducible raw measurements and plots
- `reports/`: technical explanations and experiment conclusions

Every optimization will be supported by a baseline, a measurement, and an explanation of the tradeoff.
