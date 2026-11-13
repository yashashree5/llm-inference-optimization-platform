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

## Phase 2: single-request streaming server

Install the requirements, then start the server. It loads the model once at startup; set `MODEL_NAME` or `INFERENCE_DEVICE` to override the defaults.

```bash
uvicorn serve:app --host 127.0.0.1 --port 8000
```

Check readiness and stream a greedy completion using Server-Sent Events:

```bash
curl http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The future of efficient machine learning is","max_new_tokens":32}'
```

`/generate` emits one `token` event per generated token, followed by a `done` event with time to first token (TTFT), mean inter-token latency (ITL), total time, and token count. This learning server serializes generation requests over one model; concurrent scheduling and batching are Phase 3.

Run the API tests with `pytest -q`.

Benchmark a prompt set sequentially after warming each prompt once:

```bash
python benchmark_server.py \
  --prompts benchmarks/phase2_prompts.jsonl \
  --warmup-rounds 1 \
  --repeat 3
```

The JSONL prompt file accepts one object per line with `id`, `prompt`, and optional `max_new_tokens`. The client records client-observed and server-reported TTFT, ITL, total time, and output tokens/sec, then writes request rows to `results/phase2_server_requests.jsonl` and p50/p95 summaries overall and by exact input/output token lengths to `results/phase2_server_summary.json`. Prompt text is not copied into results. This is sequential single-request latency testing, not concurrent throughput testing.

## Repository roadmap

- `phase1/`: model execution, manual decoding, KV-cache behavior
- `phase2/`: HTTP serving and streaming responses
- `phase3/`: batching and scheduling
- `phase4/`: vLLM benchmarking and workload generation
- `phase5/`: quantization and profiling
- `results/`: reproducible raw measurements and plots
- `reports/`: technical explanations and experiment conclusions

Every optimization will be supported by a baseline, a measurement, and an explanation of the tradeoff.
