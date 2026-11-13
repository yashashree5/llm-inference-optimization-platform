"""Single-model HTTP server that streams greedy generation events."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Thread
from typing import Iterator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


_END = object()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16_384)
    max_new_tokens: int = Field(default=32, ge=1, le=256)


@dataclass(frozen=True)
class _StreamFailure:
    error: BaseException


class _TokenQueueStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.items: queue.Queue[object] = queue.Queue()
        self._skip_prompt = True

    def put(self, value: torch.Tensor) -> None:
        if self._skip_prompt:
            self._skip_prompt = False
            return

        token = value.reshape(-1)
        if token.numel() != 1:
            raise ValueError("Streaming generation supports batch size one")
        self.items.put((int(token.item()), time.perf_counter()))

    def end(self) -> None:
        self.items.put(_END)


class InferenceEngine:
    def __init__(self, model, tokenizer, device: torch.device, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self._generation_lock = threading.Lock()
        self.context_limit = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
        )

    @classmethod
    def from_pretrained(cls) -> InferenceEngine:
        model_name = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
        requested_device = os.getenv("INFERENCE_DEVICE", "auto")
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("INFERENCE_DEVICE must be auto, cpu, or cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        device_name = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        return cls(model, tokenizer, device, model_name)

    def encode_prompt(self, prompt: str, max_new_tokens: int) -> torch.Tensor:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if self.context_limit and input_ids.shape[1] + max_new_tokens > self.context_limit:
            raise HTTPException(
                status_code=422,
                detail=f"prompt plus max_new_tokens exceeds model context limit ({self.context_limit})",
            )
        return input_ids

    def stream(self, input_ids: torch.Tensor, max_new_tokens: int, request_started: float) -> Iterator[str]:
        with self._generation_lock:
            streamer = _TokenQueueStreamer()
            prompt_tokens = input_ids.shape[1]

            def generate() -> None:
                try:
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = self.tokenizer.eos_token_id
                    with torch.inference_mode():
                        self.model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            use_cache=True,
                            streamer=streamer,
                            pad_token_id=pad_token_id,
                        )
                except BaseException as error:
                    streamer.items.put(_StreamFailure(error))
                finally:
                    streamer.end()

            worker = Thread(target=generate, daemon=True)
            worker.start()
            generated_ids: list[int] = []
            token_times: list[float] = []

            try:
                while True:
                    item = streamer.items.get()
                    if item is _END:
                        break
                    if isinstance(item, _StreamFailure):
                        raise RuntimeError("model generation failed") from item.error

                    token_id, generated_at = item
                    generated_ids.append(token_id)
                    token_times.append(generated_at)
                    text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    payload = {"index": len(generated_ids), "token_id": token_id, "text": text}
                    if len(token_times) == 1:
                        payload["ttft_seconds"] = generated_at - request_started
                    else:
                        payload["inter_token_latency_ms"] = (generated_at - token_times[-2]) * 1000
                    yield _sse("token", payload)

                worker.join()
                total_seconds = time.perf_counter() - request_started
                inter_token_latencies = [
                    (later - earlier) * 1000 for earlier, later in zip(token_times, token_times[1:])
                ]
                yield _sse(
                    "done",
                    {
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated_ids),
                        "ttft_seconds": token_times[0] - request_started if token_times else None,
                        "mean_inter_token_latency_ms": (
                            sum(inter_token_latencies) / len(inter_token_latencies)
                            if inter_token_latencies
                            else 0.0
                        ),
                        "total_seconds": total_seconds,
                        "tokens_per_second": len(generated_ids) / total_seconds if total_seconds else 0.0,
                    },
                )
            except Exception:
                yield _sse("error", {"message": "Generation failed"})
            finally:
                worker.join()


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine if engine is not None else InferenceEngine.from_pretrained()
        yield

    app = FastAPI(title="LLM Inference Streaming Lab", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        active_engine: InferenceEngine = request.app.state.engine
        return {
            "status": "ok",
            "model": active_engine.model_name,
            "device": str(active_engine.device),
        }

    @app.post("/generate")
    def generate(body: GenerateRequest, request: Request) -> StreamingResponse:
        request_started = time.perf_counter()
        active_engine: InferenceEngine = request.app.state.engine
        input_ids = active_engine.encode_prompt(body.prompt, body.max_new_tokens)
        return StreamingResponse(
            active_engine.stream(input_ids, body.max_new_tokens, request_started),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()