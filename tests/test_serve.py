import json
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from serve import InferenceEngine, create_app


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, prompt, return_tensors):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join({3: "Hello", 4: " world"}.get(token_id, "") for token_id in token_ids)


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=8)

    def generate(self, input_ids, streamer, max_new_tokens, **kwargs):
        streamer.put(input_ids.cpu())
        for token_id in (3, 4)[:max_new_tokens]:
            streamer.put(torch.tensor([token_id]))


def make_client() -> TestClient:
    engine = InferenceEngine(FakeModel(), FakeTokenizer(), torch.device("cpu"), "fake-model")
    return TestClient(create_app(engine))


def test_health_reports_loaded_engine() -> None:
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake-model", "device": "cpu"}


def test_generate_streams_tokens_and_latency_metrics() -> None:
    with make_client() as client:
        response = client.post("/generate", json={"prompt": "Hi", "max_new_tokens": 2})

    assert response.status_code == 200
    lines = response.text.splitlines()
    events = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    assert events == ["token", "token", "done"]
    assert [payload["text"] for payload in payloads[:2]] == ["Hello", "Hello world"]
    assert payloads[0]["ttft_seconds"] >= 0
    assert payloads[1]["inter_token_latency_ms"] >= 0
    assert payloads[2]["prompt_tokens"] == 2
    assert payloads[2]["generated_tokens"] == 2
    assert payloads[2]["mean_inter_token_latency_ms"] >= 0


def test_generate_rejects_context_overflow() -> None:
    with make_client() as client:
        response = client.post("/generate", json={"prompt": "Hi", "max_new_tokens": 7})

    assert response.status_code == 422