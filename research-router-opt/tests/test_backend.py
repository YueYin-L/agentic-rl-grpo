from __future__ import annotations

import json
from typing import Any

from research_router_opt.analysis_models import AgentMessage
from research_router_opt.backend import VLLMBackend


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": "done", "tool_calls": []}}]}
        ).encode()


def test_vllm_backend_sends_frozen_seed(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        captured.update(json.loads(request.data.decode()))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("research_router_opt.backend.urlopen", fake_urlopen)
    backend = VLLMBackend(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3.5-9B",
        seed=20260920,
    )
    reply = backend.generate(
        [AgentMessage(role="user", content="hello")],
        [],
        timeout_s=3.0,
    )

    assert reply.content == "done"
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 20260920
    assert captured["timeout"] == 3.0
