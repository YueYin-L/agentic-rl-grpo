"""Model backends kept independent from the agent runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, cast
from urllib.request import Request, urlopen

from research_router_opt.analysis_models import AgentMessage, ModelReply, ToolCall


class ModelBackend(Protocol):
    name: str

    def reset(self) -> None:
        """Reset per-episode backend state without changing model weights."""

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelReply:
        """Generate one assistant turn."""


class MockBackend:
    """Deterministic scripted backend used by CPU-only runtime tests."""

    name = "mock"

    def __init__(self, replies: Sequence[ModelReply], *, repeat_last: bool = False) -> None:
        if not replies:
            raise ValueError("MockBackend requires at least one reply.")
        self._replies = tuple(replies)
        self._repeat_last = repeat_last
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelReply:
        del messages, tools, timeout_s
        if self._cursor >= len(self._replies):
            if self._repeat_last:
                return self._replies[-1]
            raise RuntimeError("MockBackend script exhausted before the episode terminated.")
        reply = self._replies[self._cursor]
        self._cursor += 1
        return reply


class VLLMBackend:
    """Minimal OpenAI-compatible vLLM backend with no import-time GPU dependency."""

    name = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 1024,
        seed: int = 0,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url and model must be non-empty.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.seed = seed

    def reset(self) -> None:
        return None

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelReply:
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        message = cast(dict[str, Any], body["choices"][0]["message"])
        tool_calls = cast(list[dict[str, Any]], message.get("tool_calls") or [])
        if tool_calls:
            raw_call = tool_calls[0]
            function = cast(dict[str, Any], raw_call.get("function") or {})
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(str(raw_arguments))
            except json.JSONDecodeError:
                arguments = {"_malformed_arguments": str(raw_arguments)}
            if not isinstance(arguments, dict):
                arguments = {"_malformed_arguments": raw_arguments}
            return ModelReply(
                tool_call=ToolCall(
                    call_id=str(raw_call.get("id") or "missing-call-id"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )
        content = message.get("content")
        return ModelReply(content=str(content) if content is not None else None)
