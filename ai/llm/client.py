"""LLM client: one call shape, typed responses, typed errors, usage and latency tracking.

Retries are left to the Groq SDK (``max_retries``): it already backs off on 429 and 5xx and
honours Retry-After, so a second retry loop here would only multiply the attempts.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai.llm.telemetry import LLMCall, UsageTracker, tracker

# llama-3.1-8b-instant was shut down for free and developer tiers on 2026-08-16; requests to it
# return errors (console.groq.com/docs/deprecations). Groq names this model as its replacement.
DEFAULT_MODEL = "openai/gpt-oss-20b"


def configured_model() -> str:
    """Model ids get retired, so the model is configuration rather than code."""
    return os.getenv("FOCUSBOT_MODEL") or DEFAULT_MODEL


class LLMUnavailable(RuntimeError):
    """The LLM could not produce a response. Messages never contain provider details or keys."""


class LLMNotConfigured(LLMUnavailable):
    """No API key."""


class LLMRateLimited(LLMUnavailable):
    """HTTP 429, after the SDK's retries."""


class LLMTimeout(LLMUnavailable):
    """No response in time, after the SDK's retries."""


class LLMBadRequest(LLMUnavailable):
    """HTTP 4xx about the request itself, e.g. a malformed tool call the model generated, or a
    feature the configured model does not support. Retrying the same request will not help."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON from the model: untrusted until validated


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        feature: str = "unknown",
        prompt_version: str | None = None,
    ) -> LLMResponse: ...


def _error_type(exc: Exception) -> type[LLMUnavailable]:
    import groq

    if isinstance(exc, groq.RateLimitError):
        return LLMRateLimited
    if isinstance(exc, groq.APITimeoutError):
        return LLMTimeout
    # Authentication problems are ours to fix, not a bad request from this call.
    if isinstance(exc, groq.APIStatusError) and 400 <= exc.status_code < 500 and exc.status_code not in (401, 403):
        return LLMBadRequest
    return LLMUnavailable


class GroqLLM:
    """Groq chat completions."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_completion_tokens: int = 1024,
        reasoning_effort: str | None = "low",
        timeout: float = 20.0,
        max_retries: int = 2,
        api_key: str | None = None,
        client: Any = None,
        usage_tracker: UsageTracker = tracker,
    ):
        self.model = model or configured_model()
        # Low temperature: for grounded answers we want the most likely wording of the sources.
        self.temperature = temperature
        # Reasoning models spend completion tokens on thinking too, so this cannot be as tight
        # as the length of the visible answer, or the reply can come back truncated or empty.
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client = client
        self.usage_tracker = usage_tracker

    def _sdk_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _params(self, messages, tools, response_format) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.model.startswith("openai/gpt-oss"):
            # We neither show nor store the model's reasoning, so don't request it.
            params["include_reasoning"] = False
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if response_format:
            params["response_format"] = response_format
        return params

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        feature: str = "unknown",
        prompt_version: str | None = None,
    ) -> LLMResponse:
        if not self.api_key and self._client is None:
            raise LLMNotConfigured("GROQ_API_KEY is not set")
        started = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        try:
            raw = self._sdk_client().chat.completions.create(**self._params(messages, tools, response_format))
        except Exception as exc:  # noqa: BLE001 - every provider failure becomes one typed error
            error = _error_type(exc)
            self.usage_tracker.record(LLMCall(feature, self.model, elapsed(), False, error=error.__name__,
                                              prompt_version=prompt_version))
            raise error(f"LLM request failed: {error.__name__}") from exc

        latency = elapsed()
        choice = raw.choices[0]
        usage = getattr(raw, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        finish_reason = getattr(choice, "finish_reason", None)
        calls = getattr(choice.message, "tool_calls", None) or []
        self.usage_tracker.record(LLMCall(
            feature, self.model, latency, True, prompt_tokens, completion_tokens,
            getattr(details, "reasoning_tokens", 0) or 0, finish_reason, prompt_version=prompt_version,
        ))
        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tuple(ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in calls),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            model=self.model,
        )
