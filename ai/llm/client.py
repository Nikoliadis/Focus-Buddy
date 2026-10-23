"""Minimal LLM client interface. Phase 18 adds structured outputs, token and latency tracking."""
from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL = "llama-3.1-8b-instant"


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class LLMUnavailable(RuntimeError):
    """No API key configured, or the provider failed."""


class GroqLLM:
    """Groq chat completions (the provider FocusBot already uses)."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 512,
                 api_key: str | None = None):
        self.model = model
        # Low temperature: for grounded answers we want the most likely wording of what the
        # sources say, not creative variation.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        try:
            from groq import Groq

            response = Groq(api_key=self.api_key).chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors as one type
            raise LLMUnavailable(f"LLM request failed: {type(exc).__name__}") from exc
        return response.choices[0].message.content or ""
