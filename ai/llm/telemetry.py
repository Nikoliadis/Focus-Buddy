"""Token usage, latency and error tracking for LLM calls.

Only metadata is recorded: model, feature, tokens, latency, outcome. Prompts and answers
contain user text and are never logged. Records go to the ``focusbuddy.llm`` logger as one
JSON line each, and to a bounded per-process tracker for quick summaries. Durable metrics
and alerting belong to the monitoring phase.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("focusbuddy.llm")

# USD per 1M tokens (input, output), from console.groq.com/docs/models on 2026-11-02.
PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60)}


def configure_logging() -> None:
    """Send INFO records of the ``focusbuddy.*`` loggers to stderr. Python's default drops INFO,
    so without this the JSON records above go nowhere. Called once by each process entry point;
    library code never configures logging. Safe to call twice."""
    root = logging.getLogger("focusbuddy")
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD, or None for a model without a known price."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


@dataclass(frozen=True)
class LLMCall:
    feature: str
    model: str
    latency_ms: float
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None
    prompt_version: str | None = None
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; enough for a small in-process summary."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


class UsageTracker:
    def __init__(self, max_records: int = 1000):
        self._records: deque[LLMCall] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, call: LLMCall) -> None:
        with self._lock:
            self._records.append(call)
        logger.info(json.dumps({**asdict(call), "cost_usd": call.cost_usd}))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> dict[str, dict]:
        """Per feature: calls, error rate, tokens, estimated cost, latency p50/p95 of successes."""
        with self._lock:
            records = list(self._records)
        out: dict[str, dict] = {}
        for feature in sorted({r.feature for r in records}):
            group = [r for r in records if r.feature == feature]
            ok = sorted(r.latency_ms for r in group if r.ok)
            costs = [r.cost_usd for r in group if r.cost_usd is not None]
            out[feature] = {
                "calls": len(group),
                "error_rate": sum(not r.ok for r in group) / len(group),
                "prompt_tokens": sum(r.prompt_tokens for r in group),
                "completion_tokens": sum(r.completion_tokens for r in group),
                "reasoning_tokens": sum(r.reasoning_tokens for r in group),
                "cost_usd": sum(costs) if costs else None,
                "latency_p50_ms": _percentile(ok, 0.5) if ok else None,
                "latency_p95_ms": _percentile(ok, 0.95) if ok else None,
            }
        return out


tracker = UsageTracker()
