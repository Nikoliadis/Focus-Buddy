"""Retrieval-augmented generation for FocusBot.

    question ─► embed ─► retrieve (threshold, one chunk per section) ─► numbered context
             ─► prompt (rules + sources + recent turns) ─► LLM ─► answer
             ─► post-check citations ─► answer + the sources it can point to
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ai.llm.client import LLM
from ai.rag.prompt import build_context, build_messages
from ai.rag.retriever import KnowledgeRetriever, Retrieved

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]                # the sources the answer cites (or all used, if none cited)
    retrieved: list[Retrieved]
    invalid_citations: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """At least one retrieved source was available to the model."""
        return bool(self.retrieved)

    def to_json(self) -> dict:
        return {"reply": self.answer, "sources": self.sources, "grounded": self.grounded}


def check_citations(answer: str, n_sources: int) -> tuple[str, set[int], list[int]]:
    """Remove citation markers that point to no source (a cited "[7]" with 3 sources is a
    hallucinated reference) and report which valid sources were cited."""
    cited, invalid = set(), []

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            cited.add(n)
            return match.group(0)
        invalid.append(n)
        return ""

    cleaned = CITATION_RE.sub(replace, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), cited, invalid


class RAGPipeline:
    def __init__(self, retriever: KnowledgeRetriever, llm: LLM, k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> RAGAnswer:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, self.k)
        t1 = time.perf_counter()
        context, used = build_context(retrieved)
        raw = self.llm.complete(build_messages(question, context, history))
        t2 = time.perf_counter()

        text, cited, invalid = check_citations(raw, len(used))
        shown = [i for i in range(1, len(used) + 1) if i in cited] or list(range(1, len(used) + 1))
        sources = [{"n": i, "title": used[i - 1].chunk.heading, "similarity": round(used[i - 1].similarity, 3)}
                   for i in shown]
        return RAGAnswer(text, sources, used, invalid,
                         {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000})
