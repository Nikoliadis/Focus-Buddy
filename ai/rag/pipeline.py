"""FocusBot's answer pipeline.

    question ─► retrieve (threshold, one chunk per section) ─► numbered context
             ─► [logged-in user] tool round: may the model need the user's own data?
             ─► structured answer {answer, answerable, cited_sources}
             ─► post-check citations ─► answer + the sources it can point to

Two LLM calls when tools are available, because Groq does not support tools and structured
output in the same request. Tool results go into the system message as data rather than as
tool-role messages, so the final call has no dependency on tool support.

Degrades instead of failing: if the tool round is rejected the answer is produced without
user data, and if structured output fails the answer is plain text with answerable unknown.
Outages (rate limit, timeout, no key) propagate as LLMUnavailable for the endpoint to report.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from ai.llm.client import LLM, LLMBadRequest
from ai.llm.structured import GroundedAnswer, StructuredOutputError, complete_structured, parse
from ai.rag.prompt import PROMPT_VERSION, build_context, build_messages
from ai.rag.retriever import KnowledgeRetriever, Retrieved
from ai.rag.tools import ToolContext, run_tool_calls, tool_definitions

logger = logging.getLogger("focusbuddy.llm")

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]                # the sources the answer cites (none if it says it cannot answer)
    retrieved: list[Retrieved]
    answerable: bool | None = None     # None: structured output failed, so the model did not say
    tools_used: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """At least one retrieved source was available to the model."""
        return bool(self.retrieved)

    def to_json(self) -> dict:
        return {"reply": self.answer, "sources": self.sources, "grounded": self.grounded,
                "answerable": self.answerable, "tools_used": self.tools_used}


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

    def _user_data(self, question, context, history, tool_context) -> dict:
        """First call: let the model pick tools. Skipped when nobody is logged in."""
        if tool_context is None:
            return {}
        try:
            response = self.llm.chat(build_messages(question, context, history), tools=tool_definitions(),
                                     feature="focusbot.tools", prompt_version=PROMPT_VERSION)
        except LLMBadRequest:
            # Models sometimes generate malformed tool calls, which the API rejects with a 400.
            logger.warning("tool round rejected; answering without user data")
            return {}
        return run_tool_calls(response.tool_calls, tool_context)

    def _final(self, messages) -> GroundedAnswer | str:
        try:
            return complete_structured(self.llm, messages, GroundedAnswer, "focusbot.answer", PROMPT_VERSION)
        except StructuredOutputError:
            logger.warning("structured output failed; falling back to plain text")
        text = self.llm.chat(messages, feature="focusbot.answer_plain", prompt_version=PROMPT_VERSION).text
        try:
            return parse(GroundedAnswer, text)  # the prompt asks for JSON, so plain mode may still return it
        except StructuredOutputError:
            return text

    def answer(self, question: str, history: list[dict[str, str]] | None = None,
               tool_context: ToolContext | None = None) -> RAGAnswer:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, self.k)
        context, used = build_context(retrieved)
        t1 = time.perf_counter()
        user_data = self._user_data(question, context, history, tool_context)
        t2 = time.perf_counter()
        result = self._final(build_messages(question, context, history, user_data=user_data))
        t3 = time.perf_counter()

        structured = isinstance(result, GroundedAnswer)
        text, cited, invalid = check_citations(result.answer if structured else result, len(used))
        answerable = result.answerable if structured else None
        if structured:
            cited |= {n for n in result.cited_sources if 1 <= n <= len(used)}
            invalid += [n for n in result.cited_sources if not 1 <= n <= len(used) and n not in invalid]
        # A source is shown only if it was cited; a "cannot answer" reply points to nothing.
        shown = [] if answerable is False else sorted(cited)
        sources = [{"n": i, "title": used[i - 1].chunk.heading, "similarity": round(used[i - 1].similarity, 3)}
                   for i in shown]
        return RAGAnswer(text, sources, used, answerable, list(user_data), invalid,
                         {"retrieval": (t1 - t0) * 1000, "tools": (t2 - t1) * 1000, "generation": (t3 - t2) * 1000})
