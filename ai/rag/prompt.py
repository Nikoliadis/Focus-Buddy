"""Context construction and prompt templates for grounded answers.

The prompt is the main hallucination control we have:
- sources are numbered, and the model must cite them as [1], [2];
- for questions about FocusBuddy it may use ONLY the sources; if they do not contain the
  answer it must say so instead of guessing (an 8B model will otherwise invent features);
- general study advice is allowed, so FocusBot is still useful for "how do I focus?".
"""
from __future__ import annotations

from ai.rag.retriever import Retrieved

# A rough budget: ~4 characters per token for English. The model's window is far larger,
# but more context means more cost and latency, and facts buried in the middle of a long
# context are used less reliably ("lost in the middle").
CONTEXT_TOKEN_BUDGET = 1200
CHARS_PER_TOKEN = 4

SYSTEM_PROMPT = """You are FocusBot, the study assistant inside FocusBuddy, a real-time study app.

Rules:
1. For questions about FocusBuddy (its features, settings, limits or behaviour), use ONLY the numbered sources below. Cite every fact with its source number, like [1] or [2].
2. If the sources do not contain the answer to a FocusBuddy question, say that you do not know and suggest where in the app to look. Never invent features, buttons, limits or numbers.
3. For general study questions you may also use general knowledge; cite a source when one supports the answer.
4. Answer in the language of the question. Be concise and encouraging; under 150 words unless asked for more."""

NO_SOURCES = "No help articles matched this question."


def build_context(retrieved: list[Retrieved], token_budget: int = CONTEXT_TOKEN_BUDGET) -> tuple[str, list[Retrieved]]:
    """Numbered sources, most relevant first, until the budget is used. Returns the text and
    the sources actually included (their numbers are their 1-based positions)."""
    parts, used, spent = [], [], 0
    for item in retrieved:
        block = f"[{len(used) + 1}] {item.chunk.heading}\n{item.chunk.text}"
        cost = len(block) // CHARS_PER_TOKEN + 1
        if used and spent + cost > token_budget:
            break
        parts.append(block)
        used.append(item)
        spent += cost
    return ("\n\n".join(parts) if parts else NO_SOURCES), used


def build_messages(question: str, context: str, history: list[dict[str, str]] | None = None,
                   max_history: int = 6) -> list[dict[str, str]]:
    """System rules + sources, recent conversation, then the question.

    Only user/assistant turns from the client are kept: a client cannot inject a "system"
    message to override the rules."""
    turns = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSources:\n{context}"},
        *[{"role": m["role"], "content": str(m["content"])[:2000]} for m in turns[-max_history:]],
        {"role": "user", "content": question},
    ]
