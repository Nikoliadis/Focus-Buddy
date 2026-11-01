"""Check the configured model against the real API: plain text, structured output, tool call.

The test suite runs offline, so it cannot notice a retired model or a rejected parameter.
This can. It needs GROQ_API_KEY and spends a few tokens.

Usage:
    python -m ai.llm.smoke
"""
from __future__ import annotations

from dotenv import load_dotenv

from ai.llm.client import GroqLLM, LLMUnavailable
from ai.llm.structured import GroundedAnswer, StructuredOutputError, complete_structured
from ai.llm.telemetry import tracker
from ai.rag.tools import tool_definitions


def run_checks(llm) -> dict[str, tuple[bool, str]]:
    def plain():
        text = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], feature="smoke").text
        return bool(text.strip()), text.strip()[:60]

    def structured():
        messages = [{"role": "system", "content": "No sources are available. Set answerable to false."},
                    {"role": "user", "content": "How much does FocusBuddy premium cost?"}]
        answer = complete_structured(llm, messages, GroundedAnswer, "smoke")
        return answer.answerable is False, f"answerable={answer.answerable}"

    def tools():
        response = llm.chat([{"role": "user", "content": "How much did I study this week? Check my stats."}],
                            tools=tool_definitions(), feature="smoke")
        names = [c.name for c in response.tool_calls]
        return "get_my_focus_stats" in names, f"tool calls: {names}"

    results = {}
    for name, check in (("plain text", plain), ("structured output", structured), ("tool call", tools)):
        try:
            results[name] = check()
        except (LLMUnavailable, StructuredOutputError) as exc:
            results[name] = (False, str(exc))
    return results


def main() -> int:
    load_dotenv()
    llm = GroqLLM()
    print(f"Model: {llm.model}")
    results = run_checks(llm)
    for name, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print(f"Usage: {tracker.summary().get('smoke')}")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
