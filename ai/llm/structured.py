"""Structured outputs: the model returns JSON matching a Pydantic schema, not free text.

Code can then act on fields ("answerable": false) instead of searching the reply for phrases
like "I don't know". Groq's strict ``json_schema`` mode guarantees the *shape* on supported
models; Pydantic still validates here, because the response can be truncated and because
validators can check what a JSON schema cannot.

Strict mode requires every property to be listed as required and no extra properties, so
schema classes use ``extra="forbid"`` and fields without defaults.
"""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai.llm.client import LLM, LLMBadRequest

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """The model did not produce valid output for the schema, even after a retry."""


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The reply to the user, citing sources like [1].")
    answerable: bool = Field(description="False if the sources do not answer a question about FocusBuddy.")
    cited_sources: list[int] = Field(description="Numbers of the sources the answer relies on.")


class GroundednessVerdict(BaseModel):
    """An LLM acting as a judge of another answer, used in evaluation only."""

    model_config = ConfigDict(extra="forbid")

    supported: bool = Field(description="True only if every factual claim is backed by the sources.")
    unsupported_claims: list[str] = Field(description="Claims not backed by the sources; empty if none.")


def json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}}


def parse(schema: type[T], text: str) -> T:
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise StructuredOutputError(f"{exc.error_count()} validation error(s)") from exc


def complete_structured(
    llm: LLM,
    messages: list[dict[str, Any]],
    schema: type[T],
    feature: str,
    prompt_version: str | None = None,
    attempts: int = 2,
) -> T:
    """Call the LLM for ``schema``. Invalid output is sent back once with the validation error,
    since the model can usually correct it. Provider outages are not retried here (the SDK
    did), and a 400 becomes StructuredOutputError so the caller can fall back to plain text."""
    conversation = list(messages)
    for attempt in range(attempts):
        try:
            response = llm.chat(conversation, response_format=json_schema_format(schema),
                                feature=feature, prompt_version=prompt_version)
        except LLMBadRequest as exc:
            raise StructuredOutputError("the request was rejected") from exc
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as exc:
            if attempt == attempts - 1:
                raise StructuredOutputError(f"{exc.error_count()} validation error(s)") from exc
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            conversation += [
                {"role": "assistant", "content": response.text[:2000]},
                {"role": "user", "content": f"That was not valid for the required JSON schema ({problems}). "
                                            "Reply again with only valid JSON."},
            ]
    raise StructuredOutputError("no attempts were made")
