"""LLM client, structured output and telemetry.

The client tests run the real Groq SDK over a mock HTTP transport: the request JSON, the
response parsing and the SDK's exceptions are all real, only the network is not.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest
from groq import Groq
from pydantic import BaseModel, ConfigDict

from ai.llm import client as llm_client
from ai.llm.client import (
    GroqLLM,
    LLMBadRequest,
    LLMNotConfigured,
    LLMRateLimited,
    LLMResponse,
    LLMTimeout,
    LLMUnavailable,
    ToolCall,
    configured_model,
)
from ai.llm.structured import (
    GroundedAnswer,
    GroundednessVerdict,
    StructuredOutputError,
    complete_structured,
    json_schema_format,
    parse,
)
from ai.llm.telemetry import LLMCall, UsageTracker, estimate_cost

MESSAGES = [{"role": "user", "content": "my private question"}]


def completion(content="hello", tool_calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message, "logprobs": None}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                  "completion_tokens_details": {"reasoning_tokens": 15}},
    }


class Server:
    """Records requests and replies with a fixed status and body."""

    def __init__(self, status=200, body=None, raises=None, headers=None):
        self.status, self.body, self.raises, self.headers = status, body or completion(), raises, headers
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.raises:
            raise self.raises
        return httpx.Response(self.status, json=self.body, headers=self.headers)


def make_llm(server: Server, max_retries=0, **kwargs) -> tuple[GroqLLM, UsageTracker]:
    usage = UsageTracker()
    sdk = Groq(api_key="test-key", max_retries=max_retries,
               http_client=httpx.Client(transport=httpx.MockTransport(server)))
    return GroqLLM(client=sdk, usage_tracker=usage, **kwargs), usage


class TestRequest:
    def test_default_model_is_not_a_retired_one(self, monkeypatch):
        monkeypatch.delenv("FOCUSBOT_MODEL", raising=False)
        assert configured_model() == "openai/gpt-oss-20b"
        monkeypatch.setenv("FOCUSBOT_MODEL", "other/model")
        assert GroqLLM(api_key="k").model == "other/model"

    def test_body_for_a_reasoning_model(self):
        server = Server()
        llm, _ = make_llm(server, model="openai/gpt-oss-20b")

        llm.chat(MESSAGES)

        body = server.requests[0]
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["messages"] == MESSAGES
        assert body["max_completion_tokens"] == 1024
        assert body["include_reasoning"] is False and body["reasoning_effort"] == "low"
        assert "tools" not in body and "response_format" not in body

    def test_reasoning_params_are_not_sent_to_other_models(self):
        server = Server()
        make_llm(server, model="some/other-model")[0].chat(MESSAGES)
        assert "include_reasoning" not in server.requests[0] and "reasoning_effort" not in server.requests[0]

    def test_tools_and_schema_are_sent_when_given(self):
        server = Server()
        llm, _ = make_llm(server)
        tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}

        llm.chat(MESSAGES, tools=[tool])
        llm.chat(MESSAGES, response_format=json_schema_format(GroundedAnswer))

        assert server.requests[0]["tools"] == [tool] and server.requests[0]["tool_choice"] == "auto"
        assert server.requests[1]["response_format"]["json_schema"]["strict"] is True


class TestResponse:
    def test_text_and_usage(self):
        llm, _ = make_llm(Server())
        response = llm.chat(MESSAGES)

        assert (response.text, response.prompt_tokens, response.completion_tokens) == ("hello", 100, 40)
        assert response.finish_reason == "stop" and not response.truncated and response.latency_ms >= 0

    def test_tool_calls_are_parsed(self):
        call = {"id": "c1", "type": "function", "function": {"name": "get_my_focus_stats", "arguments": "{}"}}
        llm, _ = make_llm(Server(body=completion(None, [call], "tool_calls")))

        response = llm.chat(MESSAGES)

        assert response.tool_calls == (ToolCall("c1", "get_my_focus_stats", "{}"),)
        assert response.text == ""

    def test_truncation_is_visible(self):
        llm, _ = make_llm(Server(body=completion('{"answer": "cut', finish_reason="length")))
        assert llm.chat(MESSAGES).truncated


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "error"),
        [(429, LLMRateLimited), (400, LLMBadRequest), (404, LLMBadRequest),
         (401, LLMUnavailable), (500, LLMUnavailable), (503, LLMUnavailable)],
    )
    def test_status_codes_become_typed_errors(self, status, error):
        llm, usage = make_llm(Server(status, {"error": {"message": "secret provider detail", "code": "x"}}))

        with pytest.raises(error) as caught:
            llm.chat(MESSAGES)

        assert type(caught.value) is error
        assert "secret" not in str(caught.value)
        assert usage.summary()["unknown"]["error_rate"] == 1

    def test_timeout(self):
        llm, _ = make_llm(Server(raises=httpx.ReadTimeout("slow")))
        with pytest.raises(LLMTimeout):
            llm.chat(MESSAGES)

    def test_connection_failure(self):
        llm, _ = make_llm(Server(raises=httpx.ConnectError("down")))
        with pytest.raises(LLMUnavailable):
            llm.chat(MESSAGES)

    def test_sdk_retries_rate_limits_once_configured(self):
        server = Server(429, {"error": {"message": "slow down"}}, headers={"retry-after": "0"})
        llm, _ = make_llm(server, max_retries=2)
        with pytest.raises(LLMRateLimited):
            llm.chat(MESSAGES)
        assert len(server.requests) == 3  # the SDK's own retries; no second loop on top

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(LLMNotConfigured):
            GroqLLM().chat(MESSAGES)


class TestTelemetry:
    def test_records_tokens_latency_and_cost(self):
        llm, usage = make_llm(Server(), model="openai/gpt-oss-20b")

        llm.chat(MESSAGES, feature="focusbot.answer")
        llm.chat(MESSAGES, feature="focusbot.answer")

        s = usage.summary()["focusbot.answer"]
        assert s["calls"] == 2 and s["error_rate"] == 0
        assert (s["prompt_tokens"], s["completion_tokens"], s["reasoning_tokens"]) == (200, 80, 30)
        assert s["cost_usd"] == pytest.approx((200 * 0.075 + 80 * 0.30) / 1e6)
        assert s["latency_p50_ms"] is not None

    def test_prompts_and_answers_are_never_logged(self, caplog):
        llm, _ = make_llm(Server(body=completion("private answer")), model="m")

        with caplog.at_level(logging.INFO, logger="focusbuddy.llm"):
            llm.chat(MESSAGES, feature="f", prompt_version="v1")

        record = json.loads(caplog.records[-1].getMessage())
        assert record["feature"] == "f" and record["prompt_version"] == "v1"
        assert "private" not in caplog.text

    def test_percentiles(self):
        usage = UsageTracker()
        for ms in range(1, 101):
            usage.record(LLMCall("f", "m", float(ms), True))
        s = usage.summary()["f"]
        assert (s["latency_p50_ms"], s["latency_p95_ms"]) == (50.0, 95.0)

    def test_failed_calls_do_not_distort_latency(self):
        usage = UsageTracker()
        usage.record(LLMCall("f", "m", 10.0, True))
        usage.record(LLMCall("f", "m", 20000.0, False, error="LLMTimeout"))
        s = usage.summary()["f"]
        assert s["latency_p95_ms"] == 10.0 and s["error_rate"] == 0.5

    def test_bounded_memory(self):
        usage = UsageTracker(max_records=3)
        for _ in range(10):
            usage.record(LLMCall("f", "m", 1.0, True))
        assert usage.summary()["f"]["calls"] == 3

    def test_unknown_model_has_no_invented_price(self):
        assert estimate_cost("some/model", 100, 100) is None


class Scripted:
    """Fake LLM replying with scripted texts, recording what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply)


GOOD = '{"answer": "Yes [1].", "answerable": true, "cited_sources": [1]}'


class TestStructured:
    def test_schema_satisfies_strict_mode(self):
        """Strict mode needs every property required and no additional properties."""
        for model in (GroundedAnswer, GroundednessVerdict):
            schema = json_schema_format(model)["json_schema"]["schema"]
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])

    def test_valid_output(self):
        answer = complete_structured(Scripted(GOOD), MESSAGES, GroundedAnswer, "f")
        assert answer.answerable is True and answer.cited_sources == [1]

    @pytest.mark.parametrize("bad", ["not json", '{"answer": "cut', '{"answer": "no flags"}',
                                     '{"answer": "x", "answerable": true, "cited_sources": [], "extra": 1}'])
    def test_invalid_output_is_retried_with_the_error(self, bad):
        llm = Scripted(bad, GOOD)

        answer = complete_structured(llm, MESSAGES, GroundedAnswer, "f")

        assert answer.answer == "Yes [1]."
        assert llm.calls[1][-1]["role"] == "user" and "valid" in llm.calls[1][-1]["content"]

    def test_gives_up_after_the_retry(self):
        llm = Scripted("nope", "still nope", "never reached")
        with pytest.raises(StructuredOutputError):
            complete_structured(llm, MESSAGES, GroundedAnswer, "f")
        assert len(llm.calls) == 2

    def test_rejected_request_is_not_retried(self):
        llm = Scripted(LLMBadRequest("no json_schema for this model"), GOOD)
        with pytest.raises(StructuredOutputError):
            complete_structured(llm, MESSAGES, GroundedAnswer, "f")
        assert len(llm.calls) == 1

    def test_outages_propagate(self):
        with pytest.raises(LLMRateLimited):
            complete_structured(Scripted(LLMRateLimited("x")), MESSAGES, GroundedAnswer, "f")

    def test_works_with_any_schema(self):
        class Mood(BaseModel):
            model_config = ConfigDict(extra="forbid")
            label: str

        assert complete_structured(Scripted('{"label": "ok"}'), MESSAGES, Mood, "f").label == "ok"
        with pytest.raises(StructuredOutputError):
            parse(Mood, "{}")


def test_module_exposes_no_retired_model():
    assert "llama" not in llm_client.DEFAULT_MODEL
