"""FocusBot tools: read-only, scoped to the logged-in user, tolerant of bad model output."""
from __future__ import annotations

import json

from ai.llm.client import ToolCall
from ai.rag import tools
from ai.rag.tools import MAX_CALLS_PER_QUESTION, ToolContext, run_tool_call, run_tool_calls, tool_definitions
from main.db import db
from models.todo import TodoItem


def call(name, arguments="{}"):
    return ToolCall("id", name, arguments)


def test_no_tool_lets_the_model_choose_the_user():
    """The security property: identity comes from the session, never from model arguments."""
    for definition in tool_definitions():
        properties = definition["function"]["parameters"].get("properties", {})
        assert not {"user", "user_id", "username", "email"} & set(properties)


def test_definitions_are_valid_function_schemas():
    for definition in tool_definitions():
        assert definition["type"] == "function"
        assert definition["function"]["parameters"]["type"] == "object"
        assert definition["function"]["description"]


class TestScoping:
    def test_tasks_belong_to_the_context_user_only(self, app, make_user):
        alice, bob = make_user("alice"), make_user("bob")
        db.session.add_all([
            TodoItem(user_id=alice.id, body="alice open"),
            TodoItem(user_id=alice.id, body="alice done", done=True),
            TodoItem(user_id=bob.id, body="bob open"),
        ])
        db.session.commit()

        result = run_tool_call(call("list_my_open_tasks"), ToolContext(alice.id))

        assert result == {"open_tasks": ["alice open"], "count": 1}

    def test_a_user_argument_from_the_model_is_rejected(self, app, make_user):
        alice, bob = make_user("alice"), make_user("bob")
        db.session.add(TodoItem(user_id=bob.id, body="bob secret"))
        db.session.commit()

        result = run_tool_call(call("list_my_open_tasks", json.dumps({"user_id": bob.id})), ToolContext(alice.id))

        assert result == {"error": "invalid arguments"}

    def test_focus_stats(self, app, make_user):
        user = make_user()
        result = run_tool_call(call("get_my_focus_stats"), ToolContext(user.id))

        assert result["today"] == "0m" and result["sessions"] == 0
        assert "owner" in result["note"]
        assert "username" not in result and "email" not in result


class TestBadModelOutput:
    def test_unknown_tool(self):
        assert "unknown tool" in run_tool_call(call("drop_database"), ToolContext(1))["error"]

    def test_invalid_json(self):
        assert run_tool_call(call("list_my_open_tasks", "{not json"), ToolContext(1)) == {"error": "invalid arguments"}

    def test_out_of_range_argument(self):
        assert run_tool_call(call("list_my_open_tasks", '{"limit": 100000}'), ToolContext(1)) == {
            "error": "invalid arguments"}

    def test_failing_tool_does_not_leak_or_raise(self, monkeypatch):
        def boom(_ctx, _args):
            raise RuntimeError("password=hunter2")

        tool = tools.Tool("broken", "d", tools.NoArguments, boom)
        monkeypatch.setitem(tools.TOOLS, "broken", tool)

        assert run_tool_call(call("broken"), ToolContext(1)) == {"error": "this data is temporarily unavailable"}

    def test_number_of_calls_is_capped(self, monkeypatch):
        seen = []
        tool = tools.Tool("count", "d", tools.NoArguments, lambda _c, _a: seen.append(1) or {})
        monkeypatch.setitem(tools.TOOLS, "count", tool)

        run_tool_calls(tuple(call("count") for _ in range(50)), ToolContext(1))

        assert len(seen) == MAX_CALLS_PER_QUESTION
