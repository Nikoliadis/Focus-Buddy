"""Read-only tools FocusBot can call, for questions the help articles cannot answer
("how much did I study this week?").

Security model: the model chooses WHICH tool to call, never WHOSE data it reads. The user id
comes from the logged-in session through ``ToolContext`` and is not a tool argument, so no
prompt can make a tool return another user's data. Tool arguments are model output and are
validated like any other untrusted input. A failing tool returns an error object to the
model instead of failing the whole answer.

Room chat is deliberately not a tool: it is noisy (Phase 16), and text written by other
users would reach the prompt, opening the door to indirect prompt injection.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai.llm.client import ToolCall

logger = logging.getLogger("focusbuddy.llm")

MAX_CALLS_PER_QUESTION = 3


@dataclass(frozen=True)
class ToolContext:
    user_id: int
    # Opens what the handlers need for database access, around each tool run only, so no
    # connection is held during the slow LLM call. The ML service passes a Flask app context.
    scope: Callable = nullcontext


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenTasksArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, ge=1, le=20, description="How many tasks to return, 1 to 20.")


def _focus_stats(ctx: ToolContext, _args: NoArguments) -> dict:
    from main.stats_service import get_user_stats

    stats = get_user_stats(ctx.user_id)
    return {
        "today": stats["today_fmt"], "last_7_days": stats["week_fmt"], "all_time": stats["total_fmt"],
        "best_day": stats["best_day_fmt"], "sessions": stats["session_count"],
        "current_streak_days": stats["current_streak"], "best_streak_days": stats["longest_streak"],
        "daily_goal_minutes": stats["daily_goal_minutes"], "daily_goal_percent_today": stats["daily_goal_pct"],
        "note": "Sessions are credited to the room owner, so time in other people's rooms is not included.",
    }


def _open_tasks(ctx: ToolContext, args: OpenTasksArguments) -> dict:
    from models.todo import TodoItem

    tasks = (TodoItem.query.filter_by(user_id=ctx.user_id, done=False)
             .order_by(TodoItem.created_at.asc()).limit(args.limit).all())
    return {"open_tasks": [t.body for t in tasks], "count": len(tasks)}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], dict]

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.arguments.model_json_schema()}}


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("get_my_focus_stats",
         "The current user's own focus statistics: time today, last 7 days and all time, streaks and daily goal.",
         NoArguments, _focus_stats),
    Tool("list_my_open_tasks", "The current user's own unfinished tasks from My Tasks.",
         OpenTasksArguments, _open_tasks),
)}


def tool_definitions() -> list[dict]:
    return [t.definition() for t in TOOLS.values()]


def run_tool_call(call: ToolCall, ctx: ToolContext) -> dict:
    tool = TOOLS.get(call.name)
    if tool is None:
        return {"error": f"unknown tool {call.name!r}"}
    try:
        args = tool.arguments.model_validate(json.loads(call.arguments or "{}"))
    except (json.JSONDecodeError, ValidationError):
        return {"error": "invalid arguments"}
    try:
        with ctx.scope():
            return tool.handler(ctx, args)
    except Exception:  # noqa: BLE001 - one failed tool must not lose the whole answer
        logger.exception("tool %s failed", call.name)
        return {"error": "this data is temporarily unavailable"}


def run_tool_calls(calls: tuple[ToolCall, ...], ctx: ToolContext) -> dict[str, dict]:
    """Results by tool name, capped so a confused model cannot trigger unlimited queries."""
    return {call.name: run_tool_call(call, ctx) for call in calls[:MAX_CALLS_PER_QUESTION]}
