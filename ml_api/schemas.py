"""Request and response contracts of the ML service.

Inputs are validated against what the app can really produce (same limits as the app and the
data-quality rules), and extra fields are rejected. Values derived from others, such as
is_weekend or hour_sin, are computed here: a client cannot send an inconsistent combination.
"""
from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.clustering.features import USER_FEATURES
from ml.features import FEATURE_COLUMNS


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionFeatures(Strict):
    """What is known when a session starts. History fields are absent for a first session."""

    hour: int = Field(ge=0, le=23, description="Local hour of session start")
    weekday: int = Field(ge=0, le=6, description="Local weekday, Monday = 0")
    planned_minutes: float = Field(ge=1, le=180)
    room_is_private: bool | None = None
    room_age_days: float | None = Field(default=None, ge=0)
    prev_sessions: int = Field(default=0, ge=0)
    prev_mean_minutes: float | None = Field(default=None, ge=0, le=180)
    prev_full_duration_rate: float | None = Field(default=None, ge=0, le=1)
    prev_session_minutes: float | None = Field(default=None, ge=0, le=180)
    minutes_since_prev: float | None = Field(default=None, ge=0)
    sessions_today_before: int = Field(default=0, ge=0)
    sessions_7d_before: int = Field(default=0, ge=0)
    user_tenure_days: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def first_session_has_no_history(self) -> SessionFeatures:
        history = (self.prev_mean_minutes, self.prev_full_duration_rate, self.prev_session_minutes,
                   self.minutes_since_prev, self.user_tenure_days)
        if self.prev_sessions == 0 and any(v is not None for v in history):
            raise ValueError("history fields require prev_sessions > 0")
        if self.sessions_today_before > self.prev_sessions or self.sessions_7d_before > self.prev_sessions:
            raise ValueError("session counts cannot exceed prev_sessions")
        return self

    def to_frame(self) -> pd.DataFrame:
        """One row with exactly the training columns; absent values become NaN, as in training."""
        nan = float("nan")

        def number(value) -> float:
            return nan if value is None else float(value)

        row = {
            "hour": self.hour,
            "hour_sin": math.sin(2 * math.pi * self.hour / 24),
            "hour_cos": math.cos(2 * math.pi * self.hour / 24),
            "weekday": self.weekday,
            "is_weekend": int(self.weekday >= 5),
            "planned_minutes": float(self.planned_minutes),
            "room_is_private": number(self.room_is_private),
            "room_age_days": number(self.room_age_days),
            "is_first_session": int(self.prev_sessions == 0),
            "prev_sessions": float(self.prev_sessions),
            "prev_mean_minutes": number(self.prev_mean_minutes),
            "prev_full_duration_rate": number(self.prev_full_duration_rate),
            "prev_session_minutes": number(self.prev_session_minutes),
            "minutes_since_prev": number(self.minutes_since_prev),
            "sessions_today_before": float(self.sessions_today_before),
            "sessions_7d_before": float(self.sessions_7d_before),
            "user_tenure_days": number(self.user_tenure_days),
        }
        return pd.DataFrame([row])[list(FEATURE_COLUMNS)]


class ModelInfo(BaseModel):
    """Which model answered. ``data_source`` says plainly if it learned from synthetic data."""

    name: str
    version: str
    data_source: str


class ProductivityPrediction(BaseModel):
    predicted_focused_minutes: float
    model: ModelInfo


class CompletionPrediction(BaseModel):
    probability_ended_early: float
    predicted_ended_early: bool
    threshold: float
    model: ModelInfo


class UserBehaviour(Strict):
    """The per-user rates clustering uses (see ml.clustering.features)."""

    sessions_per_week: float = Field(ge=0)
    mean_focus_minutes: float = Field(ge=0, le=180)
    full_duration_rate: float = Field(ge=0, le=1)
    long_session_share: float = Field(ge=0, le=1)
    morning_share: float = Field(ge=0, le=1)
    evening_share: float = Field(ge=0, le=1)
    night_share: float = Field(ge=0, le=1)
    weekend_share: float = Field(ge=0, le=1)
    distinct_rooms: float = Field(ge=0)
    private_room_share: float = Field(ge=0, le=1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])[list(USER_FEATURES)]


class ClusterAssignment(BaseModel):
    cluster: int
    description: str | None = Field(description="Measurable differences from the average user")
    weak_structure: bool
    model: ModelInfo


class AnalyzeRequest(Strict):
    text: str = Field(min_length=1, max_length=2000)


class AnalyzeResponse(BaseModel):
    redacted_text: str
    contains_personal_data: bool
    tokens: list[str]


class SearchRequest(Strict):
    query: str = Field(min_length=1, max_length=1000)
    user_id: int = Field(ge=1, description="Results come only from rooms this user is a member of")
    room_ids: list[int] | None = Field(default=None, max_length=100, description="Optional narrowing")
    k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    message_id: int
    room_id: int
    body: str
    similarity: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class Turn(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class RagRequest(Strict):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=10)
    user_id: int | None = Field(default=None, ge=1, description="Enables tools over this user's own data")


class RagSource(BaseModel):
    n: int
    title: str
    similarity: float


class RagResponse(BaseModel):
    reply: str
    sources: list[RagSource]
    grounded: bool
    answerable: bool | None
    tools_used: list[str]


class ModelSummary(BaseModel):
    task: str
    name: str
    version: str
    data_source: str
    sklearn_version: str | None
    sklearn_matches: bool
    notes: dict


class ModelsResponse(BaseModel):
    models: list[ModelSummary]
    errors: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    rag_ready: bool
    search_ready: bool
