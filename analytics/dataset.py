"""Load + clean everything the analyses need, in one call."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics import cleaning, data_loader
from analytics.data_loader import Bind


@dataclass(frozen=True)
class AnalyticsDataset:
    users: pd.DataFrame
    rooms: pd.DataFrame
    sessions: pd.DataFrame       # cleaned, with `quality`
    logs: pd.DataFrame           # cleaned, with `quality` and `completion_ratio`
    reports: tuple[cleaning.QualityReport, ...]


def load_dataset(bind: Bind | None = None, now: pd.Timestamp | None = None) -> AnalyticsDataset:
    """Chat is not included on purpose: it is only needed for the NLP phases."""
    sessions, s_report = cleaning.clean_focus_sessions(data_loader.load_focus_sessions(bind), now=now)
    logs, l_report = cleaning.clean_focus_logs(data_loader.load_focus_logs(bind), now=now)
    return AnalyticsDataset(
        users=data_loader.load_users(bind),
        rooms=data_loader.load_rooms(bind),
        sessions=sessions,
        logs=logs,
        reports=(s_report, l_report),
    )
