"""Print a data-quality report for the configured database. Aggregate counts only, no PII.

Usage:
    python -m analytics.quality                      # uses DATABASE_URL from .env
    DATABASE_URL=postgresql://... python -m analytics.quality

Safe to point at production: it never imports the Flask app (so no ``create_all()`` or
migrations run), and on PostgreSQL the connection is opened in read-only mode.
"""
from __future__ import annotations

import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from analytics import cleaning, data_loader


def make_readonly_engine(url: str) -> Engine:
    connect_args = {}
    if url.startswith("postgresql"):
        connect_args["options"] = "-c default_transaction_read_only=on"
    return create_engine(url, connect_args=connect_args)


def _date_range(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "-"
    return f"{valid.min():%Y-%m-%d} .. {valid.max():%Y-%m-%d}"


def build_report(engine: Engine, now: pd.Timestamp | None = None) -> str:
    with engine.connect() as conn:
        users = data_loader.load_users(conn)
        rooms = data_loader.load_rooms(conn)
        sessions = data_loader.load_focus_sessions(conn)
        logs = data_loader.load_focus_logs(conn)
        chat = data_loader.load_chat_messages(conn, include_private=True)

    sessions, s_report = cleaning.clean_focus_sessions(sessions, now=now)
    logs, l_report = cleaning.clean_focus_logs(logs, now=now)
    chat, c_report = cleaning.clean_chat_messages(chat, now=now)

    ok_logs = logs[logs["quality"] == cleaning.OK]
    lines = [
        "== FocusBuddy data quality report ==",
        f"users: {len(users)} (verified: {int(users['email_verified'].sum()) if len(users) else 0})",
        f"rooms: {len(rooms)} (private: {int(rooms['is_private'].sum()) if len(rooms) else 0})",
        "",
        str(s_report),
        f"  deleted-room sessions: {int((~sessions['room_exists'].astype(bool)).sum()) if len(sessions) else 0}",
        f"  date range: {_date_range(sessions['created_at'])}",
        "",
        str(l_report),
        f"  distinct credited users: {ok_logs['user_id'].nunique()}",
        f"  date range: {_date_range(logs['logged_at'])}",
        "",
        str(c_report),
        f"  in private rooms: {int(chat['room_is_private'].astype(bool).sum()) if len(chat) else 0}",
        f"  distinct authors: {chat['user_id'].nunique()}",
    ]
    return "\n".join(lines)


def engine_from_env() -> Engine | None:
    """Read-only engine for DATABASE_URL (normalised by the app's own Config)."""
    load_dotenv()
    from config import Config  # imported after load_dotenv so .env is honoured

    url = Config.SQLALCHEMY_DATABASE_URI
    return make_readonly_engine(url) if url else None


def main(argv: list[str] | None = None) -> int:
    engine = engine_from_env()
    if engine is None:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1
    print(build_report(engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
