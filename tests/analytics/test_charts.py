"""Charts: they render, they show exactly the data they were given, and they handle no data."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analytics import analysis, charts, cleaning
from analytics.dataset import load_dataset
from main.db import db
from models.focus import FocusLog, FocusSession
from rooms.service import create_room

NOW = pd.Timestamp("2026-03-20 12:00", tz="UTC")


def make_logs(rows):
    records = []
    for i, (user, start, minutes) in enumerate(rows, start=1):
        t = pd.Timestamp(start, tz="UTC")
        records.append({
            "log_id": i, "user_id": user, "room_id": 1, "session_id": i, "focused_seconds": minutes * 60,
            "logged_at": t + pd.Timedelta(minutes=minutes), "session_duration_seconds": 3000,
            "session_started_at": t, "session_ended_at": t, "session_paused_seconds": 0, "session_started_by": user,
        })
    logs, _ = cleaning.clean_focus_logs(pd.DataFrame(records), now=NOW)
    return logs


LOGS = make_logs([
    (1, "2026-03-02 07:00", 25), (1, "2026-03-02 07:30", 50), (2, "2026-03-04 18:00", 10), (2, "2026-03-07 08:00", 25),
])
EMPTY = LOGS.iloc[0:0]


def all_text(fig: Figure) -> str:
    FigureCanvasAgg(fig).draw()  # tick labels are only resolved at draw time
    return " ".join(t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text")))


def bar_heights(fig: Figure) -> list[float]:
    ax = fig.axes[0]
    return [p.get_height() for p in ax.patches]


def bar_widths(fig: Figure) -> list[float]:
    return [p.get_width() for p in fig.axes[0].patches]


class TestChartsShowTheData:
    def test_focus_by_hour_bars_match_summary(self):
        by_hour = analysis.focus_by_hour(LOGS)

        fig = charts.focus_by_hour(by_hour)

        assert bar_heights(fig) == pytest.approx(by_hour["total_minutes"].tolist())
        assert "Focus by hour of day" in all_text(fig)
        assert "4 sessions" in all_text(fig)

    def test_focus_over_time_has_bars_line_and_legend(self):
        daily = analysis.daily_focus(LOGS, window=3)

        fig = charts.focus_over_time(daily)

        ax = fig.axes[0]
        assert bar_heights(fig) == pytest.approx(daily["focus_minutes"].tolist())
        assert ax.lines[0].get_ydata() == pytest.approx(daily["rolling_3d_mean"].to_numpy())
        assert ax.get_legend() is not None  # two series -> legend required
        assert "3-day moving average" in all_text(fig)

    def test_heatmap_values(self):
        matrix = analysis.weekday_hour_matrix(LOGS)

        fig = charts.weekday_hour_heatmap(matrix)

        mesh = fig.axes[0].collections[0]
        assert np.asarray(mesh.get_array()).reshape(7, 24) == pytest.approx(matrix.to_numpy())

    def test_leaderboard_uses_ids_unless_labels_given(self):
        users = analysis.user_focus_summary(LOGS)

        anonymous = charts.user_leaderboard(users)
        named = charts.user_leaderboard(users, labels={1: "alice", 2: "bob"})

        assert "user 1" in all_text(anonymous) and "alice" not in all_text(anonymous)
        assert "alice" in all_text(named)
        assert sorted(bar_widths(anonymous)) == pytest.approx(sorted(users["total_hours"].tolist()))

    def test_leaderboard_top_n(self):
        many = make_logs([(u, "2026-03-02 07:00", 25) for u in range(1, 21)])
        fig = charts.user_leaderboard(analysis.user_focus_summary(many), top=5)
        assert len(fig.axes[0].patches) == 5

    def test_completion_breakdown(self):
        rate = pd.DataFrame([{
            analysis.FULL_DURATION: 6, analysis.ENDED_EARLY: 3, analysis.ABANDONED: 1,
            "decided_sessions": 10, "completion_rate": 0.6,
        }])

        fig = charts.completion_breakdown(rate)

        assert sorted(bar_widths(fig)) == [1, 3, 6]
        assert "60% ran their full duration" in all_text(fig)
        assert "derived" in all_text(fig)  # the caveat travels with the chart

    def test_room_scatter_states_the_snapshot_caveat(self):
        rooms = pd.DataFrame({"room_id": [1, 2], "current_member_count": [3, 1], "total_hours": [5.0, 0.0]})

        fig = charts.room_size_vs_focus(rooms)

        offsets = fig.axes[0].collections[0].get_offsets()
        assert offsets.tolist() == [[3, 5.0], [1, 0.0]]
        assert "today's snapshot" in all_text(fig)
        assert "not causal" in all_text(fig)

    def test_session_length_histogram_counts(self):
        fig = charts.session_length_distribution(LOGS)
        assert sum(bar_heights(fig)) == len(LOGS)
        assert "median (25 min)" in all_text(fig)


class TestDesignRules:
    @pytest.mark.parametrize("theme", [charts.LIGHT, charts.DARK], ids=["light", "dark"])
    def test_text_never_uses_the_data_colour(self, theme):
        fig = charts.focus_by_hour(analysis.focus_by_hour(LOGS), theme=theme)

        from matplotlib.colors import to_hex
        text_colours = {to_hex(t.get_color()) for t in fig.findobj(lambda o: hasattr(o, "get_text")) if t.get_text()}

        assert theme.accent not in text_colours
        assert to_hex(fig.get_facecolor()) == theme.surface

    def test_no_pyplot_figures_are_leaked(self):
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        for _ in range(5):
            charts.focus_by_hour(analysis.focus_by_hour(LOGS))
        assert len(plt.get_fignums()) == before

    def test_global_rcparams_untouched(self):
        import matplotlib

        before = list(matplotlib.rcParams["font.sans-serif"])
        charts.focus_by_hour(analysis.focus_by_hour(LOGS))
        assert matplotlib.rcParams["font.sans-serif"] == before


@pytest.mark.parametrize(
    "build",
    [
        lambda: charts.focus_by_hour(analysis.focus_by_hour(EMPTY)),
        lambda: charts.focus_over_time(analysis.daily_focus(EMPTY)),
        lambda: charts.weekday_hour_heatmap(analysis.weekday_hour_matrix(EMPTY)),
        lambda: charts.user_leaderboard(analysis.user_focus_summary(EMPTY)),
        lambda: charts.session_length_distribution(EMPTY),
        lambda: charts.room_size_vs_focus(pd.DataFrame(columns=["current_member_count", "total_hours"])),
        lambda: charts.completion_breakdown(pd.DataFrame([{
            analysis.FULL_DURATION: 0, analysis.ENDED_EARLY: 0, analysis.ABANDONED: 0,
            "decided_sessions": 0, "completion_rate": float("nan"),
        }])),
    ],
    ids=["hour", "time", "heatmap", "leaderboard", "length", "rooms", "completion"],
)
def test_empty_data_renders_a_message(build):
    fig = build()
    assert isinstance(fig, Figure)
    assert "No valid sessions yet" in all_text(fig)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_render_all_writes_pngs(app, make_user, tmp_path, theme):
    user = make_user()
    room = create_room(user.id, "Room")
    start = datetime(2026, 3, 2, 7, 0)
    s = FocusSession(room_id=room.id, started_by=user.id, status="ended", duration_seconds=1500,
                     started_at=start, ended_at=start + timedelta(minutes=25), paused_seconds=0)
    db.session.add(s)
    db.session.flush()
    db.session.add(FocusLog(user_id=user.id, room_id=room.id, session_id=s.id, focused_seconds=1500,
                            created_at=start + timedelta(minutes=25)))
    db.session.commit()

    paths = charts.render_all(load_dataset(), tmp_path, theme=charts.THEMES[theme])

    assert len(paths) == 7
    for p in paths:
        assert p.name.endswith(f"_{theme}.png")
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
