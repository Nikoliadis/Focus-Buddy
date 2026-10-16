"""Reusable static charts (matplotlib + seaborn) for the FocusBuddy analyses.

Each chart function takes the *summary* DataFrame produced by ``analytics.analysis``
(it does no aggregation itself) and returns a ``matplotlib.figure.Figure``.

Design choices:
- Object-oriented matplotlib (``Figure``), never ``pyplot``: no global figure registry,
  so charts can be rendered in a server or a loop without leaking memory or state.
- One quiet visual system: thin marks, recessive hairline grid, text in ink colours
  (never in the data colour), a single hue for single-series charts and a one-hue
  sequential ramp for magnitude. Colours come from a CVD-validated reference palette.
- Light and dark themes are separately chosen steps of the same hues, not an inversion.
- Users appear as ids unless the caller passes display labels explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from analytics.analysis import ABANDONED, ENDED_EARLY, FULL_DURATION


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    accent: str          # categorical slot 1: the colour of every single-series chart
    accent_soft: str     # a light step of the same hue, for context marks
    sequential: tuple[str, ...]


LIGHT = Theme(
    name="light", surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7", accent="#2a78d6", accent_soft="#b7d3f6",
    sequential=("#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark", surface="#1a1a19", ink="#f0efec", ink_secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", baseline="#383835", accent="#3987e5", accent_soft="#184f95",
    sequential=("#2c2c2a", "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"),
)
THEMES = {"light": LIGHT, "dark": DARK}

FONT_STACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
OUTCOME_LABELS = {FULL_DURATION: "Ran full duration", ENDED_EARLY: "Ended early", ABANDONED: "Abandoned"}


def _figure(theme: Theme, size: tuple[float, float] = (9, 4.5)) -> tuple[Figure, object]:
    fig = Figure(figsize=size, dpi=110, facecolor=theme.surface, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(theme.surface)
    return fig, ax


def _style(ax, theme: Theme, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.baseline)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, length=0, labelsize=9)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)
    if grid_axis:
        ax.grid(axis=grid_axis, color=theme.grid, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _title(ax, theme: Theme, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=theme.ink, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=theme.ink_secondary, va="bottom")


def _empty(fig: Figure, ax, theme: Theme, title: str) -> Figure:
    _title(ax, theme, title)
    ax.axis("off")
    ax.text(0.5, 0.5, "No valid sessions yet", ha="center", va="center", color=theme.muted, fontsize=11,
            transform=ax.transAxes)
    return fig


def _themed(func: Callable[..., Figure]) -> Callable[..., Figure]:
    """Run a chart with the font stack applied, without touching global rcParams."""
    @wraps(func)
    def wrapper(*args, theme: Theme = LIGHT, **kwargs) -> Figure:
        with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": FONT_STACK}):
            return func(*args, theme=theme, **kwargs)
    return wrapper


@_themed
def focus_by_hour(by_hour: pd.DataFrame, theme: Theme = LIGHT, tz: str = "Europe/Athens") -> Figure:
    """Columns: total focused minutes per local hour of session start (from analysis.focus_by_hour)."""
    fig, ax = _figure(theme)
    title = "Focus by hour of day"
    if by_hour["sessions"].sum() == 0:
        return _empty(fig, ax, theme, title)
    hours = by_hour["hour"].to_numpy()
    minutes = by_hour["total_minutes"].to_numpy()
    ax.bar(hours, minutes, width=0.62, color=theme.accent)
    peak = int(np.argmax(minutes))
    ax.annotate(f"{minutes[peak]:,.0f} min", (hours[peak], minutes[peak]), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9, color=theme.ink_secondary)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylabel("Focused minutes")
    _style(ax, theme)
    _title(ax, theme, title, f"Total focused minutes by session start hour ({tz}), "
                             f"{int(by_hour['sessions'].sum())} sessions")
    return fig


@_themed
def focus_over_time(daily: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Daily focused minutes (light columns) with the rolling mean (line), from analysis.daily_focus."""
    fig, ax = _figure(theme)
    title = "Focus over time"
    if daily.empty:
        return _empty(fig, ax, theme, title)
    rolling_col = next(c for c in daily.columns if c.startswith("rolling_"))
    window = rolling_col.removeprefix("rolling_").removesuffix("d_mean")
    dates = pd.to_datetime(daily["date"])
    ax.bar(dates, daily["focus_minutes"], width=0.7, color=theme.accent_soft, label="Daily focus")
    ax.plot(dates, daily[rolling_col], color=theme.accent, linewidth=2, solid_capstyle="round",
            label=f"{window}-day moving average")
    last = daily[rolling_col].iloc[-1]
    ax.plot(dates.iloc[-1], last, "o", markersize=8, color=theme.accent, markeredgecolor=theme.surface,
            markeredgewidth=2)
    ax.annotate(f"{last:,.0f} min/day", (dates.iloc[-1], last), textcoords="offset points", xytext=(6, 6),
                fontsize=9, color=theme.ink_secondary)
    ax.set_ylabel("Focused minutes")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=theme.ink_secondary)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(ax, theme)
    _title(ax, theme, title, f"{dates.min():%d %b} to {dates.max():%d %b %Y}; days without sessions count as 0")
    return fig


@_themed
def weekday_hour_heatmap(matrix: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """7 × 24 heatmap of focused minutes (from analysis.weekday_hour_matrix), one-hue sequential ramp."""
    fig, ax = _figure(theme, (10, 3.8))
    title = "When does focus happen?"
    if matrix.to_numpy().sum() == 0:
        return _empty(fig, ax, theme, title)
    cmap = LinearSegmentedColormap.from_list("focus", list(theme.sequential))
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=0, linewidths=2, linecolor=theme.surface, square=False,
        cbar_kws={"label": "Focused minutes", "shrink": 0.8, "pad": 0.02},
    )
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, length=0, labelsize=8)
    cbar.ax.yaxis.label.set_color(theme.ink_secondary)
    ax.set_xticks(np.arange(0, 24, 3) + 0.5, [f"{h:02d}" for h in range(0, 24, 3)], rotation=0)
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    _style(ax, theme, grid_axis="")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(ax, theme, title, "Focused minutes by weekday and session start hour")
    return fig


@_themed
def user_leaderboard(
    users: pd.DataFrame,
    theme: Theme = LIGHT,
    top: int = 10,
    labels: Mapping[int, str] | None = None,
) -> Figure:
    """Horizontal bars of total focus hours (from analysis.user_focus_summary).
    Pass ``labels`` (user_id -> name) only for an audience allowed to see names."""
    fig, ax = _figure(theme, (8, 0.42 * min(top, max(len(users), 1)) + 1.4))
    title = "Leaderboard"
    if users.empty:
        return _empty(fig, ax, theme, title)
    df = users.nlargest(top, "total_hours").iloc[::-1]
    names = [labels.get(u, f"user {u}") if labels else f"user {u}" for u in df["user_id"]]
    ax.barh(names, df["total_hours"], height=0.55, color=theme.accent)
    for y, hours in enumerate(df["total_hours"]):
        ax.text(hours, y, f"  {hours:.1f} h", va="center", fontsize=9, color=theme.ink_secondary)
    ax.set_xlim(0, df["total_hours"].max() * 1.15)
    ax.set_xlabel("Total focus (hours)")
    _style(ax, theme, grid_axis="x")
    ax.spines["left"].set_visible(False)
    _title(ax, theme, title, f"Top {len(df)} users by total credited focus time")
    return fig


@_themed
def completion_breakdown(rate: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Outcome counts behind the completion rate (from analysis.completion_rate, no grouping)."""
    fig, ax = _figure(theme, (8, 2.8))
    row = rate.iloc[0]
    decided = int(row["decided_sessions"])
    title = "Session completion"
    if decided == 0:
        return _empty(fig, ax, theme, title)
    keys = [FULL_DURATION, ENDED_EARLY, ABANDONED]
    counts = [int(row[k]) for k in keys][::-1]
    names = [OUTCOME_LABELS[k] for k in keys][::-1]
    ax.barh(names, counts, height=0.5, color=theme.accent)
    for y, n in enumerate(counts):
        ax.text(n, y, f"  {n}  ({n / decided:.0%})", va="center", fontsize=9, color=theme.ink_secondary)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_xlabel("Sessions")
    _style(ax, theme, grid_axis="x")
    ax.spines["left"].set_visible(False)
    _title(ax, theme, f"{title}: {row['completion_rate']:.0%} ran their full duration",
           f"{decided} decided sessions. Outcomes are derived: the app does not record why a session ended")
    return fig


@_themed
def room_size_vs_focus(rooms: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Scatter of current member count vs total focus hours (from analysis.room_activity)."""
    fig, ax = _figure(theme, (7, 4.5))
    title = "Room size vs focus time"
    if rooms.empty:
        return _empty(fig, ax, theme, title)
    ax.scatter(rooms["current_member_count"], rooms["total_hours"], s=64, color=theme.accent,
               edgecolors=theme.surface, linewidths=2, zorder=3)
    ax.set_xlabel("Current members (today's snapshot)")
    ax.set_ylabel("Total focus (hours)")
    ax.xaxis.get_major_locator().set_params(integer=True)
    _style(ax, theme, grid_axis="both")
    _title(ax, theme, title, f"{len(rooms)} rooms. Membership is today's, not at session time: "
                             "descriptive only, not causal")
    return fig


@_themed
def session_length_distribution(logs: pd.DataFrame, theme: Theme = LIGHT) -> Figure:
    """Histogram of focused minutes per valid session. Shows the ties at 25/50 min."""
    fig, ax = _figure(theme, (8, 3.8))
    title = "Session length"
    minutes = logs.loc[logs["quality"] == "ok", "focused_seconds"] / 60
    if minutes.empty:
        return _empty(fig, ax, theme, title)
    sns.histplot(minutes, ax=ax, binwidth=5, binrange=(0, max(60, minutes.max() + 5)), color=theme.accent,
                 edgecolor=theme.surface, linewidth=2, alpha=1)
    median = float(minutes.median())
    ax.axvline(median, color=theme.ink_secondary, linewidth=1)
    ax.set_xlabel("Focused minutes per session")
    ax.set_ylabel("Sessions")
    _style(ax, theme)
    _title(ax, theme, title,
           f"{len(minutes)} valid sessions in 5-minute bins; vertical line = median ({median:.0f} min)")
    return fig


def render_all(
    ds,
    out_dir: str | Path,
    theme: Theme = LIGHT,
    tz: str = "Europe/Athens",
    window: int = 7,
) -> list[Path]:
    """Compute every summary from an ``AnalyticsDataset`` and save one PNG per chart."""
    from analytics import analysis

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = analysis.session_outcomes(ds.sessions, ds.logs)
    charts = {
        "focus_by_hour": focus_by_hour(analysis.focus_by_hour(ds.logs, tz), theme=theme, tz=tz),
        "focus_over_time": focus_over_time(analysis.daily_focus(ds.logs, window=window, tz=tz), theme=theme),
        "weekday_hour_heatmap": weekday_hour_heatmap(analysis.weekday_hour_matrix(ds.logs, tz), theme=theme),
        "user_leaderboard": user_leaderboard(analysis.user_focus_summary(ds.logs, tz), theme=theme),
        "completion": completion_breakdown(analysis.completion_rate(outcomes), theme=theme),
        "room_size_vs_focus": room_size_vs_focus(analysis.room_activity(ds.logs, ds.rooms, tz), theme=theme),
        "session_length": session_length_distribution(ds.logs, theme=theme),
    }
    paths = []
    for name, fig in charts.items():
        path = out / f"{name}_{theme.name}.png"
        fig.savefig(path, facecolor=theme.surface)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from analytics.dataset import load_dataset
    from analytics.quality import engine_from_env

    parser = argparse.ArgumentParser(description="Render FocusBuddy analytics charts to PNG.")
    parser.add_argument("--out", default="reports/figures", help="output directory (git-ignored)")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument("--tz", default="Europe/Athens")
    args = parser.parse_args(argv)

    engine = engine_from_env()
    if engine is None:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1
    with engine.connect() as conn:
        ds = load_dataset(conn)
    for path in render_all(ds, args.out, THEMES[args.theme], args.tz):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
