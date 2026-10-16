"""SYNTHETIC focus-session data for developing and testing ML pipelines.

Nothing produced here describes real FocusBuddy users. It exists because (a) the demo
seed only creates sessions that run their full length, so there is nothing to learn, and
(b) with synthetic data the ground truth is *known*, so we can check whether a model
recovers it. Any metric computed on this data must be labelled synthetic.

Generative story (the "ground truth" a model could learn):
- Each user has a latent ``discipline`` in (0, 1) and a chronotype: a preferred hour
  around 10:00 (morning) or 21:00 (evening).
- Sessions are 25 or 50 minutes; users differ in how often they pick 50.
- P(session runs its full length) = sigmoid(
      1.2 + 3.0 * (discipline - 0.6)
      - 1.0 * late_night            (local start 23:00-04:59)
      - 0.6 * fatigue               (>= 3 sessions earlier that local day)
      - 0.4 * weekend
      - 0.5 * long_session          (planned 50 minutes)
  )
- A session that does not run full is ended early after planned * Beta(2, 2) minutes.
The latent variables are not exposed as features; only their observable traces are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import cleaning

SYNTHETIC = "synthetic"

COEFFICIENTS = {
    "intercept": 1.2, "discipline": 3.0, "late_night": -1.0, "fatigue": -0.6, "weekend": -0.4, "long_session": -0.5,
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def generate(
    n_users: int = 60,
    days: int = 90,
    seed: int = 0,
    start: str = "2026-01-05",
    tz: str = cleaning.DEFAULT_LOCAL_TZ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(logs, rooms)`` shaped like the cleaned frames from ``analytics.dataset``."""
    rng = np.random.default_rng(seed)
    first_day = pd.Timestamp(start, tz=tz)
    n_rooms = max(1, n_users // 3)

    rooms = pd.DataFrame({
        "room_id": np.arange(1, n_rooms + 1),
        "owner_id": rng.integers(1, n_users + 1, n_rooms),
        "is_private": rng.random(n_rooms) < 0.3,
        "created_at": (first_day - pd.to_timedelta(rng.integers(1, 60, n_rooms), unit="D")).tz_convert("UTC"),
        "current_member_count": rng.integers(1, 8, n_rooms),
    })

    records = []
    log_id = 0
    for user in range(1, n_users + 1):
        discipline = rng.beta(5, 2.5)
        peak_hour = 10 if rng.random() < 0.45 else 21
        sessions_per_day = rng.gamma(2.0, 0.6)
        p_long = rng.uniform(0.05, 0.5)
        room = int(rng.integers(1, n_rooms + 1))
        last_end = None

        for day in range(days):
            n_today = rng.poisson(sessions_per_day)
            if n_today == 0:
                continue
            day_start = first_day + pd.Timedelta(days=day)
            hours = np.sort(np.clip(rng.normal(peak_hour, 2.0, n_today), 6, 28.5))
            for i, h in enumerate(hours):
                t = day_start + pd.Timedelta(hours=float(h))
                if last_end is not None and t < last_end + pd.Timedelta(minutes=5):
                    t = last_end + pd.Timedelta(minutes=int(rng.integers(5, 30)))
                planned = 50 if rng.random() < p_long else 25
                local_hour = t.hour
                z = (
                    COEFFICIENTS["intercept"]
                    + COEFFICIENTS["discipline"] * (discipline - 0.6)
                    + COEFFICIENTS["late_night"] * (local_hour >= 23 or local_hour < 5)
                    + COEFFICIENTS["fatigue"] * (i >= 3)
                    + COEFFICIENTS["weekend"] * (t.dayofweek >= 5)
                    + COEFFICIENTS["long_session"] * (planned == 50)
                )
                full = rng.random() < _sigmoid(np.array(z))
                focused = planned if full else max(1, int(round(planned * rng.beta(2, 2))))
                end = t + pd.Timedelta(minutes=focused)
                log_id += 1
                records.append({
                    "log_id": log_id, "user_id": user, "room_id": room, "session_id": log_id,
                    "focused_seconds": focused * 60, "logged_at": end.tz_convert("UTC"),
                    "session_duration_seconds": planned * 60, "session_started_at": t.tz_convert("UTC"),
                    "session_ended_at": end.tz_convert("UTC"), "session_paused_seconds": 0,
                    "session_started_by": user,
                })
                last_end = end

    raw = pd.DataFrame.from_records(records)
    now = first_day + pd.Timedelta(days=days + 1)
    logs, _ = cleaning.clean_focus_logs(raw, now=now.tz_convert("UTC"))
    return logs, rooms
