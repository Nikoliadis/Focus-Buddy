"""Inference helpers: hand-checked values plus simulations where the truth is known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import cleaning
from analytics import inference as inf

RNG = np.random.default_rng(42)


class TestConfidenceIntervals:
    def test_mean_ci_hand_calculation(self):
        # mean 3, s = sqrt(2.5), n = 5, t*(0.975, df=4) = 2.776
        mean, lo, hi = inf.mean_ci([1, 2, 3, 4, 5])

        half_width = 2.7764451 * np.sqrt(2.5) / np.sqrt(5)
        assert mean == 3
        assert lo == pytest.approx(3 - half_width, rel=1e-6)
        assert hi == pytest.approx(3 + half_width, rel=1e-6)

    def test_ci_narrows_with_more_data(self):
        small = inf.mean_ci(RNG.normal(30, 10, 20))
        large = inf.mean_ci(RNG.normal(30, 10, 2000))
        assert (large[2] - large[1]) < (small[2] - small[1]) / 5

    def test_95_percent_coverage_in_simulation(self):
        """The defining property: ~95% of intervals from repeated samples contain the true mean."""
        rng = np.random.default_rng(7)
        hits = 0
        for _ in range(1000):
            _, lo, hi = inf.mean_ci(rng.normal(25, 8, 30))
            hits += lo <= 25 <= hi
        assert 0.93 <= hits / 1000 <= 0.97

    def test_degenerate_inputs(self):
        assert np.isnan(inf.mean_ci([])[0])
        assert np.isnan(inf.mean_ci([25])[1])
        assert inf.mean_ci([25, 25, 25]) == (25, 25, 25)

    def test_bootstrap_median_ci_contains_median(self):
        x = RNG.exponential(25, 300)

        est, lo, hi = inf.bootstrap_ci(x, np.median)

        assert est == pytest.approx(np.median(x))
        assert lo < est < hi

    def test_bootstrap_is_reproducible_and_handles_ties(self):
        x = [25] * 40 + [50] * 8
        assert inf.bootstrap_ci(x, seed=1) == inf.bootstrap_ci(x, seed=1)
        assert inf.bootstrap_ci([25] * 10) == (25, 25, 25)


class TestTwoGroups:
    def test_welch_hand_checked(self):
        # Welch: t = -3/sqrt(1/3 + 1/3) = -3.674, df = 4, p = 0.0213
        r = inf.welch_t_test([1, 2, 3], [4, 5, 6], min_n=2)

        assert r.estimate == -3
        assert r.statistic == pytest.approx(-3.6742, abs=1e-4)
        assert r.p_value == pytest.approx(0.0213, abs=1e-4)
        assert r.ci_low < -3 < r.ci_high

    def test_detects_a_real_difference(self):
        a, b = RNG.normal(35, 8, 200), RNG.normal(25, 8, 200)

        r = inf.welch_t_test(a, b)

        assert r.reject_null
        assert r.ci_low < 10 < r.ci_high       # true difference is inside the CI
        assert r.effect_size == pytest.approx(1.25, abs=0.25)  # (35-25)/8

    def test_false_positive_rate_is_about_alpha(self):
        """When H0 is true, p < 0.05 should happen ~5% of the time, not never."""
        rng = np.random.default_rng(3)
        rejections = sum(
            inf.welch_t_test(rng.normal(25, 8, 40), rng.normal(25, 8, 40)).reject_null for _ in range(1000)
        )
        assert 0.03 <= rejections / 1000 <= 0.07

    def test_tiny_effect_becomes_significant_with_huge_n(self):
        """Significant does not mean important: 0.3 min difference, n = 50,000 each."""
        a, b = RNG.normal(25.3, 8, 50_000), RNG.normal(25.0, 8, 50_000)

        r = inf.welch_t_test(a, b)

        assert r.reject_null
        assert abs(r.effect_size) < 0.1   # negligible effect

    def test_insufficient_data_is_never_significant(self):
        r = inf.welch_t_test([1, 2, 3], [10, 11, 12])  # default min_n = 10

        assert r.p_value < 0.05
        assert not r.sufficient_data
        assert not r.reject_null
        assert "too little data" in r.summary()

    def test_constant_groups(self):
        r = inf.welch_t_test([25] * 12, [25] * 12)
        assert np.isnan(r.p_value)
        assert not r.reject_null

    def test_mann_whitney_full_separation(self):
        r = inf.mann_whitney([10, 11, 12], [20, 21, 22], min_n=3)

        assert r.statistic == 0
        assert r.effect_size == -1       # every A smaller than every B
        assert r.estimate == -10         # difference in medians

    def test_mann_whitney_robust_to_outlier(self):
        a = [25] * 15 + [24] * 5
        b = [25] * 15 + [24] * 4 + [10_000]  # one absurd value

        t = inf.welch_t_test(a, b)
        u = inf.mann_whitney(a, b)

        assert abs(t.estimate) > 400      # mean difference is dominated by the outlier
        assert u.estimate == 0            # medians are identical
        assert not u.reject_null


class TestPaired:
    def test_pairing_detects_what_independent_test_misses(self):
        # Users differ a lot from each other, but each one is ~3 min longer on weekends.
        rng = np.random.default_rng(11)
        baseline = rng.normal(30, 15, 25)
        weekday = baseline + rng.normal(0, 1, 25)
        weekend = baseline + 3 + rng.normal(0, 1, 25)

        paired = inf.paired_t_test(weekend, weekday)
        independent = inf.welch_t_test(weekend, weekday)

        assert paired.reject_null
        assert not independent.reject_null
        assert paired.estimate == pytest.approx(3, abs=1)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            inf.paired_t_test([1, 2], [1, 2, 3])

    def test_constant_difference(self):
        r = inf.paired_t_test([1, 2, 3], [2, 3, 4])
        assert np.isnan(r.p_value)


class TestCorrelation:
    def test_perfect_linear(self):
        x = np.arange(20)
        r = inf.correlation(x, 2 * x + 1)
        assert r.estimate == pytest.approx(1)
        assert r.reject_null

    def test_monotonic_nonlinear_pearson_vs_spearman(self):
        x = np.linspace(1, 10, 30)
        y = np.exp(x)

        assert inf.correlation(x, y, "spearman").estimate == pytest.approx(1)
        assert inf.correlation(x, y, "pearson").estimate < 0.8

    def test_pearson_ci_contains_true_rho(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=500)
        y = 0.5 * x + np.sqrt(1 - 0.25) * rng.normal(size=500)  # true rho = 0.5

        r = inf.correlation(x, y)

        assert r.ci_low < 0.5 < r.ci_high

    def test_constant_variable(self):
        r = inf.correlation([1, 2, 3, 4], [5, 5, 5, 5], min_n=3)
        assert np.isnan(r.estimate)

    def test_summary_never_claims_causation(self):
        x = np.arange(30)
        text = inf.correlation(x, x + RNG.normal(0, 1, 30)).summary()
        assert "not evidence of a causal effect" in text


def make_logs(rows):
    """rows: (user_id, local-Athens start 'YYYY-MM-DD HH:MM', minutes)."""
    records = []
    for i, (user, start, minutes) in enumerate(rows, start=1):
        t = pd.Timestamp(start, tz="Europe/Athens").tz_convert("UTC")
        records.append({
            "log_id": i, "user_id": user, "room_id": 1, "session_id": i,
            "focused_seconds": minutes * 60, "logged_at": t + pd.Timedelta(minutes=minutes),
            "session_duration_seconds": 60 * 60, "session_started_at": t,
            "session_ended_at": t, "session_paused_seconds": 0, "session_started_by": user,
        })
    logs, _ = cleaning.clean_focus_logs(pd.DataFrame(records), now=pd.Timestamp("2026-06-01", tz="UTC"))
    return logs


class TestFocusBuddyQuestions:
    def test_weekend_vs_weekday_per_user(self):
        rows = []
        for user in range(1, 13):
            rows += [(user, "2026-03-02 10:00", 20 + user), (user, "2026-03-03 10:00", 20 + user)]  # Mon, Tue
            rows += [(user, "2026-03-07 10:00", 30 + user + (user % 3))]                         # Sat
        r = inf.weekend_vs_weekday(make_logs(rows), unit="user")

        assert r.test == "paired t-test"
        assert r.n == (12,)
        assert r.estimate == pytest.approx(10 + np.mean([u % 3 for u in range(1, 13)]))
        assert r.reject_null

    def test_users_active_only_on_one_side_are_excluded_from_pairing(self):
        rows = [(1, "2026-03-02 10:00", 25), (1, "2026-03-07 10:00", 30), (2, "2026-03-02 10:00", 25)]
        r = inf.weekend_vs_weekday(make_logs(rows), unit="user", min_n=1)
        assert r.n == (1,)

    def test_weekend_is_decided_in_local_time(self):
        # Fri 23:30 UTC == Sat 01:30 in Athens (UTC+2): a weekend session.
        logs = make_logs([(1, "2026-03-07 01:30", 30), (1, "2026-03-06 10:00", 25)])
        r = inf.weekend_vs_weekday(logs, unit="session", min_n=1)
        assert r.n == (1, 1)

    def test_activity_vs_session_length(self):
        rows = [(u, f"2026-03-0{d} 10:00", 10 + 3 * u) for u in range(1, 13) for d in range(1, 1 + (u % 6) + 1)]
        r = inf.activity_vs_session_length(make_logs(rows))

        assert r.test == "Spearman rho"
        assert r.n == (12,)
