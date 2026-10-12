"""NumPy statistics, cross-checked against hand calculations and pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import statistics as st

SESSIONS = [25, 25, 25, 50, 10]  # minutes


class TestDescribe:
    def test_matches_hand_calculation(self):
        d = st.describe(SESSIONS)

        assert d.n == 5
        assert d.mean == pytest.approx(27.0)           # 135 / 5
        assert d.median == 25
        assert d.min == 10 and d.max == 50
        # sample variance: sum((x - 27)^2) / (5 - 1) = (4 + 4 + 4 + 529 + 289) / 4 = 207.5
        assert d.std == pytest.approx(np.sqrt(207.5))

    def test_matches_pandas(self):
        s = pd.Series(SESSIONS, dtype=float)
        d = st.describe(SESSIONS)

        assert d.std == pytest.approx(s.std())          # pandas also uses ddof=1
        assert d.p25 == pytest.approx(s.quantile(0.25))
        assert d.p90 == pytest.approx(s.quantile(0.90))

    def test_ignores_nan_and_inf(self):
        d = st.describe([25, np.nan, 35, np.inf])
        assert d.n == 2
        assert d.mean == 30

    def test_empty(self):
        d = st.describe([])
        assert d.n == 0
        assert np.isnan(d.mean) and np.isnan(d.iqr)

    def test_single_value_has_undefined_std(self):
        d = st.describe([25])
        assert d.mean == 25
        assert np.isnan(d.std)  # ddof=1 with n=1 divides by zero


class TestMovingAverage:
    def test_values(self):
        out = st.moving_average([1, 2, 3, 4, 5], window=3)

        assert np.isnan(out[:2]).all()
        assert out[2:].tolist() == [2.0, 3.0, 4.0]

    def test_matches_pandas_rolling(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 120, size=200)

        expected = pd.Series(x).rolling(7).mean().to_numpy()

        np.testing.assert_allclose(st.moving_average(x, 7), expected, equal_nan=True)

    def test_window_one_is_identity(self):
        assert st.moving_average([3, 1, 2], 1).tolist() == [3, 1, 2]

    def test_shorter_than_window(self):
        assert np.isnan(st.moving_average([1, 2], 5)).all()

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            st.moving_average([1, 2, 3], 0)


class TestTransformations:
    def test_zscore_broadcasts_over_columns(self):
        # rows = sessions, columns = (focused minutes, hour of day)
        x = np.array([[25, 9], [50, 21], [10, 15]], dtype=float)

        z = st.zscore(x)

        assert z.shape == x.shape
        np.testing.assert_allclose(z.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(z.std(axis=0, ddof=1), [1, 1])

    def test_zscore_constant_column_is_zero(self):
        z = st.zscore(np.array([[25, 1], [25, 2], [25, 3]], dtype=float))
        assert (z[:, 0] == 0).all()
        assert not np.isnan(z).any()

    def test_min_max_scale(self):
        out = st.min_max_scale([10, 25, 50])
        np.testing.assert_allclose(out, [0, 0.375, 1])

    def test_log1p_compresses_long_tail(self):
        out = st.log1p_transform([0, 25, 180])

        assert out[0] == 0
        # 180 is 7.2x of 25 in raw minutes, but under 2x after the transform
        assert out[2] / out[1] < 2

    def test_log1p_rejects_negative(self):
        with pytest.raises(ValueError):
            st.log1p_transform([25, -1])

    def test_winsorize_clips_extremes_only(self):
        x = np.arange(1, 101, dtype=float)
        out = st.winsorize(x, 5, 95)

        assert out.min() == pytest.approx(np.percentile(x, 5))
        assert out.max() == pytest.approx(np.percentile(x, 95))
        assert out[50] == x[50]

    def test_input_is_not_mutated(self):
        x = np.array([1.0, 100.0, 3.0])
        st.winsorize(x, 10, 90)
        st.zscore(x)
        assert x.tolist() == [1.0, 100.0, 3.0]


class TestOutliers:
    def test_tukey_rule(self):
        mask = st.iqr_outliers([20, 22, 25, 25, 27, 30, 180])
        assert mask.tolist() == [False] * 6 + [True]

    def test_nan_is_not_an_outlier(self):
        assert st.iqr_outliers([25, np.nan, 26]).tolist() == [False, False, False]

    def test_degenerate_iqr_on_tied_data(self):
        """Known limitation: when most sessions are exactly 25 min, IQR = 0 and every
        other duration is flagged. The rule assumes a continuous distribution."""
        mask = st.iqr_outliers([25] * 8 + [50, 30])
        assert mask.sum() == 2


class TestGroupMeans:
    def test_matches_pandas_groupby(self):
        values = [25, 25, 50, 10, 10]
        groups = [1, 1, 1, 2, 2]

        keys, means = st.group_means(values, groups)

        expected = pd.Series(values).groupby(groups).mean()
        assert keys.tolist() == expected.index.tolist()
        np.testing.assert_allclose(means, expected.to_numpy())

    def test_balanced_vs_pooled_mean(self):
        # A heavy user (10 x 50 min) and a light user (1 x 10 min).
        values = [50] * 10 + [10]
        groups = [1] * 10 + [2]

        pooled = np.mean(values)
        balanced = st.balanced_mean(values, groups)

        assert pooled == pytest.approx(500 / 11 + 10 / 11)  # ~46.4, dominated by user 1
        assert balanced == pytest.approx(30)                 # (50 + 10) / 2

    def test_balanced_mean_empty(self):
        assert np.isnan(st.balanced_mean([], []))


def test_focus_distribution_uses_only_valid_logs():
    logs = pd.DataFrame({
        "user_id": [1, 1, 2],
        "focused_seconds": [1500, 3000, 99999],
        "quality": ["ok", "ok", "invalid_duration"],
    })

    out = st.focus_distribution(logs)

    assert out["n"] == 2
    assert out["mean"] == pytest.approx(37.5)
    assert out["skew_hint_mean_minus_median"] == pytest.approx(0)
