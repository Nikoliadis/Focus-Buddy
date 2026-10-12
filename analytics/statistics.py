"""Descriptive statistics and numerical transformations with NumPy.

Everything here is vectorized: operations run on whole arrays in compiled code instead of
Python loops. Functions accept any array-like, ignore NaN where that is meaningful, and
never mutate their input.

Statistical conventions:
- ``std`` is the *sample* standard deviation (ddof=1), because our sessions are a sample
  of user behaviour, not the whole population. With ddof=0 the spread is underestimated.
- Percentiles use NumPy's default linear interpolation (same as pandas ``quantile``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from analytics.cleaning import OK

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


def _as_float(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _finite(values: ArrayLike) -> FloatArray:
    x = _as_float(values)
    return x[np.isfinite(x)]  # boolean mask indexing: drops NaN and ±inf in one step


@dataclass(frozen=True)
class Descriptive:
    n: int
    mean: float
    median: float
    std: float
    min: float
    p25: float
    p75: float
    p90: float
    max: float

    @property
    def iqr(self) -> float:
        return self.p75 - self.p25

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "iqr": self.iqr}


def describe(values: ArrayLike) -> Descriptive:
    """Summary statistics of the finite values. Empty input gives n=0 and NaN everywhere."""
    x = _finite(values)
    if x.size == 0:
        nan = float("nan")
        return Descriptive(0, nan, nan, nan, nan, nan, nan, nan, nan)
    # One call computes all percentiles: a single sort instead of five.
    p0, p25, p50, p75, p90, p100 = np.percentile(x, [0, 25, 50, 75, 90, 100])
    return Descriptive(
        n=int(x.size),
        mean=float(x.mean()),
        median=float(p50),
        std=float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        min=float(p0),
        p25=float(p25),
        p75=float(p75),
        p90=float(p90),
        max=float(p100),
    )


def moving_average(values: ArrayLike, window: int) -> FloatArray:
    """Trailing moving average with a full window; the first ``window - 1`` values are NaN.

    Uses the cumulative-sum trick: sum(x[i-w+1 .. i]) = c[i+1] - c[i+1-w], which is O(n)
    regardless of the window size (a naive loop is O(n * w)).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x = _as_float(values)
    out = np.full(x.shape, np.nan)
    if x.size < window:
        return out
    c = np.concatenate(([0.0], np.cumsum(x)))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def zscore(values: ArrayLike) -> FloatArray:
    """(x - mean) / std along axis 0.

    For a 2-D (n_samples, n_features) matrix, ``mean`` and ``std`` have shape
    (n_features,) and *broadcast* across all rows, so every column is standardised at once.
    Constant columns (std = 0) become 0 instead of dividing by zero.
    """
    x = _as_float(values)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=1)
    safe_std = np.where(std > 0, std, 1.0)
    return np.where(std > 0, (x - mean) / safe_std, 0.0)


def min_max_scale(values: ArrayLike) -> FloatArray:
    """Rescale each column to [0, 1]. Constant columns become 0."""
    x = _as_float(values)
    lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    span = hi - lo
    return np.where(span > 0, (x - lo) / np.where(span > 0, span, 1.0), 0.0)


def log1p_transform(values: ArrayLike) -> FloatArray:
    """log(1 + x): compresses a right-skewed distribution (a few very long sessions)
    and is defined at 0. Negative inputs are invalid durations and raise."""
    x = _as_float(values)
    if np.any(x[np.isfinite(x)] < 0):
        raise ValueError("log1p_transform expects non-negative values")
    return np.log1p(x)


def iqr_outliers(values: ArrayLike, k: float = 1.5) -> npt.NDArray[np.bool_]:
    """Tukey's rule: True where x < Q1 - k·IQR or x > Q3 + k·IQR. NaN is never an outlier."""
    x = _as_float(values)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=bool)
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    with np.errstate(invalid="ignore"):
        return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def winsorize(values: ArrayLike, lower_pct: float = 1, upper_pct: float = 99) -> FloatArray:
    """Clip values to the given percentiles: keeps extreme rows but limits their pull on the mean."""
    x = _as_float(values)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x.copy()
    lo, hi = np.percentile(finite, [lower_pct, upper_pct])
    return np.clip(x, lo, hi)


def group_means(values: ArrayLike, groups: ArrayLike) -> tuple[np.ndarray, FloatArray]:
    """Mean per group without a Python loop.

    ``np.unique(return_inverse=True)`` maps every row to a group index 0..k-1, and
    ``np.bincount`` sums values per index in one pass.
    """
    x = _as_float(values)
    g = np.asarray(groups)
    keys, idx = np.unique(g, return_inverse=True)
    sums = np.bincount(idx, weights=x, minlength=keys.size)
    counts = np.bincount(idx, minlength=keys.size)
    return keys, sums / counts


def balanced_mean(values: ArrayLike, groups: ArrayLike) -> float:
    """Mean of per-group means: every user counts once, however many sessions they have.

    The pooled mean answers "what does an average *session* look like"; this answers
    "what does an average *user* look like". They diverge when heavy users behave differently.
    """
    x = _as_float(values)
    if x.size == 0:
        return float("nan")
    _, means = group_means(x, groups)
    return float(means.mean())


def focus_distribution(logs: pd.DataFrame) -> dict[str, float]:
    """Distribution of focused minutes for valid logs, ready for a report."""
    ok = logs[logs["quality"] == OK]
    minutes = ok["focused_seconds"].to_numpy(dtype=np.float64) / 60
    stats = describe(minutes).as_dict()
    stats["balanced_mean_per_user"] = balanced_mean(minutes, ok["user_id"].to_numpy())
    stats["iqr_outliers"] = int(iqr_outliers(minutes).sum())
    stats["skew_hint_mean_minus_median"] = stats["mean"] - stats["median"]
    return stats
