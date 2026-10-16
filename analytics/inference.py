"""Statistical inference with SciPy: confidence intervals, group comparisons, correlation.

Every test returns a ``TestResult`` that carries the estimate, its confidence interval and
an effect size next to the p-value. A p-value alone says nothing about *how big* a
difference is, and with few observations nothing is reported as significant.

FocusBuddy data is observational. Nothing here can establish causation: users choose when
to study, so a difference between weekend and weekday sessions may reflect *who* studies
on weekends rather than the weekend itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from analytics.cleaning import DEFAULT_LOCAL_TZ, OK

DEFAULT_ALPHA = 0.05
# Below this many observations per group we still compute, but flag the result as not
# interpretable: the test has little power and its normal approximation is shaky.
DEFAULT_MIN_N = 10

NOT_CAUSAL = "Observational data: this is an association, not evidence of a causal effect."


@dataclass(frozen=True)
class TestResult:
    test: str
    estimate: float              # difference in means/medians, mean paired difference, or r
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    n: tuple[int, ...]
    alpha: float = DEFAULT_ALPHA
    min_n: int = DEFAULT_MIN_N
    note: str = ""

    @property
    def sufficient_data(self) -> bool:
        return min(self.n) >= self.min_n and np.isfinite(self.p_value)

    @property
    def reject_null(self) -> bool:
        """Only meaningful with enough data; never True otherwise."""
        return self.sufficient_data and self.p_value < self.alpha

    def summary(self) -> str:
        sizes = " vs ".join(str(n) for n in self.n)
        head = f"{self.test} (n = {sizes})"
        if not self.sufficient_data:
            return f"{head}: too little data (need >= {self.min_n} observations per sample); not interpreted."
        ci = f"{1 - self.alpha:.0%} CI [{self.ci_low:.2f}, {self.ci_high:.2f}]" if np.isfinite(self.ci_low) else "no CI"
        verdict = "reject H0" if self.reject_null else "fail to reject H0"
        lines = [
            f"{head}: estimate {self.estimate:.2f}, {ci}",
            f"  statistic {self.statistic:.3f}, p = {self.p_value:.4f} -> {verdict} at alpha = {self.alpha}",
            f"  {self.effect_size_name} = {self.effect_size:.2f}",
        ]
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def _clean(values: npt.ArrayLike) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x)]


def _nan_result(test: str, n: tuple[int, ...], effect_name: str, note: str, alpha: float, min_n: int) -> TestResult:
    nan = float("nan")
    return TestResult(test, nan, nan, nan, nan, nan, nan, effect_name, n, alpha, min_n, note)


# --- Confidence intervals ---------------------------------------------------

def mean_ci(values: npt.ArrayLike, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and its t-based confidence interval: mean ± t* · s/√n.

    The t distribution (not the normal) accounts for estimating the std from the sample;
    it has heavier tails for small n and converges to the normal as n grows.
    """
    x = _clean(values)
    if x.size < 2:
        return (float(x.mean()) if x.size else float("nan"), float("nan"), float("nan"))
    mean, sem = x.mean(), stats.sem(x)  # sem = s / sqrt(n), with ddof=1
    if sem == 0:
        return float(mean), float(mean), float(mean)
    lo, hi = stats.t.interval(confidence, df=x.size - 1, loc=mean, scale=sem)
    return float(mean), float(lo), float(hi)


def bootstrap_ci(
    values: npt.ArrayLike,
    statistic=np.median,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile-bootstrap CI for any statistic (e.g. the median).

    Resample the data with replacement many times, compute the statistic each time, and
    take the middle ``confidence`` share of those values. No normality assumption.
    The percentile method is used because BCa breaks down on heavily tied data
    (most sessions are exactly 25 minutes).
    """
    x = _clean(values)
    if x.size < 2:
        return (float(statistic(x)) if x.size else float("nan"), float("nan"), float("nan"))
    if np.all(x == x[0]):
        return float(x[0]), float(x[0]), float(x[0])
    res = stats.bootstrap(
        (x,), statistic, n_resamples=n_resamples, confidence_level=confidence,
        method="percentile", rng=np.random.default_rng(seed),
    )
    return float(statistic(x)), float(res.confidence_interval.low), float(res.confidence_interval.high)


# --- Two independent groups -------------------------------------------------

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with small-sample bias correction.
    Rough guide: 0.2 small, 0.5 medium, 0.8 large."""
    na, nb = a.size, b.size
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / pooled * correction)


def welch_t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
    note: str = "",
) -> TestResult:
    """H0: the two population means are equal. H1: they differ (two-sided).

    Welch's version does not assume equal variances, which is the safer default.
    """
    x, y = _clean(a), _clean(b)
    n = (x.size, y.size)
    if min(n) < 2:
        return _nan_result("Welch t-test", n, "Hedges' g", note, alpha, min_n)
    if x.var() == 0 and y.var() == 0:  # t is undefined; SciPy would warn and return NaN
        return _nan_result("Welch t-test", n, "Hedges' g", "both groups have zero variance", alpha, min_n)
    res = stats.ttest_ind(x, y, equal_var=False)
    ci = res.confidence_interval(1 - alpha)
    return TestResult(
        "Welch t-test", float(x.mean() - y.mean()), float(ci.low), float(ci.high),
        float(res.statistic), float(res.pvalue), hedges_g(x, y), "Hedges' g", n, alpha, min_n, note,
    )


def mann_whitney(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
    note: str = "",
) -> TestResult:
    """Rank-based alternative to the t-test. H0: a value from A is equally likely to be
    larger or smaller than a value from B. Robust to outliers and non-normal data.

    Effect size: rank-biserial correlation in [-1, 1]; +1 means every A > every B.
    """
    x, y = _clean(a), _clean(b)
    n = (x.size, y.size)
    if min(n) < 1:
        return _nan_result("Mann-Whitney U", n, "rank-biserial r", note, alpha, min_n)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    rank_biserial = 2 * res.statistic / (x.size * y.size) - 1
    nan = float("nan")
    return TestResult(
        "Mann-Whitney U", float(np.median(x) - np.median(y)), nan, nan,
        float(res.statistic), float(res.pvalue), float(rank_biserial), "rank-biserial r", n, alpha, min_n, note,
    )


# --- Paired observations ----------------------------------------------------

def paired_t_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
    note: str = "",
) -> TestResult:
    """Same subjects measured twice (e.g. each user's weekend vs weekday mean).

    Tests H0: mean(a - b) = 0. Pairing removes between-user variation, so it is more
    powerful than comparing two independent groups. Effect size: d_z = mean(diff) / sd(diff).
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    keep = np.isfinite(x) & np.isfinite(y)
    diff = x[keep] - y[keep]
    n = (int(diff.size),)
    if diff.size < 2 or diff.std(ddof=1) == 0:
        return _nan_result("paired t-test", n, "d_z", note or "fewer than 2 pairs or constant difference",
                           alpha, min_n)
    res = stats.ttest_rel(x[keep], y[keep])
    ci = res.confidence_interval(1 - alpha)
    return TestResult(
        "paired t-test", float(diff.mean()), float(ci.low), float(ci.high),
        float(res.statistic), float(res.pvalue), float(diff.mean() / diff.std(ddof=1)), "d_z", n, alpha, min_n, note,
    )


# --- Correlation ------------------------------------------------------------

def correlation(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    method: Literal["pearson", "spearman"] = "pearson",
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
    note: str = NOT_CAUSAL,
) -> TestResult:
    """H0: no correlation (rho = 0).

    Pearson measures *linear* association and is sensitive to outliers. Spearman is
    Pearson on ranks: it captures any monotonic relationship.
    """
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    name = "Pearson r" if method == "pearson" else "Spearman rho"
    n = (int(a.size),)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return _nan_result(name, n, name, "fewer than 3 points or a constant variable", alpha, min_n)

    nan = float("nan")
    if method == "pearson":
        res = stats.pearsonr(a, b)
        ci = res.confidence_interval(1 - alpha)
        lo, hi = float(ci.low), float(ci.high)
    else:
        res = stats.spearmanr(a, b)
        lo = hi = nan
    r = float(res.statistic)
    return TestResult(name, r, lo, hi, r, float(res.pvalue), r, name, n, alpha, min_n, note)


# --- FocusBuddy questions ---------------------------------------------------

def _local_logs(logs: pd.DataFrame, tz: str) -> pd.DataFrame:
    ok = logs[(logs["quality"] == OK) & logs["session_started_at"].notna()]
    local = ok["session_started_at"].dt.tz_convert(tz)
    return ok.assign(minutes=ok["focused_seconds"] / 60, is_weekend=local.dt.dayofweek >= 5)


def weekend_vs_weekday(
    logs: pd.DataFrame,
    unit: Literal["user", "session"] = "user",
    tz: str = DEFAULT_LOCAL_TZ,
    min_n: int = DEFAULT_MIN_N,
) -> TestResult:
    """Are weekend sessions longer or shorter than weekday ones?

    unit="user" (default): one weekend mean and one weekday mean per user who studied on
    both, compared with a paired t-test. Each user counts once.
    unit="session": every session is one observation (Welch t-test). Sessions of the same
    user are not independent, so this overstates the evidence (pseudo-replication).
    """
    df = _local_logs(logs, tz)
    if unit == "session":
        return welch_t_test(
            df.loc[df["is_weekend"], "minutes"], df.loc[~df["is_weekend"], "minutes"], min_n=min_n,
            note="sessions of the same user are not independent; prefer unit='user'. " + NOT_CAUSAL,
        )
    per_user = df.pivot_table(index="user_id", columns="is_weekend", values="minutes", aggfunc="mean")
    per_user = per_user.reindex(columns=[True, False]).dropna()
    return paired_t_test(per_user[True], per_user[False], min_n=min_n, note=NOT_CAUSAL)


def activity_vs_session_length(logs: pd.DataFrame, min_n: int = DEFAULT_MIN_N) -> TestResult:
    """Do users with more sessions have longer sessions? One point per user, Spearman
    (session counts are skewed and the relationship need not be linear)."""
    ok = logs[logs["quality"] == OK]
    per_user = ok.groupby("user_id")["focused_seconds"].agg(["count", "mean"])
    return correlation(per_user["count"], per_user["mean"] / 60, method="spearman", min_n=min_n)
