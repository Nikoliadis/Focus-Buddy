"""Over/underfitting diagnostics, learning curves and bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from ml.evaluation.tasks import Task
from ml.features import FEATURE_COLUMNS

# Validation loss this much worse (relative) than training loss is called overfitting.
OVERFIT_GAP = 0.25


@dataclass(frozen=True)
class FitDiagnosis:
    verdict: str            # "underfitting" | "overfitting" | "reasonable fit"
    train_loss: float
    validation_loss: float
    baseline_validation_loss: float
    relative_gap: float
    explanation: str


def diagnose_fit(task: Task, train_score: float, val_score: float, baseline_val_score: float) -> FitDiagnosis:
    """Compare training and validation performance on the task's primary metric.

    underfitting   validation is no better than the baseline: the model has not captured
                   signal beyond the simplest rule (high bias), or there is little to capture
    overfitting    training is much better than validation: the model memorised the
                   training data (high variance)
    reasonable     beats the baseline and generalises without a large gap
    """
    tr, va, base = task.loss(train_score), task.loss(val_score), task.loss(baseline_val_score)
    gap = (va - tr) / abs(va) if va else float("inf")
    if va >= base:
        verdict = "underfitting"
        why = (f"validation loss {va:.3f} is not better than the baseline's {base:.3f}: no signal captured "
               "beyond the simplest rule (high bias), or the features carry little signal")
    elif gap > OVERFIT_GAP:
        verdict = "overfitting"
        why = (f"training loss {tr:.3f} vs validation {va:.3f} ({gap:.0%} worse): the model fits training "
               "data much better than new data (high variance)")
    else:
        verdict = "reasonable fit"
        why = f"beats the baseline ({va:.3f} vs {base:.3f}) with a {gap:.0%} train/validation gap"
    return FitDiagnosis(verdict, tr, va, base, gap, why)


def learning_curve(
    model: BaseEstimator,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    task: Task,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Train on the most recent ``fraction`` of the fit period, score on validation.

    Read it like this: if validation keeps improving with more data, collecting data helps;
    if train and validation converge at a poor value, the model is too simple (bias);
    if a gap stays open, the model is too flexible for the data (variance)."""
    ordered = fit.sort_values("as_of", kind="stable")
    X_va, y_va = validation[list(FEATURE_COLUMNS)], task.target(validation)
    rows = []
    for fraction in fractions:
        part = ordered.iloc[int(len(ordered) * (1 - fraction)):]
        X_tr, y_tr = part[list(FEATURE_COLUMNS)], task.target(part)
        if len(part) < 20 or (task.name == "classification" and y_tr.nunique() < 2):
            continue
        fitted = clone(model).fit(X_tr, y_tr)
        rows.append({
            "fraction": fraction, "n_train": len(part),
            "train": task.score(fitted, X_tr, y_tr)[task.primary_metric],
            "validation": task.score(fitted, X_va, y_va)[task.primary_metric],
        })
    return pd.DataFrame(rows)


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    skip_single_class: bool = False,
) -> tuple[float, float, float]:
    """Point estimate and percentile-bootstrap CI of a metric on a fixed test set."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if skip_single_class and len(np.unique(y[idx])) < 2:  # ranking metrics need both classes
            continue
        stats.append(metric(y[idx], pred[idx]))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return float(metric(y, pred)), float(lo), float(hi)


def paired_bootstrap_difference(
    y: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    skip_single_class: bool = False,
) -> tuple[float, float, float]:
    """CI of metric(model) - metric(baseline), resampling the *same* rows for both.

    Pairing matters: both models are scored on identical resamples, so session-to-session
    noise cancels and the CI reflects the difference between the models only. If the CI
    contains 0, the test set cannot tell the two apart."""
    y, a, b = np.asarray(y), np.asarray(pred_model), np.asarray(pred_baseline)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if skip_single_class and len(np.unique(y[idx])) < 2:  # ranking metrics need both classes
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return float(metric(y, a) - metric(y, b)), float(lo), float(hi)
