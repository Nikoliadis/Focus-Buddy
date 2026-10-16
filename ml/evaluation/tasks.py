"""A ``Task`` bundles what differs between supervised problems, so one evaluation framework
can serve both: how to get y, which models exist, how to score, and which metric decides.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ml.evaluation.metrics import classification_metrics, regression_metrics


@dataclass(frozen=True)
class Task:
    name: str
    target_description: str
    target: Callable[[pd.DataFrame], pd.Series]
    make_models: Callable[..., dict[str, BaseEstimator]]
    primary_metric: str
    higher_is_better: bool
    predict: Callable[[BaseEstimator, pd.DataFrame], np.ndarray]
    metrics: Callable[[np.ndarray, np.ndarray], dict[str, float]]

    def score(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        return self.metrics(np.asarray(y), self.predict(model, X))

    def loss(self, value: float) -> float:
        """Primary metric turned into "lower is better", for fit diagnostics."""
        return 1 - value if self.higher_is_better else value

    def better(self, a: float, b: float) -> bool:
        return a > b if self.higher_is_better else a < b


def _regression() -> Task:
    from ml.regression.model import make_models

    return Task(
        name="regression",
        target_description="target_focused_minutes: minutes actually focused in the session",
        target=lambda t: t["target_focused_minutes"],
        make_models=make_models,
        primary_metric="mae",
        higher_is_better=False,
        predict=lambda m, X: m.predict(X),
        metrics=regression_metrics,
    )


def _classification() -> Task:
    from ml.classification.model import ended_early, make_models

    return Task(
        name="classification",
        target_description="ended_early: 1 if the session did not run its full planned length",
        target=ended_early,
        make_models=make_models,
        primary_metric="pr_auc",
        higher_is_better=True,
        predict=lambda m, X: m.predict_proba(X)[:, 1],
        # Threshold-free metrics are what matter across folds; the threshold is a later decision.
        metrics=lambda y, p: classification_metrics(y, p, 0.5),
    )


TASKS: dict[str, Callable[[], Task]] = {"regression": _regression, "classification": _classification}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose one of {sorted(TASKS)}")
    return TASKS[name]()
