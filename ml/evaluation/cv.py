"""Cross-validation that respects how the model will be used.

Two questions, two schemes:

temporal (expanding window), "does it predict the future?"
    fold 1: train [■■      ] validate [  ■     ]
    fold 2: train [■■■     ] validate [   ■    ]
    fold 3: train [■■■■    ] validate [    ■   ]  ...
    Training rows whose label arrived after the validation start are purged.

by user (GroupKFold), "does it work for users it has never seen?"
    Every user is entirely in train or entirely in validation, so the model cannot
    memorise individuals. It ignores time, so it answers a different question.

Plain shuffled k-fold is not offered: it mixes the future into training and puts the same
user on both sides, so it overestimates performance on both questions.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, GroupKFold

from ml.evaluation.tasks import Task
from ml.features import FEATURE_COLUMNS

MIN_FOLD_TRAIN = 100
MIN_FOLD_VALIDATION = 30


def temporal_fold_positions(
    as_of: np.ndarray, label_available_at: np.ndarray, n_splits: int = 5,
    min_train: int = MIN_FOLD_TRAIN, min_validation: int = MIN_FOLD_VALIDATION,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds as *row positions*, with purging. Shared by our CV and by
    scikit-learn searches (via ``PurgedTemporalSplit``)."""
    as_of, label_available_at = np.asarray(as_of), np.asarray(label_available_at)
    order = np.argsort(as_of, kind="stable")
    edges = np.linspace(0, len(order), n_splits + 2).astype(int)
    for i in range(1, n_splits + 1):
        val_pos = order[edges[i]:edges[i + 1]]
        if val_pos.size == 0:
            continue
        start = as_of[val_pos].min()
        train_pos = order[:edges[i]]
        train_pos = train_pos[label_available_at[train_pos] < start]
        if train_pos.size >= min_train and val_pos.size >= min_validation:
            yield train_pos, val_pos


def temporal_folds(table: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    for train_pos, val_pos in temporal_fold_positions(
        table["as_of"].to_numpy(), table["label_available_at"].to_numpy(), n_splits
    ):
        yield table.iloc[train_pos], table.iloc[val_pos]


class PurgedTemporalSplit(BaseCrossValidator):
    """scikit-learn CV splitter for GridSearchCV & co. Needs the timestamps of the rows,
    in the same order as the X later passed to ``split``."""

    def __init__(self, as_of, label_available_at, n_splits: int = 5):
        self.as_of = np.asarray(as_of)
        self.label_available_at = np.asarray(label_available_at)
        self.n_splits = n_splits
        self._folds = list(temporal_fold_positions(self.as_of, self.label_available_at, n_splits))

    def split(self, X=None, y=None, groups=None):
        if X is not None and len(X) != len(self.as_of):
            raise ValueError("X must have the same rows, in the same order, as the timestamps")
        yield from self._folds

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._folds)


def user_folds(table: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    n_users = table["user_id"].nunique()
    splitter = GroupKFold(n_splits=min(n_splits, n_users))
    for train_idx, val_idx in splitter.split(table, groups=table["user_id"]):
        yield table.iloc[train_idx], table.iloc[val_idx]


SCHEMES = {"temporal": temporal_folds, "user": user_folds}


def cross_validate(
    model: BaseEstimator,
    table: pd.DataFrame,
    task: Task,
    scheme: str = "temporal",
    n_splits: int = 5,
) -> pd.DataFrame:
    """One row per fold: sizes plus train and validation metrics (``train_*`` / ``val_*``)."""
    rows = []
    for fold, (train, validation) in enumerate(SCHEMES[scheme](table, n_splits), start=1):
        X_tr, y_tr = train[list(FEATURE_COLUMNS)], task.target(train)
        X_va, y_va = validation[list(FEATURE_COLUMNS)], task.target(validation)
        if task.name == "classification" and (y_tr.nunique() < 2 or y_va.nunique() < 2):
            continue  # a fold with one class has no defined PR-AUC / ROC-AUC
        fitted = clone(model).fit(X_tr, y_tr)
        train_scores = task.score(fitted, X_tr, y_tr)
        val_scores = task.score(fitted, X_va, y_va)
        rows.append({
            "fold": fold, "n_train": len(train), "n_validation": len(validation),
            **{f"train_{k}": v for k, v in train_scores.items() if k != "n"},
            **{f"val_{k}": v for k, v in val_scores.items() if k != "n"},
        })
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame, metric: str) -> dict[str, float]:
    """Mean and spread of a validation metric across folds. The spread is a direct
    look at variance: how much the score depends on which data the model happened to see."""
    values = folds[f"val_{metric}"].dropna()
    mean = float(values.mean()) if len(values) else float("nan")
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {"folds": int(len(values)), "mean": mean, "std": std,
            "min": float(values.min()) if len(values) else float("nan"),
            "max": float(values.max()) if len(values) else float("nan")}
