"""Inference: load a trained model and predict. Never trains."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator

from ml.artifacts import load
from ml.features import FEATURE_COLUMNS

__all__ = ["load", "predict"]


def predict(model: BaseEstimator, features: pd.DataFrame) -> pd.Series:
    """Predicted focused minutes, one per row. ``features`` must contain every feature column."""
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = features[list(FEATURE_COLUMNS)]
    return pd.Series(model.predict(X), index=features.index, name="predicted_focused_minutes")
