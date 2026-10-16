"""Inference for the "ended early" classifier. Never trains."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator

from ml.artifacts import load
from ml.features import FEATURE_COLUMNS

__all__ = ["load", "predict"]


def predict(model: BaseEstimator, features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """P(ended early) and the decision at the threshold stored with the model."""
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    proba = model.predict_proba(features[list(FEATURE_COLUMNS)])[:, 1]
    return pd.DataFrame(
        {"p_ended_early": proba, "predicted_ended_early": (proba >= threshold).astype(int)},
        index=features.index,
    )
