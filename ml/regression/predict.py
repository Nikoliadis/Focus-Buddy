"""Inference: load a trained model and predict. Never trains."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from ml.features import FEATURE_COLUMNS


def load(model_path: str | Path) -> tuple[BaseEstimator, dict]:
    path = Path(model_path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["features"] != list(FEATURE_COLUMNS):
        raise ValueError("model was trained on a different feature set; retrain it")
    return joblib.load(path), meta


def predict(model: BaseEstimator, features: pd.DataFrame) -> pd.Series:
    """Predicted focused minutes, one per row. ``features`` must contain every feature column."""
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = features[list(FEATURE_COLUMNS)]
    return pd.Series(model.predict(X), index=features.index, name="predicted_focused_minutes")
