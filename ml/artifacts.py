"""Persist and load fitted models with a JSON metadata file side by side.

Security note: joblib/pickle files can execute code when loaded. Only load artifacts this
project produced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.base import BaseEstimator

from ml.features import FEATURE_COLUMNS


def save(model: BaseEstimator, metadata: dict[str, Any], out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{name}.joblib"
    joblib.dump(model, model_path)
    meta = {**metadata, "model_file": model_path.name, "sklearn_version": sklearn.__version__}
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return model_path


def load(model_path: str | Path) -> tuple[BaseEstimator, dict[str, Any]]:
    path = Path(model_path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["features"] != list(FEATURE_COLUMNS):
        raise ValueError("model was trained on a different feature set; retrain it")
    return joblib.load(path), meta
