"""Load trained models from disk for serving. Training happens elsewhere (ml.*.train).

A model that fails to load is reported, and its endpoint answers 503; the service and the
other models stay up. For now "current" means the most recently trained artifact of each
task; an explicit production pointer comes with the model registry phase.

Security: joblib files can execute code when loaded, so only the configured artifact folder
is read, and the service mounts it read-only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sklearn

from ml import artifacts
from ml.clustering.features import USER_FEATURES
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger("focusbuddy.ml_api")

TASKS = {"regression": FEATURE_COLUMNS, "classification": FEATURE_COLUMNS, "clustering": list(USER_FEATURES)}
NOTE_FIELDS = ("beats_best_baseline", "beats_baselines", "best_baseline", "threshold", "k", "silhouette",
               "stability_ari", "weak_structure")


@dataclass(frozen=True)
class LoadedModel:
    task: str
    file: str
    model: Any
    metadata: dict

    @property
    def name(self) -> str:
        return self.metadata.get("selected_model") or self.file.split("_")[0]

    @property
    def version(self) -> str:
        return self.metadata["trained_at"]

    @property
    def data_source(self) -> str:
        return self.metadata.get("data_source", "unknown")

    @property
    def sklearn_matches(self) -> bool:
        """A pickle from another scikit-learn version may load and still behave differently."""
        return self.metadata.get("sklearn_version") == sklearn.__version__


@dataclass
class ModelRegistry:
    root: Path
    models: dict[str, LoadedModel] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def load_all(self) -> ModelRegistry:
        for task, features in TASKS.items():
            try:
                loaded = self._load_latest(task, features)
            except Exception as exc:  # noqa: BLE001 - one broken artifact must not stop the service
                logger.exception("could not load the %s model", task)
                self.errors[task] = f"{type(exc).__name__}: could not load the artifact"
                continue
            if loaded is None:
                self.errors[task] = "no trained model found"
                continue
            self.models[task] = loaded
            if not loaded.sklearn_matches:
                logger.warning("%s model was saved with scikit-learn %s, running %s", task,
                               loaded.metadata.get("sklearn_version"), sklearn.__version__)
        return self

    def _load_latest(self, task: str, features) -> LoadedModel | None:
        candidates = []
        for path in (Path(self.root) / task).glob("*.joblib"):
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                candidates.append((json.loads(sidecar.read_text(encoding="utf-8"))["trained_at"], path))
        if not candidates:
            return None
        path = max(candidates)[1]
        model, metadata = artifacts.load(path, expected_features=features)
        return LoadedModel(task, path.name, model, metadata)

    def notes(self, loaded: LoadedModel) -> dict:
        return {k: loaded.metadata[k] for k in NOTE_FIELDS if k in loaded.metadata}
