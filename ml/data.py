"""Shared data access and splitting for every model family."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.features import TemporalSplit, build_feature_table, temporal_split

MIN_FIT_ROWS = 200
MIN_TEST_ROWS = 50


class InsufficientDataError(ValueError):
    """Raised instead of training a model whose metrics would be meaningless."""


@dataclass(frozen=True)
class ModelSplits:
    """fit + validation = train (the past); test = the future, used once."""

    fit: pd.DataFrame
    validation: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    purged: int


def model_splits(table: pd.DataFrame, test_fraction: float = 0.2, validation_fraction: float = 0.25) -> ModelSplits:
    outer: TemporalSplit = temporal_split(table, test_fraction)
    inner: TemporalSplit = temporal_split(outer.train, validation_fraction)
    if len(inner.train) < MIN_FIT_ROWS or len(outer.test) < MIN_TEST_ROWS:
        raise InsufficientDataError(
            f"need >= {MIN_FIT_ROWS} fit rows and >= {MIN_TEST_ROWS} test rows, "
            f"got {len(inner.train)} and {len(outer.test)}: not enough data to train meaningfully"
        )
    return ModelSplits(inner.train, inner.test, outer.train, outer.test, outer.cutoff,
                       outer.purged + inner.purged)


def load_logs_rooms(source: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned ``(logs, rooms)`` from ``synthetic`` data or the configured database (``db``, read-only)."""
    if source == "synthetic":
        from ml import synthetic

        return synthetic.generate(seed=seed)
    if source != "db":
        raise ValueError(f"unknown source {source!r}")

    from analytics.dataset import load_dataset
    from analytics.quality import engine_from_env

    engine = engine_from_env()
    if engine is None:
        raise SystemExit("DATABASE_URL is not set.")
    with engine.connect() as conn:
        ds = load_dataset(conn)
    return ds.logs, ds.rooms


def load_table(source: str, seed: int = 0) -> pd.DataFrame:
    """Session-level feature table (see ml.features) for ``synthetic`` or ``db``."""
    return build_feature_table(*load_logs_rooms(source, seed))
