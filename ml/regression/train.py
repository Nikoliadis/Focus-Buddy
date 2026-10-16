"""Train and evaluate focus-time regression models.

Protocol (time-ordered, so the model never learns from the future):

    all sessions ──temporal_split──► train (past 80%)          │ test (future 20%)
    train        ──temporal_split──► fit (75%) │ validation    │
                                                                │
    1. fit every candidate on `fit`, score on `validation`      │
    2. select the best non-baseline model by validation MAE     │
    3. refit the selected model on all of `train`               │
    4. score every model ONCE on `test` ────────────────────────┘

The test set never influences a decision, so its numbers are an honest estimate of
performance on future sessions. Choosing a model by its test score would make that score
optimistic (selection on the test set is a subtle form of leakage).

Usage:
    python -m ml.regression.train --source synthetic
    python -m ml.regression.train --source db          # DATABASE_URL, read-only
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ml import artifacts
from ml.data import InsufficientDataError, load_table, model_splits
from ml.evaluation.metrics import regression_metrics
from ml.features import FEATURE_COLUMNS, features_and_target
from ml.regression.model import is_baseline, make_models

__all__ = ["InsufficientDataError", "RegressionReport", "load_table", "save", "train_and_evaluate"]

TARGET = "target_focused_minutes"
DEFAULT_ARTIFACT_DIR = Path("artifacts/regression")


@dataclass
class RegressionReport:
    data_source: str
    target: str
    n_rows: int
    n_users: int
    n_fit: int
    n_validation: int
    n_train: int
    n_test: int
    purged: int
    cutoff: str
    validation: dict[str, dict[str, float]]
    test: dict[str, dict[str, float]]
    selected_model: str
    best_baseline: str
    beats_best_baseline: bool
    random_split_test: dict[str, float]
    importances: dict[str, float]
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_text(self) -> str:
        banner = "=" * 72
        lines = []
        if self.data_source != "db":
            lines += [banner, f"  {self.data_source.upper()} DATA: these metrics say nothing about real users",
                      banner]
        lines += [
            f"Target: {self.target}   rows: {self.n_rows}   users: {self.n_users}",
            f"Split by time at {self.cutoff}: fit {self.n_fit} / validation {self.n_validation} "
            f"/ test {self.n_test} (purged {self.purged})",
            "",
            f"{'model':<24}{'val MAE':>9}{'test MAE':>10}{'RMSE':>8}{'R2':>8}",
        ]
        for name, m in self.test.items():
            marker = "  <- selected" if name == self.selected_model else ""
            lines.append(f"{name:<24}{self.validation[name]['mae']:>9.2f}{m['mae']:>10.2f}"
                         f"{m['rmse']:>8.2f}{m['r2']:>8.3f}{marker}")
        sel, base = self.test[self.selected_model], self.test[self.best_baseline]
        verdict = "beats" if self.beats_best_baseline else "does NOT beat"
        lines += [
            "",
            f"Selected model {verdict} the best baseline ({self.best_baseline}) on test MAE: "
            f"{sel['mae']:.2f} vs {base['mae']:.2f} min",
            f"Same model with a RANDOM split (for comparison only): MAE {self.random_split_test['mae']:.2f}, "
            f"R2 {self.random_split_test['r2']:.3f}",
            "",
            "Permutation importance on test (MAE increase when a feature is shuffled):",
        ]
        lines += [f"  {k:<26}{v:>8.3f}" for k, v in list(self.importances.items())[:8]]
        return "\n".join(lines)


def _fit_score(models: dict[str, BaseEstimator], X_fit, y_fit, X_eval, y_eval):
    fitted, scores = {}, {}
    for name, model in models.items():
        m = clone(model).fit(X_fit, y_fit)
        fitted[name] = m
        scores[name] = regression_metrics(y_eval, m.predict(X_eval))
    return fitted, scores


def train_and_evaluate(
    table: pd.DataFrame,
    data_source: str,
    random_state: int = 0,
    test_fraction: float = 0.2,
    n_jobs: int = 1,
) -> tuple[RegressionReport, BaseEstimator]:
    splits = model_splits(table, test_fraction)

    models = make_models(random_state, n_jobs=n_jobs)
    X_fit, y_fit = features_and_target(splits.fit, TARGET)
    X_val, y_val = features_and_target(splits.validation, TARGET)
    X_train, y_train = features_and_target(splits.train, TARGET)
    X_test, y_test = features_and_target(splits.test, TARGET)

    # 1-2. Select on validation; baselines are compared but never selected.
    _, val_scores = _fit_score(models, X_fit, y_fit, X_val, y_val)
    candidates = {n: s for n, s in val_scores.items() if not is_baseline(n)}
    selected = min(candidates, key=lambda n: candidates[n]["mae"])

    # 3-4. Refit on the full training period, score once on the future.
    fitted, test_scores = _fit_score(models, X_train, y_train, X_test, y_test)
    best_baseline = min((n for n in test_scores if is_baseline(n)), key=lambda n: test_scores[n]["mae"])
    final = fitted[selected]

    perm = permutation_importance(final, X_test, y_test, scoring="neg_mean_absolute_error",
                                  n_repeats=10, random_state=random_state, n_jobs=1)
    importances = dict(sorted(zip(FEATURE_COLUMNS, perm.importances_mean, strict=True),
                              key=lambda kv: -kv[1]))

    # For teaching: the same model evaluated with a shuffled split.
    X_all, y_all = features_and_target(table, TARGET)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_all, y_all, test_size=test_fraction, random_state=random_state)
    random_split = regression_metrics(yr_te, clone(models[selected]).fit(Xr_tr, yr_tr).predict(Xr_te))

    report = RegressionReport(
        data_source=data_source, target=TARGET, n_rows=len(table), n_users=int(table["user_id"].nunique()),
        n_fit=len(splits.fit), n_validation=len(splits.validation), n_train=len(splits.train),
        n_test=len(splits.test), purged=splits.purged, cutoff=str(splits.cutoff),
        validation=val_scores, test=test_scores, selected_model=selected, best_baseline=best_baseline,
        beats_best_baseline=test_scores[selected]["mae"] < test_scores[best_baseline]["mae"],
        random_split_test=random_split, importances={k: float(v) for k, v in importances.items()},
    )
    return report, final


def save(model: BaseEstimator, report: RegressionReport, out_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> Path:
    """Persist the fitted model and its metadata side by side. Returns the model path."""
    stamp = report.trained_at.replace(":", "").replace("-", "")
    return artifacts.save(model, asdict(report), out_dir, f"{report.selected_model}_{report.data_source}_{stamp}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train focus-time regression models.")
    parser.add_argument("--source", choices=["synthetic", "db"], required=True)
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (never use >1 under gevent)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    table = load_table(args.source, args.seed)
    started = time.perf_counter()
    try:
        report, model = train_and_evaluate(table, args.source, random_state=args.seed, n_jobs=args.n_jobs)
    except InsufficientDataError as exc:
        print(f"Not training: {exc}")
        return 2
    print(report.to_text())
    print(f"\n(trained in {time.perf_counter() - started:.1f}s)")
    if not args.no_save:
        print(f"saved: {save(model, report, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
