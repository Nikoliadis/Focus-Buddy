"""Hyperparameter tuning: grid search, randomized search and Optuna, done carefully.

    parameter       learned from data by fit():  coefficients, tree splits, scaler means
    hyperparameter  chosen before fit(), controls *how* the model learns: C, min_samples_leaf

Rules this module enforces:
1. Search on the training period only, scored with purged temporal CV. The test set is
   touched once, after the search, to compare the tuned model with the default one.
2. Report the default configuration next to the best one. An "improvement" smaller than
   the fold-to-fold noise is not an improvement.
3. One-standard-error rule (paired): among configurations not distinguishable from the
   best, fold by fold, pick the *simplest*. The best CV score is itself an optimistic,
   noisy estimate (we picked the maximum of many noisy numbers).
4. Log every search (space, scores, choice, data, code version) to an experiment log.
   MLflow replaces this file in Phase 20.

Usage:
    python -m ml.tuning --task classification --model random_forest --method random
    python -m ml.tuning --task classification --model logistic_regression --method grid
    python -m ml.tuning --task classification --model random_forest --method optuna --trials 40
"""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV

from ml.data import model_splits
from ml.evaluation.card import PRIMARY_METRIC_FN
from ml.evaluation.cv import PurgedTemporalSplit
from ml.evaluation.diagnostics import paired_bootstrap_difference
from ml.evaluation.tasks import Task, get_task
from ml.features import FEATURE_COLUMNS
from ml.pipelines import make_preprocessor

SCORING = {"classification": "average_precision", "regression": "neg_mean_absolute_error"}
EXPERIMENT_LOG = Path("artifacts/experiments/tuning.jsonl")


@dataclass(frozen=True)
class SearchSpace:
    """Candidate values per hyperparameter, plus how to rank configurations by simplicity.

    ``complexity`` maps a parameter dict to a sortable key: smaller = simpler model."""

    grid: dict[str, list]
    complexity: Callable[[dict], tuple]
    rationale: str
    fixed: dict[str, Any] = field(default_factory=dict)


def hour_encoding(prep: BaseEstimator) -> str:
    """'cyclic' or 'onehot' for a preprocessor built by ml.pipelines.make_preprocessor."""
    names = [name for name, _, _ in prep.named_steps["columns"].transformers]
    return "cyclic" if "cyclic" in names else "onehot"


def _rf_complexity(p: dict) -> tuple:
    depth = p.get("max_depth") or 10_000  # None = unlimited = most complex
    return (-p.get("min_samples_leaf", 1), depth, p.get("max_features", 1.0))


SPACES: dict[tuple[str, str], SearchSpace] = {
    ("classification", "logistic_regression"): SearchSpace(
        grid={
            # C = 1 / regularisation strength, searched on a log scale: 0.01 vs 0.1 matters
            # as much as 1 vs 10.
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "prep__columns__rate__shrink__strength": [0.0, 5.0, 25.0],
            "prep": [make_preprocessor("onehot"), make_preprocessor("cyclic")],
        },
        # Smaller C = stronger regularisation = simpler; a 2-term wave is simpler than 23 dummies.
        complexity=lambda p: (p["model__C"], 0 if hour_encoding(p["prep"]) == "cyclic" else 1),
        rationale="regularisation strength, history shrinkage and hour encoding (24 dummies vs a 2-term wave)",
    ),
    ("classification", "random_forest"): SearchSpace(
        grid={
            "min_samples_leaf": [1, 2, 5, 10, 25, 50, 100, 200],
            "max_depth": [None, 4, 6, 8, 12, 16],
            "max_features": [0.2, 0.35, 0.5, 0.7, 1.0],
        },
        complexity=_rf_complexity,
        rationale="tree size limits (leaf size, depth) and per-split feature sampling: the levers against "
                  "the overfitting found in Phase 10",
        # More trees never overfit; they only cost time. Not a tuning target.
        fixed={"n_estimators": 200},
    ),
    ("regression", "random_forest"): SearchSpace(
        grid={
            "min_samples_leaf": [1, 5, 10, 25, 50, 100, 200],
            "max_depth": [None, 4, 6, 8, 12],
            "max_features": [0.3, 0.5, 0.7, 1.0],
        },
        complexity=_rf_complexity,
        rationale="tree size limits and per-split feature sampling",
        fixed={"n_estimators": 200},
    ),
}


def one_standard_error_choice(results: pd.DataFrame, complexity: Callable[[dict], tuple]) -> int:
    """Index of the simplest configuration that is not distinguishable from the best.

    Paired version of the one-standard-error rule: every configuration was scored on the
    *same* folds, and folds differ a lot in difficulty, which inflates each configuration's
    own SE and would let clearly worse settings count as "equivalent". So we compare each
    candidate with the best fold by fold and use the SE of those differences:

        eligible  <=>  mean(best - candidate) <= SE(best - candidate)

    ``results`` needs columns: params, mean (higher is better), fold_scores (one per fold).
    """
    best = results["mean"].idxmax()
    best_folds = np.asarray(results.loc[best, "fold_scores"])
    k = best_folds.size

    def equivalent(i: int) -> bool:
        diff = best_folds - np.asarray(results.loc[i, "fold_scores"])
        se = diff.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
        return diff.mean() <= se

    eligible = [i for i in results.index if equivalent(i)]
    return int(min(eligible, key=lambda i: complexity(results.loc[i, "params"])))


@dataclass
class SearchResult:
    task: str
    model: str
    method: str
    data_source: str
    scoring: str
    space: dict[str, list]
    rationale: str
    n_candidates: int
    n_folds: int
    seconds: float
    results: pd.DataFrame
    best_index: int
    selected_index: int
    default_cv: dict[str, float]
    test: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> pd.Series:
        return self.results.loc[self.best_index]

    @property
    def selected(self) -> pd.Series:
        return self.results.loc[self.selected_index]

    @property
    def improvement_over_default(self) -> float:
        return float(self.selected["mean"] - self.default_cv["mean"])

    @property
    def paired_gain(self) -> tuple[float, float]:
        """Mean and standard error of (selected - default), fold by fold."""
        diff = np.asarray(self.selected["fold_scores"]) - np.asarray(self.default_cv["fold_scores"])
        return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0

    @property
    def beyond_noise(self) -> bool:
        """Is the fold-by-fold gain over the default larger than twice its standard error?"""
        mean, se = self.paired_gain
        return mean > 2 * se

    def to_text(self, top: int = 8) -> str:
        sign = -1 if self.scoring.startswith("neg_") else 1
        label = self.scoring.removeprefix("neg_")
        fmt = lambda v: f"{sign * v:.4f}"  # noqa: E731
        banner = "=" * 78
        lines = []
        if self.data_source != "db":
            lines += [banner, f"  {self.data_source.upper()} DATA: results describe the pipeline, not real users",
                      banner]
        lines += [
            f"{self.method} search: {self.model} ({self.task}), {self.n_candidates} distinct configurations x "
            f"{self.n_folds} purged temporal folds, {self.seconds:.0f}s",
            f"Space: {self.rationale}",
            "",
            f"Top configurations by mean CV {label} (+/- std across folds):",
        ]
        ranked = self.results.sort_values("mean", ascending=False).head(top)
        for i, row in ranked.iterrows():
            mark = " <- best" if i == self.best_index else ""
            mark += " <- selected" if i == self.selected_index else ""
            lines.append(f"  {fmt(row['mean'])} +/- {row['std']:.4f}  {_short(row['params'])}{mark}")
        sel = self.selected
        lines += [
            "",
            "Selected (simplest not distinguishable from the best, paired 1-SE rule):",
            f"  {fmt(sel['mean'])} +/- {sel['std']:.4f}  {_short(sel['params'])}",
            f"Default configuration: {fmt(self.default_cv['mean'])} +/- {self.default_cv['std']:.4f}",
            f"Best vs default (CV): {(self.best['mean'] - self.default_cv['mean']) * sign:+.4f}",
            f"Selected vs default (CV, paired by fold): {self.paired_gain[0] * sign:+.4f} "
            f"(se {self.paired_gain[1]:.4f}) -> "
            + ("more than 2 SE: a consistent gain across folds" if self.beyond_noise
               else "not consistent across folds: no reliable gain"),
            f"Train vs CV for the selected model: {fmt(self.selected['train_mean'])} vs "
            f"{fmt(self.selected['mean'])} (default: {fmt(self.default_cv['train_mean'])} vs "
            f"{fmt(self.default_cv['mean'])})",
        ]
        if self.test:
            t = self.test
            lines += [
                "",
                f"Held-out test ({t['n']} sessions, evaluated once): selected {t['selected']:.4f}, "
                f"default {t['default']:.4f}",
                f"Paired bootstrap difference (selected - default): {t['diff']:+.4f}, "
                f"95% CI [{t['diff_lo']:+.4f}, {t['diff_hi']:+.4f}]",
            ]
        return "\n".join(lines)

    def log(self, path: Path = EXPERIMENT_LOG) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "task": self.task, "model": self.model, "method": self.method, "data_source": self.data_source,
            "scoring": self.scoring, "space": {k: [_short(v) for v in vs] for k, vs in self.space.items()},
            "n_candidates": self.n_candidates, "n_folds": self.n_folds, "seconds": round(self.seconds, 1),
            "best": {"params": _short(self.best["params"]), "mean": self.best["mean"], "std": self.best["std"]},
            "selected": {"params": _short(self.selected["params"]), "mean": self.selected["mean"],
                         "std": self.selected["std"]},
            "default": self.default_cv, "beyond_noise": self.beyond_noise, "test": self.test,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
        return path


def _short(value) -> Any:
    if isinstance(value, dict):
        return {k.replace("prep__columns__", "").replace("model__", ""): _short(v) for k, v in value.items()}
    if isinstance(value, BaseEstimator):
        return f"prep({hour_encoding(value)})"
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _base_estimator(task: Task, model_name: str, space: SearchSpace, random_state: int) -> BaseEstimator:
    model = clone(task.make_models(random_state, n_jobs=1)[model_name])
    return model.set_params(**space.fixed) if space.fixed else model


def _results_frame(cv_results: dict) -> pd.DataFrame:
    n_folds = sum(1 for k in cv_results if k.startswith("split") and k.endswith("_test_score"))
    folds = np.column_stack([cv_results[f"split{i}_test_score"] for i in range(n_folds)])
    return pd.DataFrame({
        "params": cv_results["params"],
        "mean": cv_results["mean_test_score"],
        "std": cv_results["std_test_score"],
        "se": cv_results["std_test_score"] / np.sqrt(n_folds),
        "train_mean": cv_results["mean_train_score"],
        "fold_scores": list(folds),
    })


def _optuna_search(estimator, space, X, y, cv, scoring, n_trials, seed, n_jobs) -> pd.DataFrame:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if "n_jobs" in estimator.get_params():  # trials run one at a time, so parallelise inside the model
        estimator = clone(estimator).set_params(n_jobs=n_jobs)
    rows: dict[tuple, dict] = {}
    scorer = check_scoring(estimator, scoring=scoring)

    def objective(trial: optuna.Trial) -> float:
        # Values are suggested by index: Optuna stores categorical choices, and estimator
        # objects (the preprocessors) are not storable.
        choice = tuple(trial.suggest_categorical(name, list(range(len(values))))
                       for name, values in space.grid.items())
        if choice in rows:  # TPE often re-proposes a categorical combination: reuse, don't refit
            return float(rows[choice]["mean"])
        params = {name: values[i] for (name, values), i in zip(space.grid.items(), choice, strict=True)}
        candidate = clone(estimator).set_params(**params)
        scores = []
        for train_idx, val_idx in cv.split(X):
            fitted = clone(candidate).fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append((scorer(fitted, X.iloc[val_idx], y.iloc[val_idx]),
                           scorer(fitted, X.iloc[train_idx], y.iloc[train_idx])))
        val, train = np.array(scores).T
        rows[choice] = {"params": params, "mean": val.mean(), "std": val.std(),
                        "se": val.std() / np.sqrt(len(val)), "train_mean": train.mean(), "fold_scores": val}
        return float(val.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return pd.DataFrame(list(rows.values()))


def tune(
    table: pd.DataFrame,
    task_name: str,
    model_name: str,
    method: str = "random",
    data_source: str = "synthetic",
    n_iter: int = 30,
    n_splits: int = 5,
    random_state: int = 0,
    n_jobs: int = 1,
    evaluate_test: bool = True,
) -> SearchResult:
    """``n_jobs`` > 1 parallelises over candidates and folds; only from a plain process (not gevent)."""
    task = get_task(task_name)
    if (task_name, model_name) not in SPACES:
        raise ValueError(f"no search space for {task_name}/{model_name}; available: {sorted(SPACES)}")
    space = SPACES[(task_name, model_name)]
    splits = model_splits(table)
    train = splits.train.reset_index(drop=True)
    X, y = train[list(FEATURE_COLUMNS)], task.target(train)
    cv = PurgedTemporalSplit(train["as_of"], train["label_available_at"], n_splits)
    estimator = _base_estimator(task, model_name, space, random_state)
    scoring = SCORING[task_name]
    started = time.perf_counter()

    common = {"scoring": scoring, "cv": cv, "refit": False, "return_train_score": True, "n_jobs": n_jobs}
    if method == "grid":
        search = GridSearchCV(estimator, space.grid, **common).fit(X, y)
        results = _results_frame(search.cv_results_)
    elif method == "random":
        search = RandomizedSearchCV(estimator, space.grid, n_iter=n_iter, random_state=random_state,
                                    **common).fit(X, y)
        results = _results_frame(search.cv_results_)
    elif method == "optuna":
        results = _optuna_search(estimator, space, X, y, cv, scoring, n_iter, random_state, n_jobs)
    else:
        raise ValueError("method must be 'grid', 'random' or 'optuna'")
    elapsed = time.perf_counter() - started

    default = GridSearchCV(estimator, {k: [v] for k, v in estimator.get_params().items() if k in space.grid},
                           **common).fit(X, y)
    d = _results_frame(default.cv_results_).iloc[0]
    default_cv = {"mean": float(d["mean"]), "std": float(d["std"]), "train_mean": float(d["train_mean"]),
                  "fold_scores": [float(v) for v in d["fold_scores"]]}

    result = SearchResult(
        task=task_name, model=model_name, method=method, data_source=data_source, scoring=scoring,
        space=space.grid, rationale=space.rationale, n_candidates=len(results), n_folds=cv.get_n_splits(),
        seconds=elapsed, results=results, best_index=int(results["mean"].idxmax()),
        selected_index=one_standard_error_choice(results, space.complexity), default_cv=default_cv,
    )
    if evaluate_test:
        result.test = _test_comparison(task, estimator, result.selected["params"], splits, random_state)
    return result


def _test_comparison(task: Task, estimator, params: dict, splits, random_state: int) -> dict:
    cols = list(FEATURE_COLUMNS)
    X_tr, y_tr = splits.train[cols], task.target(splits.train)
    X_te, y_te = splits.test[cols], task.target(splits.test).to_numpy()
    tuned = clone(estimator).set_params(**params).fit(X_tr, y_tr)
    default = clone(estimator).fit(X_tr, y_tr)
    metric = PRIMARY_METRIC_FN[task.primary_metric]
    p_tuned, p_default = task.predict(tuned, X_te), task.predict(default, X_te)
    diff, lo, hi = paired_bootstrap_difference(y_te, p_tuned, p_default, metric, seed=random_state,
                                               skip_single_class=task.name == "classification")
    return {"metric": task.primary_metric, "n": len(y_te), "selected": float(metric(y_te, p_tuned)),
            "default": float(metric(y_te, p_default)), "diff": diff, "diff_lo": lo, "diff_hi": hi}


def grid_size(task_name: str, model_name: str) -> int:
    return len(ParameterGrid(SPACES[(task_name, model_name)].grid))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ml.data import InsufficientDataError, load_table

    parser = argparse.ArgumentParser(description="Tune hyperparameters with purged temporal CV.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--model", choices=["logistic_regression", "random_forest"], required=True)
    parser.add_argument("--method", choices=["grid", "random", "optuna"], default="random")
    parser.add_argument("--source", choices=["synthetic", "db"], default="synthetic")
    parser.add_argument("--trials", type=int, default=30, help="candidates for random/optuna")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if args.method == "grid":
        print(f"grid size: {grid_size(args.task, args.model)} configurations")
    try:
        result = tune(load_table(args.source, args.seed), args.task, args.model, args.method, args.source,
                      n_iter=args.trials, random_state=args.seed, n_jobs=args.n_jobs)
    except InsufficientDataError as exc:
        print(f"Not tuning: {exc}")
        return 2
    print(result.to_text())
    if not args.no_log:
        print(f"\nlogged to {result.log()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
