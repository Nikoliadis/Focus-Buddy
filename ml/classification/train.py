"""Train and evaluate "will this session end early?" classifiers.

Same time-ordered protocol as regression (see ml.regression.train), plus a decision
threshold that is also chosen on validation:

    fit ──► validation: select the model by PR-AUC, then pick its F1-maximising threshold
    train ─► test: score every model once, at its own validation-chosen threshold

PR-AUC is the selection metric because it is threshold-free and, unlike accuracy and
ROC-AUC, it focuses on how well the *minority* class is found.

Usage:
    python -m ml.classification.train --source synthetic
    python -m ml.classification.train --source db
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, clone

from ml import artifacts
from ml.classification.model import ended_early, is_baseline, logistic_coefficients, make_models
from ml.data import InsufficientDataError, load_table, model_splits
from ml.evaluation.metrics import best_f1_threshold, classification_metrics
from ml.features import FEATURE_COLUMNS

TARGET = "ended_early"
SELECTION_METRIC = "pr_auc"
DEFAULT_ARTIFACT_DIR = Path("artifacts/classification")


@dataclass
class ClassificationReport:
    data_source: str
    target: str
    n_rows: int
    n_users: int
    n_fit: int
    n_validation: int
    n_test: int
    cutoff: str
    positive_rate_train: float
    positive_rate_test: float
    thresholds: dict[str, float]
    validation: dict[str, dict[str, float]]
    test: dict[str, dict[str, float]]
    test_at_default_threshold: dict[str, float]
    selected_model: str
    beats_baselines: bool
    coefficients: dict[str, float]
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def threshold(self) -> float:
        return self.thresholds[self.selected_model]

    def to_text(self) -> str:
        banner = "=" * 78
        lines = []
        if self.data_source != "db":
            lines += [banner, f"  {self.data_source.upper()} DATA: these metrics say nothing about real users",
                      banner]
        lines += [
            f"Target: {self.target} (1 = session ended early)   rows: {self.n_rows}   users: {self.n_users}",
            f"Class balance: {self.positive_rate_train:.1%} positive in train, {self.positive_rate_test:.1%} in test",
            f"Split by time at {self.cutoff}: fit {self.n_fit} / validation {self.n_validation} / test {self.n_test}",
            "",
            f"{'model':<30}{'thr':>6}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'ROC':>7}{'PR':>7}{'Brier':>7}",
        ]
        for name, m in self.test.items():
            marker = "  <- selected" if name == self.selected_model else ""
            lines.append(
                f"{name:<30}{self.thresholds[name]:>6.2f}{m['accuracy']:>7.3f}{m['precision']:>7.3f}"
                f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['roc_auc']:>7.3f}{m['pr_auc']:>7.3f}{m['brier']:>7.3f}{marker}"
            )
        s = self.test[self.selected_model]
        d = self.test_at_default_threshold
        lines += [
            "",
            f"PR-AUC chance level = positive rate = {self.positive_rate_test:.3f}",
            f"Selected model {'beats' if self.beats_baselines else 'does NOT beat'} every baseline on test PR-AUC.",
            "",
            f"Confusion matrix of {self.selected_model} at threshold {self.threshold:.2f} (test):",
            "                     predicted full   predicted early",
            f"  actually full      {s['tn']:>14}   {s['fp']:>15}",
            f"  actually early     {s['fn']:>14}   {s['tp']:>15}",
            f"At the default 0.50 threshold instead: precision {d['precision']:.3f}, recall {d['recall']:.3f}, "
            f"F1 {d['f1']:.3f}",
        ]
        if self.coefficients:
            lines += ["", "Logistic regression, standardised coefficients (log-odds of ending early per +1 SD):"]
            lines += [f"  {k:<36}{v:>+8.3f}" for k, v in list(self.coefficients.items())[:10]]
        return "\n".join(lines)


def _proba(model: BaseEstimator, X: pd.DataFrame):
    return model.predict_proba(X)[:, 1]


def train_and_evaluate(
    table: pd.DataFrame,
    data_source: str,
    random_state: int = 0,
    test_fraction: float = 0.2,
    n_jobs: int = 1,
) -> tuple[ClassificationReport, BaseEstimator]:
    splits = model_splits(table, test_fraction)
    y = {name: ended_early(part) for name, part in
         [("fit", splits.fit), ("val", splits.validation), ("train", splits.train), ("test", splits.test)]}
    if y["fit"].nunique() < 2:
        raise InsufficientDataError("the training data contains only one class: there is nothing to learn")
    X = {name: part[list(FEATURE_COLUMNS)] for name, part in
         [("fit", splits.fit), ("val", splits.validation), ("train", splits.train), ("test", splits.test)]}

    models = make_models(random_state, n_jobs=n_jobs)
    thresholds, val_scores = {}, {}
    for name, model in models.items():
        p_val = _proba(clone(model).fit(X["fit"], y["fit"]), X["val"])
        thresholds[name] = 0.5 if is_baseline(name) else best_f1_threshold(y["val"], p_val)
        val_scores[name] = classification_metrics(y["val"], p_val, thresholds[name])

    candidates = {n: s for n, s in val_scores.items() if not is_baseline(n)}
    selected = max(candidates, key=lambda n: candidates[n][SELECTION_METRIC])

    fitted, test_scores = {}, {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(X["train"], y["train"])
        test_scores[name] = classification_metrics(y["test"], _proba(fitted[name], X["test"]), thresholds[name])

    final = fitted[selected]
    best_baseline_pr = max(s["pr_auc"] for n, s in test_scores.items() if is_baseline(n))
    coefs = logistic_coefficients(fitted["logistic_regression"])

    report = ClassificationReport(
        data_source=data_source, target=TARGET, n_rows=len(table), n_users=int(table["user_id"].nunique()),
        n_fit=len(splits.fit), n_validation=len(splits.validation), n_test=len(splits.test),
        cutoff=str(splits.cutoff),
        positive_rate_train=float(y["train"].mean()), positive_rate_test=float(y["test"].mean()),
        thresholds=thresholds, validation=val_scores, test=test_scores,
        test_at_default_threshold=classification_metrics(y["test"], _proba(final, X["test"]), 0.5),
        selected_model=selected, beats_baselines=test_scores[selected]["pr_auc"] > best_baseline_pr,
        coefficients={k: float(v) for k, v in coefs.items()},
    )
    return report, final


def save(model: BaseEstimator, report: ClassificationReport, out_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> Path:
    stamp = report.trained_at.replace(":", "").replace("-", "")
    meta = {**asdict(report), "threshold": report.threshold}
    return artifacts.save(model, meta, out_dir, f"{report.selected_model}_{report.data_source}_{stamp}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train 'ended early' classifiers.")
    parser.add_argument("--source", choices=["synthetic", "db"], required=True)
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (never use >1 under gevent)")
    args = parser.parse_args(argv)

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
