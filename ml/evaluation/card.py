"""Evaluation cards: one standard, honest summary for every supervised model.

Every card answers the same questions:
    what data (size, users, period, source), which features, which target,
    how it was split, which baseline, what metrics (with uncertainty),
    how it behaves across folds, whether it under- or overfits, and its limitations.

Protocol (no decision ever looks at the test set):
    fit → validation : choose the model AND the baseline it is compared with
    temporal CV      : on the training period only, for variance and fit diagnosis
    train → test     : one final score, with bootstrap CIs
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml.data import model_splits
from ml.evaluation import cv
from ml.evaluation.diagnostics import (
    FitDiagnosis,
    bootstrap_ci,
    diagnose_fit,
    learning_curve,
    paired_bootstrap_difference,
)
from ml.evaluation.tasks import Task, get_task
from ml.features import FEATURE_COLUMNS

PRIMARY_METRIC_FN = {"mae": mean_absolute_error, "pr_auc": average_precision_score}
METRIC_LABELS = {"mae": "MAE", "pr_auc": "PR-AUC", "rmse": "RMSE", "r2": "R2", "roc_auc": "ROC-AUC", "brier": "Brier"}
HIGH_CV_VARIATION = 0.15  # fold std / |mean| above this: the score depends a lot on the period


@dataclass
class EvaluationCard:
    task: str
    model: str
    target: str
    data_source: str
    dataset: dict
    features: list[str]
    split: dict
    baseline: str
    primary_metric: str
    higher_is_better: bool
    test_model: dict
    test_baseline: dict
    primary_ci: tuple[float, float, float]
    difference_vs_baseline_ci: tuple[float, float, float]
    cv: dict
    diagnosis: dict
    learning_curve: list[dict]
    limitations: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def significantly_better_than_baseline(self) -> bool:
        _, lo, hi = self.difference_vs_baseline_ci
        return lo > 0 if self.higher_is_better else hi < 0

    def to_markdown(self) -> str:
        m, pm = self.primary_metric, METRIC_LABELS[self.primary_metric]
        est, lo, hi = self.primary_ci
        d, dlo, dhi = self.difference_vs_baseline_ci
        sig = ("yes: the 95% CI of the difference excludes 0" if self.significantly_better_than_baseline
               else "no: the 95% CI of the difference includes 0 or favours the baseline")
        lines = [f"# Evaluation card: {self.model} ({self.task})", ""]
        if self.data_source != "db":
            lines += [f"> **{self.data_source.upper()} DATA.** These numbers describe the pipeline, "
                      "not real users.", ""]
        lines += [
            "## Data",
            f"- Source: `{self.data_source}`; {self.dataset['rows']} sessions from {self.dataset['users']} users, "
            f"{self.dataset['start']} to {self.dataset['end']}",
            f"- Target: {self.target}",
            f"- Features ({len(self.features)}): {', '.join(f'`{f}`' for f in self.features)}",
            "",
            "## Split (by time)",
            f"- fit {self.split['fit']} / validation {self.split['validation']} / test {self.split['test']} "
            f"(purged {self.split['purged']}); test starts {self.split['cutoff']}",
            "- Model and baseline chosen on validation; test used once",
            "",
            "## Test performance",
            "",
            f"| | {pm} | " + " | ".join(METRIC_LABELS[k] for k in self._extra_metrics()) + " |",
            "|---|---|" + "---|" * len(self._extra_metrics()),
            f"| **{self.model}** | {self.test_model[m]:.3f} | "
            + " | ".join(f"{self.test_model[k]:.3f}" for k in self._extra_metrics()) + " |",
            f"| baseline `{self.baseline}` | {self.test_baseline[m]:.3f} | "
            + " | ".join(f"{self.test_baseline[k]:.3f}" for k in self._extra_metrics()) + " |",
            "",
            f"- {pm} {est:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]",
            f"- Difference vs baseline: {d:+.3f}, 95% CI [{dlo:+.3f}, {dhi:+.3f}] "
            f"({'higher' if self.higher_is_better else 'lower'} is better)",
            f"- Better than the baseline? **{sig}**",
            "",
            "## Cross-validation (training period only)",
        ]
        for scheme, s in self.cv.items():
            lines.append(f"- {scheme}: {pm} {s['model']['mean']:.3f} +/- {s['model']['std']:.3f} over "
                         f"{s['model']['folds']} folds (baseline {s['baseline']['mean']:.3f} "
                         f"+/- {s['baseline']['std']:.3f})")
        lines += [
            "",
            "## Fit diagnosis",
            f"- **{self.diagnosis['verdict']}**: {self.diagnosis['explanation']}",
            "",
            "Learning curve (validation " + pm + " vs share of the fit period used):",
            "",
            "| share | rows | train | validation |",
            "|---|---|---|---|",
        ]
        lines += [f"| {r['fraction']:.0%} | {r['n_train']} | {r['train']:.3f} | {r['validation']:.3f} |"
                  for r in self.learning_curve]
        lines += ["", "## Limitations", *[f"- {item}" for item in self.limitations], ""]
        return "\n".join(lines)

    def _extra_metrics(self) -> list[str]:
        if self.task == "regression":
            return ["rmse", "r2"]
        return ["roc_auc", "brier"]

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = self.created_at.replace(":", "").replace("-", "")
        path = out / f"card_{self.task}_{self.model}_{self.data_source}_{stamp}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps(asdict(self), indent=2, default=float), encoding="utf-8")
        return path


def _limitations(card_inputs: dict, task: Task, data_source: str) -> list[str]:
    items = []
    if data_source != "db":
        items.append(f"Trained and evaluated on {data_source} data generated from assumptions, not on real users.")
    else:
        items.append("Focus logs exist only for the user who ended a session (the room owner): other "
                     "participants are missing, so results describe owners only.")
        items.append("Break sessions are logged like focus sessions; the phase is not stored.")
    if task.name == "classification":
        rate = card_inputs["positive_rate"]
        items.append(f"Class imbalance: {rate:.0%} of test sessions ended early; PR-AUC chance level is {rate:.2f}.")
        items.append("Outcome labels are derived from focused/planned time; the app does not record why a "
                     "session ended.")
    temporal = card_inputs["cv"]["temporal"]["model"]
    variation = temporal["std"] / abs(temporal["mean"]) if temporal["mean"] else float("nan")
    if np.isfinite(variation) and variation > HIGH_CV_VARIATION:
        items.append(f"Unstable across periods: fold-to-fold std is {variation:.0%} of the mean.")
    if not card_inputs["significant"]:
        items.append("Not distinguishable from the baseline on this test set: prefer the baseline unless "
                     "other criteria (ranking, calibration) justify the model.")
    if card_inputs["users"] < 30:
        items.append(f"Only {card_inputs['users']} users: generalisation to new users is weakly supported.")
    items.append("Observational data: feature importances and coefficients describe associations, not causes.")
    return items


def evaluate(table: pd.DataFrame, task_name: str, data_source: str, n_splits: int = 5,
             random_state: int = 0, n_jobs: int = 1) -> EvaluationCard:
    """``n_jobs`` > 1 only from a plain process (joblib deadlocks under gevent)."""
    task = get_task(task_name)
    splits = model_splits(table)
    models = task.make_models(random_state, n_jobs=n_jobs)
    cols = list(FEATURE_COLUMNS)

    # 1. Choose the model and its baseline on validation.
    val = {}
    for name, model in models.items():
        fitted = clone(model).fit(splits.fit[cols], task.target(splits.fit))
        val[name] = task.score(fitted, splits.validation[cols], task.target(splits.validation))[task.primary_metric]
    pick = max if task.higher_is_better else min
    model_name = pick((n for n in val if not n.startswith("baseline_")), key=val.get)
    baseline_name = pick((n for n in val if n.startswith("baseline_")), key=val.get)

    # 2. Cross-validation on the training period only (the test set stays untouched).
    cv_results = {}
    for scheme in ("temporal", "user"):
        folds = {name: cv.cross_validate(models[name], splits.train, task, scheme, n_splits)
                 for name in (model_name, baseline_name)}
        cv_results[scheme] = {
            "model": cv.summarise(folds[model_name], task.primary_metric),
            "baseline": cv.summarise(folds[baseline_name], task.primary_metric),
            "model_train_mean": float(folds[model_name][f"train_{task.primary_metric}"].mean()),
        }

    temporal = cv_results["temporal"]
    diagnosis: FitDiagnosis = diagnose_fit(task, temporal["model_train_mean"], temporal["model"]["mean"],
                                           temporal["baseline"]["mean"])
    curve = learning_curve(models[model_name], splits.fit, splits.validation, task)

    # 3. One final score on the future.
    y_test = task.target(splits.test).to_numpy()
    final = clone(models[model_name]).fit(splits.train[cols], task.target(splits.train))
    base = clone(models[baseline_name]).fit(splits.train[cols], task.target(splits.train))
    pred_model, pred_base = task.predict(final, splits.test[cols]), task.predict(base, splits.test[cols])
    metric_fn = PRIMARY_METRIC_FN[task.primary_metric]
    single = task.name == "classification"
    primary_ci = bootstrap_ci(y_test, pred_model, metric_fn, seed=random_state, skip_single_class=single)
    diff = paired_bootstrap_difference(y_test, pred_model, pred_base, metric_fn, seed=random_state,
                                       skip_single_class=single)
    _, dlo, dhi = diff
    significant = dlo > 0 if task.higher_is_better else dhi < 0

    users = int(table["user_id"].nunique())
    inputs = {"cv": cv_results, "significant": significant, "users": users,
              "positive_rate": float(np.mean(y_test)) if single else None}
    return EvaluationCard(
        task=task.name, model=model_name, target=task.target_description, data_source=data_source,
        dataset={"rows": len(table), "users": users,
                 "start": f"{table['as_of'].min():%Y-%m-%d}", "end": f"{table['as_of'].max():%Y-%m-%d}"},
        features=cols,
        split={"fit": len(splits.fit), "validation": len(splits.validation), "train": len(splits.train),
               "test": len(splits.test), "purged": splits.purged, "cutoff": f"{splits.cutoff:%Y-%m-%d %H:%M}"},
        baseline=baseline_name, primary_metric=task.primary_metric, higher_is_better=task.higher_is_better,
        test_model=task.metrics(y_test, pred_model), test_baseline=task.metrics(y_test, pred_base),
        primary_ci=primary_ci, difference_vs_baseline_ci=diff, cv=cv_results,
        diagnosis=asdict(diagnosis), learning_curve=curve.to_dict("records"),
        limitations=_limitations(inputs, task, data_source),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ml.data import InsufficientDataError, load_table

    parser = argparse.ArgumentParser(description="Produce an evaluation card for the best model of a task.")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--source", choices=["synthetic", "db"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/cards")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    try:
        card = evaluate(load_table(args.source, args.seed), args.task, args.source, random_state=args.seed,
                        n_jobs=args.n_jobs)
    except InsufficientDataError as exc:
        print(f"Not evaluating: {exc}")
        return 2
    print(card.to_markdown())
    if not args.no_save:
        print(f"saved: {card.save(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
