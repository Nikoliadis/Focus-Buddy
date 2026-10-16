"""Does a neural network beat the classical models here? Compare honestly.

Same protocol as every other model: purged temporal CV on the training period, then one
evaluation on the held-out future with a paired bootstrap CI against the best classical
model. Expectation (Phase 0): on a few thousand rows of ~17 tabular features, an MLP rarely
beats a well-regularised tree ensemble. The point is to learn PyTorch, and to measure.

Usage:
    python -m ml.deep_learning.train --source synthetic
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from ml.classification.model import make_logistic
from ml.data import model_splits
from ml.deep_learning.mlp import TorchMLPClassifier, resolve_device
from ml.evaluation import cv
from ml.evaluation.diagnostics import paired_bootstrap_difference
from ml.evaluation.tasks import get_task
from ml.features import FEATURE_COLUMNS
from ml.pipelines import make_preprocessor

# Selected by the paired 1-SE rule in Phase 12 (synthetic data).
TUNED_FOREST = {"n_estimators": 300, "min_samples_leaf": 100, "max_depth": 8, "max_features": 0.35}


def make_candidates(random_state: int = 0, n_jobs: int = 1, **mlp_params) -> dict[str, BaseEstimator]:
    return {
        "logistic_regression": make_logistic(),
        "random_forest_tuned": RandomForestClassifier(**TUNED_FOREST, n_jobs=n_jobs, random_state=random_state),
        # The MLP needs the same preprocessing as the linear model: scaled, imputed, one-hot.
        "mlp": Pipeline([("prep", make_preprocessor()),
                         ("model", TorchMLPClassifier(random_state=random_state, **mlp_params))]),
    }


@dataclass
class DeepLearningReport:
    data_source: str
    device: str
    cv: dict[str, dict[str, float]]
    test: dict[str, float]
    best_classical: str
    mlp_vs_best: tuple[float, float, float]
    mlp_info: dict
    history: pd.DataFrame

    def to_text(self) -> str:
        banner = "=" * 76
        lines = []
        if self.data_source != "db":
            lines += [banner, f"  {self.data_source.upper()} DATA: results describe the pipeline, not real users",
                      banner]
        info = self.mlp_info
        lines += [
            f"Device: {self.device}",
            f"MLP: hidden layers {info['hidden']}, {info['parameters']:,} parameters, "
            f"stopped at epoch {info['epochs']} (best {info['best_epoch']}), {info['seconds']:.1f}s",
            "",
            f"{'model':<24}{'CV PR-AUC':>18}{'test PR-AUC':>14}",
        ]
        for name, s in self.cv.items():
            lines.append(f"{name:<24}{s['mean']:>10.4f} +/- {s['std']:.4f}{self.test[name]:>12.4f}")
        d, lo, hi = self.mlp_vs_best
        verdict = ("better" if lo > 0 else "worse" if hi < 0 else "not distinguishable from")
        lines += [
            "",
            f"MLP - {self.best_classical} on test: {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> MLP is {verdict} "
            "the best classical model",
        ]
        return "\n".join(lines)


def compare(table: pd.DataFrame, data_source: str, n_splits: int = 3, random_state: int = 0,
            n_jobs: int = 1, **mlp_params) -> DeepLearningReport:
    task = get_task("classification")
    splits = model_splits(table)
    models = make_candidates(random_state, n_jobs, **mlp_params)
    cols = list(FEATURE_COLUMNS)

    cv_scores = {name: cv.summarise(cv.cross_validate(model, splits.train, task, "temporal", n_splits), "pr_auc")
                 for name, model in models.items()}
    best_classical = max((n for n in cv_scores if n != "mlp"), key=lambda n: cv_scores[n]["mean"])

    y_test = task.target(splits.test).to_numpy()
    fitted = {name: clone(m).fit(splits.train[cols], task.target(splits.train)) for name, m in models.items()}
    proba = {name: m.predict_proba(splits.test[cols])[:, 1] for name, m in fitted.items()}
    test = {name: float(average_precision_score(y_test, p)) for name, p in proba.items()}
    diff = paired_bootstrap_difference(y_test, proba["mlp"], proba[best_classical], average_precision_score,
                                       seed=random_state, skip_single_class=True)

    mlp: TorchMLPClassifier = fitted["mlp"].named_steps["model"]
    return DeepLearningReport(
        data_source=data_source, device=str(mlp.device_),
        cv={n: {"mean": s["mean"], "std": s["std"]} for n, s in cv_scores.items()},
        test=test, best_classical=best_classical, mlp_vs_best=diff,
        mlp_info={"hidden": tuple(mlp.hidden), "parameters": mlp.n_parameters(), "epochs": mlp.n_epochs_,
                  "best_epoch": mlp.best_epoch_, "seconds": mlp.fit_seconds_},
        history=pd.DataFrame(mlp.history_),
    )


def training_curve(history: pd.DataFrame, best_epoch: int) -> Figure:
    """Train vs validation loss per epoch: the picture behind early stopping."""
    from analytics.charts import LIGHT, _figure, _style, _title

    fig, ax = _figure(LIGHT, (8, 4))
    ax.plot(history["epoch"], history["train_loss"], color=LIGHT.accent, linewidth=2, label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], color="#eb6834", linewidth=2, label="Validation loss")
    ax.axvline(best_epoch, color=LIGHT.ink_secondary, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.legend(frameon=False, fontsize=9, labelcolor=LIGHT.ink_secondary)
    _style(ax, LIGHT)
    _title(ax, LIGHT, "MLP training", f"Vertical line: best validation epoch ({best_epoch}), weights restored there")
    return fig


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ml.data import InsufficientDataError, load_table

    parser = argparse.ArgumentParser(description="Compare a PyTorch MLP with the classical models.")
    parser.add_argument("--source", choices=["synthetic", "db"], default="synthetic")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/deep_learning")
    args = parser.parse_args(argv)

    print(f"Requested device: {args.device} -> {resolve_device(args.device)}")
    try:
        report = compare(load_table(args.source, args.seed), args.source, random_state=args.seed, n_jobs=-1,
                         device=args.device)
    except InsufficientDataError as exc:
        print(f"Not training: {exc}")
        return 2
    print(report.to_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mlp_training_{args.source}.png"
    training_curve(report.history, report.mlp_info["best_epoch"]).savefig(path)
    print(f"\nsaved training curve: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
