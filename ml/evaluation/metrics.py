"""Model evaluation metrics."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """MAE, MSE, RMSE and R².

    MAE  = mean |y - ŷ|        average miss, in minutes; robust to a few big misses
    MSE  = mean (y - ŷ)²       squares errors, so big misses dominate
    RMSE = √MSE                MSE back in minutes; RMSE >= MAE, a big gap means a few large misses
    R²   = 1 - SSE / SST       share of variance explained versus always predicting the mean
                               of y_true; 0 = no better than that mean, < 0 = worse
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "n": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def classification_metrics(y_true: npt.ArrayLike, proba: npt.ArrayLike, threshold: float = 0.5) -> dict[str, float]:
    """Metrics for a binary classifier, given P(y = 1) and a decision threshold.

    Threshold-dependent (computed on ŷ = proba >= threshold):
      accuracy   (TP + TN) / n         misleading under imbalance: "always 0" scores the majority share
      precision  TP / (TP + FP)        of the sessions we flag, how many really end early
      recall     TP / (TP + FN)        of the sessions that end early, how many we flag
      f1         harmonic mean of precision and recall
    Threshold-free (computed on proba):
      roc_auc    P(a random positive is scored above a random negative); 0.5 = chance
      pr_auc     average precision; chance level = the positive rate, so it reflects imbalance
      brier      mean (proba - y)²: calibration + sharpness, lower is better
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(y.size),
        "positive_rate": float(y.mean()) if y.size else float("nan"),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true: npt.ArrayLike, proba: npt.ArrayLike) -> float:
    """Threshold that maximises F1. Choose it on validation data, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true, dtype=int), proba)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=(precision + recall) > 0)
    # precision/recall have one more entry than thresholds (the final point has no threshold)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if thresholds.size else 0.5
