"""Model evaluation metrics."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
