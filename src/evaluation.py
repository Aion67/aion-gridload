"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def regression_metrics(y_true, y_pred) -> dict:
    """Standard regression scorecard: RMSE, MAE, R^2, MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Guard against divide-by-zero in MAPE.
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def evaluate_model(model, features, target) -> dict:
    """Fit-free evaluation: predict on the given features and score."""
    preds = model.predict(features)
    return regression_metrics(target, preds)


def cross_validate_rmse(model, X, y, n_splits: int = 5, random_state: int = 42):
    """K-fold CV returning per-fold RMSE (positive) plus mean and std.

    Uses shuffled KFold because the data is cross-sectional (no temporal
    ordering to preserve); shuffling gives representative folds across regions.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    neg_mse = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse = np.sqrt(-neg_mse)
    return {"fold_rmse": rmse.tolist(), "mean_rmse": float(rmse.mean()), "std_rmse": float(rmse.std())}


def scoreboard(results: dict) -> pd.DataFrame:
    """Turn a {model_name: metrics_dict} mapping into a sorted DataFrame."""
    board = pd.DataFrame(results).T
    if "RMSE" in board.columns:
        board = board.sort_values("RMSE")
    return board.round(3)
