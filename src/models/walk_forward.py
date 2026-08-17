"""Phase 6: expanding-window walk-forward validation with label-horizon purge.

Purge: the last `purge` rows of each train window are dropped because their
labels embed price data from beyond the train boundary (the label horizon),
which would otherwise leak into the validation fold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from .evaluate import classification_metrics

log = logging.getLogger(__name__)

FitAndPredict = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], tuple[np.ndarray, np.ndarray]
]


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    min_train_rows: int,
    purge: int,
    fit_and_predict: FitAndPredict,
    ts: np.ndarray | None = None,
) -> dict:
    """Expanding-window walk-forward evaluation.

    fit_and_predict(X_train, y_train, X_val, y_val) -> (pred, proba)

    Returns {"folds": [...], "aggregate": {...}}.
    """
    n = len(X)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n < min_train_rows + n // n_splits:
        raise ValueError(
            f"too few rows ({n}) for {n_splits} folds with min_train_rows={min_train_rows}"
        )

    val_size = n // n_splits
    folds: list[dict] = []
    for i in range(1, n_splits):
        val_start = i * val_size
        val_end = n if i == n_splits - 1 else (i + 1) * val_size
        train_end = val_start - purge
        if train_end < min_train_rows:
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]
        pred, proba = fit_and_predict(X_train, y_train, X_val, y_val)

        fold: dict[str, float] = {
            "fold": i,
            "train_rows": train_end,
            "val_rows": val_end - val_start,
        }
        if ts is not None:
            fold["train_last_ts_ms"] = int(ts[train_end - 1])
            fold["val_first_ts_ms"] = int(ts[val_start])
        fold.update(classification_metrics(y_val.to_numpy(), pred, proba))
        folds.append(fold)

    if not folds:
        raise RuntimeError("walk-forward produced no folds; increase data or lower min_train_rows")

    expected = n_splits - 1
    if len(folds) < expected:
        log.warning(
            "walk-forward ran only %d/%d folds — increase data or lower min_train_rows",
            len(folds),
            expected,
        )

    agg = {
        f"mean_{k}": float(np.mean([f[k] for f in folds]))
        for k in folds[0]
        if isinstance(folds[0][k], (int, float)) and not isinstance(folds[0][k], bool)
    }
    return {
        "folds": folds,
        "aggregate": agg,
        "n_folds_executed": len(folds),
        "expected_folds": expected,
    }
