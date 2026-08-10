"""Walk-forward tests: chronological folds, purge enforcement, aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward import walk_forward


def _data(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.integers(0, 3, size=n), dtype=int)
    ts = np.arange(1_700_000_000_000, 1_700_000_000_000 + n * 60_000, 60_000)
    return X, y, ts


def test_folds_are_chronological_and_purged():
    X, y, ts = _data()
    seen = {}

    def fit_and_predict(X_train, y_train, X_val, y_val):
        seen["train_len"] = seen.get("train_len", []) + [len(X_train)]
        seen["train_last_ts"] = seen.get("train_last_ts", []) + [None]  # set below
        seen["val_first_ts"] = seen.get("val_first_ts", []) + [None]
        return y_val.to_numpy(), np.full((len(y_val), 3), 1 / 3)

    result = walk_forward(X, y, n_splits=5, min_train_rows=150, purge=5,
                          fit_and_predict=fit_and_predict, ts=ts)
    folds = result["folds"]
    assert len(folds) == 4

    for i, fold in enumerate(folds):
        # train rows exclude the purge window: train_end = val_start - purge
        val_size = len(X) // 5
        val_start = (i + 1) * val_size
        assert seen["train_len"][i] == val_start - 5
        # validation strictly after training, with purge gap of exactly 5 rows
        assert fold["train_last_ts_ms"] < fold["val_first_ts_ms"]
        gap_ms = fold["val_first_ts_ms"] - fold["train_last_ts_ms"]
        assert gap_ms == 6 * 60_000  # 5 purged rows + 1 interval


def test_aggregate_metrics_present():
    X, y, ts = _data()

    def fit_and_predict(X_train, y_train, X_val, y_val):
        return y_val.to_numpy(), np.full((len(y_val), 3), 1 / 3)

    result = walk_forward(X, y, n_splits=5, min_train_rows=200, purge=3,
                          fit_and_predict=fit_and_predict)
    for key in ("mean_accuracy", "mean_f1_macro", "mean_log_loss"):
        assert key in result["aggregate"]
    assert len(result["folds"]) >= 2


def test_raises_on_too_little_data():
    X, y, _ = _data(n=100)

    def fit_and_predict(X_train, y_train, X_val, y_val):
        return y_val.to_numpy(), np.full((len(y_val), 3), 1 / 3)

    with pytest.raises(ValueError, match="too few rows"):
        walk_forward(X, y, n_splits=5, min_train_rows=500, purge=5,
                     fit_and_predict=fit_and_predict)


def test_degenerate_folds_are_reported_and_warned(caplog):
    """When early folds lack enough train rows they are skipped; the result
    must say how many ran vs how many were expected, and warn."""
    X, y, ts = _data(n=700)

    def fit_and_predict(X_train, y_train, X_val, y_val):
        return y_val.to_numpy(), np.full((len(y_val), 3), 1 / 3)

    with caplog.at_level("WARNING"):
        result = walk_forward(X, y, n_splits=8, min_train_rows=300, purge=0,
                              fit_and_predict=fit_and_predict, ts=ts)

    assert result["expected_folds"] == 7
    assert result["n_folds_executed"] == 4  # folds 4..7 only (train_rows >= 300)
    assert result["n_folds_executed"] == len(result["folds"])
    assert "ran only 4/7 folds" in caplog.text
