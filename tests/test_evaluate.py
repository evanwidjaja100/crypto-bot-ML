"""Evaluation tests: classification metrics + the honest trading proxy."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_candles

from src.models.evaluate import classification_metrics, simulate_trading


def test_classification_metrics_shape():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_proba = np.full((9, 3), 1 / 3)
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert set(m) >= {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "n",
    }
    assert m["n"] == 9


def test_proxy_all_long_on_uptrend():
    df = make_candles(300, seed=8, drift=0.002)
    preds = np.full(len(df), 2)  # all long
    m = simulate_trading(df, preds, taker_fee=0.00055, slippage=0.0002)
    assert m["total_return"] > 0
    assert m["max_drawdown"] <= 0.0
    assert m["n_trades"] == 1  # one entry, one exit


def test_proxy_all_flat_is_zero():
    df = make_candles(300, seed=8, drift=0.002)
    preds = np.full(len(df), 1)
    m = simulate_trading(df, preds)
    assert m["total_return"] == pytest.approx(0.0)
    assert m["n_trades"] == 0


def test_proxy_misaligned_raises():
    df = make_candles(100, seed=8)
    with pytest.raises(ValueError, match="aligned"):
        simulate_trading(df, np.full(50, 1))


def test_proxy_fees_apply_on_flip():
    df = make_candles(300, seed=8, drift=0.0)
    preds = np.full(len(df), 2)
    preds[150:] = 0  # long then short
    m = simulate_trading(df, preds, taker_fee=0.00055, slippage=0.0002)
    # transitions: one entry into long, one flip long->short. A flip = 1 transition.
    assert m["n_trades"] == 2


def test_proxy_zero_losses_profit_factor_inf():
    df = make_candles(300, seed=8, drift=0.01, vol=0.001)  # every body > fee
    preds = np.full(len(df), 2)
    m = simulate_trading(df, preds, taker_fee=0.0001, slippage=0.0)
    assert m["profit_factor"] == np.inf


def test_proxy_short_allwin_curve_annualized_is_nan_not_overflow():
    import warnings

    df = make_candles(5, seed=8, drift=0.02, vol=0.0005)
    preds = np.full(len(df), 2)  # all long, all wins
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # RuntimeWarning -> error
        m = simulate_trading(df, preds, taker_fee=0.0001, slippage=0.0)
    assert np.isnan(m["annualized_return"])  # too short to annualize honestly
