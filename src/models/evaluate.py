"""Phase 6: model evaluation — classification metrics plus an honest trading proxy.

The trading proxy here is deliberately simple (position held from next candle
open to close, taker fee + slippage on every position change). It exists only
to sanity-check models before the full event-driven backtester (Phase 7)
exists. Do not treat it as the final backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

CLASS_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Accuracy, precision/recall/F1 (macro), log loss, per-class recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    valid = y_true >= 0
    metrics = {
        "accuracy": float(accuracy_score(y_true[valid], y_pred[valid])),
        "balanced_accuracy": float(balanced_accuracy_score(y_true[valid], y_pred[valid])),
        "precision_macro": float(precision_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)),
        "log_loss": float(
            log_loss(
                y_true[valid],
                np.clip(y_proba[valid], 1e-6, 1 - 1e-6),
                labels=[0, 1, 2],
            )
        ),
    }
    for cls, name in ((0, "short"), (1, "flat"), (2, "long")):
        mask = y_true[valid] == cls
        metrics[f"recall_{name}"] = (
            float((y_pred[valid][mask] == cls).mean()) if mask.sum() else float("nan")
        )
    metrics["n"] = int(valid.sum())
    return metrics


def _map_preds(preds: np.ndarray) -> np.ndarray:
    return np.array([CLASS_MAP[p] for p in preds], dtype=float)


def simulate_trading(
    df: pd.DataFrame,
    preds: np.ndarray,
    *,
    taker_fee: float = 0.00055,
    slippage: float = 0.0002,
    interval_ms: int = 300_000,
) -> dict:
    """Proxy equity curve. See module docstring for caveats.

    Decision at close of candle t -> position effective over candle t+1
    (open[t+1] -> close[t+1]).
    """
    preds = np.asarray(preds)
    if len(df) != len(preds):
        raise ValueError("df and preds must be aligned")

    r = (df["close"].to_numpy() / df["open"].to_numpy() - 1.0)
    pos = np.zeros(len(r))
    pos[1:] = _map_preds(preds[:-1])
    turnover = np.abs(np.diff(pos, prepend=0.0))
    ret = pos * r - turnover * (taker_fee + slippage)
    equity = np.cumprod(1.0 + ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    candles_per_year = (365 * 86_400_000) / interval_ms
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    n_trades = int((turnover > 0).sum())
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "total_return": float(equity[-1] - 1.0),
        "annualized_return": float(equity[-1] ** (candles_per_year / len(equity)) - 1.0)
        if equity[-1] > 0
        else -1.0,
        "sharpe": float(
            (ret.mean() / ret.std() * np.sqrt(candles_per_year)) if ret.std() > 0 else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": profit_factor,
        "win_rate": float((ret > 0).mean()) if len(ret) else 0.0,
        "n_trades": n_trades,
        "final_equity": float(equity[-1]),
    }
