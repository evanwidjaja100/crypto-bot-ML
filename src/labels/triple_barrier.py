"""Path-dependent Triple-Barrier Event Labeler (Marcos López de Prado).

Evaluates intra-horizon price action (High/Low wicks) against volatility-scaled
take-profit (upper), stop-loss (lower), and vertical (time-limit) barriers.

Eliminates the lookahead bias and intra-horizon stop-out blindspots of fixed-horizon labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_SHORT_WIN = 0
CLASS_FLAT = 1
CLASS_LONG_WIN = 2


def add_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding_bars: int = 12,
    vol_window: int = 50,
    min_barrier_pct: float = 0.001,
) -> pd.DataFrame:
    """Computes path-dependent triple-barrier labels.

    Class definitions:
        2 (CLASS_LONG_WIN): Price touched Long Take-Profit barrier before Stop-Loss.
        0 (CLASS_SHORT_WIN): Price touched Long Stop-Loss (or Short Take-Profit) before TP.
        1 (CLASS_FLAT): Vertical time barrier reached before either horizontal barrier.

    All barrier distances are calculated using strictly TRAILING realized volatility or ATR.
    """
    out = df.copy()
    n = len(out)
    if n < max_holding_bars + 2:
        out["tb_label"] = np.nan
        out["tb_ret"] = np.nan
        out["tb_bars"] = np.nan
        out["tb_barrier_pct"] = np.nan
        return out

    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)

    # Trailing realized 1-bar volatility (strictly past data)
    ret_1 = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1.0)
    ret_series = pd.Series(ret_1)
    vol = ret_series.rolling(vol_window, min_periods=2).std().to_numpy()
    barrier_pct = np.nan_to_num(
        np.clip(vol, a_min=min_barrier_pct, a_max=None), nan=min_barrier_pct
    )

    labels = np.full(n, np.nan, dtype=float)
    realized_rets = np.full(n, np.nan, dtype=float)
    bars_held = np.full(n, np.nan, dtype=float)

    # Evaluate forward paths up to n - max_holding_bars
    for t in range(n - max_holding_bars):
        c0 = close[t]
        b_pct = barrier_pct[t]
        tp_price = c0 * (1.0 + tp_mult * b_pct)
        sl_price = c0 * (1.0 - sl_mult * b_pct)

        event_label = CLASS_FLAT
        exit_ret = 0.0
        exit_bar = max_holding_bars

        for h in range(1, max_holding_bars + 1):
            curr_high = high[t + h]
            curr_low = low[t + h]

            tp_hit = curr_high >= tp_price
            sl_hit = curr_low <= sl_price

            if tp_hit and sl_hit:
                # Ambiguous double-wick: conservatively treat as stop-loss
                event_label = CLASS_SHORT_WIN
                exit_ret = (sl_price - c0) / c0
                exit_bar = h
                break
            elif tp_hit:
                event_label = CLASS_LONG_WIN
                exit_ret = (tp_price - c0) / c0
                exit_bar = h
                break
            elif sl_hit:
                event_label = CLASS_SHORT_WIN
                exit_ret = (sl_price - c0) / c0
                exit_bar = h
                break

        if event_label == CLASS_FLAT:
            # Vertical barrier reached
            c_end = close[t + max_holding_bars]
            exit_ret = (c_end - c0) / c0
            exit_bar = max_holding_bars

        labels[t] = event_label
        realized_rets[t] = exit_ret
        bars_held[t] = exit_bar

    out["tb_label"] = labels
    out["tb_ret"] = realized_rets
    out["tb_bars"] = bars_held
    out["tb_barrier_pct"] = barrier_pct

    return out
