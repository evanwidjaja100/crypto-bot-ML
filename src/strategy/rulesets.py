"""Trend-following rulesets for benchmarking against the ML pipeline.

These produce SignalDecision objects so the SAME event-driven backtester,
risk sizing, fees, slippage, stops and funding apply. The intent is an
apples-to-apples question: does 5m BTCUSDT OHLCV contain a tradeable trend
edge at all, independent of ML?

Indicator columns are added to the candle frame by add_trend_columns() and
ride along through BacktestEngine untouched. Every signal uses only past data
(lagged extremes / past closes), so there is no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signal_engine import FLAT, HOLD, OPEN_LONG, OPEN_SHORT, PositionState, SignalDecision

# Column names added to the frame.
EMA_FAST = "trend_ema_10"
EMA_SLOW = "trend_ema_30"
DON_HI = "trend_donchian_hi_20"
DON_LO = "trend_donchian_lo_20"
MOM = "trend_mom_close_20"
ATR = "trend_atr_14"

TREND_COLUMNS = [EMA_FAST, EMA_SLOW, DON_HI, DON_LO, MOM, ATR]


def add_trend_columns(
    df: pd.DataFrame, *, donchian: int = 20, momentum: int = 20, atr_period: int = 14
) -> pd.DataFrame:
    """Add lagged trend indicator columns. Assumes forward-unshifted candles."""
    from ..features import indicators as ind

    out = df.copy()
    close = out["close"]

    out[EMA_FAST] = close.ewm(span=10, adjust=False).mean()
    out[EMA_SLOW] = close.ewm(span=30, adjust=False).mean()

    # extremes as of the PREVIOUS close (no lookahead to the current bar)
    hi = out["high"].rolling(donchian, min_periods=donchian).max().shift(1)
    lo = out["low"].rolling(donchian, min_periods=donchian).min().shift(1)
    out[DON_HI] = hi
    out[DON_LO] = lo

    out[MOM] = close.shift(momentum)

    out[ATR] = ind.atr(out[["high", "low", "close"]], atr_period)
    return out


def _atr(row) -> float | None:
    value = row[ATR]
    return float(value) if value == value else None  # NaN -> None


# ------------------------------------------------------------------ rules
def decide_donchian(row: pd.Series, state: PositionState) -> SignalDecision:
    """Turtle-style channel breakout: enter on a lagged-20 extreme break,
    exit back through the opposite extreme. Stops handle the rest."""
    close = float(row["close"])
    hi = row[DON_HI]
    lo = row[DON_LO]
    if np.isnan(hi) or np.isnan(lo):
        return SignalDecision(FLAT, ["warmup"], atr_value=_atr(row))
    atr = _atr(row)

    if state.direction == 0:
        if close > hi:
            return SignalDecision(
                OPEN_LONG, [f"close {close:.2f} > channel high {hi:.2f}"], atr_value=atr
            )
        if close < lo:
            return SignalDecision(
                OPEN_SHORT, [f"close {close:.2f} < channel low {lo:.2f}"], atr_value=atr
            )
        return SignalDecision(FLAT, ["inside channel"], atr_value=atr)

    if state.direction == 1 and close < lo:
        return SignalDecision(
            OPEN_SHORT, [f"long exit: close {close:.2f} < channel low {lo:.2f}"], atr_value=atr
        )
    if state.direction == -1 and close > hi:
        return SignalDecision(
            OPEN_LONG, [f"short exit: close {close:.2f} > channel high {hi:.2f}"], atr_value=atr
        )
    return SignalDecision(HOLD, ["trend intact"], atr_value=atr)


def decide_ema(row: pd.Series, state: PositionState) -> SignalDecision:
    """EMA 10/30 crossover, always in market, reverses directly."""
    f, s = float(row[EMA_FAST]), float(row[EMA_SLOW])
    atr = _atr(row)
    if np.isnan(f) or np.isnan(s):
        return SignalDecision(FLAT, ["warmup"], atr_value=atr)
    if state.direction == 0:
        action, reason = (OPEN_LONG, "ema10 > ema30") if f > s else (OPEN_SHORT, "ema10 < ema30")
        return SignalDecision(action, [reason], atr_value=atr)
    if state.direction == 1 and f <= s:
        return SignalDecision(OPEN_SHORT, ["cross down"], atr_value=atr)
    if state.direction == -1 and f >= s:
        return SignalDecision(OPEN_LONG, ["cross up"], atr_value=atr)
    return SignalDecision(HOLD, ["aligned"], atr_value=atr)


def decide_momentum(row: pd.Series, state: PositionState) -> SignalDecision:
    """Time-series momentum: sign of close vs close 20 bars ago."""
    close = float(row["close"])
    ref = row[MOM]
    if np.isnan(ref):
        return SignalDecision(FLAT, ["warmup"], atr_value=_atr(row))
    atr = _atr(row)
    if state.direction == 0:
        action, reason = (OPEN_LONG, "momentum +") if close > ref else (OPEN_SHORT, "momentum -")
        return SignalDecision(action, [reason], atr_value=atr)
    if state.direction == 1 and close < ref:
        return SignalDecision(OPEN_SHORT, ["momentum flipped -"], atr_value=atr)
    if state.direction == -1 and close > ref:
        return SignalDecision(OPEN_LONG, ["momentum flipped +"], atr_value=atr)
    return SignalDecision(HOLD, ["momentum aligned"], atr_value=atr)


RULESETS = {
    "donchian20": decide_donchian,
    "ema10x30": decide_ema,
    "momentum20": decide_momentum,
}
