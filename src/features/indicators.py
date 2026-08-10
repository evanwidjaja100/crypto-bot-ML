"""Phase 4: pure, causal indicator functions (no lookahead by construction).

Every function only combines values at or before index t. The leakage probe in
tests/test_indicators.py enforces this property mechanically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~np.isinf(out), 100.0)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def rolling_returns(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    return pd.concat({f"ret_{p}": close.pct_change(p) for p in periods}, axis=1)


def rolling_volatility(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    rets = close.pct_change()
    return pd.concat(
        {f"vol_{p}": rets.rolling(p, min_periods=p).std() for p in periods}, axis=1
    )


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std()
    return (volume - mean) / std


def ema_ratios(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    cols = {}
    for fast, slow in zip(periods, periods[1:]):
        cols[f"ema_{fast}_{slow}"] = ema(close, fast) / ema(close, slow)
    return pd.DataFrame(cols)


def sma_ratios(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    cols = {}
    for fast, slow in zip(periods, periods[1:]):
        cols[f"sma_{fast}_{slow}"] = sma(close, fast) / sma(close, slow)
    return pd.DataFrame(cols)


def candle_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Body, range and wick features relative to close (unit-free)."""
    body = (df["close"] - df["open"]) / df["open"]
    hl_range = (df["high"] - df["low"]) / df["close"]
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    return pd.DataFrame(
        {
            "body": body,
            "hl_range": hl_range,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
        }
    )


def time_encodings(dt) -> pd.DataFrame:
    """Cyclic time-of-day and day-of-week encodings (never leak future)."""
    idx = pd.DatetimeIndex(dt)
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        }
    )
