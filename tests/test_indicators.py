"""Indicator unit tests: math sanity + mechanical lookahead probe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_candles

from src.features import indicators as ind


def test_rsi_bounds():
    close = pd.Series(np.random.default_rng(3).normal(0, 0.01, 500)).cumsum() + 100
    r = ind.rsi(close, 14)
    valid = r.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_max_on_streak():
    up = pd.Series(np.arange(50, dtype=float) + 100)
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_atr_positive_and_sized():
    df = make_candles(500, seed=2, vol=0.002)
    a = ind.atr(df, 14)
    valid = a.dropna()
    assert (valid > 0).all()
    assert valid.mean() < df["close"].mean()


def test_atr_scales_with_volatility():
    low_vol = make_candles(500, seed=5, vol=0.0005)
    high_vol = make_candles(500, seed=5, vol=0.005)
    a_low = ind.atr(low_vol, 14).dropna().mean()
    a_high = ind.atr(high_vol, 14).dropna().mean()
    assert a_high > 5 * a_low


def test_ema_ratio_above_one_on_uptrend():
    close = pd.Series(np.linspace(100, 200, 300))
    ratio = ind.ema_ratios(close, [10, 30, 90])
    assert (ratio.dropna() > 1.0).all().all()


def test_volume_zscore_finite():
    df = make_candles(400, seed=1)
    z = ind.volume_zscore(df["volume"], 20)
    assert z.dropna().abs().max() < 100  # sanity, no runaway values
    assert z.notna().sum() == len(df) - 19


def test_time_encodings_periodic():
    dt = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
    enc = ind.time_encodings(dt)
    assert set(enc.columns) == {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    assert enc["hour_sin"].abs().max() <= 1.0
    # 24h apart -> same sin/cos pair
    assert enc["hour_sin"].iloc[0] == pytest.approx(enc["hour_sin"].iloc[24])
    assert enc["hour_cos"].iloc[0] == pytest.approx(enc["hour_cos"].iloc[24])


def test_indicators_do_not_use_future_rows():
    """Perturb row i; every indicator output before i must be unchanged."""
    df = make_candles(400, seed=9)
    i = 200
    original = {
        "rsi": ind.rsi(df["close"], 14),
        "atr": ind.atr(df, 14),
        "ema_ratio": ind.ema_ratios(df["close"], [10, 30, 90]).iloc[:, 0],
        "vol_z": ind.volume_zscore(df["volume"], 20),
    }

    df2 = df.copy()
    df2.loc[i, "close"] *= 10.0
    df2.loc[i, "high"] *= 10.0
    df2.loc[i, "low"] *= 10.0
    df2.loc[i, "open"] *= 10.0
    df2.loc[i, "volume"] *= 10.0

    perturbed = {
        "rsi": ind.rsi(df2["close"], 14),
        "atr": ind.atr(df2, 14),
        "ema_ratio": ind.ema_ratios(df2["close"], [10, 30, 90]).iloc[:, 0],
        "vol_z": ind.volume_zscore(df2["volume"], 20),
    }
    for name in original:
        pd.testing.assert_series_equal(
            original[name].iloc[:i], perturbed[name].iloc[:i], check_names=False
        )
