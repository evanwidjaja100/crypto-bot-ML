"""Shared fixtures: deterministic synthetic OHLCV generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import _patch_lightgbm_windows

_patch_lightgbm_windows()


def make_candles(
    n: int = 400,
    seed: int = 7,
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 300_000,
    drift: float = 0.0,
    vol: float = 0.001,
) -> pd.DataFrame:
    """Deterministic random-walk candles with realistic open/gap dynamics.

    open[t] = close[t-1] * exp(gap), close[t] = open[t] * exp(body_ret),
    so body returns carry the drift (this is what the trading sims see).
    """
    rng = np.random.default_rng(seed)
    body = rng.normal(drift, vol, n) + 0.0002 * np.sin(np.arange(n) / 50.0)
    gap = rng.normal(0.0, 0.15 * vol, n)
    close = 30000.0 * np.exp(np.cumsum(body))
    open_ = np.empty(n)
    open_[0] = close[0] * np.exp(gap[0])
    open_[1:] = close[:-1] * np.exp(gap[1:])
    ext = np.abs(rng.normal(0.0, 0.5 * vol, n))
    high = np.maximum(open_, close) * (1.0 + ext)
    low = np.minimum(open_, close) * (1.0 - ext)
    volume = rng.uniform(10.0, 100.0, n) * (1.0 + np.sin(np.arange(n) / 17.0))
    ts = start_ms + np.arange(n) * interval_ms
    return pd.DataFrame(
        {
            "ts_ms": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover": volume * close,
        }
    )


@pytest.fixture
def candles() -> pd.DataFrame:
    return make_candles(400, seed=7)


@pytest.fixture
def feature_settings():
    from src.config import FeatureSettings

    return FeatureSettings()


@pytest.fixture
def label_settings():
    from src.config import LabelSettings

    return LabelSettings()
