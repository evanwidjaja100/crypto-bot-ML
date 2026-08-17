"""Phase 4: feature pipeline — turns raw candles into a labeled-ready feature frame."""

from __future__ import annotations

import pandas as pd

from ..config import FeatureSettings
from . import indicators as ind

FEATURE_PREFIX = "f_"


def build_feature_frame(
    df: pd.DataFrame,
    cfg: FeatureSettings,
    *,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the causal feature matrix from raw candles.

    Returns (df, feature_cols). The returned frame keeps the original OHLCV
    columns plus features prefixed with "f_". Rows with any missing feature
    value are dropped (warm-up period) unless drop_na=False.
    """
    work = df.copy()
    if "dt" not in work.columns:
        work["dt"] = pd.to_datetime(work["ts_ms"], unit="ms", utc=True)

    close = work["close"]
    for p in cfg.return_periods:
        work[f"{FEATURE_PREFIX}ret_{p}"] = close.pct_change(p)
    for p in cfg.vol_periods:
        work[f"{FEATURE_PREFIX}vol_{p}"] = close.pct_change().rolling(p, min_periods=p).std()

    work[f"{FEATURE_PREFIX}vol_zscore_{cfg.vol_zscore_period}"] = ind.volume_zscore(
        work["volume"], cfg.vol_zscore_period
    )
    work[f"{FEATURE_PREFIX}rsi_{cfg.rsi_period}"] = ind.rsi(close, cfg.rsi_period)
    # raw ATR stays in price units for stop anchoring (NOT a model feature);
    # the model feature is normalized by price so its scale is price-level invariant
    work[f"atr_raw_{cfg.atr_period}"] = ind.atr(work, cfg.atr_period)
    work[f"{FEATURE_PREFIX}atr_{cfg.atr_period}"] = work[f"atr_raw_{cfg.atr_period}"] / close

    ema_df = ind.ema_ratios(close, cfg.ema_periods)
    for fast, slow in zip(cfg.ema_periods, cfg.ema_periods[1:]):
        work[f"{FEATURE_PREFIX}ema_{fast}_{slow}"] = ema_df[f"ema_{fast}_{slow}"]

    sma_df = ind.sma_ratios(close, cfg.sma_ratio_periods)
    for fast, slow in zip(cfg.sma_ratio_periods, cfg.sma_ratio_periods[1:]):
        work[f"{FEATURE_PREFIX}sma_{fast}_{slow}"] = sma_df[f"sma_{fast}_{slow}"]

    shape = ind.candle_shape(work)
    for col in shape.columns:
        work[f"{FEATURE_PREFIX}{col}"] = shape[col]

    tenc = ind.time_encodings(work["dt"])
    for col in tenc.columns:
        work[f"{FEATURE_PREFIX}{col}"] = tenc[col].to_numpy()

    feature_cols = sorted(c for c in work.columns if c.startswith(FEATURE_PREFIX))
    if drop_na:
        work = work.dropna(subset=feature_cols).reset_index(drop=True)
    return work, feature_cols
