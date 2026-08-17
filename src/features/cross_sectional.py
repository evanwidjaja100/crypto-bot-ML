"""Cross-Sectional Relative Momentum & Dispersion Features.

Computes cross-sectional ranking and residual momentum metrics across a liquid basket.
All features use strictly past-observable data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_basket(frames_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Aligns multiple symbol OHLCV dataframes onto a shared timestamp index.

    Returns:
        wide_df: MultiIndex or prefixed column DataFrame indexed by ts_ms.
        symbols: Sorted list of valid symbols.
    """
    symbols = sorted(frames_by_symbol.keys())
    if not symbols:
        return pd.DataFrame(), []

    dfs = []
    for sym in symbols:
        df = frames_by_symbol[sym].copy().sort_values("ts_ms").drop_duplicates(subset="ts_ms")
        df = df.set_index("ts_ms")
        cols = {
            c: f"{sym}_{c}"
            for c in ["open", "high", "low", "close", "volume", "turnover"]
            if c in df.columns
        }
        dfs.append(df[list(cols.keys())].rename(columns=cols))

    wide = pd.concat(dfs, axis=1, join="inner").sort_index().reset_index()
    return wide, symbols


def compute_cross_sectional_features(
    wide_df: pd.DataFrame,
    symbols: list[str],
    benchmark_symbol: str = "BTCUSDT",
    *,
    beta_window: int = 720,  # 30 days of 60m candles
) -> dict[str, pd.DataFrame]:
    """Generates cross-sectional relative strength and market-residual features for each symbol.

    Features generated per symbol:
        f_cs_rank_ret_24h: Percentile rank of 24h return relative to basket [0.0, 1.0].
        f_cs_rank_ret_7d: Percentile rank of 7-day return relative to basket [0.0, 1.0].
        f_cs_beta_btc: Rolling beta vs benchmark (BTCUSDT).
        f_cs_residual_mom: Idiosyncratic return (symbol return - beta * btc return).
        f_cs_vol_share_24h: Fraction of basket 24h volume.
    """
    out: dict[str, pd.DataFrame] = {
        sym: pd.DataFrame({"ts_ms": wide_df["ts_ms"]}) for sym in symbols
    }
    n_syms = len(symbols)
    if n_syms == 0 or wide_df.empty:
        return out

    # 1. 24h and 7d returns for all symbols
    rets_24h: dict[str, pd.Series] = {}
    rets_7d: dict[str, pd.Series] = {}
    rets_1h: dict[str, pd.Series] = {}
    vols_24h: dict[str, pd.Series] = {}

    for sym in symbols:
        c = wide_df[f"{sym}_close"]
        v = (
            wide_df[f"{sym}_turnover"]
            if f"{sym}_turnover" in wide_df
            else (c * wide_df[f"{sym}_volume"])
        )
        rets_1h[sym] = c.pct_change(1)
        rets_24h[sym] = c.pct_change(24)
        rets_7d[sym] = c.pct_change(168)
        vols_24h[sym] = v.rolling(24, min_periods=12).sum()

    rets_24h_df = pd.DataFrame(rets_24h)
    rets_7d_df = pd.DataFrame(rets_7d)
    vols_24h_df = pd.DataFrame(vols_24h)

    # 2. Cross-sectional ranking at each timestamp (percentile rank: 0.0 to 1.0)
    # rank(axis=1, pct=True) ranks across columns (symbols) per row (timestamp)
    rank_24h_df = rets_24h_df.rank(axis=1, pct=True)
    rank_7d_df = rets_7d_df.rank(axis=1, pct=True)

    # 3. Volume share
    total_vol_24h = vols_24h_df.sum(axis=1).replace(0, np.nan)
    vol_share_df = vols_24h_df.div(total_vol_24h, axis=0)

    # 4. Rolling Beta vs Benchmark
    btc_1h = rets_1h.get(benchmark_symbol, rets_1h[symbols[0]])
    btc_var = btc_1h.rolling(beta_window, min_periods=beta_window // 4).var()

    for sym in symbols:
        cov = rets_1h[sym].rolling(beta_window, min_periods=beta_window // 4).cov(btc_1h)
        beta = cov / btc_var.replace(0, np.nan)
        beta = beta.fillna(1.0).clip(-3.0, 5.0)

        # Residual momentum: 24h return minus beta * btc 24h return
        btc_24h = rets_24h.get(benchmark_symbol, rets_24h[symbols[0]])
        residual_mom = rets_24h[sym] - beta * btc_24h

        sym_feat = pd.DataFrame(
            {
                "ts_ms": wide_df["ts_ms"],
                "f_cs_rank_ret_24h": rank_24h_df[sym].fillna(0.5),
                "f_cs_rank_ret_7d": rank_7d_df[sym].fillna(0.5),
                "f_cs_beta_btc": beta,
                "f_cs_residual_mom": residual_mom.fillna(0.0),
                "f_cs_vol_share_24h": vol_share_df[sym].fillna(1.0 / n_syms),
            }
        )
        out[sym] = sym_feat

    return out
