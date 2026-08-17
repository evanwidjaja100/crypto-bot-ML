"""Unit tests for Cross-Sectional features."""

from __future__ import annotations

import pandas as pd

from src.features.cross_sectional import align_basket, compute_cross_sectional_features


def test_align_basket():
    df_btc = pd.DataFrame(
        {
            "ts_ms": [1000, 2000, 3000],
            "open": [10, 11, 12],
            "high": [10, 11, 12],
            "low": [10, 11, 12],
            "close": [10, 11, 12],
            "volume": [1, 1, 1],
            "turnover": [10, 11, 12],
        }
    )
    df_eth = pd.DataFrame(
        {
            "ts_ms": [2000, 3000, 4000],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [2, 2, 2],
            "turnover": [2, 4, 6],
        }
    )
    wide, symbols = align_basket({"BTCUSDT": df_btc, "ETHUSDT": df_eth})

    assert symbols == ["BTCUSDT", "ETHUSDT"]
    # Inner join on timestamps -> [2000, 3000]
    assert list(wide["ts_ms"]) == [2000, 3000]
    assert "BTCUSDT_close" in wide.columns
    assert "ETHUSDT_close" in wide.columns


def test_compute_cross_sectional_features():
    # Construct 30 bars for 3 assets where Asset A out-performs Asset B & C
    n = 40
    ts = [1000 * i for i in range(n)]

    p_a = [100.0 * (1.02**i) for i in range(n)]  # strong uptrend
    p_b = [100.0 * (1.00**i) for i in range(n)]  # flat
    p_c = [100.0 * (0.98**i) for i in range(n)]  # downtrend

    wide_df = pd.DataFrame(
        {
            "ts_ms": ts,
            "A_close": p_a,
            "A_volume": [10.0] * n,
            "A_turnover": [1000.0] * n,
            "B_close": p_b,
            "B_volume": [10.0] * n,
            "B_turnover": [1000.0] * n,
            "C_close": p_c,
            "C_volume": [10.0] * n,
            "C_turnover": [1000.0] * n,
        }
    )
    feats = compute_cross_sectional_features(
        wide_df, ["A", "B", "C"], benchmark_symbol="B", beta_window=10
    )

    assert set(feats.keys()) == {"A", "B", "C"}
    # At index 25, A should have highest 24h return rank (1.0), C lowest (approx 0.33)
    assert feats["A"]["f_cs_rank_ret_24h"].iloc[25] == 1.0
    assert feats["C"]["f_cs_rank_ret_24h"].iloc[25] < feats["B"]["f_cs_rank_ret_24h"].iloc[25]
    assert "f_cs_residual_mom" in feats["A"].columns
    assert "f_cs_vol_share_24h" in feats["A"].columns
