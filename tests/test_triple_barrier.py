"""Unit tests for Triple-Barrier event labeler."""

from __future__ import annotations

import pandas as pd

from src.labels.triple_barrier import (
    CLASS_FLAT,
    CLASS_LONG_WIN,
    CLASS_SHORT_WIN,
    add_triple_barrier_labels,
)


def test_triple_barrier_long_win():
    # Construct 10 bars where bar 5 moves sharply upward
    df = pd.DataFrame(
        {
            "ts_ms": [1000 * i for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0, 101.0, 101.0, 101.0, 106.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0] * 10,
            "close": [100.0] * 10,
        }
    )
    res = add_triple_barrier_labels(
        df, tp_mult=2.0, sl_mult=2.0, max_holding_bars=3, vol_window=5, min_barrier_pct=0.01
    )

    # Bar 3 has forward window bars 4, 5, 6. At bar 4 high=106.0 >= 100 * (1 + 2*0.01) = 102.0 -> Long Win
    assert res["tb_label"].iloc[3] == CLASS_LONG_WIN
    assert res["tb_bars"].iloc[3] == 1


def test_triple_barrier_short_win_or_stop_out():
    # Construct 10 bars where bar 2 drops sharply
    df = pd.DataFrame(
        {
            "ts_ms": [1000 * i for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0, 99.0, 94.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0] * 10,
        }
    )
    res = add_triple_barrier_labels(
        df, tp_mult=2.0, sl_mult=2.0, max_holding_bars=3, vol_window=5, min_barrier_pct=0.01
    )

    # Bar 0 has forward window bars 1, 2, 3. At bar 2 (h=2) low=94.0 <= 98.0 -> SL Hit / Short Win
    assert res["tb_label"].iloc[0] == CLASS_SHORT_WIN
    assert res["tb_bars"].iloc[0] == 2


def test_triple_barrier_vertical_expiration():
    # Tight range: no high or low breaches barriers
    df = pd.DataFrame(
        {
            "ts_ms": [1000 * i for i in range(10)],
            "open": [100.0] * 10,
            "high": [100.2] * 10,
            "low": [99.8] * 10,
            "close": [100.0] * 10,
        }
    )
    res = add_triple_barrier_labels(
        df, tp_mult=2.0, sl_mult=2.0, max_holding_bars=3, vol_window=5, min_barrier_pct=0.01
    )

    assert res["tb_label"].iloc[0] == CLASS_FLAT
    assert res["tb_bars"].iloc[0] == 3


def test_triple_barrier_tail_nans():
    df = pd.DataFrame(
        {
            "ts_ms": [1000 * i for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
        }
    )
    H = 4
    res = add_triple_barrier_labels(df, max_holding_bars=H, vol_window=5, min_barrier_pct=0.01)

    # The last H bars must be NaN
    assert res["tb_label"].iloc[-H:].isna().all()
    assert res["tb_label"].iloc[:-H].notna().all()
