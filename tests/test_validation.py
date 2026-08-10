"""Data validation tests: duplicates, gaps, ordering, price bounds."""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_candles
from src.data_ingestion.validation import validate_candles

IV = 300_000


def test_clean_frame_passes(candles):
    report = validate_candles(candles, IV)
    assert report.ok
    assert report.n_duplicates == 0
    assert report.n_gaps == 0


def test_duplicates_detected(candles):
    df = pd.concat([candles, candles.iloc[[100]]], ignore_index=True)
    report = validate_candles(df, IV)
    assert not report.ok
    assert report.n_duplicates == 1


def test_out_of_order_detected(candles):
    df = candles.copy()
    df.loc[150, "ts_ms"] = df["ts_ms"].iloc[0] - 1
    report = validate_candles(df, IV)
    assert not report.ok
    assert report.out_of_order >= 1


def test_gap_is_warning_by_default(candles):
    df = candles[candles["ts_ms"] != candles["ts_ms"].iloc[200]]
    report = validate_candles(df, IV)
    assert report.ok  # gap allowed -> only warning
    assert report.n_gaps == 1


def test_gap_is_error_when_disallowed(candles):
    df = candles[candles["ts_ms"] != candles["ts_ms"].iloc[200]]
    report = validate_candles(df, IV, allow_gaps=False)
    assert not report.ok
    assert report.first_gap_ms == int(df["ts_ms"].iloc[200])


def test_price_bounds_detected(candles):
    df = candles.copy()
    df.loc[50, "high"] = df["low"].iloc[50] - 1.0  # high < low
    report = validate_candles(df, IV)
    assert not report.ok


def test_close_outside_range_detected(candles):
    df = candles.copy()
    df.loc[50, "close"] = df["high"].iloc[50] * 2.0
    report = validate_candles(df, IV)
    assert not report.ok


def test_nan_detected(candles):
    df = candles.copy()
    df.loc[50, "close"] = float("nan")
    report = validate_candles(df, IV)
    assert not report.ok
    assert report.n_nan == 1


def test_missing_columns_reported(candles):
    report = validate_candles(candles.drop(columns=["volume"]), IV)
    assert not report.ok
    assert "volume" in report.errors[0]


def test_empty_frame_fails():
    report = validate_candles(pd.DataFrame(), IV)
    assert not report.ok
