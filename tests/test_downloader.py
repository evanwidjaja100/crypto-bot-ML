"""Downloader tests: pagination, chunking, incremental merge, store round-trip."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_ingestion.candle_downloader import (
    CandleStore,
    download_range,
    incremental_update,
)

IV = 300_000
START = 1_700_000_000_000


def _frame(ts_list):
    return pd.DataFrame(
        {
            "ts_ms": ts_list,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "turnover": 1005.0,
        }
    )


class FakeClient:
    """Mimics Bybit: returns the MOST RECENT `limit` candles in [start_ms, end_ms)."""

    def __init__(self, all_ts, page=3):
        self.all_ts = all_ts
        self.page = page
        self.calls = 0

    def server_time_ms(self):
        return int(self.all_ts[-1]) + IV

    def fetch_candles(self, symbol, interval, *, limit, start_ms, end_ms):
        self.calls += 1
        ts = [t for t in self.all_ts if start_ms <= t < end_ms]
        return _frame(ts[-limit:])


def test_download_range_paginates_and_dedupes(tmp_path):
    ts = list(range(START, START + 10 * IV, IV))
    client = FakeClient(ts, page=4)
    df = download_range(client, "BTCUSDT", "5", START, START + 10 * IV, page_size=4)
    assert len(df) == 10
    assert list(df["ts_ms"]) == sorted(df["ts_ms"])
    assert len(df["ts_ms"].drop_duplicates()) == 10


def test_download_range_chunks_over_long_spans(tmp_path):
    # 40 hourly candles = 40h, chunk_days=1 -> 2 chunks
    ts = [START + k * 3_600_000 for k in range(40)]
    client = FakeClient(ts, page=5)
    df = download_range(
        client,
        "BTCUSDT",
        "60",
        START,
        ts[-1] + 3_600_000,
        chunk_days=1,
        page_size=5,
    )
    assert len(df) == 40
    assert client.calls >= 8  # 2 chunks, 4 pages each


def test_incremental_update_merges_without_duplicates(tmp_path):
    store = CandleStore(tmp_path)
    client = FakeClient(list(range(START, START + 10 * IV, IV)))

    df1, r1 = incremental_update(client, "BTCUSDT", "5", store, end_ms=START + 6 * IV)
    assert len(df1) == 6 and r1.ok

    df2, r2 = incremental_update(client, "BTCUSDT", "5", store, end_ms=START + 10 * IV)
    assert len(df2) == 10
    assert len(df2["ts_ms"].drop_duplicates()) == 10  # no dup across merges
    assert r2.ok

    # third call with no new data -> unchanged
    df3, _ = incremental_update(client, "BTCUSDT", "5", store, end_ms=START + 10 * IV)
    assert len(df3) == 10


def test_store_roundtrip(tmp_path):
    store = CandleStore(tmp_path)
    df = _frame(list(range(START, START + 4 * IV, IV)))
    store.write(df, "BTCUSDT", "5")
    loaded = store.load("BTCUSDT", "5")
    df = df.copy()
    df["network"] = "mainnet"  # write() stamps provenance
    pd.testing.assert_frame_equal(df, loaded)
    assert store.raw_path("BTCUSDT", "5").exists()


def test_incremental_update_rejects_invalid_data(tmp_path):
    """Corrupt frames (duplicates) must raise, never write to cache."""
    store = CandleStore(tmp_path)

    class BadClient(FakeClient):
        def fetch_candles(self, *a, **k):
            out = super().fetch_candles(*a, **k)
            if out.empty:
                return out
            out = out.copy()
            out.loc[out.index[0], "close"] = float("nan")  # corrupt data that survives dedupe
            return out

    client = BadClient(list(range(START, START + 5 * IV, IV)))
    with pytest.raises(ValueError, match="validation failed"):
        incremental_update(client, "BTCUSDT", "5", store, end_ms=START + 5 * IV)
    assert store.load("BTCUSDT", "5") is None  # cache untouched


def test_store_is_network_keyed_in_filename_and_column(tmp_path):
    store = CandleStore(tmp_path)
    df = _frame(list(range(START, START + 4 * IV, IV)))
    path = store.write(df, "BTCUSDT", "5")
    assert path.name == "BTCUSDT_5_mainnet.parquet"
    loaded = store.load("BTCUSDT", "5")
    assert loaded["network"].eq("mainnet").all()
    assert not (tmp_path / "raw" / "BTCUSDT_5.parquet").exists()


def test_store_testnet_writes_separate_file(tmp_path):
    main = CandleStore(tmp_path)
    test = CandleStore(tmp_path, network="testnet")
    df = _frame(list(range(START, START + 4 * IV, IV)))
    main.write(df, "BTCUSDT", "5")
    test.write(df, "BTCUSDT", "5")
    main_df = main.load("BTCUSDT", "5")
    test_df = test.load("BTCUSDT", "5")
    assert main_df["network"].eq("mainnet").all()
    assert test_df["network"].eq("testnet").all()


def test_store_refuses_mixed_network_write(tmp_path):
    store = CandleStore(tmp_path)
    df = _frame(list(range(START, START + 4 * IV, IV)))
    df.loc[1, "network"] = "testnet"
    with pytest.raises(ValueError, match="refusing to write"):
        store.write(df, "BTCUSDT", "5")


def test_store_load_rejects_wrong_network_file(tmp_path):
    store = CandleStore(tmp_path)
    df = _frame(list(range(START, START + 4 * IV, IV)))
    df["network"] = "testnet"
    path = store.raw_path("BTCUSDT", "5")
    df[["ts_ms", "open", "high", "low", "close", "volume", "turnover", "network"]].to_parquet(
        path, index=False
    )
    with pytest.raises(ValueError, match="network"):
        store.load("BTCUSDT", "5")


def test_store_load_rejects_unkeyed_legacy_file(tmp_path):
    store = CandleStore(tmp_path)
    df = _frame(list(range(START, START + 4 * IV, IV)))  # no network column: pre-F1 file
    df.to_parquet(store.raw_path("BTCUSDT", "5"), index=False)
    with pytest.raises(ValueError, match="network"):
        store.load("BTCUSDT", "5")


def test_store_rejects_invalid_network(tmp_path):
    with pytest.raises(ValueError, match="network"):
        CandleStore(tmp_path, network="sidechain")
