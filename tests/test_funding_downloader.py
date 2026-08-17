"""Unit tests for funding rate downloader and storage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_ingestion.bybit_client import BybitClient
from src.data_ingestion.funding_downloader import FundingStore, download_funding_range


class FakeFundingSession:
    def __init__(self, settlements: list[dict]):
        self.settlements = sorted(
            settlements, key=lambda x: int(x["fundingRateTimestamp"]), reverse=True
        )

    def get_funding_rate_history(self, **kwargs):
        symbol = kwargs.get("symbol")
        start = kwargs.get("startTime", 0)
        end = kwargs.get("endTime", int(1e15))
        limit = kwargs.get("limit", 200)

        filtered = [
            s
            for s in self.settlements
            if s["symbol"] == symbol
            and int(s["fundingRateTimestamp"]) >= start
            and int(s["fundingRateTimestamp"]) <= end
        ]
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": filtered[:limit]},
        }


def test_funding_store_roundtrip(tmp_path):
    store = FundingStore(tmp_path)
    assert store.load("BTCUSDT") is None

    df = pd.DataFrame(
        [
            {"ts_ms": 1700000000000, "symbol": "BTCUSDT", "funding_rate": 0.0001},
            {"ts_ms": 1700028800000, "symbol": "BTCUSDT", "funding_rate": 0.00015},
        ]
    )
    store.write(df, "BTCUSDT")

    loaded = store.load("BTCUSDT")
    assert loaded is not None
    assert len(loaded) == 2
    assert list(loaded["funding_rate"]) == [0.0001, 0.00015]


def test_download_funding_range_pagination():
    base_ts = 1_700_000_000_000
    eight_h_ms = 8 * 3600 * 1000
    mock_data = [
        {
            "symbol": "BTCUSDT",
            "fundingRate": str(0.0001 * (i % 3)),
            "fundingRateTimestamp": str(base_ts + i * eight_h_ms),
        }
        for i in range(10)
    ]
    session = FakeFundingSession(mock_data)
    client = BybitClient(session=session)

    start = base_ts
    end = base_ts + 10 * eight_h_ms

    res = download_funding_range(client, "BTCUSDT", start, end, page_size=3)
    assert len(res) == 10
    assert res["ts_ms"].iloc[0] == start
    assert res["ts_ms"].iloc[-1] == base_ts + 9 * eight_h_ms


def test_download_funding_range_invalid_range():
    client = BybitClient(session=FakeFundingSession([]))
    with pytest.raises(ValueError, match="must be greater than"):
        download_funding_range(client, "BTCUSDT", 200, 100)
