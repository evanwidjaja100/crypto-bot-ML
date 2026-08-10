"""BybitClient tests against a fake session: parsing, sorting, retries, throttling."""
from __future__ import annotations

import time

import pytest
import requests

from src.data_ingestion.bybit_client import BybitClient


class FakeSession:
    """Mimics the pybit methods the client uses, with fault injection."""

    def __init__(self, kline_batches=None, fail_first_n: int = 0, fail_with=None):
        self.kline_batches = kline_batches or []
        self.fail_first_n = fail_first_n
        self.fail_with = fail_with or requests.exceptions.ConnectionError("boom")
        self.kline_calls = 0
        self.time_calls = 0
        self.last_kwargs = None

    def get_kline(self, **kwargs):
        self.kline_calls += 1
        self.last_kwargs = kwargs
        if self.kline_calls <= self.fail_first_n:
            raise self.fail_with
        batch = self.kline_batches[self.kline_calls - 1 - self.fail_first_n]
        return {"retCode": 0, "result": {"list": batch}}

    def get_server_time(self, **kwargs):
        self.time_calls += 1
        return {"retCode": 0, "result": {"timeSecond": "1700000000"}}


def _rows(ts_list, base=30000.0):
    return [
        [str(ts), str(base), str(base + 10), str(base - 5), str(base + 4), "10.5", "315000.0"]
        for ts in ts_list
    ]


def test_fetch_candles_parses_sorts_and_dedupes():
    batch = _rows([2, 1, 1])  # out of order + duplicate on purpose
    client = BybitClient(session=FakeSession([batch]), requests_per_second=1000)
    df = client.fetch_candles("BTCUSDT", "5", limit=1000)
    assert list(df["ts_ms"]) == [1, 2]
    assert (df["close"] == 30004.0).all()
    assert df["open"].dtype == float


def test_fetch_candles_passes_through_params():
    session = FakeSession([_rows([1, 2])])
    client = BybitClient(session=session, requests_per_second=1000)
    client.fetch_candles("BTCUSDT", "5", limit=500, start_ms=10, end_ms=20)
    assert session.last_kwargs["symbol"] == "BTCUSDT"
    assert session.last_kwargs["interval"] == "5"
    assert session.last_kwargs["limit"] == 500
    assert session.last_kwargs["start"] == 10
    assert session.last_kwargs["end"] == 20
    assert session.last_kwargs["category"] == "linear"


def test_retry_on_transient_failure_then_success():
    batch = _rows([1, 2, 3])
    session = FakeSession([batch], fail_first_n=2)
    client = BybitClient(session=session, requests_per_second=1000, max_retries=5)
    df = client.fetch_candles("BTCUSDT", "5")
    assert len(df) == 3
    assert session.kline_calls == 3


def test_raises_after_max_retries():
    session = FakeSession([], fail_first_n=99)
    client = BybitClient(session=session, requests_per_second=1000, max_retries=2)
    with pytest.raises(RuntimeError, match="after 2 retries"):
        client.fetch_candles("BTCUSDT", "5")
    assert session.kline_calls == 3


def test_server_time():
    client = BybitClient(session=FakeSession(), requests_per_second=1000)
    assert client.server_time_ms() == 1_700_000_000_000


def test_throttle_enforces_minimum_interval():
    client = BybitClient(session=FakeSession(), requests_per_second=5)
    client._last_call = 0.0
    t0 = time.monotonic()
    client._request("get_server_time")
    client._request("get_server_time")
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.19  # 2 calls @ 5 rps -> at least ~0.2s of spacing
