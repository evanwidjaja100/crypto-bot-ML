"""Phase 2: minimal Bybit V5 connectivity (public market data only).

Trading (signed) endpoints live in src/execution/bybit_executor.py (Phase 10).
This module never touches API keys.
"""

from __future__ import annotations

import random
import time
from typing import Any

import pandas as pd
import requests
from pybit.exceptions import FailedRequestError
from pybit.unified_trading import HTTP

KLINE_COLUMNS = ["ts_ms", "open", "high", "low", "close", "volume", "turnover"]

# Transient failures worth retrying: API-level errors and network errors.
_RETRYABLE = (FailedRequestError, requests.exceptions.RequestException)


class BybitClient:
    """Throttled, retrying REST client for Bybit V5 public endpoints.

    Args:
        testnet: use api-testnet.bybit.com (public data requires no keys).
            Defaults to False: market data is always mainnet. Testnet klines
            exist for connectivity smoke-tests only and must never reach the
            mainnet CandleStore (the network is stamped per row).
        session: injectable pybit HTTP instance (tests use a fake).
        max_retries: retry count for transient failures (backoff, jittered).
        requests_per_second: self-imposed rate cap (Bybit allows far more for
            public klines; we stay conservative).
    """

    def __init__(
        self,
        testnet: bool = False,
        session: Any | None = None,
        *,
        max_retries: int = 5,
        requests_per_second: float = 12.0,
    ) -> None:
        self.testnet = testnet
        self._session = session if session is not None else HTTP(testnet=testnet, timeout=15)
        self._max_retries = max_retries
        self._min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._rng = random.Random(0)

    # ------------------------------------------------------------------ rate
    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._last_call + self._min_interval - now
        if wait > 0.0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    # ---------------------------------------------------------------- request
    def _request(self, method: str, **kwargs: Any) -> dict:
        """Call a session method, retrying transient failures with backoff.

        Every call here is a GET-style, idempotent data request, so blind retry
        is safe. (Order placement must NOT reuse this pattern — see Phase 10.)
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = getattr(self._session, method)(**kwargs)
                if resp.get("retCode") != 0:
                    raise FailedRequestError(
                        request=str(kwargs),
                        message=resp.get("retMsg", "unknown error"),
                        status_code=resp.get("retCode"),
                        time=None,
                        resp_headers=None,
                    )
                return resp
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                time.sleep((2.0**attempt) * (0.5 + self._rng.random()))
        raise RuntimeError(f"Bybit {method} failed after {self._max_retries} retries") from last_exc

    # ---------------------------------------------------------------- methods
    def server_time_ms(self) -> int:
        resp = self._request("get_server_time")
        return int(resp["result"]["timeSecond"]) * 1000

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 1000,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch one page of klines as a DataFrame, sorted ascending, deduped.

        Columns: ts_ms, open, high, low, close, volume, turnover.
        """
        resp = self._request(
            "get_kline",
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
            start=start_ms,
            end=end_ms,
        )
        rows = resp["result"]["list"]
        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        for col in ("open", "high", "low", "close", "volume", "turnover"):
            df[col] = df[col].astype(float)
        df["ts_ms"] = df["ts_ms"].astype("int64")
        return df.sort_values("ts_ms").drop_duplicates(subset="ts_ms").reset_index(drop=True)
