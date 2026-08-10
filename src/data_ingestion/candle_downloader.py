"""Phase 3: historical + incremental candle download to Parquet."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .bybit_client import BybitClient
from .intervals import INTERVAL_MS
from .validation import ValidationReport, validate_candles

log = logging.getLogger(__name__)

CANDLE_COLUMNS = ["ts_ms", "open", "high", "low", "close", "volume", "turnover"]


class CandleStore:
    """Parquet-backed OHLCV cache, one file per symbol_interval."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)

    def raw_path(self, symbol: str, interval: str) -> Path:
        return self.data_dir / "raw" / f"{symbol}_{interval}.parquet"

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        path = self.raw_path(symbol, interval)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return (
            df.sort_values("ts_ms")
            .drop_duplicates(subset="ts_ms")
            .reset_index(drop=True)
        )

    def write(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        path = self.raw_path(symbol, interval)
        df = (
            df.sort_values("ts_ms")
            .drop_duplicates(subset="ts_ms", keep="last")
            .reset_index(drop=True)
        )
        df[CANDLE_COLUMNS].to_parquet(path, index=False)
        return path


def download_range(
    client: BybitClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    chunk_days: int = 30,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Download [start_ms, end_ms) in time chunks with paginated pages.

    Returns a sorted, deduplicated frame covering the full range.
    """
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be after start_ms ({start_ms})")
    interval_ms = INTERVAL_MS[interval]
    chunk_ms = chunk_days * 86_400_000

    frames: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        frames.append(
            _download_chunk(client, symbol, interval, cursor, chunk_end, page_size)
        )
        cursor = chunk_end

    if not frames:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = (
        out.sort_values("ts_ms")
        .drop_duplicates(subset="ts_ms", keep="last")
        .reset_index(drop=True)
    )
    return out


def _download_chunk(
    client: BybitClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_size: int,
) -> pd.DataFrame:
    """Paginate [start_ms, end_ms) backward.

    Bybit's kline endpoint returns the MOST RECENT `limit` candles within the
    requested range, so forward pagination would skip the oldest data. We walk
    the cursor from the end of the chunk toward the start.
    """
    interval_ms = INTERVAL_MS[interval]
    frames: list[pd.DataFrame] = []
    end_cursor = end_ms - interval_ms  # inclusive latest candle timestamp
    while end_cursor >= start_ms:
        batch = client.fetch_candles(
            symbol, interval, limit=page_size, start_ms=start_ms, end_ms=end_cursor + interval_ms
        )
        if batch.empty:
            break
        first_ts = int(batch["ts_ms"].iloc[0])
        frames.append(batch)
        if len(batch) < page_size:
            break  # reached the start of the range
        new_cursor = first_ts - interval_ms
        if new_cursor >= end_cursor:
            break  # no progress -> stop to avoid an infinite loop
        end_cursor = new_cursor
    if not frames:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def incremental_update(
    client: BybitClient,
    symbol: str,
    interval: str,
    store: CandleStore,
    end_ms: int | None = None,
    *,
    history_days: int = 365,
    chunk_days: int = 30,
    page_size: int = 1000,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Fetch new candles and merge with the local cache.

    Raises ValueError if the merged frame fails validation (never writes a
    corrupted cache). Gaps are allowed (warnings).
    """
    end_ms = end_ms if end_ms is not None else client.server_time_ms()
    existing = store.load(symbol, interval)

    if existing is None or existing.empty:
        start_ms = end_ms - history_days * 86_400_000
        df = download_range(
            client, symbol, interval, start_ms, end_ms,
            chunk_days=chunk_days, page_size=page_size,
        )
    else:
        start_ms = int(existing["ts_ms"].iloc[-1]) + INTERVAL_MS[interval]
        if start_ms >= end_ms:
            # cache is already up to date
            df = existing
        else:
            fresh = download_range(
                client, symbol, interval, start_ms, end_ms,
                chunk_days=chunk_days, page_size=page_size,
            )
            df = pd.concat([existing, fresh], ignore_index=True)
            df = (
                df.sort_values("ts_ms")
                .drop_duplicates(subset="ts_ms", keep="last")
                .reset_index(drop=True)
            )

    report = validate_candles(df, INTERVAL_MS[interval])
    if not report.ok:
        raise ValueError(f"candle validation failed: {report.summary()} -> {report.errors}")
    store.write(df, symbol, interval)
    log.info(
        "store updated symbol=%s interval=%s rows=%d %s",
        symbol, interval, len(df), report.summary(),
    )
    return df, report
