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
NETWORKS = ("mainnet", "testnet")


class CandleStore:
    """Parquet-backed OHLCV cache, one file per symbol_interval_network.

    Market data is mainnet by default (public, keyless). Testnet klines are a
    connectivity smoke-test surface only and must never reach the mainnet store:
    the network is part of the filename AND stamped per-row, so the two
    universes can never share a file, and a hand-moved file is detectable.
    """

    def __init__(self, data_dir: str | Path, *, network: str = "mainnet") -> None:
        if network not in NETWORKS:
            raise ValueError(f"network={network!r} not in {NETWORKS}")
        self.network = network
        self.data_dir = Path(data_dir)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)

    def raw_path(self, symbol: str, interval: str) -> Path:
        return self.data_dir / "raw" / f"{symbol}_{interval}_{self.network}.parquet"

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        path = self.raw_path(symbol, interval)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "network" not in df.columns:
            raise ValueError(
                f"{path.name} has no network column — it predates the F1 fix; "
                "re-download with scripts/download_data.py"
            )
        foreign = sorted(set(df["network"].dropna().unique()) - {self.network})
        if foreign:
            raise ValueError(
                f"network mismatch: {path.name} contains {foreign} rows but "
                f"this store serves network={self.network!r}"
            )
        return (
            df.sort_values("ts_ms")
            .drop_duplicates(subset="ts_ms")
            .reset_index(drop=True)
        )

    def write(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        *,
        validate: bool = True,
        max_bar_move_pct: float | None = 25.0,
    ) -> Path:
        df = df.copy()
        if "network" in df.columns:
            foreign = sorted(set(df["network"].dropna().unique()) - {self.network})
            if foreign:
                raise ValueError(
                    f"refusing to write {foreign} rows into the {self.network} store"
                )
        df["network"] = self.network
        df = (
            df.sort_values("ts_ms")
            .drop_duplicates(subset="ts_ms", keep="last")
            .reset_index(drop=True)
        )
        if validate:
            report = validate_candles(df, INTERVAL_MS[interval], max_bar_move_pct=max_bar_move_pct)
            if not report.ok:
                raise ValueError(f"refusing corrupt write: {report.summary()} -> {report.errors}")
        df[CANDLE_COLUMNS + ["network"]].to_parquet(self.raw_path(symbol, interval), index=False)
        return self.raw_path(symbol, interval)


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
    max_bar_move_pct: float | None = 25.0,
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

    report = validate_candles(df, INTERVAL_MS[interval], max_bar_move_pct=max_bar_move_pct)
    if not report.ok:
        raise ValueError(f"candle validation failed: {report.summary()} -> {report.errors}")
    store.write(df, symbol, interval, max_bar_move_pct=max_bar_move_pct)
    log.info(
        "store updated symbol=%s interval=%s rows=%d %s",
        symbol, interval, len(df), report.summary(),
    )
    return df, report
