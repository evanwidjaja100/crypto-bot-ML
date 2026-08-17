"""Funding rate settlement history downloader and store.

Downloads 8h funding settlements from Bybit V5 and caches per symbol in Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .bybit_client import BybitClient

log = logging.getLogger(__name__)

FUNDING_COLUMNS = ["ts_ms", "symbol", "funding_rate"]


class FundingStore:
    """Parquet cache for 8h funding rate settlements."""

    def __init__(self, data_dir: str | Path, *, network: str = "mainnet") -> None:
        self.network = network
        self.data_dir = Path(data_dir)
        (self.data_dir / "funding").mkdir(parents=True, exist_ok=True)

    def funding_path(self, symbol: str) -> Path:
        return self.data_dir / "funding" / f"{symbol}_funding_{self.network}.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.funding_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return df.sort_values("ts_ms").drop_duplicates(subset="ts_ms").reset_index(drop=True)

    def write(self, df: pd.DataFrame, symbol: str) -> Path:
        out = df.copy()
        out["symbol"] = symbol
        out = out.sort_values("ts_ms").drop_duplicates(subset="ts_ms").reset_index(drop=True)
        out[FUNDING_COLUMNS].to_parquet(self.funding_path(symbol), index=False)
        return self.funding_path(symbol)


def download_funding_range(
    client: BybitClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    page_size: int = 200,
) -> pd.DataFrame:
    """Download funding settlements in [start_ms, end_ms) using pagination."""
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be greater than start_ms ({start_ms})")

    frames: list[pd.DataFrame] = []
    cursor_end = end_ms

    while cursor_end > start_ms:
        page = client.fetch_funding_history(
            symbol,
            limit=page_size,
            start_ms=start_ms,
            end_ms=cursor_end,
        )
        if page.empty:
            break
        frames.append(page)
        oldest = int(page["ts_ms"].min())
        if oldest >= cursor_end or len(page) < page_size:
            break
        cursor_end = oldest - 1

    if not frames:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values("ts_ms")
        .drop_duplicates(subset="ts_ms")
        .reset_index(drop=True)
    )
    return merged[(merged["ts_ms"] >= start_ms) & (merged["ts_ms"] < end_ms)].reset_index(drop=True)
