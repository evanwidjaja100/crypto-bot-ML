"""Download historical OHLCV candles from Bybit (public API, no credentials).

Examples:
    python scripts/download_data.py --days 365
    python scripts/download_data.py --start 2024-01-01 --end 2024-06-01
    python scripts/download_data.py --update
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

from src.config import load_settings
from src.data_ingestion.bybit_client import BybitClient
from src.data_ingestion.candle_downloader import CandleStore, download_range, incremental_update
from src.monitoring.logging_setup import setup_logging

log = logging.getLogger("download_data")


def _parse_date(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=settings.symbol)
    parser.add_argument("--interval", default=settings.interval)
    parser.add_argument("--start", type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", type=_parse_date, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--days", type=int, default=settings.data.history_days)
    parser.add_argument("--update", action="store_true", help="incremental update of local cache")
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="connectivity smoke-test data; writes a separate testnet file, NEVER mainnet",
    )
    parser.add_argument(
        "--allow-jumps",
        action="store_true",
        help="bypass the bar-to-bar price-jump check (a genuine flash-crash bar)",
    )
    args = parser.parse_args(argv)

    setup_logging(settings.logging.log_level)
    network = "testnet" if args.testnet else "mainnet"
    client = BybitClient(testnet=args.testnet)
    store = CandleStore(settings.data.data_dir, network=network)
    if args.testnet:
        print(
            "WARNING: testnet klines are connectivity smoke-test data. "
            "They are written to a separate testnet file and are NOT usable for training."
        )
    max_move = None if args.allow_jumps else settings.data.max_bar_move_pct

    if args.update:
        df, report = incremental_update(
            client,
            args.symbol,
            args.interval,
            store,
            history_days=args.days,
            chunk_days=settings.data.chunk_days,
            page_size=settings.data.page_size,
            max_bar_move_pct=max_move,
        )
        print(
            f"updated {args.symbol} {args.interval} [{network}]: {len(df)} rows [{report.summary()}]"
        )
        return 0

    end_ms = args.end or client.server_time_ms()
    start_ms = args.start or (end_ms - args.days * 86_400_000)
    df = download_range(
        client,
        args.symbol,
        args.interval,
        start_ms,
        end_ms,
        chunk_days=settings.data.chunk_days,
        page_size=settings.data.page_size,
    )
    if df.empty:
        log.error("no candles returned for %s %s", args.symbol, args.interval)
        return 1
    store.write(df, args.symbol, args.interval, max_bar_move_pct=max_move)
    first = datetime.fromtimestamp(df["ts_ms"].iloc[0] / 1000, tz=timezone.utc)
    last = datetime.fromtimestamp(df["ts_ms"].iloc[-1] / 1000, tz=timezone.utc)
    print(
        f"downloaded {len(df)} candles {first} -> {last} -> {store.raw_path(args.symbol, args.interval)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
