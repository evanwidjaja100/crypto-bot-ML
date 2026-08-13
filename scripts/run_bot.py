"""Phase 10: run the bot loop in paper (default), testnet or live mode.

Usage:
    python scripts/run_bot.py [--once] [--sleep-secs N] [--warmup-bars N]

Paper mode simulates fills locally (no credentials needed). Testnet/live
require BYBIT_API_KEY / BYBIT_API_SECRET in .env and mirror every fill to the
exchange. Exit codes: 0 ok, 1 init/config error, 2 no model, 3 kill switch.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from src.config import load_settings
from src.data_ingestion.bybit_client import BybitClient
from src.data_ingestion.candle_downloader import CandleStore
from src.features.manifest import feature_set_id
from src.features.pipeline import build_feature_frame
from src.monitoring.logging_setup import setup_logging
from src.models.store import latest_model
from src.runner.runner import BotRunner

log = logging.getLogger("run_bot")


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    settings.check_credentials()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run a single tick and exit")
    parser.add_argument("--sleep-secs", type=float, default=None,
                        help="poll interval; defaults to the candle interval")
    parser.add_argument("--warmup-bars", type=int, default=2000)
    parser.add_argument("--journal-dir", default="data/runner")
    parser.add_argument("--state-path", default="data/runner/state.json")
    args = parser.parse_args(argv)

    setup_logging(settings.logging.log_level)

    loaded = latest_model("artifacts")
    if loaded is None:
        log.error("no trained model found; run scripts/train_model.py first")
        return 2
    model, meta = loaded
    log.info("model: %s (framework=%s)", meta["model_id"], meta["framework"])

    client = BybitClient(testnet=False)  # market data is public and always mainnet
    store = CandleStore(settings.data.data_dir, network=settings.market_data_network)

    # refuse to run a stale model against the current feature pipeline:
    # a mismatched feature_set_id silently defeats the staleness guard
    df = store.load(settings.symbol, settings.interval)
    if df is None or df.empty:
        log.error("no cached candles; run scripts/download_data.py then scripts/build_features.py")
        return 1
    _, feature_cols = build_feature_frame(df, settings.features)
    current_fid = feature_set_id(
        settings.features.version, feature_cols, settings.features.model_dump()
    )
    if meta["feature_set_id"] != current_fid:
        log.error(
            "model feature_set_id=%s does not match the current pipeline %s "
            "(version=%s) — rebuild + retrain: build_features.py -> train_model.py",
            meta["feature_set_id"], current_fid, settings.features.version,
        )
        return 1
    log.info("feature_set_id OK: %s", current_fid)

    executor = None
    if settings.mode in ("testnet", "live"):
        from pybit.unified_trading import HTTP

        session = HTTP(
            testnet=settings.order_endpoints_testnet,  # derived from mode, not env
            api_key=settings.env.bybit_api_key,
            api_secret=settings.env.bybit_api_secret,
            timeout=15,
        )
        from src.execution.bybit_executor import BybitExecutor

        executor = BybitExecutor(session, settings.symbol)
        log.warning("ORDERS -> %s (mode=%s)", settings.trading_network, settings.mode)
    else:
        log.info("paper mode: local fill simulation only")

    runner = BotRunner(
        settings=settings,
        client=client,
        store=store,
        model=model,
        meta=meta,
        executor=executor,
        journal_dir=args.journal_dir,
        state_path=args.state_path,
        warmup_bars=args.warmup_bars,
    )

    if runner.gate.kill_switch.is_tripped():
        log.error(
            "KILL SWITCH ACTIVE: %s (tripped %s). Investigate, then: "
            "python scripts/reset_kill_switch.py --reason '<what you fixed>'",
            runner.gate.kill_switch.describe(), runner.gate.kill_switch.tripped_at(),
        )
        return 3

    log.info("warmup: loading %d bars of history...", args.warmup_bars)
    runner.warmup()
    log.info("warmup done: last candle ts=%d pending=%s",
             runner.last_ts, runner.pending.action if runner.pending else None)

    try:
        runner.tick()
        if args.once:
            log.info("--once: single tick done")
            return 0
        while True:
            sleep_s = args.sleep_secs or (runner.interval_ms / 1000.0)
            time.sleep(sleep_s)
            try:
                runner.tick()
            except RuntimeError as exc:
                log.error("run aborted: %s", exc)
                return 3
    except KeyboardInterrupt:
        log.info("interrupted; state snapshot already saved per bar")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
