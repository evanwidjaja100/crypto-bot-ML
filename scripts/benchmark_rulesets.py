"""Benchmark trend rulesets through the real backtester (no ML).

Runs each ruleset from src/strategy/rulesets through BacktestEngine with the
same fees/slippage/funding/stops/risk sizing as the ML backtests, on the full
cached history. Verdict: any rule with total_return > 0 and profit_factor >= 1.0
shows tradeable trend structure at this timeframe.

Usage:
    python scripts/benchmark_rulesets.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from src.backtesting.engine import BacktestEngine
from src.config import load_settings
from src.data_ingestion.candle_downloader import CandleStore
from src.data_ingestion.intervals import INTERVAL_MS
from src.monitoring.logging_setup import setup_logging
from src.strategy import rulesets


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    setup_logging(settings.logging.log_level)

    store = CandleStore(settings.data.data_dir)
    df = store.load(settings.symbol, settings.interval)
    if df is None:
        print("no candles cached; run scripts/download_data.py first")
        return 1

    frame = rulesets.add_trend_columns(df)
    frame = frame.dropna(subset=rulesets.TREND_COLUMNS).reset_index(drop=True)
    risk = settings.risk.model_copy(update={"stop_loss_atr_mult": 5.0, "take_profit_atr_mult": 100.0})
    print(f"candles={len(df)} usable={len(frame)} (trend-fair risk: 5xATR stop, no TP)")

    header = (f"{'ruleset':<12} {'gross':>9} {'ret':>9} {'sharpe':>7} {'max_dd':>8} "
              f"{'pf':>6} {'win':>6} {'n_trades':>9} {'fees':>10}")
    print(header)
    results: dict[str, dict] = {}
    for name, decider in rulesets.RULESETS.items():
        m = (
            BacktestEngine(
                frame, decider, initial_equity=settings.backtest.initial_equity,
                taker_fee=settings.execution.taker_fee,
                slippage_bps=settings.execution.slippage_bps,
                funding_rate=settings.backtest.funding_rate,
                risk_cfg=risk, interval_ms=INTERVAL_MS[settings.interval],
            )
            .run()["metrics"]
        )
        g = (
            BacktestEngine(
                frame, decider, initial_equity=settings.backtest.initial_equity,
                taker_fee=0.0, slippage_bps=0.0, funding_rate=0.0,
                risk_cfg=risk, interval_ms=INTERVAL_MS[settings.interval],
            )
            .run()["metrics"]
        )
        m["gross_pnl"] = g["total_gross_pnl"]
        results[name] = m
        print(
            f"{name:<12} {m['gross_pnl']:>9.1f} {m['total_return']:>9.4f} {m['sharpe']:>7.2f} "
            f"{m['max_drawdown']:>8.4f} {m['profit_factor']:>6.2f} "
            f"{m['win_rate']:>6.3f} {m['n_trades']:>9} {m['total_fees']:>10.1f}"
        )

    print("\nverdict (positive return AND pf >= 1.0, gross pnl should be > 0 too):")
    any_pass = False
    for name, m in results.items():
        ok = m["total_return"] > 0.0 and m["profit_factor"] >= 1.0 and m["gross_pnl"] > 0.0
        any_pass |= ok
        print(f"  {name:<12} {'TRADEABLE EDGE' if ok else 'no edge'}")
    print("reference: ML baseline (threshold 0.40) was -11.5% total, pf 0.42; its gross pnl was -303.")
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())