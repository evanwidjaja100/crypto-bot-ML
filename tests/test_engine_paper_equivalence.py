"""BacktestEngine <-> PaperBroker equivalence harness (Bugs 1, 3, 9).

Drives the same bar sequence and the same decision sequence through the
`BacktestEngine` (via a `decision_fn`) and the `PaperBroker` (manually:
execute the pending decision at bar open, then `enter_bar`), and asserts the
two produce identical observable results: per-bar equity, every per-trade
field, and the funding payments.

This is the acceptance test for Bug 1 (funding) and Bug 3 (cooldown), and
doubles as the maker-fee check for Bug 9.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.config import RiskSettings, StrategySettings
from src.execution.paper_broker import PaperBroker
from src.strategy.signal_engine import FLAT, HOLD, OPEN_LONG, OPEN_SHORT, SignalDecision, decide

IV = 300_000
FUNDING_BOUNDARY = 1_704_096_000_000  # 2024-01-01 08:00 UTC (a funding boundary)

RISK = RiskSettings(
    risk_per_trade_pct=0.5, leverage_cap=3, max_notional_pct=100.0,
    stop_loss_atr_mult=2.0, take_profit_atr_mult=3.0,
    min_hold_bars=1, max_hold_bars=60, cooldown_bars=5,
)
STRAT = StrategySettings()

TRADE_FIELDS = ("entry_ts_ms", "entry_price", "direction", "qty", "exit_ts_ms",
                "exit_price", "exit_reason", "gross_pnl", "fees", "funding",
                "net_pnl", "bars_held")
INT_FIELDS = ("entry_ts_ms", "exit_ts_ms", "direction", "bars_held")
FLOAT_FIELDS = ("entry_price", "qty", "exit_price", "gross_pnl", "fees", "funding", "net_pnl")


def candles(n, opens, highs, lows, closes, start_ms=FUNDING_BOUNDARY - IV, atr=1.0):
    return pd.DataFrame(
        {
            "ts_ms": [start_ms + i * IV for i in range(n)],
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": 10.0, "turnover": 1000.0,
            "atr_raw_14": atr,
        }
    )


def flat_series(n, price=100.0, high=100.5, low=99.5):
    return [float(price)] * n, [float(high)] * n, [float(low)] * n, [float(price)] * n


def run_equivalence(
    df,
    decision_fn,
    *,
    initial_equity=10_000.0,
    taker_fee=0.001,
    maker_fee=0.0002,
    slippage_bps=0.0,
    funding_rate=0.0001,
    risk_cfg=RISK,
):
    """Run the identical scenario through both engines.

    Returns (engine, result, paper_trades, broker): the BacktestEngine
    instance, its run() dict, the paper trade records, and the paper broker.
    Per-bar equity equality is asserted inside the loop.
    """
    engine = BacktestEngine(
        df, decision_fn, initial_equity=initial_equity, taker_fee=taker_fee,
        maker_fee=maker_fee, slippage_bps=slippage_bps,
        funding_rate=funding_rate, risk_cfg=risk_cfg, interval_ms=IV,
    )
    result = engine.run()

    broker = PaperBroker(
        initial_equity=initial_equity, taker_fee=taker_fee, maker_fee=maker_fee,
        slippage_bps=slippage_bps, funding_rate=funding_rate, risk_cfg=risk_cfg,
    )
    pending = decision_fn(df.iloc[0], broker.state)
    open_trade: dict | None = None
    bars_held = 0
    paper_trades: list[dict] = []

    def close_trade(fill):
        nonlocal open_trade
        trade = dict(open_trade)
        trade.update(
            exit_ts_ms=fill.ts_ms, exit_price=fill.price, exit_reason=fill.reason,
            gross_pnl=fill.gross_pnl, fees=fill.fee, funding=fill.funding,
            net_pnl=fill.realized_pnl, bars_held=bars_held,
        )
        paper_trades.append(trade)
        open_trade = None

    def open_trade_(ts, price, side, atr_value):
        nonlocal open_trade, bars_held
        fill = broker.open_position(ts, price, side, atr_value)
        if fill is not None:
            open_trade = {
                "entry_ts_ms": fill.ts_ms, "entry_price": fill.price,
                "direction": side, "qty": fill.qty,
            }
            bars_held = 0

    for i in range(1, len(df)):
        bar = df.iloc[i]
        ts, open_p = int(bar["ts_ms"]), float(bar["open"])

        if pending is not None and pending.action in (OPEN_LONG, OPEN_SHORT):
            side = 1 if pending.action == OPEN_LONG else -1
            if broker.direction == 0:
                open_trade_(ts, open_p, side, pending.atr_value)
            elif side != broker.direction:  # reverse: close then re-open on the same open
                close_trade(broker.close_position(ts, open_p, "reverse"))
                open_trade_(ts, open_p, side, pending.atr_value)
        elif pending is not None and pending.action == FLAT and broker.direction != 0:
            close_trade(broker.close_position(ts, open_p, "signal_flat"))
        pending = None

        if broker.direction != 0:
            bars_held += 1
        fills, _ = broker.enter_bar(bar)
        for fill in fills:
            close_trade(fill)
        pending = decision_fn(bar, broker.state)

        assert broker.equity(float(bar["close"])) == pytest.approx(
            result["equity"].iloc[i - 1]["equity"], rel=1e-9
        )

    if broker.direction != 0:
        last = df.iloc[-1]
        close_trade(broker.close_position(int(last["ts_ms"]), float(last["close"]), "end_of_backtest"))

    return engine, result, paper_trades, broker


def assert_trades_match(engine_result, paper_trades):
    et = engine_result["trades"].reset_index(drop=True)
    assert len(et) == len(paper_trades), (len(et), len(paper_trades))
    for row, paper in zip(et.itertuples(index=False), paper_trades):
        for field in TRADE_FIELDS:
            if field in INT_FIELDS:
                assert getattr(row, field) == paper[field], field
            elif field in FLOAT_FIELDS:
                assert getattr(row, field) == pytest.approx(paper[field], rel=1e-9, abs=1e-9), field
            else:
                assert getattr(row, field) == paper[field], field


# ------------------------------------------------------------- Bug 1: funding
def test_funding_two_boundaries_equivalent():
    """A long spanning two 8h boundaries: identical equity and per-trade pnl."""
    n = 110
    o, h, l, c = flat_series(n)
    df = candles(n, o, h, l, c)  # bar 1 = boundary FUNDING_BOUNDARY, bar 97 = second boundary

    def fn(row, state):
        ts = int(row["ts_ms"])
        if ts == FUNDING_BOUNDARY - IV:
            return SignalDecision(OPEN_LONG, ["enter"], atr_value=1.0)
        if ts == FUNDING_BOUNDARY + 96 * IV:
            return SignalDecision(FLAT, ["exit"], atr_value=1.0)
        return SignalDecision(HOLD, ["hold"], atr_value=1.0)

    engine, result, trades, broker = run_equivalence(df, fn, funding_rate=0.001)

    assert len(trades) == 1
    assert trades[0]["funding"] == pytest.approx(-25.0 * 100.0 * 0.001 * 2)
    assert_trades_match(result, trades)
    assert broker.equity() == pytest.approx(result["equity"]["equity"].iloc[-1], rel=1e-9)
    assert sum(t["funding"] for t in trades) == pytest.approx(broker._funding_total)


def test_funding_does_not_leak_into_next_trade():
    # trade 1 crosses one boundary; trade 2 crosses none -> its realied_pnl
    # must contain zero funding even though trade 1 paid funding.
    b = PaperBroker(
        initial_equity=10_000.0, taker_fee=0.001, maker_fee=0.0002,
        slippage_bps=0.0, funding_rate=0.001, risk_cfg=RISK,
    )
    b.open_position(FUNDING_BOUNDARY - IV, 100.0, 1, atr_value=1.0)
    fills, funding = b.enter_bar(pd.Series({"ts_ms": FUNDING_BOUNDARY, "open": 101, "high": 102, "low": 100, "close": 101}))
    assert fills == [] and funding == pytest.approx(-25.0 * 101.0 * 0.001)
    _, funding2 = b.enter_bar(pd.Series({"ts_ms": FUNDING_BOUNDARY + IV, "open": 101, "high": 102, "low": 100, "close": 101}))
    assert funding2 == 0.0
    fill1 = b.close_position(FUNDING_BOUNDARY + 2 * IV, 102.0, "signal_flat")

    fill2_open = b.open_position(FUNDING_BOUNDARY + 3 * IV, 102.0, 1, atr_value=1.0)
    assert b.enter_bar(pd.Series({"ts_ms": FUNDING_BOUNDARY + 4 * IV, "open": 102, "high": 103, "low": 101, "close": 102})) == ([], 0.0)
    fill2 = b.close_position(FUNDING_BOUNDARY + 5 * IV, 102.0, "signal_flat")

    assert fill1.realized_pnl == pytest.approx(fill1.gross_pnl - fill1.fee + fill1.funding)
    assert fill2.realized_pnl == pytest.approx(fill2.gross_pnl - fill2.fee)  # zero funding
    assert fill2.funding == 0.0


# ------------------------------------------------------- Bug 3: cooldown bars
def test_stop_loss_cooldown_equivalence():
    # Entry bar 1, stop bar 5 (low 96 -> fill at stop 98), cooldown_bars=5, then
    # a re-entry allowed by decide() exactly when the cooldown expires.
    n = 25
    o, h, l, c = flat_series(n)
    l[5] = 96.0  # stop bar: low breaches the 98 stop
    df = candles(n, o, h, l, c, start_ms=1_700_000_000_000)  # non-boundary start

    def fn(row, state):
        return decide(row, state, STRAT, RISK, np.array([0.2, 0.1, 0.7]))

    engine, result, trades, broker = run_equivalence(df, fn, funding_rate=0.0)

    assert [t["exit_reason"] for t in trades] == ["stop_loss", "end_of_backtest"]
    assert trades[0]["exit_ts_ms"] == trades[0]["entry_ts_ms"] + 4 * IV
    # cooldown expired during bar 10 -> re-entry at bar 11 open, one bar earlier
    assert trades[1]["entry_ts_ms"] == trades[0]["exit_ts_ms"] + 6 * IV
    assert broker.state.cooldown_bars_left == 0
    assert engine._state.cooldown_bars_left == 0
    assert_trades_match(result, trades)


# ------------------------------------------------------------- Bug 9: maker fee
def test_take_profit_charges_maker_fee_equivalent():
    # TP is a resting limit fill -> maker fee on the exit side, identical in both.
    n = 12
    o, h, l, c = flat_series(n)
    c[5] = 104.0  # close beyond target 103 -> TP
    df = candles(n, o, h, l, c, start_ms=1_700_000_000_000)

    def fn(row, state):
        return SignalDecision(
            OPEN_LONG if int(row["ts_ms"]) == 1_700_000_000_000 else HOLD,
            ["x"], atr_value=1.0,
        )

    engine, result, trades, _ = run_equivalence(df, fn)
    qty = 25.0
    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_price"] == pytest.approx(103.0)
    assert trades[0]["fees"] == pytest.approx(qty * 100.0 * 0.001 + qty * 103.0 * 0.0002)
    assert_trades_match(result, trades)