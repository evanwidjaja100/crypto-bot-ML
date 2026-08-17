"""Backtester tests: fill timing, fees, stops, gaps, TP, funding, flat runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.config import RiskSettings, StrategySettings
from src.strategy.signal_engine import (
    FLAT,
    HOLD,
    OPEN_LONG,
    OPEN_SHORT,
    SignalDecision,
    decide,
)

IV = 300_000
START = 1_700_000_000_000

RISK = RiskSettings(
    risk_per_trade_pct=0.5,
    leverage_cap=3,
    max_notional_pct=100.0,
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=3.0,
    min_hold_bars=1,
    max_hold_bars=60,
    cooldown_bars=0,
)


def candles(opens, highs, lows, closes, start_ms=START):
    n = len(opens)
    return pd.DataFrame(
        {
            "ts_ms": start_ms + np.arange(n) * IV,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 10.0,
            "turnover": 1000.0,
        }
    )


def const_signal(action, atr=1.0):
    def fn(row, state):
        return SignalDecision(action, ["test"], atr_value=atr)

    return fn


def run(df, decision_fn, **kw):
    risk = kw.pop("risk_cfg", RISK)
    taker_fee = kw.pop("taker_fee", 0.001)
    slippage_bps = kw.pop("slippage_bps", 0.0)
    funding_rate = kw.pop("funding_rate", 0.0)
    return BacktestEngine(
        df,
        decision_fn,
        initial_equity=10_000.0,
        taker_fee=taker_fee,
        slippage_bps=slippage_bps,
        funding_rate=funding_rate,
        risk_cfg=risk,
        interval_ms=IV,
        **kw,
    ).run()


# ------------------------------------------------------------ timing & fees
def test_entry_at_next_open_no_lookahead():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100.5] * 6)
    result = run(df, const_signal(OPEN_LONG))
    trade = result["trades"].iloc[0]

    # decision at close of bar0 -> fill at open of bar1 (strictly later)
    assert result["decisions"].iloc[0]["ts_ms"] == START
    assert trade["entry_ts_ms"] == START + IV
    assert len(result["decisions"]) == len(df) - 1


def exit_after(max_bars):
    """Decision fn: enter long exactly once, FLAT once held `max_bars` bars."""
    entered = exited = False

    def fn(row, state):
        nonlocal entered, exited
        if state.direction == 0 and not entered:
            entered = True
            return SignalDecision(OPEN_LONG, ["enter"], atr_value=1.0)
        if entered and not exited and state.bars_in_position >= max_bars:
            exited = True
            return SignalDecision(FLAT, ["want out"], atr_value=1.0)
        if exited:
            return SignalDecision(FLAT, ["done"], atr_value=1.0)
        return SignalDecision(HOLD, ["hold"], atr_value=1.0)

    return fn


def test_fees_and_sizing_exact():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100, 101, 102, 101, 100, 101])
    result = run(df, exit_after(2))
    trade = result["trades"].iloc[0]

    # qty = 0.5% of 10k / (100 - 98) = 25; entry 100, FLAT exit at open[3] = 100
    assert trade["qty"] == pytest.approx(25.0)
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_reason"] == "signal_flat"
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["gross_pnl"] == pytest.approx(0.0)
    assert trade["fees"] == pytest.approx(25 * 100 * 0.001 * 2)  # entry + exit
    assert trade["net_pnl"] == pytest.approx(-5.0)
    assert result["metrics"]["total_fees"] == pytest.approx(5.0)
    assert result["equity"]["equity"].iloc[-1] == pytest.approx(9_995.0)
    assert trade["bars_held"] == 2


# -------------------------------------------------------------- stop losses
def test_stop_loss_fills_at_stop_price():
    # bar1 low 96 breaches stop 98 -> fill at 98, exactly the risked $50
    df = candles([100, 100, 100], [100.5, 100.5, 100.5], [99.5, 96.0, 95.0], [100, 99, 98])
    result = run(df, const_signal(OPEN_LONG))
    trade = result["trades"].iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(98.0)
    assert trade["gross_pnl"] == pytest.approx(-50.0)  # 25 * (98-100)
    assert trade["bars_held"] == 1


def test_stop_loss_gap_fills_at_worse_open():
    # bar2 opens at 95, gapping through the stop (98, anchored to entry 100)
    # -> fill at open 95, worse than the stop
    df = candles([100, 100, 95], [100.5, 101, 96], [99.5, 99, 94], [100, 99, 94.5])
    result = run(df, const_signal(OPEN_LONG))
    trade = result["trades"].iloc[0]
    assert trade["exit_reason"] == "stop_loss_gap"
    assert trade["exit_price"] == pytest.approx(95.0)
    assert trade["gross_pnl"] == pytest.approx(25 * (95 - 100))


# ----------------------------------------------------------- take profit
def test_take_profit_requires_close_beyond_target():
    # target 103: bar1 wick touches 104 but closes 102 -> no fill
    df = candles([100, 100, 100], [101, 104, 104.5], [99, 99, 99], [100, 102, 104])
    result = run(df, const_signal(OPEN_LONG))
    # after bar1: still in position with unrealized 25*(102-100)=50
    assert result["equity"].iloc[0]["unrealized"] == pytest.approx(50.0)
    trade = result["trades"].iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(103.0)
    assert trade["entry_ts_ms"] == START + IV
    assert trade["exit_ts_ms"] == START + 2 * IV


# --------------------------------------------------------------- hold rules
def test_position_persists_to_end_of_data():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100.5] * 6)
    result = run(df, const_signal(OPEN_LONG))
    trade = result["trades"].iloc[0]
    assert trade["bars_held"] == 5  # entered bar1, held through bar5
    assert trade["exit_reason"] == "end_of_backtest"


def test_min_hold_protects_position():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100, 101, 102, 101, 100, 101])
    risk = RISK.model_copy(update={"min_hold_bars": 5, "max_hold_bars": 60})

    # min-hold is enforced by the strategy decide(); strong long while flat,
    # weak signal while in position must NOT exit before min_hold_bars
    def fn(row, state):
        if state.direction == 0:
            proba = np.array([0.2, 0.2, 0.6])
        else:
            proba = np.array([0.3, 0.4, 0.3])
        return decide(row, state, StrategySettings(), risk, proba)

    result = run(df, fn, risk_cfg=risk)
    trade = result["trades"].iloc[0]
    assert trade["bars_held"] >= 5
    assert trade["exit_reason"] == "end_of_backtest"


# ------------------------------------------------------------------ reverse
def test_reverse_closes_and_reopens_same_bar():
    df = candles([100] * 5, [101] * 5, [99] * 5, [100] * 5)

    def fn(row, state):
        if row["ts_ms"] == START:
            return SignalDecision(OPEN_LONG, ["in"], atr_value=1.0)
        if row["ts_ms"] == START + IV:
            return SignalDecision(OPEN_SHORT, ["reverse"], atr_value=1.0)
        return SignalDecision(FLAT, ["out"], atr_value=1.0)

    result = run(df, fn)
    assert len(result["trades"]) == 2
    first, second = result["trades"].iloc[0], result["trades"].iloc[1]
    assert first["exit_reason"] == "reverse"
    assert first["direction"] == 1 and second["direction"] == -1
    assert first["exit_ts_ms"] == second["entry_ts_ms"] == START + 2 * IV


# ------------------------------------------------------------------ funding
def test_funding_applied_on_boundary():
    # bar close ts = 08:00 UTC = funding boundary (00/08/16)
    boundary = 1_704_096_000_000  # 2024-01-01 08:00:00 UTC
    df = candles(
        [100] * 6, [101] * 6, [99] * 6, [100, 101, 100, 100, 100, 100], start_ms=boundary - IV
    )
    result = BacktestEngine(
        df,
        const_signal(OPEN_LONG),
        initial_equity=10_000,
        taker_fee=0.0,
        slippage_bps=0,
        funding_rate=0.001,
        risk_cfg=RISK,
        interval_ms=IV,
    ).run()
    trade = result["trades"].iloc[0]
    # funding = -direction * qty * close * rate = -25 * 101 * 0.001
    assert trade["funding"] == pytest.approx(-25.0 * 101.0 * 0.001)
    assert result["metrics"]["total_funding"] == pytest.approx(-2.525)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - 2.525)


# --------------------------------------------------------------------- flat
def test_flat_strategy_no_trades_no_equity_change():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100.5] * 6)
    result = run(df, const_signal(FLAT))
    assert result["trades"].empty
    assert result["metrics"]["n_trades"] == 0
    assert result["equity"]["equity"].iloc[-1] == pytest.approx(10_000.0)


def test_slippage_applied_on_market_fills():
    df = candles([100] * 6, [101.5] * 6, [99] * 6, [100, 101, 102, 102, 102, 102])
    result = run(df, exit_after(2), slippage_bps=10.0)
    trade = result["trades"].iloc[0]
    # entry: long buys at open*(1+slip); stop anchored to slipped entry
    assert trade["entry_price"] == pytest.approx(100.0 * 1.001)
    # FLAT exit at open[3]: long sells at open*(1-slip)
    assert trade["exit_price"] == pytest.approx(100.0 * 0.999)
