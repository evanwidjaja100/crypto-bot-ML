"""PaperBroker tests: fill semantics must match the backtester exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import RiskSettings
from src.execution.paper_broker import PaperBroker

RISK = RiskSettings(
    risk_per_trade_pct=0.5,
    leverage_cap=3,
    max_notional_pct=100.0,
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=3.0,
    min_hold_bars=1,
    max_hold_bars=60,
    cooldown_bars=5,
)

IV = 300_000
START = 1_700_000_000_000


def bar(ts, o, h, low, c):
    return pd.Series({"ts_ms": ts, "open": o, "high": h, "low": low, "close": c})


def broker(**kw):
    kw.setdefault("slippage_bps", 0.0)
    return PaperBroker(
        initial_equity=10_000.0,
        taker_fee=0.001,
        slippage_bps=kw.pop("slippage_bps"),
        funding_rate=0.0001,
        risk_cfg=RISK,
        **kw,
    )


def test_open_sizes_and_fees():
    b = broker()
    fill = b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    assert fill is not None
    assert fill.qty == pytest.approx(25.0)  # 0.5% of 10k / (2*atr)
    assert fill.price == pytest.approx(100.0)
    assert fill.fee == pytest.approx(25 * 100 * 0.001)
    assert b.equity() == pytest.approx(10_000 - 2.5)
    assert b.state.stop_price == pytest.approx(98.0)
    assert b.state.target_price == pytest.approx(103.0)


def test_open_without_stop_uses_caps():
    b = broker()
    fill = b.open_position(START + IV, 100.0, 1, atr_value=None)
    assert fill is not None
    assert fill.qty == pytest.approx(100.0)  # min(leverage 300, notional 100%) = 100
    assert b.state.stop_price is None
    assert b.state.target_price is None


def test_stop_touch_fills_at_stop():
    b = broker()
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    fills, _ = b.enter_bar(bar(START + 2 * IV, 100, 100.5, 96.0, 99))
    assert len(fills) == 1
    f = fills[0]
    assert f.reason == "stop_loss"
    assert f.price == pytest.approx(98.0)
    assert f.realized_pnl == pytest.approx(25 * (98 - 100) - 25 * 98 * 0.001 - 2.5)
    assert b.direction == 0
    assert b.state.cooldown_bars_left == 5  # stop exit arms cooldown


def test_gap_through_stop_fills_at_open():
    b = broker()
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    fills, _ = b.enter_bar(bar(START + 2 * IV, 95.0, 95.5, 94, 94.5))
    assert fills[0].reason == "stop_loss_gap"
    assert fills[0].price == pytest.approx(95.0)


def test_take_profit_requires_close_beyond_target():
    b = broker()
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    # wick touches 104 but closes 102 -> no fill
    fills, _ = b.enter_bar(bar(START + 2 * IV, 100, 104.0, 99, 102))
    assert fills == []
    assert b.direction == 1
    fills, _ = b.enter_bar(bar(START + 3 * IV, 102, 105, 101, 104))
    assert len(fills) == 1
    assert fills[0].reason == "take_profit"
    assert fills[0].price == pytest.approx(103.0)  # limit fill, no slippage
    assert b.state.cooldown_bars_left == 0  # TP does not arm cooldown


def test_funding_applied_on_boundary():
    b = broker()
    funding_ts = 1_728_000_000_000  # 00:00 UTC boundary (multiple of 8h)
    b.open_position(funding_ts, 100.0, 1, atr_value=1.0)
    before = b.equity()
    fills, funding = b.enter_bar(bar(funding_ts, 101, 102, 100, 101))  # boundary bar
    assert fills == []
    assert funding == pytest.approx(-25 * 101 * 0.0001)  # -direction*qty*close*rate
    assert b.equity() == pytest.approx(before + funding)
    # still in position: no TP (close 101 < 103), no stop (low 100 > 98)
    assert b.direction == 1
    # a non-boundary bar charges no funding
    _, funding2 = b.enter_bar(bar(funding_ts + IV, 101, 102, 100, 101))
    assert funding2 == 0.0


def test_flat_close_is_market_with_slippage():
    b = broker(slippage_bps=10.0)
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    fill = b.close_position(START + 2 * IV, 100.0, "signal_flat")
    assert fill.price == pytest.approx(100.0 * 0.999)  # long sells low
    # entry 100.1 (slipped), exit 99.9 -> gross -5, fees 2.5025 + 2.4975
    assert fill.realized_pnl == pytest.approx(
        25 * (99.9 - 100.1) - 25 * 100.1 * 0.001 - 25 * 99.9 * 0.001
    )
    assert b.direction == 0


def test_snapshot_restore_roundtrip():
    b = broker()
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    snap = b.snapshot()
    b2 = broker()
    b2.restore(snap)
    assert b2.state.direction == 1
    assert b2.state.qty == pytest.approx(25.0)
    assert b2.equity() == pytest.approx(b.equity())
    # the restored account can still manage exits
    fills, _ = b2.enter_bar(bar(START + 2 * IV, 100, 100.5, 96, 99))
    assert fills[0].reason == "stop_loss"


def test_enter_bar_increments_bars_in_position():
    b = broker()
    b.open_position(START + IV, 100.0, 1, atr_value=1.0)
    assert b.state.bars_in_position == 0
    b.enter_bar(bar(START + 2 * IV, 100, 100.5, 99, 100))
    assert b.state.bars_in_position == 1
    b.enter_bar(bar(START + 3 * IV, 100, 100.5, 99, 100))
    assert b.state.bars_in_position == 2
