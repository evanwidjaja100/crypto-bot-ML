"""Risk engine tests: sizing math, caps, stops, kill switch, daily loss, gate."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RiskSettings
from src.risk.gate import RiskGate
from src.risk.limits import DailyLossTracker, KillSwitch
from src.risk.sizing import compute_stops, size_position


# ---------------------------------------------------------------- sizing
def test_size_risk_budget():
    # 0.5% of 10k = $50 risk; entry 100, stop 99 -> $1/share -> 50 BTC
    qty, info = size_position(
        10_000, 100.0, 99.0, risk_per_trade_pct=0.5, leverage_cap=3, max_notional_pct=100.0
    )
    assert qty == pytest.approx(50.0)
    assert info["notional"] == pytest.approx(5000.0)
    assert info["effective_risk_pct"] == pytest.approx(0.5)


def test_size_leverage_cap():
    qty, info = size_position(
        10_000, 100.0, 99.0, risk_per_trade_pct=5.0, leverage_cap=3, max_notional_pct=1000.0
    )
    assert qty == pytest.approx(300.0)  # capped at 3x leverage
    assert info["effective_risk_pct"] == pytest.approx(3.0)


def test_size_notional_cap():
    qty, _ = size_position(
        10_000, 100.0, 99.0, risk_per_trade_pct=5.0, leverage_cap=10, max_notional_pct=20.0
    )
    assert qty == pytest.approx(20.0)  # 20% of equity


def test_size_zero_risk_raises():
    with pytest.raises(ValueError, match="zero risk budget"):
        size_position(
            10_000, 100.0, 100.0, risk_per_trade_pct=0.5, leverage_cap=3, max_notional_pct=20.0
        )


def test_size_without_stop_uses_caps():
    qty, info = size_position(
        10_000, 100.0, None, risk_per_trade_pct=0.5, leverage_cap=3, max_notional_pct=20.0
    )
    assert qty == pytest.approx(20.0)  # min(leverage 300, notional 20)
    assert np.isnan(info["price_risk"])


def test_compute_stops_long_and_short():
    sl, tp = compute_stops(100.0, 2.0, sl_atr_mult=2.0, tp_atr_mult=3.0, direction=1)
    assert sl == pytest.approx(96.0)
    assert tp == pytest.approx(106.0)
    sl, tp = compute_stops(100.0, 2.0, sl_atr_mult=2.0, tp_atr_mult=3.0, direction=-1)
    assert sl == pytest.approx(104.0)
    assert tp == pytest.approx(94.0)


def test_compute_stops_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_stops(100.0, 0.0, sl_atr_mult=2, tp_atr_mult=3, direction=1)
    with pytest.raises(ValueError):
        compute_stops(100.0, 2.0, sl_atr_mult=2, tp_atr_mult=3, direction=0)


# --------------------------------------------------------------- limits
def test_kill_switch_trips_after_streak():
    ks = KillSwitch(max_api_error_streak=5)
    for _ in range(4):
        ks.on_api_error()
        assert not ks.is_tripped()
    ks.on_api_error()
    assert ks.is_tripped()
    assert "streak" in ks.describe()


def test_kill_switch_reset_by_success_and_operator():
    ks = KillSwitch(max_api_error_streak=3)
    ks.on_api_error()
    ks.on_api_error()
    ks.on_api_success()  # resets streak
    ks.on_api_error()
    ks.on_api_error()
    assert not ks.is_tripped()
    ks.trip("manual")
    assert ks.is_tripped()
    ks.reset()
    assert not ks.is_tripped()


def test_daily_loss_limit():
    tracker = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    ts = 1_700_000_000_000
    assert tracker.allowed(ts)
    tracker.update(-150.0, ts, 9_850)
    assert tracker.allowed(ts)
    tracker.update(-100.0, ts, 9_750)  # total -250 < -200 limit
    assert not tracker.allowed(ts)


def test_daily_loss_resets_on_new_day():
    tracker = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    day1 = 1_700_000_000_000
    day2 = day1 + 86_400_000
    tracker.update(-500.0, day1, 9_500)
    assert not tracker.allowed(day1)
    assert tracker.allowed(day2)  # new day, nothing recorded yet
    tracker.update(-10.0, day2, 9_490)
    assert tracker.allowed(day2)


def test_daily_loss_snapshot_restore_roundtrip():
    """The review's scenario: lose 2%+, restart, and the limit must still hold."""
    tracker = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    ts = 1_700_000_000_000
    tracker.update(-250.0, ts, 9_750)  # over the 2% limit
    snap = tracker.snapshot()

    fresh = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    assert fresh.allowed(ts)  # a fresh tracker forgets the loss...
    fresh.restore(snap)
    assert not fresh.allowed(ts)  # ...but restore brings it back
    assert fresh.allowed(ts + 86_400_000)  # a new day must still reset


def test_daily_loss_restore_rejects_missing_fields():
    tracker = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    tracker.restore({"pnl": -500.0})  # no day: like a fresh tracker
    assert tracker.allowed(1_700_000_000_000)


def test_daily_loss_reset_restores_equity_base():
    tracker = DailyLossTracker(max_daily_loss_pct=2.0, initial_equity=10_000)
    ts = 1_700_000_000_000
    tracker.update(-100.0, ts, 9_000)  # equity base moves to 9k
    tracker.reset()
    assert tracker.snapshot() == {"day": None, "pnl": 0.0, "equity_base": 10_000.0}


def test_kill_switch_tombstone_survives_restart(tmp_path):
    path = tmp_path / "KILL_SWITCH.json"
    ks = KillSwitch(tombstone_path=path)
    ks.trip("reconciliation mismatch")
    assert path.exists()
    assert ks.tripped_at() is not None

    ks2 = KillSwitch(tombstone_path=path)  # a fresh process on the same path
    assert ks2.is_tripped()
    assert ks2.describe() == "reconciliation mismatch"
    assert ks2.tripped_at() == ks.tripped_at()

    ks2.reset()
    assert not path.exists()

    ks3 = KillSwitch(tombstone_path=path)
    assert not ks3.is_tripped()  # reset really is reset


def test_kill_switch_without_tombstone_behaves_as_before():
    ks = KillSwitch(max_api_error_streak=2)
    ks.on_api_error()
    assert not ks.is_tripped()
    ks.on_api_error()
    assert ks.is_tripped()
    assert "streak" in ks.describe()
    ks.reset()
    assert not ks.is_tripped()
    assert ks.tripped_at() is None


# ------------------------------------------------------------------ gate
def _gate():
    cfg = RiskSettings(
        max_open_positions=1,
        max_notional_pct=20.0,
        leverage_cap=3,
        max_daily_loss_pct=2.0,
        max_api_error_streak=5,
    )
    return RiskGate(cfg, initial_equity=10_000)


def test_gate_approves_clean_entry():
    gate = _gate()
    approval = gate.approve_entry(
        direction=1,
        qty=10,
        entry_price=100,
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert approval.approved
    assert approval.reasons == []


def test_gate_rejects_over_max_positions():
    gate = _gate()
    approval = gate.approve_entry(
        direction=1,
        qty=10,
        entry_price=100,
        equity=10_000,
        open_positions=1,
        ts_ms=1_700_000_000_000,
    )
    assert not approval.approved
    assert any("max open positions" in r for r in approval.reasons)


def test_gate_rejects_over_notional():
    gate = _gate()
    approval = gate.approve_entry(
        direction=1,
        qty=100,
        entry_price=100,  # notional 10k = 100% equity
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert not approval.approved
    assert any("notional" in r for r in approval.reasons)


def test_gate_rejects_over_leverage():
    gate = RiskGate(
        RiskSettings(
            max_open_positions=1,
            max_notional_pct=1000.0,
            leverage_cap=3,
            max_daily_loss_pct=2.0,
            max_api_error_streak=5,
        ),
        initial_equity=10_000,
    )
    approval = gate.approve_entry(
        direction=1,
        qty=10,
        entry_price=100,  # notional 1k, ok
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert approval.approved
    approval = gate.approve_entry(
        direction=1,
        qty=400,
        entry_price=100,  # notional 40k = 4x
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert not approval.approved
    assert any("leverage" in r for r in approval.reasons)


def test_gate_rejects_zero_qty():
    gate = _gate()
    approval = gate.approve_entry(
        direction=1,
        qty=0.0,
        entry_price=100,
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert not approval.approved


def test_gate_rejects_on_kill_switch():
    gate = _gate()
    gate.on_api_error()
    approval = gate.approve_entry(
        direction=1,
        qty=10,
        entry_price=100,
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert approval.approved  # 1 error is below the streak threshold
    for _ in range(4):
        gate.on_api_error()
    approval = gate.approve_entry(
        direction=1,
        qty=10,
        entry_price=100,
        equity=10_000,
        open_positions=0,
        ts_ms=1_700_000_000_000,
    )
    assert not approval.approved
    assert any("kill switch" in r for r in approval.reasons)


def test_gate_rejects_on_daily_loss():
    gate = _gate()
    ts = 1_700_000_000_000
    gate.on_position_closed(-250.0, ts, 9_750)  # over the 2% daily limit
    approval = gate.approve_entry(
        direction=1, qty=10, entry_price=100, equity=9_750, open_positions=0, ts_ms=ts
    )
    assert not approval.approved
    assert any("daily loss" in r for r in approval.reasons)
