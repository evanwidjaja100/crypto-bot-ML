"""Strategy engine tests: thresholds, hold rules, cooldown, reverse."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RiskSettings, StrategySettings
from src.strategy.signal_engine import (
    FLAT,
    HOLD,
    OPEN_LONG,
    OPEN_SHORT,
    PositionState,
    decide,
)

STRAT = StrategySettings(confidence_long=0.55, confidence_short=0.55, confidence_reverse=0.60)
RISK = RiskSettings(min_hold_bars=3, max_hold_bars=60, cooldown_bars=5)


def row(atr: float = 100.0) -> pd.Series:
    return pd.Series({"ts_ms": 0, "f_atr_14": atr})


def test_open_long_above_threshold():
    d = decide(row(), PositionState(), STRAT, RISK, np.array([0.2, 0.2, 0.6]))
    assert d.action == OPEN_LONG
    assert d.atr_value == 100.0
    assert any("p_long" in r for r in d.reasons)


def test_open_short_above_threshold():
    d = decide(row(), PositionState(), STRAT, RISK, np.array([0.6, 0.2, 0.2]))
    assert d.action == OPEN_SHORT


def test_flat_below_thresholds():
    d = decide(row(), PositionState(), STRAT, RISK, np.array([0.3, 0.4, 0.3]))
    assert d.action == FLAT


def test_flat_when_strongest_side_below_threshold():
    # p_long 0.6 above threshold but p_short even higher -> OPEN_SHORT wins
    d = decide(row(), PositionState(), STRAT, RISK, np.array([0.6, 0.0, 0.4]))
    assert d.action == OPEN_SHORT
    # both sides below their thresholds -> flat
    d = decide(row(), PositionState(), STRAT, RISK, np.array([0.4, 0.3, 0.3]))
    assert d.action == FLAT


def test_cooldown_blocks_entries():
    state = PositionState(cooldown_bars_left=2)
    d = decide(row(), state, STRAT, RISK, np.array([0.2, 0.2, 0.6]))
    assert d.action == FLAT
    assert any("cooldown" in r for r in d.reasons)


def test_min_hold_prevents_early_exit():
    state = PositionState(direction=1, qty=1, entry_price=100, bars_in_position=2)
    d = decide(row(), state, STRAT, RISK, np.array([0.7, 0.2, 0.1]))
    assert d.action == HOLD  # strong opposite signal but min_hold not reached


def test_max_hold_forces_exit():
    state = PositionState(direction=1, qty=1, entry_price=100, bars_in_position=60)
    d = decide(row(), state, STRAT, RISK, np.array([0.3, 0.4, 0.3]))
    assert d.action == FLAT
    assert any("max_hold_bars" in r for r in d.reasons)


def test_reverse_long_to_short():
    state = PositionState(direction=1, qty=1, entry_price=100, bars_in_position=10)
    d = decide(row(), state, STRAT, RISK, np.array([0.7, 0.2, 0.1]))
    assert d.action == OPEN_SHORT


def test_hold_when_signal_weak_in_position():
    state = PositionState(direction=1, qty=1, entry_price=100, bars_in_position=10)
    d = decide(row(), state, STRAT, RISK, np.array([0.3, 0.4, 0.3]))
    assert d.action == HOLD


def test_atr_absent_when_feature_missing():
    d = decide(pd.Series({"ts_ms": 0}), PositionState(), STRAT, RISK, np.array([0.2, 0.2, 0.6]))
    assert d.action == OPEN_LONG
    assert d.atr_value is None
