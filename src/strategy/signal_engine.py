"""Phase 8: strategy engine — model probabilities -> trade decisions.

The strategy only proposes direction and exit reasons. Position sizing, stop
prices and hard limits are owned by the risk engine (Phase 9). Stops/targets
are anchored to the actual fill price at execution time, so the strategy
passes along the ATR it saw and nothing else price-sensitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RiskSettings, StrategySettings

OPEN_LONG = "OPEN_LONG"
OPEN_SHORT = "OPEN_SHORT"
FLAT = "FLAT"
HOLD = "HOLD"


@dataclass
class PositionState:
    """Current runtime position. Mutated by the backtester / runner, read by the strategy."""

    direction: int = 0  # +1 long, -1 short, 0 flat
    qty: float = 0.0
    entry_price: float = 0.0
    stop_price: float | None = None
    target_price: float | None = None
    bars_in_position: int = 0
    cooldown_bars_left: int = 0
    entry_ts_ms: int = 0


@dataclass
class SignalDecision:
    action: str
    reasons: list[str] = field(default_factory=list)
    proba_long: float = 0.0
    proba_short: float = 0.0
    atr_value: float | None = None


def decide(
    row,
    state: PositionState,
    strategy_cfg: StrategySettings,
    risk_cfg: RiskSettings,
    proba: np.ndarray,
) -> SignalDecision:
    """Translate model probabilities (0=short, 1=flat, 2=long) into an action.

    Every return value is fully explained in `reasons` so the journal can
    reconstruct why a trade was or was not taken.
    """
    p_long = float(proba[2])
    p_short = float(proba[0])
    atr_value = _row_atr(row)

    if state.direction != 0:
        if state.bars_in_position >= risk_cfg.max_hold_bars:
            return SignalDecision(
                FLAT,
                [f"max_hold_bars reached ({risk_cfg.max_hold_bars})"],
                p_long,
                p_short,
                atr_value,
            )
        if state.bars_in_position < risk_cfg.min_hold_bars:
            return SignalDecision(
                HOLD,
                [f"min_hold_bars not reached ({state.bars_in_position}/{risk_cfg.min_hold_bars})"],
                p_long,
                p_short,
                atr_value,
            )
        if state.direction == 1 and p_short > strategy_cfg.confidence_reverse:
            return SignalDecision(
                OPEN_SHORT,
                [f"strong opposite signal (p_short={p_short:.3f})"],
                p_long,
                p_short,
                atr_value,
            )
        if state.direction == -1 and p_long > strategy_cfg.confidence_reverse:
            return SignalDecision(
                OPEN_LONG,
                [f"strong opposite signal (p_long={p_long:.3f})"],
                p_long,
                p_short,
                atr_value,
            )
        return SignalDecision(
            HOLD,
            [f"position maintained (p_long={p_long:.3f}, p_short={p_short:.3f})"],
            p_long,
            p_short,
            atr_value,
        )

    if state.cooldown_bars_left > 0:
        return SignalDecision(
            FLAT,
            [f"cooldown active ({state.cooldown_bars_left} bars left)"],
            p_long,
            p_short,
            atr_value,
        )

    if p_long > strategy_cfg.confidence_long and p_long > p_short:
        return SignalDecision(
            OPEN_LONG,
            [f"p_long={p_long:.3f} > {strategy_cfg.confidence_long}"],
            p_long,
            p_short,
            atr_value,
        )
    if p_short > strategy_cfg.confidence_short and p_short > p_long:
        return SignalDecision(
            OPEN_SHORT,
            [f"p_short={p_short:.3f} > {strategy_cfg.confidence_short}"],
            p_long,
            p_short,
            atr_value,
        )
    return SignalDecision(
        FLAT,
        [f"no signal above confidence (p_long={p_long:.3f}, p_short={p_short:.3f})"],
        p_long,
        p_short,
        atr_value,
    )


def _row_atr(row) -> float | None:
    """Raw price-unit ATR from the feature row, if present (stops stay in price units).

    The normalized `f_atr_` columns are a model feature, not the anchoring
    value; prefer the `atr_raw_*` column when available.
    """
    for key in ("atr_raw_14", "atr_raw", "f_atr_14", "f_atr"):
        try:
            value = row[key]
        except (KeyError, IndexError):
            continue
        return float(value) if value == value else None  # NaN -> None
    return None
