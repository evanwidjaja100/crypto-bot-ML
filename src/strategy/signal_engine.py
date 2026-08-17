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
        if value == value:  # not NaN
            # If it's a normalized fraction like 0.02, convert back to price units if close is available
            val = float(value)
            if key.startswith("f_atr") and val < 1.0 and "close" in row:
                return float(val * float(row["close"]))
            return val
    if "close" in row:
        return float(row["close"]) * 0.02
    return None


def decide_triple_barrier(
    row,
    state: PositionState,
    strategy_cfg: StrategySettings,
    risk_cfg: RiskSettings,
    proba: np.ndarray,
) -> SignalDecision:
    """Decide action using triple-barrier model probabilities (0=short win, 1=flat, 2=long win)."""
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
                [f"strong opposite triple-barrier signal (p_short={p_short:.3f})"],
                p_long,
                p_short,
                atr_value,
            )
        if state.direction == -1 and p_long > strategy_cfg.confidence_reverse:
            return SignalDecision(
                OPEN_LONG,
                [f"strong opposite triple-barrier signal (p_long={p_long:.3f})"],
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
            [f"p_long_win={p_long:.3f} > {strategy_cfg.confidence_long}"],
            p_long,
            p_short,
            atr_value,
        )
    if p_short > strategy_cfg.confidence_short and p_short > p_long:
        return SignalDecision(
            OPEN_SHORT,
            [f"p_short_win={p_short:.3f} > {strategy_cfg.confidence_short}"],
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


def decide_cross_sectional(
    row,
    state: PositionState,
    risk_cfg: RiskSettings,
    *,
    rank_threshold: float = 0.90,
    exit_rank_threshold: float = 0.50,
) -> SignalDecision:
    """Decide action using cross-sectional relative strength and residual momentum."""
    rank = float(row.get("f_cs_rank_ret_24h", 0.5))
    res_mom = float(row.get("f_cs_residual_mom", 0.0))
    atr_value = _row_atr(row)

    if state.direction != 0:
        if state.bars_in_position >= risk_cfg.max_hold_bars:
            return SignalDecision(
                FLAT, [f"max_hold_bars reached ({risk_cfg.max_hold_bars})"], rank, 0.0, atr_value
            )
        if state.bars_in_position < risk_cfg.min_hold_bars:
            return SignalDecision(
                HOLD,
                [f"min_hold_bars not reached ({state.bars_in_position})"],
                rank,
                0.0,
                atr_value,
            )
        if rank < exit_rank_threshold:
            return SignalDecision(
                FLAT,
                [f"rank dropped below threshold ({rank:.2f} < {exit_rank_threshold:.2f})"],
                rank,
                0.0,
                atr_value,
            )
        return SignalDecision(
            HOLD, [f"maintaining leader position (rank={rank:.2f})"], rank, 0.0, atr_value
        )

    if state.cooldown_bars_left > 0:
        return SignalDecision(
            FLAT, [f"cooldown active ({state.cooldown_bars_left} bars left)"], rank, 0.0, atr_value
        )

    if rank >= rank_threshold and res_mom > 0:
        return SignalDecision(
            OPEN_LONG,
            [
                f"cross-sectional leader: rank={rank:.2f} >= {rank_threshold:.2f}, res_mom={res_mom:.4f}"
            ],
            rank,
            0.0,
            atr_value,
        )
    return SignalDecision(FLAT, [f"below leader threshold (rank={rank:.2f})"], rank, 0.0, atr_value)


def decide_funding_squeeze(
    row,
    state: PositionState,
    risk_cfg: RiskSettings,
    *,
    z_threshold: float = -2.0,
) -> SignalDecision:
    """Decide action using extreme negative funding rate anomaly (short squeeze setup)."""
    z = float(row.get("f_funding_zscore", 0.0))
    atr_value = _row_atr(row)

    if state.direction != 0:
        if state.bars_in_position >= risk_cfg.max_hold_bars:
            return SignalDecision(
                FLAT, [f"max_hold_bars reached ({risk_cfg.max_hold_bars})"], 0.0, 0.0, atr_value
            )
        if state.bars_in_position < risk_cfg.min_hold_bars:
            return SignalDecision(
                HOLD, [f"min_hold_bars not reached ({state.bars_in_position})"], 0.0, 0.0, atr_value
            )
        if z > 0.0:
            return SignalDecision(
                FLAT, [f"funding normalized (z={z:.2f} > 0.0)"], 0.0, 0.0, atr_value
            )
        return SignalDecision(HOLD, [f"holding squeeze trade (z={z:.2f})"], 0.0, 0.0, atr_value)

    if state.cooldown_bars_left > 0:
        return SignalDecision(
            FLAT, [f"cooldown active ({state.cooldown_bars_left} bars left)"], 0.0, 0.0, atr_value
        )

    if z <= z_threshold:
        return SignalDecision(
            OPEN_LONG,
            [f"negative funding squeeze anomaly: z={z:.2f} <= {z_threshold:.2f}"],
            0.0,
            0.0,
            atr_value,
        )
    return SignalDecision(FLAT, [f"normal funding (z={z:.2f})"], 0.0, 0.0, atr_value)
