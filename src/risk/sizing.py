"""Phase 9: risk-based position sizing and ATR stop/target prices."""

from __future__ import annotations

import numpy as np


def compute_stops(
    entry_price: float,
    atr_value: float,
    *,
    sl_atr_mult: float,
    tp_atr_mult: float,
    direction: int,
) -> tuple[float, float]:
    """Stop-loss and take-profit anchored to the actual entry price.

    direction: +1 long (stop below entry), -1 short (stop above entry).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if atr_value <= 0:
        raise ValueError(f"atr_value must be positive, got {atr_value}")
    stop = entry_price - direction * sl_atr_mult * atr_value
    target = entry_price + direction * tp_atr_mult * atr_value
    if (direction == 1 and stop >= entry_price) or (direction == -1 and stop <= entry_price):
        raise ValueError("stop-loss not on the correct side of entry")
    return stop, target


def size_position(
    equity: float,
    entry_price: float,
    stop_price: float | None,
    *,
    risk_per_trade_pct: float,
    leverage_cap: int,
    max_notional_pct: float,
) -> tuple[float, dict]:
    """Risk-based position size in base units, capped by leverage and notional.

    If stop_price is None the risk budget is unavailable -> size by caps only.
    Returns (qty, info) where info explains the caps applied. qty = 0 is a
    valid result (e.g. risk budget of 0) and must be rejected by callers.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")

    cap_leverage = equity * leverage_cap / entry_price
    cap_notional = equity * (max_notional_pct / 100.0) / entry_price

    if stop_price is None:
        qty = min(cap_leverage, cap_notional)
        price_risk = float("nan")
        budget = float("nan")
    else:
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0:
            raise ValueError("stop_loss equals entry price -> zero risk budget; refusing to size")
        budget = equity * (risk_per_trade_pct / 100.0)
        qty = min(budget / price_risk, cap_leverage, cap_notional)

    notional = qty * entry_price
    info = {
        "notional": notional,
        "leverage": notional / equity if equity else 0.0,
        "effective_risk_pct": (qty * price_risk / equity * 100.0)
        if equity and not np.isnan(price_risk)
        else 0.0,
        "price_risk": price_risk,
        "caps_applied": {"leverage_cap": cap_leverage, "max_notional": cap_notional},
    }
    return qty, info
