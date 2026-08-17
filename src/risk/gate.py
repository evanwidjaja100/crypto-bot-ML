"""Phase 9: risk gate — the single approval point before any order is sent."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RiskSettings
from .limits import DailyLossTracker, KillSwitch


@dataclass
class Approval:
    approved: bool
    reasons: list[str] = field(default_factory=list)


class RiskGate:
    """Enforces hard limits before order submission. A rejection is logged
    with its reasons; a rejection due to the kill switch is final until the
    operator resets it."""

    def __init__(
        self,
        cfg: RiskSettings,
        initial_equity: float,
        *,
        tombstone_path=None,
    ) -> None:
        self.cfg = cfg
        self.kill_switch = KillSwitch(cfg.max_api_error_streak, tombstone_path=tombstone_path)
        self.daily_loss = DailyLossTracker(cfg.max_daily_loss_pct, initial_equity)

    def approve_entry(
        self,
        *,
        direction: int,
        qty: float,
        entry_price: float,
        equity: float,
        open_positions: int,
        ts_ms: int,
    ) -> Approval:
        reasons: list[str] = []
        if qty <= 0:
            reasons.append("qty <= 0 after sizing")
        if self.kill_switch.is_tripped():
            reasons.append(f"kill switch tripped: {self.kill_switch.describe()}")
        if open_positions >= self.cfg.max_open_positions:
            reasons.append(f"max open positions reached ({self.cfg.max_open_positions})")
        notional = qty * entry_price
        if notional > equity * (self.cfg.max_notional_pct / 100.0):
            reasons.append(
                f"notional {notional:.2f} exceeds {self.cfg.max_notional_pct}% of equity"
            )
        if notional > equity * self.cfg.leverage_cap:
            reasons.append(f"notional {notional:.2f} exceeds leverage cap x{self.cfg.leverage_cap}")
        if not self.daily_loss.allowed(ts_ms):
            reasons.append(f"daily loss limit hit (pnl today {self.daily_loss.day_pnl():.2f})")
        return Approval(approved=not reasons, reasons=reasons)

    def on_position_closed(self, realized_pnl: float, ts_ms: int, equity: float) -> None:
        self.daily_loss.update(realized_pnl, ts_ms, equity)

    def on_api_error(self) -> None:
        self.kill_switch.on_api_error()

    def on_api_success(self) -> None:
        self.kill_switch.on_api_success()
