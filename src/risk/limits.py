"""Phase 9: hard limits — kill switch, daily loss tracker, API error streak."""
from __future__ import annotations

from datetime import UTC, datetime


class KillSwitch:
    """Permanent (until operator reset) halt on detected anomalies."""

    def __init__(self, max_api_error_streak: int = 5) -> None:
        self._max_streak = max_api_error_streak
        self._streak = 0
        self._tripped = False
        self._reason: str | None = None
        self._tripped_at: str | None = None

    def on_api_error(self) -> None:
        """Call on every API failure; trips the switch after a streak."""
        self._streak += 1
        if self._streak >= self._max_streak:
            self.trip(f"api error streak reached ({self._streak})")

    def on_api_success(self) -> None:
        self._streak = 0

    def trip(self, reason: str) -> None:
        if not self._tripped:
            self._tripped = True
            self._reason = reason
            self._tripped_at = datetime.now(UTC).isoformat()

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._streak = 0
        self._tripped = False
        self._reason = None
        self._tripped_at = None

    def describe(self) -> str | None:
        return self._reason


class DailyLossTracker:
    """Tracks realized PnL per UTC day against a hard daily loss limit."""

    def __init__(self, max_daily_loss_pct: float, initial_equity: float) -> None:
        self._max_pct = max_daily_loss_pct
        self._day: str | None = None
        self._pnl = 0.0
        self._equity_base = initial_equity

    @staticmethod
    def _key(ts_ms: int) -> str:
        return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")

    def update(self, realized_pnl: float, ts_ms: int, equity: float) -> None:
        day = self._key(ts_ms)
        if day != self._day:
            self._day = day
            self._pnl = 0.0
            self._equity_base = equity
        self._pnl += realized_pnl

    def day_pnl(self) -> float:
        return self._pnl

    def restore_day_pnl(self, pnl: float) -> None:
        """Restore the current day's realized PnL from a state snapshot."""
        self._pnl = float(pnl)

    def allowed(self, ts_ms: int) -> bool:
        if self._day is None or self._key(ts_ms) != self._day:
            return True  # new day resets on the next update; nothing recorded yet
        limit = self._equity_base * (self._max_pct / 100.0)
        return self._pnl >= -limit

    def reset(self) -> None:
        self._day = None
        self._pnl = 0.0
