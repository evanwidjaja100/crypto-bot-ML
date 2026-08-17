"""Phase 9: hard limits — kill switch, daily loss tracker, API error streak."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class KillSwitch:
    """Permanent (until operator reset) halt on detected anomalies.

    When a tombstone_path is given, a trip is written to disk so a restart
    (process crash, supervisor, reboot) cannot forget it. The tombstone lives
    beside the state snapshot — never in /tmp, which vanishes on reboot.
    """

    def __init__(
        self,
        max_api_error_streak: int = 5,
        *,
        tombstone_path: str | Path | None = None,
    ) -> None:
        self._max_streak = max_api_error_streak
        self._streak = 0
        self._tripped = False
        self._reason: str | None = None
        self._tripped_at: str | None = None
        self._tombstone = Path(tombstone_path) if tombstone_path else None
        if self._tombstone is not None and self._tombstone.exists():
            data = json.loads(self._tombstone.read_text(encoding="utf-8"))
            self._tripped, self._reason = True, data.get("reason", "unknown")
            self._tripped_at = data.get("tripped_at")

    def on_api_error(self) -> None:
        """Call on every API failure; trips the switch after a streak."""
        self._streak += 1
        if self._streak >= self._max_streak:
            self.trip(f"api error streak reached ({self._streak})")

    def on_api_success(self) -> None:
        self._streak = 0

    def trip(self, reason: str) -> None:
        if self._tripped:
            return
        self._tripped = True
        self._reason = reason
        self._tripped_at = datetime.now(UTC).isoformat()
        self._write_tombstone()

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._streak = 0
        self._tripped = False
        self._reason = None
        self._tripped_at = None
        if self._tombstone is not None:
            self._tombstone.unlink(missing_ok=True)

    def describe(self) -> str | None:
        return self._reason

    def tripped_at(self) -> str | None:
        return self._tripped_at

    def _write_tombstone(self) -> None:
        if self._tombstone is None:
            return
        self._tombstone.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._tombstone.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"reason": self._reason, "tripped_at": self._tripped_at}, fh)
            os.replace(tmp, self._tombstone)
        except Exception:
            os.unlink(tmp)
            raise


class DailyLossTracker:
    """Tracks realized PnL per UTC day against a hard daily loss limit."""

    def __init__(self, max_daily_loss_pct: float, initial_equity: float) -> None:
        self._max_pct = max_daily_loss_pct
        self._initial_equity = initial_equity
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

    def snapshot(self) -> dict:
        return {"day": self._day, "pnl": self._pnl, "equity_base": self._equity_base}

    def restore(self, snap: dict) -> None:
        """Full round-trip: without the day, a restart treats the loss as new."""
        self._day = snap.get("day")
        self._pnl = float(snap.get("pnl", 0.0))
        self._equity_base = float(snap.get("equity_base", self._equity_base))

    def allowed(self, ts_ms: int) -> bool:
        if self._day is None or self._key(ts_ms) != self._day:
            return True  # new day resets on the next update; nothing recorded yet
        limit = self._equity_base * (self._max_pct / 100.0)
        return self._pnl >= -limit

    def reset(self) -> None:
        self._day = None
        self._pnl = 0.0
        self._equity_base = self._initial_equity
