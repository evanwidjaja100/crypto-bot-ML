"""Phase 10: paper-mode account — simulated fills with backtester-identical semantics.

Every fill rule mirrors src/backtesting/engine.py so paper results are directly
comparable to the backtest:
  - Market fills at the bar open with taker fee + slippage (long buys high,
    long sells low).
  - Stop-loss fills on a range touch (fill at the stop) or on a gap through
    the stop at the open (fill at the open — the worse fill).
  - Take-profit fills ONLY on a close beyond the target (wick touches do not
    fill), at the target price without slippage.
  - Funding is a constant per-8h rate charged on funding boundaries
    (00/08/16 UTC) while in position.
  - A stop-loss exit arms the cooldown (risk_cfg.cooldown_bars), which both
    the backtester and paper mode respect before allowing a new entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskSettings
from ..data_ingestion.intervals import FUNDING_INTERVAL_MS
from ..risk.sizing import compute_stops, size_position
from ..strategy.signal_engine import OPEN_LONG, OPEN_SHORT, PositionState


@dataclass
class PaperFill:
    ts_ms: int
    action: str  # OPEN_LONG / OPEN_SHORT / CLOSE_LONG / CLOSE_SHORT
    price: float
    qty: float
    reason: str  # "entry", "signal_flat", "reverse", "stop_loss", ...
    fee: float
    realized_pnl: float  # 0.0 for opening fills
    gross_pnl: float = 0.0  # price-only P&L, before fees and funding
    funding: float = 0.0  # funding accrued over the life of the trade (closing fills)
    gate_applied: bool = False  # the runner already applied this fill to the risk gate


class PaperBroker:
    def __init__(
        self,
        *,
        initial_equity: float,
        taker_fee: float,
        slippage_bps: float,
        funding_rate: float,
        risk_cfg: RiskSettings,
        maker_fee: float = 0.0002,
    ) -> None:
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage_bps / 10_000.0
        self.funding_rate = funding_rate
        self.risk_cfg = risk_cfg

        self._cash = initial_equity
        self._realized = 0.0
        self._fees_total = 0.0
        self._funding_total = 0.0
        self._funding_on_open = 0.0
        self._entry_fee = 0.0
        self.state = PositionState()

    # ------------------------------------------------------------- account
    @property
    def direction(self) -> int:
        return self.state.direction

    def unrealized(self, close_price: float) -> float:
        if self.state.direction == 0:
            return 0.0
        return self.state.direction * self.state.qty * (close_price - self.state.entry_price)

    def equity(self, close_price: float | None = None) -> float:
        return self._cash + (self.unrealized(close_price) if close_price is not None else 0.0)

    def propose_qty(self, direction: int, entry_price: float, atr_value: float | None) -> float:
        stop = None
        if atr_value:
            stop, _ = compute_stops(
                entry_price,
                atr_value,
                sl_atr_mult=self.risk_cfg.stop_loss_atr_mult,
                tp_atr_mult=self.risk_cfg.take_profit_atr_mult,
                direction=direction,
            )
        qty, _ = size_position(
            self.equity(),
            entry_price,
            stop,
            risk_per_trade_pct=self.risk_cfg.risk_per_trade_pct,
            leverage_cap=self.risk_cfg.leverage_cap,
            max_notional_pct=self.risk_cfg.max_notional_pct,
        )
        return qty

    # ------------------------------------------------------------ execution
    def open_position(
        self,
        ts_ms: int,
        open_price: float,
        direction: int,
        atr_value: float | None,
        *,
        qty: float | None = None,
    ) -> PaperFill | None:
        if self.state.direction != 0:
            raise ValueError("cannot open while a position is held")
        entry_price = open_price * (1.0 + direction * self.slippage)

        stop = target = None
        if atr_value:
            stop, target = compute_stops(
                entry_price,
                atr_value,
                sl_atr_mult=self.risk_cfg.stop_loss_atr_mult,
                tp_atr_mult=self.risk_cfg.take_profit_atr_mult,
                direction=direction,
            )
        if qty is None:
            qty, _ = size_position(
                self.equity(),
                entry_price,
                stop,
                risk_per_trade_pct=self.risk_cfg.risk_per_trade_pct,
                leverage_cap=self.risk_cfg.leverage_cap,
                max_notional_pct=self.risk_cfg.max_notional_pct,
            )
        if qty <= 0:
            return None

        fee = qty * entry_price * self.taker_fee
        self._cash -= fee
        self._fees_total += fee
        self._entry_fee = fee
        self._funding_on_open = 0.0
        self.state.direction = direction
        self.state.qty = qty
        self.state.entry_price = entry_price
        self.state.stop_price = stop
        self.state.target_price = target
        self.state.bars_in_position = 0
        self.state.entry_ts_ms = ts_ms
        return PaperFill(
            ts_ms=ts_ms,
            action=OPEN_LONG if direction == 1 else OPEN_SHORT,
            price=entry_price,
            qty=qty,
            reason="entry",
            fee=fee,
            realized_pnl=0.0,
        )

    def close_position(
        self, ts_ms: int, basis_price: float, reason: str, *, limit_fill: bool = False
    ) -> PaperFill:
        if self.state.direction == 0:
            raise ValueError("cannot close with no position held")
        side = self.state.direction
        exit_price = basis_price if limit_fill else basis_price * (1.0 - side * self.slippage)
        gross = side * self.state.qty * (exit_price - self.state.entry_price)
        fee_rate = self.maker_fee if limit_fill else self.taker_fee
        exit_fee = self.state.qty * exit_price * fee_rate
        fees = self._entry_fee + exit_fee
        net = gross - fees + self._funding_on_open
        self._cash += gross - exit_fee
        self._realized += net
        self._fees_total += exit_fee

        fill = PaperFill(
            ts_ms=ts_ms,
            action="CLOSE_LONG" if side == 1 else "CLOSE_SHORT",
            price=exit_price,
            qty=self.state.qty,
            reason=reason,
            fee=fees,
            realized_pnl=net,
            gross_pnl=gross,
            funding=self._funding_on_open,
        )
        self._funding_on_open = 0.0

        cooldown = reason in ("stop_loss", "stop_loss_gap")
        self.state.direction = 0
        self.state.qty = 0.0
        self.state.entry_price = 0.0
        self.state.stop_price = None
        self.state.target_price = None
        self.state.bars_in_position = 0
        self.state.cooldown_bars_left = self.risk_cfg.cooldown_bars if cooldown else 0
        return fill

    # ------------------------------------------------------------- per-bar
    def enter_bar(self, bar) -> tuple[list[PaperFill], float]:
        """Process one closed candle: exit checks (gap, stop, TP) then funding.

        Returns (fills, funding_pnl) — funding_pnl is nonzero on a funding
        boundary while a position is held.
        """
        fills: list[PaperFill] = []
        if self.state.cooldown_bars_left > 0:
            self.state.cooldown_bars_left -= 1
        side = self.state.direction
        if side != 0:
            self.state.bars_in_position += 1
            fills += self._check_exits(bar)

        ts = int(bar["ts_ms"])
        funding_pnl = 0.0
        if self.state.direction != 0 and ts % FUNDING_INTERVAL_MS == 0:
            funding_pnl = (
                -self.state.direction * self.state.qty * float(bar["close"]) * self.funding_rate
            )
            self._cash += funding_pnl
            self._funding_total += funding_pnl
            self._funding_on_open += funding_pnl
        return fills, funding_pnl

    def _check_exits(self, bar) -> list[PaperFill]:
        side = self.state.direction
        open_p, high, low, close_p = (
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
        )
        ts = int(bar["ts_ms"])
        stop = self.state.stop_price
        target = self.state.target_price

        if side == 1:  # long
            if stop is not None:
                if open_p <= stop:
                    return [self.close_position(ts, open_p, "stop_loss_gap")]
                if low <= stop:
                    return [self.close_position(ts, stop, "stop_loss")]
            if target is not None and close_p >= target:
                return [self.close_position(ts, target, "take_profit", limit_fill=True)]
        else:  # short
            if stop is not None:
                if open_p >= stop:
                    return [self.close_position(ts, open_p, "stop_loss_gap")]
                if high >= stop:
                    return [self.close_position(ts, stop, "stop_loss")]
            if target is not None and close_p <= target:
                return [self.close_position(ts, target, "take_profit", limit_fill=True)]
        return []

    # ------------------------------------------------------------- state
    def snapshot(self) -> dict:
        return {
            "cash": self._cash,
            "realized": self._realized,
            "fees_total": self._fees_total,
            "funding_total": self._funding_total,
            "funding_on_open": self._funding_on_open,
            "entry_fee": self._entry_fee,
            "state": {
                "direction": self.state.direction,
                "qty": self.state.qty,
                "entry_price": self.state.entry_price,
                "stop_price": self.state.stop_price,
                "target_price": self.state.target_price,
                "bars_in_position": self.state.bars_in_position,
                "cooldown_bars_left": self.state.cooldown_bars_left,
                "entry_ts_ms": self.state.entry_ts_ms,
            },
        }

    def restore(self, snap: dict) -> None:
        self._cash = float(snap["cash"])
        self._realized = float(snap.get("realized", 0.0))
        self._fees_total = float(snap.get("fees_total", 0.0))
        self._funding_total = float(snap.get("funding_total", 0.0))
        self._funding_on_open = float(snap.get("funding_on_open", 0.0))
        self._entry_fee = float(snap.get("entry_fee", 0.0))
        st = snap["state"]
        self.state.direction = int(st["direction"])
        self.state.qty = float(st["qty"])
        self.state.entry_price = float(st["entry_price"])
        self.state.stop_price = float(st["stop_price"]) if st["stop_price"] else None
        self.state.target_price = float(st["target_price"]) if st["target_price"] else None
        self.state.bars_in_position = int(st["bars_in_position"])
        self.state.cooldown_bars_left = int(st["cooldown_bars_left"])
        self.state.entry_ts_ms = int(st["entry_ts_ms"])
