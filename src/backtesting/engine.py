"""Phase 7: event-driven backtester with conservative fill assumptions.

Execution model (matches what the live runner will do):
  - The strategy decision is made at the CLOSE of candle t and executed at the
    OPEN of candle t+1 (taker fee + slippage on market fills).
  - Stop-loss: assumed hit if the candle range touches the stop. Fill at the
    stop price (or the open if it gapped beyond the stop — the worse fill),
    plus slippage. Conservative: an intrabar touch below/above the stop always
    fills even if price later recovered.
  - Take-profit: filled ONLY if the candle close is beyond the target. An
    intrabar wick that touches the target without closing beyond it does NOT
    fill (we cannot know the path, so we do not assume the best case).
  - Funding: a constant per-8h funding rate, applied only on bars whose close
    timestamp is a funding boundary (00:00/08:00/16:00 UTC) while in position.
  - No lookahead: decisions are produced by decision_fn(row_at_t, state) and
    only rows up to t are visible to it; fills happen on row t+1.

The risk limits enforced here are the sizing caps (leverage / notional / risk
budget). The hard live limits (kill switch, daily loss) are enforced by
src/risk/gate.py in the runner, not simulated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..config import RiskSettings
from ..data_ingestion.intervals import FUNDING_INTERVAL_MS
from ..risk.sizing import compute_stops, size_position
from ..strategy.signal_engine import (
    FLAT,
    OPEN_LONG,
    OPEN_SHORT,
    PositionState,
    SignalDecision,
)

DecisionFn = Callable[[pd.Series, PositionState], SignalDecision]


@dataclass
class TradeRecord:
    entry_ts_ms: int
    entry_price: float
    direction: int
    qty: float
    exit_ts_ms: int
    exit_price: float
    exit_reason: str
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    bars_held: int


class BacktestEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        decision_fn: DecisionFn,
        *,
        initial_equity: float = 10_000.0,
        taker_fee: float = 0.00055,
        maker_fee: float = 0.0002,
        slippage_bps: float = 2.0,
        funding_rate: float = 0.0001,
        risk_cfg: RiskSettings,
        interval_ms: int = 300_000,
    ) -> None:
        if df is None or len(df) < 2:
            raise ValueError("backtest needs at least 2 candle rows")
        self.df = df.reset_index(drop=True)
        self.decision_fn = decision_fn
        self.initial_equity = initial_equity
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage_bps / 10_000.0
        self.funding_rate = funding_rate
        self.risk_cfg = risk_cfg
        self.interval_ms = interval_ms

        self._cash = initial_equity
        self._equity_last = initial_equity
        self._realized = 0.0
        self._fees_total = 0.0
        self._funding_total = 0.0
        self._state = PositionState()
        self._open_trade: TradeRecord | None = None
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict] = []
        self.decisions: list[dict] = []

    # ------------------------------------------------------------- execution
    def _market_exit_price(self, basis: float) -> float:
        """Market exit with slippage. Long sells (worse: lower), short buys."""
        return basis * (1.0 - self._state.direction * self.slippage)

    def _open_position(self, ts_ms: int, open_price: float, decision: SignalDecision) -> None:
        side = 1 if decision.action == OPEN_LONG else -1
        entry_price = open_price * (1.0 + side * self.slippage)

        stop = target = None
        if decision.atr_value:
            stop, target = compute_stops(
                entry_price,
                decision.atr_value,
                sl_atr_mult=self.risk_cfg.stop_loss_atr_mult,
                tp_atr_mult=self.risk_cfg.take_profit_atr_mult,
                direction=side,
            )
        qty, size_info = size_position(
            self._equity_last,
            entry_price,
            stop,
            risk_per_trade_pct=self.risk_cfg.risk_per_trade_pct,
            leverage_cap=self.risk_cfg.leverage_cap,
            max_notional_pct=self.risk_cfg.max_notional_pct,
        )
        if qty <= 0:
            self.decisions[-1]["size_rejected"] = True
            return

        fee = qty * entry_price * self.taker_fee
        self._cash -= fee
        self._fees_total += fee
        self._state.direction = side
        self._state.qty = qty
        self._state.entry_price = entry_price
        self._state.stop_price = stop
        self._state.target_price = target
        self._state.bars_in_position = 0
        self._state.entry_ts_ms = ts_ms
        self._open_trade = TradeRecord(
            entry_ts_ms=ts_ms,
            entry_price=entry_price,
            direction=side,
            qty=qty,
            exit_ts_ms=0,
            exit_price=0.0,
            exit_reason="",
            gross_pnl=0.0,
            fees=fee,
            funding=0.0,
            net_pnl=0.0,
            bars_held=0,
        )

    def _close_position(
        self,
        ts_ms: int,
        basis_price: float,
        reason: str,
        *,
        limit_fill: bool = False,
    ) -> None:
        open_trade = self._open_trade
        if open_trade is None:
            raise RuntimeError("_close_position called with no open trade")
        side = self._state.direction
        exit_price = basis_price if limit_fill else self._market_exit_price(basis_price)
        gross = side * self._state.qty * (exit_price - self._state.entry_price)
        fee = self._state.qty * exit_price * (self.maker_fee if limit_fill else self.taker_fee)
        self._cash += gross - fee
        self._fees_total += fee
        self._realized += gross - fee + open_trade.funding

        open_trade.exit_ts_ms = ts_ms
        open_trade.exit_price = exit_price
        open_trade.exit_reason = reason
        open_trade.gross_pnl = gross
        open_trade.fees += fee
        open_trade.net_pnl = gross - open_trade.fees + open_trade.funding
        open_trade.bars_held = self._state.bars_in_position
        self.trades.append(open_trade)
        self._open_trade = None

        if reason in ("stop_loss", "stop_loss_gap"):
            self._state.cooldown_bars_left = self.risk_cfg.cooldown_bars
        self._state.direction = 0
        self._state.qty = 0.0
        self._state.entry_price = 0.0
        self._state.stop_price = None
        self._state.target_price = None

    def _check_exits(self, bar: pd.Series) -> None:
        side = self._state.direction
        open_p, high, low, close_p = (
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
        )
        ts = int(bar["ts_ms"])
        stop = self._state.stop_price
        target = self._state.target_price

        if side == 1:  # long
            if stop is not None:
                if open_p <= stop:  # gap through the stop -> fill at open (worse)
                    self._close_position(ts, open_p, "stop_loss_gap")
                    return
                if low <= stop:
                    self._close_position(ts, stop, "stop_loss")
                    return
            if target is not None and close_p >= target:
                self._close_position(ts, target, "take_profit", limit_fill=True)
                return
        else:  # short
            if stop is not None:
                if open_p >= stop:
                    self._close_position(ts, open_p, "stop_loss_gap")
                    return
                if high >= stop:
                    self._close_position(ts, stop, "stop_loss")
                    return
            if target is not None and close_p <= target:
                self._close_position(ts, target, "take_profit", limit_fill=True)
                return

    def _execute_decision(self, bar: pd.Series, decision: SignalDecision) -> None:
        ts = int(bar["ts_ms"])
        open_p = float(bar["open"])

        if self._state.direction == 0:
            if decision.action in (OPEN_LONG, OPEN_SHORT):
                self._open_position(ts, open_p, decision)
            return

        if decision.action == FLAT:
            self._close_position(ts, open_p, "signal_flat")
            return

        side = 1 if decision.action == OPEN_LONG else -1
        if decision.action in (OPEN_LONG, OPEN_SHORT) and side != self._state.direction:
            self._close_position(ts, open_p, "reverse")
            self._open_position(ts, open_p, decision)
        # same-direction OPEN while in position: no-op (position maintained)

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        n = len(self.df)
        for i in range(1, n):
            bar = self.df.iloc[i]
            prev = self.df.iloc[i - 1]
            ts = int(bar["ts_ms"])

            # decision at previous close, executed at this bar's open
            decision = self.decision_fn(prev, self._state)
            self.decisions.append(
                {
                    "ts_ms": int(prev["ts_ms"]),
                    "action": decision.action,
                    "reasons": "; ".join(decision.reasons),
                    "proba_long": decision.proba_long,
                    "proba_short": decision.proba_short,
                }
            )
            self._execute_decision(bar, decision)
            # cooldown decays once per bar, after the decision: a stop armed on
            # this bar keeps the full cooldown for the decision on this bar's close
            if self._state.cooldown_bars_left > 0:
                self._state.cooldown_bars_left -= 1

            if self._state.direction != 0:
                self._state.bars_in_position += 1
                self._check_exits(bar)

            if self._state.direction != 0 and ts % FUNDING_INTERVAL_MS == 0:
                funding_pnl = (
                    -self._state.direction
                    * self._state.qty
                    * float(bar["close"])
                    * self.funding_rate
                )
                self._cash += funding_pnl
                self._realized += funding_pnl
                self._funding_total += funding_pnl
                if self._open_trade is not None:
                    self._open_trade.funding += funding_pnl

            unrealized = (
                self._state.direction
                * self._state.qty
                * (float(bar["close"]) - self._state.entry_price)
                if self._state.direction != 0
                else 0.0
            )
            equity = self._cash + unrealized
            self._equity_last = equity
            self.equity_curve.append(
                {"ts_ms": ts, "equity": equity, "cash": self._cash, "unrealized": unrealized}
            )

        if self._state.direction != 0:  # force close at last close for completeness
            last = self.df.iloc[-1]
            self._close_position(int(last["ts_ms"]), float(last["close"]), "end_of_backtest")

        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = (
            pd.DataFrame([t.__dict__ for t in self.trades]) if self.trades else pd.DataFrame()
        )
        decisions_df = pd.DataFrame(self.decisions)
        metrics = backtest_metrics(equity_df, trades_df, self.interval_ms, self.initial_equity)
        return {
            "equity": equity_df,
            "trades": trades_df,
            "decisions": decisions_df,
            "metrics": metrics,
        }


def backtest_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    interval_ms: int,
    initial_equity: float,
) -> dict:
    """Performance summary: returns, drawdown, Sharpe, trade statistics."""
    if equity_df.empty:
        return {"n_trades": 0}
    equity = equity_df["equity"].to_numpy(dtype=float)
    rets = np.diff(equity) / equity[:-1]
    candles_per_year = (365 * 86_400_000) / interval_ms
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())

    metrics = {
        "final_equity": float(equity[-1]),
        "total_return": float(equity[-1] / initial_equity - 1.0),
        "annualized_return": float(
            (equity[-1] / initial_equity) ** (candles_per_year / len(equity)) - 1.0
        )
        if equity[-1] > 0
        else -1.0,
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(candles_per_year))
        if rets.std() > 0
        else 0.0,
        "max_drawdown": max_dd,
        "n_candles": len(equity),
    }

    if trades_df.empty:
        metrics.update({"n_trades": 0, "total_fees": 0.0, "total_funding": 0.0})
        return metrics

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    gross_profit = float(wins["net_pnl"].sum())
    gross_loss = float(-losses["net_pnl"].sum())
    metrics.update(
        {
            "n_trades": int(len(trades_df)),
            "win_rate": float((trades_df["net_pnl"] > 0).mean()),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "avg_bars_held": float(trades_df["bars_held"].mean()),
            "total_fees": float(trades_df["fees"].sum()),
            "total_funding": float(trades_df["funding"].sum()),
            "total_gross_pnl": float(trades_df["gross_pnl"].sum()),
            "total_net_pnl": float(trades_df["net_pnl"].sum()),
            "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        }
    )
    return metrics
