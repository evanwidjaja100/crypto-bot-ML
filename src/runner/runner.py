"""Phase 10: the bot runner loop — candles -> features -> model -> strategy
-> risk gate -> execution, journaled bar by bar.

Loop semantics match the backtester exactly:
  - At the close of candle t we decide (features/predictions computed on
    closed bars only) and execute the decision at the OPEN of candle t+1.
  - Exits (stop / TP / funding) are applied per closed candle by the
    PaperBroker with the same fill rules as the backtester.
  - Paper mode accounts everything locally; testnet/live additionally mirror
    every fill to the exchange via an injected BybitExecutor.
  - Every decision, fill, funding payment and equity mark is appended to a
    JSONL journal; a state snapshot allows restart without re-arming a trade.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..data_ingestion.intervals import INTERVAL_MS
from ..data_ingestion.validation import validate_candles
from ..execution.paper_broker import PaperBroker, PaperFill
from ..features.pipeline import build_feature_frame
from ..risk.gate import RiskGate
from ..strategy.signal_engine import FLAT, OPEN_LONG, OPEN_SHORT, SignalDecision, decide

log = logging.getLogger("runner")


def _f(x) -> float:
    return float(x)


class BotRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        client,
        store,
        model,
        meta: dict,
        executor=None,
        journal_dir: str | Path = "data/runner",
        state_path: str | Path = "data/runner/state.json",
        warmup_bars: int = 2000,
    ) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self.model = model
        self.meta = meta
        self.executor = executor
        self.interval_ms = INTERVAL_MS[settings.interval]
        self.warmup_bars = warmup_bars

        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.gate = RiskGate(settings.risk, settings.backtest.initial_equity)
        self.broker = PaperBroker(
            initial_equity=settings.backtest.initial_equity,
            taker_fee=settings.execution.taker_fee,
            maker_fee=settings.execution.maker_fee,
            slippage_bps=settings.execution.slippage_bps,
            funding_rate=settings.backtest.funding_rate,
            risk_cfg=settings.risk,
        )
        self.pending: SignalDecision | None = None
        self.last_ts: int = 0
        self.ctx: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------ io
    def _journal(self, record: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        record = {k: (_f(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in record.items()}
        with open(self.journal_dir / f"journal_{day}.jsonl", "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _save_snapshot(self) -> None:
        snap = {
            "last_ts": self.last_ts,
            "broker": self.broker.snapshot(),
            "daily_loss_pnl": self.gate.daily_loss.day_pnl(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, default=str))
        tmp.rename(self.state_path)

    def _restore_snapshot(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            snap = json.loads(self.state_path.read_text())
            self.broker.restore(snap["broker"])
            self.last_ts = int(snap["last_ts"])
            if "daily_loss_pnl" in snap:
                self.gate.daily_loss.restore_day_pnl(snap["daily_loss_pnl"])
            log.info("restored state: last_ts=%d direction=%d", self.last_ts, self.broker.direction)
            return True
        except Exception as exc:  # noqa: BLE001 — a corrupt snapshot must not block startup
            log.warning("state snapshot unreadable (%s); starting fresh", exc)
            return False

    # ------------------------------------------------------------ warmup
    def warmup(self) -> None:
        df = self.store.load(self.settings.symbol, self.settings.interval)
        stale = False
        if df is not None and not df.empty:
            try:
                today = self.client.server_time_ms()
                stale = today - int(df["ts_ms"].iloc[-1]) > 2 * self.interval_ms
            except Exception:  # noqa: BLE001 — clock unavailable: fall back to row-count only
                log.warning("could not check cache staleness (server time unavailable)")
        if df is None or len(df) < self.warmup_bars or stale:
            from ..data_ingestion.candle_downloader import incremental_update

            df, _ = incremental_update(
                self.client, self.settings.symbol, self.settings.interval, self.store,
                history_days=self.settings.data.history_days,
            )
        self.ctx = df.tail(self.warmup_bars).reset_index(drop=True)
        self.last_ts = int(self.ctx["ts_ms"].iloc[-1])

        if self._restore_snapshot():
            self._decide_on_last()  # pending decision is recomputed on the same bars below
        else:
            self._decide_on_last()

        if self.executor is not None:
            self._exchange_setup_and_reconcile()

    # ----------------------------------------------------------- decision
    def _feature_row(self) -> pd.DataFrame:
        frame, _ = build_feature_frame(self.ctx, self.settings.features, drop_na=False)
        return frame

    def _decide_on_last(self) -> None:
        frame = self._feature_row()
        matches = frame.loc[frame["ts_ms"] == self.last_ts]
        if matches.empty:
            return
        row = matches.iloc[-1]
        cols = sorted(c for c in frame.columns if c.startswith("f_"))
        proba = self.model.predict_proba(pd.DataFrame([row[cols]], columns=cols))[0]
        self.pending = decide(row, self.broker.state, self.settings.strategy, self.settings.risk, proba)
        self._journal(
            {
                "type": "decision", "ts_ms": int(row["ts_ms"]),
                "action": self.pending.action,
                "reasons": "; ".join(self.pending.reasons),
                "proba_long": self.pending.proba_long,
                "proba_short": self.pending.proba_short,
            }
        )

    # ----------------------------------------------------------------- tick
    def tick(self, now_ms: int | None = None) -> dict:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        try:
            page = self.client.fetch_candles(self.settings.symbol, self.settings.interval, limit=10)
            self.gate.on_api_success()
        except Exception:
            self.gate.on_api_error()
            if self.gate.kill_switch.is_tripped():
                log.error("kill switch tripped: %s", self.gate.kill_switch.describe())
                raise RuntimeError("kill switch tripped") from None
            raise

        closed = page[page["ts_ms"] <= now - self.interval_ms]
        if closed.empty:
            return {"records": [], "new_bars": 0}

        # if the bot was down longer than the fetch window, backfill the gap so
        # no bar after last_ts is skipped (stops/funding must be evaluated per bar)
        first_new = int(closed["ts_ms"].iloc[0])
        if first_new > self.last_ts + self.interval_ms:
            from ..data_ingestion.candle_downloader import download_range

            log.warning("gap detected: cache last_ts=%d, fetch starts at %d — backfilling", self.last_ts, first_new)
            backfill = download_range(
                self.client, self.settings.symbol, self.settings.interval,
                self.last_ts + self.interval_ms, first_new + self.interval_ms,
                chunk_days=self.settings.data.chunk_days, page_size=self.settings.data.page_size,
            )
            closed = (
                pd.concat([backfill, closed], ignore_index=True)
                .sort_values("ts_ms")
                .drop_duplicates(subset="ts_ms")
                .reset_index(drop=True)
            )

        new_bars = closed[closed["ts_ms"] > self.last_ts]
        if not new_bars.empty:
            # validate on ingest, not on persist: the bar has not been traded
            # on yet, and a bad feed halts the bot instead of being written to
            # the cache. Splicing the cached tail catches a jump between the
            # cache and the first new bar, not just within the batch.
            recent = pd.concat([self.ctx.tail(2), new_bars], ignore_index=True)
            report = validate_candles(
                recent, self.interval_ms,
                max_bar_move_pct=self.settings.data.max_bar_move_pct,
            )
            if not report.ok:
                self.gate.kill_switch.trip(f"candle validation failed: {report.errors}")
                log.error("candle validation failed: %s -> %s", report.summary(), report.errors)
                raise RuntimeError("kill switch tripped: candle validation failed")
        records: list[dict] = []
        for _, bar in new_bars.iterrows():
            records += self._process_bar(bar)
        self._persist_bars(new_bars)
        if self.gate.kill_switch.is_tripped():
            log.error("kill switch tripped: %s", self.gate.kill_switch.describe())
            raise RuntimeError("kill switch tripped")
        return {"records": records, "new_bars": int(len(new_bars))}

    def _persist_bars(self, bars: pd.DataFrame) -> None:
        """Append processed closed bars to the parquet cache so a restart resumes from real data."""
        if bars is None or bars.empty:
            return
        existing = self.store.load(self.settings.symbol, self.settings.interval)
        if existing is None or existing.empty:
            self.store.write(bars, self.settings.symbol, self.settings.interval)
        else:
            merged = pd.concat([existing, bars], ignore_index=True)
            self.store.write(merged, self.settings.symbol, self.settings.interval)

    def _process_bar(self, bar: pd.Series) -> list[dict]:
        ts = int(bar["ts_ms"])
        records: list[dict] = []

        fills = self._execute_pending(bar)
        for fill in fills:
            records += self._record_fill(fill)

        bar_fills, funding = self.broker.enter_bar(bar)
        for fill in bar_fills:
            records += self._record_fill(fill)
        if funding:
            record = {"type": "funding", "ts_ms": ts, "pnl": funding}
            records.append(record)
            self._journal(record)

        equity, unrealized = self.broker.equity(float(bar["close"])), self.broker.unrealized(float(bar["close"]))
        mark = {"type": "mark", "ts_ms": ts, "equity": equity, "unrealized": unrealized}
        records.append(mark)
        self._journal(mark)

        self.ctx = pd.concat([self.ctx, bar.to_frame().T], ignore_index=True).tail(self.warmup_bars).reset_index(drop=True)
        self.last_ts = ts
        self._decide_on_last()
        self._save_snapshot()
        return records

    def _record_fill(self, fill: PaperFill) -> list[dict]:
        record = {
            "type": "fill", "ts_ms": fill.ts_ms, "action": fill.action, "price": fill.price,
            "qty": fill.qty, "reason": fill.reason, "fee": fill.fee,
            "realized_pnl": fill.realized_pnl,
        }
        self._journal(record)
        if fill.action.startswith("CLOSE") and not fill.gate_applied:
            self.gate.on_position_closed(fill.realized_pnl, fill.ts_ms, self.broker.equity())
        self._send_to_exchange(fill)
        return [record]

    def _execute_pending(self, bar: pd.Series) -> list[PaperFill]:
        decision = self.pending
        self.pending = None
        open_p = float(bar["open"])
        ts = int(bar["ts_ms"])
        atr = decision.atr_value if decision else None

        if self.broker.direction == 0:
            if decision and decision.action in (OPEN_LONG, OPEN_SHORT):
                direction = 1 if decision.action == OPEN_LONG else -1
                entry = open_p * (1.0 + direction * self.broker.slippage)
                qty = self._execution_qty(decision, direction, entry, ts)
                if qty is None:
                    return []
                approval = self.gate.approve_entry(
                    direction=direction, qty=qty, entry_price=entry,
                    equity=self.broker.equity(),
                    open_positions=0,
                    ts_ms=ts,
                )
                if not approval.approved:
                    self._journal(
                        {"type": "rejected", "ts_ms": ts, "action": decision.action,
                         "reasons": "; ".join(approval.reasons)}
                    )
                    return []
                fill = self.broker.open_position(ts, open_p, direction, atr, qty=qty)
                return [fill] if fill else []
            return []

        if decision is None:
            return []
        if decision.action in (OPEN_LONG, OPEN_SHORT):
            side = 1 if decision.action == OPEN_LONG else -1
            if side == self.broker.direction:
                return []  # same direction: position maintained
            close_fill = self.broker.close_position(ts, open_p, "reverse")
            # apply the closed leg's realized P&L to the daily-loss tracker NOW so
            # the gate can reject the fresh leg on a day that just hit its limit
            self.gate.on_position_closed(close_fill.realized_pnl, ts, self.broker.equity())
            close_fill.gate_applied = True
            entry = open_p * (1.0 + side * self.broker.slippage)
            qty = self._execution_qty(decision, side, entry, ts)
            if qty is None:
                return [close_fill]
            approval = self.gate.approve_entry(
                direction=side, qty=qty, entry_price=entry,
                equity=self.broker.equity(), open_positions=0, ts_ms=ts,
            )
            if not approval.approved:
                self._journal(
                    {"type": "rejected", "ts_ms": ts, "action": decision.action,
                     "reasons": "; ".join(approval.reasons)}
                )
                return [close_fill]
            open_fill = self.broker.open_position(ts, open_p, side, decision.atr_value, qty=qty)
            return [close_fill] + ([open_fill] if open_fill else [])
        if decision.action == FLAT:
            return [self.broker.close_position(ts, open_p, "signal_flat")]
        return []

    def _execution_qty(self, decision: SignalDecision, direction: int, entry: float, ts: int) -> float | None:
        """Size + exchange-sanitize a proposed qty. Returns None if rejected."""
        qty = self.broker.propose_qty(direction, entry, decision.atr_value)
        if self.executor is not None:
            qty, reasons = self.executor.sanitize_qty(qty, entry)
            if reasons:
                self._journal(
                    {"type": "rejected", "ts_ms": ts, "action": decision.action,
                     "reasons": "; ".join(reasons)}
                )
                return None
        return qty

    def _send_to_exchange(self, fill: PaperFill) -> None:
        if self.executor is None:
            return
        is_close = fill.action.startswith("CLOSE")
        side = "Sell" if fill.action in ("CLOSE_LONG", "OPEN_SHORT") else "Buy"
        result = self.executor.market_order(side, fill.qty, reduce_only=is_close)
        log.info("order %s %s %s -> %s", side, fill.qty, "reduce" if is_close else "open", result["status"])
        if result["status"] == "failed":
            self.gate.kill_switch.trip(
                f"order {side} {fill.qty} {fill.action} failed to reach the exchange"
            )
            log.error("ORDER FAILED to reach the exchange — kill switch tripped: %s", result)
            raise RuntimeError("kill switch tripped: order failed to reach the exchange")
        if result["status"] == "already_placed":
            log.warning("order idempotently re-placed — reconciling with the exchange")
            self._reconcile_position()
        # "submitted" is a clean placement; nothing to do

    # ------------------------------------------------- exchange testnet/live
    def _exchange_setup_and_reconcile(self) -> None:
        setup = getattr(self.executor, "setup", None)
        if callable(setup):
            setup(self.settings.risk.leverage_cap)
        self._reconcile_position()
        if self.gate.kill_switch.is_tripped():
            raise RuntimeError(f"kill switch tripped: {self.gate.kill_switch.describe()}")

    def _reconcile_position(self) -> None:
        """Compare the exchange position with the local ledger; trip the kill switch on mismatch."""
        if self.executor is None:
            return
        get_pos = getattr(self.executor, "get_position", None)
        if get_pos is None:
            return
        try:
            pos = get_pos()
        except Exception as exc:  # noqa: BLE001 — a network hiccup is not itself a mismatch
            log.warning("position reconciliation fetch failed: %s", exc)
            return
        want = None
        if self.broker.direction == 1:
            want = ("Buy", self.broker.state.qty)
        elif self.broker.direction == -1:
            want = ("Sell", self.broker.state.qty)
        if pos is None:
            if want is not None:
                self.gate.kill_switch.trip("startup reconciliation: exchange is flat, ledger holds a position")
                log.error("reconciliation mismatch: exchange flat, ledger=%s", want)
            return
        side = "Buy" if pos["side"] == "Buy" else "Sell"
        qty = pos["size"]
        if want is None or side != want[0] or (qty - want[1]) > max(1e-6, want[1] * 0.01):
            self.gate.kill_switch.trip(
                f"startup reconciliation mismatch: exchange={pos} ledger={want}"
            )
            log.error("reconciliation mismatch: exchange=%s ledger=%s", pos, want)
