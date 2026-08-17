"""Multi-Asset Basket Paper Runner.

Coordinates real-time feed updates, cross-sectional features, parallel paper brokers,
and unified portfolio equity tracking across a basket of crypto perpetuals.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..data_ingestion.candle_downloader import CandleStore
from ..data_ingestion.funding_downloader import FundingStore
from ..data_ingestion.intervals import INTERVAL_MS
from ..data_ingestion.validation import validate_candles
from ..execution.paper_broker import PaperBroker, PaperFill
from ..features.cross_sectional import align_basket, compute_cross_sectional_features
from ..features.pipeline import build_feature_frame
from ..monitoring.heartbeat import Heartbeat
from ..monitoring.notify import Notifier
from ..risk.exceptions import KillSwitchTripped
from ..risk.gate import RiskGate
from ..strategy.signal_engine import (
    SignalDecision,
    decide_cross_sectional,
    decide_funding_squeeze,
    decide_triple_barrier,
)

log = logging.getLogger("basket_runner")


def _f(x) -> float:
    return float(x)


class BasketRunner:
    """Manages multi-symbol live paper trading across a basket of symbols."""

    def __init__(
        self,
        *,
        settings: Settings,
        symbols: list[str],
        client,
        store: CandleStore,
        funding_store: FundingStore | None = None,
        strategy_mode: str = "cross_sectional",  # "triple_barrier" | "cross_sectional" | "funding_squeeze" | "basket"
        models_by_symbol: dict[str, Any] | None = None,
        journal_dir: str | Path = "data/runner",
        state_path: str | Path = "data/runner/basket_state.json",
        warmup_bars: int = 1000,
    ) -> None:
        self.settings = settings
        self.symbols = sorted(symbols)
        self.client = client
        self.store = store
        self.funding_store = funding_store or FundingStore(
            settings.data.data_dir, network=settings.market_data_network
        )
        self.strategy_mode = strategy_mode
        self.models_by_symbol = models_by_symbol or {}
        self.interval_ms = INTERVAL_MS[settings.interval]
        self.warmup_bars = warmup_bars

        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.notifier = Notifier(settings.notifications)
        self.heartbeat = Heartbeat(settings.notifications.healthcheck_url)

        # Per-symbol state
        self.ctx: dict[str, pd.DataFrame] = {sym: pd.DataFrame() for sym in self.symbols}
        self.last_ts: dict[str, int] = {sym: 0 for sym in self.symbols}
        self.pending: dict[str, SignalDecision | None] = {sym: None for sym in self.symbols}

        # Capital per symbol
        cap_per_sym = settings.backtest.initial_equity / max(1, len(self.symbols))
        self.gates: dict[str, RiskGate] = {}
        self.brokers: dict[str, PaperBroker] = {}

        for sym in self.symbols:
            self.gates[sym] = RiskGate(
                settings.risk,
                cap_per_sym,
                tombstone_path=self.state_path.parent / f"KILL_SWITCH_{sym}.json",
            )
            self.brokers[sym] = PaperBroker(
                initial_equity=cap_per_sym,
                taker_fee=settings.execution.taker_fee,
                maker_fee=settings.execution.maker_fee,
                slippage_bps=settings.execution.slippage_bps,
                funding_rate=settings.backtest.funding_rate,
                risk_cfg=settings.risk,
            )

    # ------------------------------------------------------------------ io
    def _journal(self, record: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        record = {
            k: (_f(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in record.items()
        }
        with open(self.journal_dir / f"basket_journal_{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _save_snapshot(self) -> None:
        snap = {
            "last_ts": self.last_ts,
            "brokers": {sym: self.brokers[sym].snapshot() for sym in self.symbols},
            "daily_losses": {sym: self.gates[sym].daily_loss.snapshot() for sym in self.symbols},
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _restore_snapshot(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            snap = json.loads(self.state_path.read_text(encoding="utf-8"))
            if "brokers" not in snap or "daily_losses" not in snap:
                return False
            for sym in self.symbols:
                if sym in snap["brokers"]:
                    self.brokers[sym].restore(snap["brokers"][sym])
                if sym in snap["daily_losses"]:
                    self.gates[sym].daily_loss.restore(snap["daily_losses"][sym])
                if sym in snap.get("last_ts", {}):
                    self.last_ts[sym] = int(snap["last_ts"][sym])
            log.info("restored basket state across %d symbols", len(self.symbols))
            return True
        except Exception as exc:
            log.warning("basket snapshot unreadable (%s); starting fresh", exc)
            return False

    # ------------------------------------------------------------ warmup
    def warmup(self) -> None:
        for sym in self.symbols:
            df = self.store.load(sym, self.settings.interval)
            if df is None or len(df) < self.warmup_bars:
                from ..data_ingestion.candle_downloader import incremental_update

                df, _ = incremental_update(
                    self.client,
                    sym,
                    self.settings.interval,
                    self.store,
                    history_days=self.settings.data.history_days,
                )
            self.ctx[sym] = df.tail(self.warmup_bars).reset_index(drop=True)
            self.last_ts[sym] = int(self.ctx[sym]["ts_ms"].iloc[-1])

        self._restore_snapshot()
        self._decide_all_on_last()

    # ----------------------------------------------------------- decisions
    def _decide_all_on_last(self) -> None:
        if self.strategy_mode == "cross_sectional":
            wide, syms = align_basket(self.ctx)
            if len(syms) >= 2 and not wide.empty:
                cs_feats = compute_cross_sectional_features(wide, syms)
                for sym in syms:
                    row = cs_feats[sym].iloc[-1]
                    self.pending[sym] = decide_cross_sectional(
                        row,
                        self.brokers[sym].state,
                        self.settings.risk,
                    )
        elif self.strategy_mode == "funding_squeeze":
            for sym in self.symbols:
                fdf = self.funding_store.load(sym)
                cdf = self.ctx[sym]
                if fdf is not None and not fdf.empty:
                    merged = cdf.merge(fdf[["ts_ms", "funding_rate"]], on="ts_ms", how="left")
                    merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0001)
                    mean = merged["funding_rate"].rolling(720, min_periods=50).mean()
                    std = merged["funding_rate"].rolling(720, min_periods=50).std().replace(0, 1e-6)
                    merged["f_funding_zscore"] = (merged["funding_rate"] - mean) / std
                    row = merged.iloc[-1]
                    self.pending[sym] = decide_funding_squeeze(
                        row, self.brokers[sym].state, self.settings.risk
                    )
        elif self.strategy_mode == "triple_barrier":
            for sym in self.symbols:
                frame, cols = build_feature_frame(
                    self.ctx[sym], self.settings.features, drop_na=False
                )
                row = frame.iloc[-1]
                model = self.models_by_symbol.get(sym)
                if model is not None:
                    f_cols = sorted(c for c in frame.columns if c.startswith("f_"))
                    proba = model.predict_proba(pd.DataFrame([row[f_cols]], columns=f_cols))[0]
                    self.pending[sym] = decide_triple_barrier(
                        row,
                        self.brokers[sym].state,
                        self.settings.strategy,
                        self.settings.risk,
                        proba,
                    )

    # --------------------------------------------------------------- tick
    def tick(self, now_ms: int | None = None) -> dict[str, Any]:
        now = now_ms if now_ms is not None else self.client.server_time_ms()
        total_records: list[dict] = []
        new_bars_count = 0

        for sym in self.symbols:
            page = self.client.fetch_candles(sym, self.settings.interval, limit=10)
            closed = page[page["ts_ms"] <= now - self.interval_ms]
            if closed.empty:
                continue

            new_bars = closed[closed["ts_ms"] > self.last_ts[sym]]
            if new_bars.empty:
                continue

            recent = pd.concat([self.ctx[sym].tail(2), new_bars], ignore_index=True)
            report = validate_candles(
                recent, self.interval_ms, max_bar_move_pct=self.settings.data.max_bar_move_pct
            )
            if not report.ok:
                self.gates[sym].kill_switch.trip(
                    f"candle validation failed on {sym}: {report.errors}"
                )
                raise KillSwitchTripped(f"kill switch tripped on {sym}")

            for _, bar in new_bars.iterrows():
                total_records += self._process_bar_for_symbol(sym, bar)
                new_bars_count += 1

            self.store.append(new_bars, sym, self.settings.interval, validate=False)

        # Portfolio equity log
        total_equity = sum(b.equity() for b in self.brokers.values())
        self._journal(
            {
                "type": "portfolio_mark",
                "ts_ms": now,
                "total_equity": total_equity,
                "symbol_equities": {sym: self.brokers[sym].equity() for sym in self.symbols},
            }
        )
        self.heartbeat.ping(f"basket_equity={total_equity:.2f}")

        return {"records": total_records, "new_bars": new_bars_count, "total_equity": total_equity}

    def _process_bar_for_symbol(self, sym: str, bar: pd.Series) -> list[dict]:
        records: list[dict] = []
        ts = int(bar["ts_ms"])

        # 1. Execute pending decision from prior bar
        decision = self.pending[sym]
        if decision is not None:
            fills = self._execute_decision_for_symbol(sym, bar, decision)
            for fill in fills:
                records.append(
                    {
                        "type": "fill",
                        "symbol": sym,
                        "ts_ms": fill.ts_ms,
                        "action": fill.action,
                        "qty": fill.qty,
                        "price": fill.price,
                        "fee": fill.fee,
                        "reason": fill.reason,
                    }
                )
                if fill.action.startswith("OPEN"):
                    self.notifier.notify_fill(
                        sym, fill.action, fill.price, fill.qty, fill.reason, fill.fee
                    )
            self.pending[sym] = None

        # 2. Exits and funding on bar
        bar_fills, funding = self.brokers[sym].enter_bar(bar)
        for fill in bar_fills:
            records.append(
                {
                    "type": "exit",
                    "symbol": sym,
                    "ts_ms": fill.ts_ms,
                    "action": fill.action,
                    "qty": fill.qty,
                    "price": fill.price,
                    "fee": fill.fee,
                    "reason": fill.reason,
                }
            )
            self.notifier.notify_exit(
                sym, fill.action, fill.price, fill.qty, fill.reason, fill.realized_pnl
            )
            if not fill.gate_applied:
                self.gates[sym].on_position_closed(
                    fill.realized_pnl, fill.ts_ms, self.brokers[sym].equity()
                )

        if funding:
            record = {"type": "funding", "symbol": sym, "ts_ms": ts, "pnl": funding}
            records.append(record)
            self._journal(record)

        # 3. Update context
        self.ctx[sym] = (
            pd.concat([self.ctx[sym], pd.DataFrame([bar])], ignore_index=True)
            .drop_duplicates(subset="ts_ms")
            .tail(self.warmup_bars)
            .reset_index(drop=True)
        )
        self.last_ts[sym] = ts

        # 4. Generate next pending decision
        self._decide_symbol_on_bar(sym, bar)
        self._save_snapshot()

        return records

    def _execute_decision_for_symbol(
        self, sym: str, bar: pd.Series, decision: SignalDecision
    ) -> list[PaperFill]:
        broker = self.brokers[sym]
        gate = self.gates[sym]
        action = decision.action
        open_p = float(bar["open"])
        ts = int(bar["ts_ms"])
        atr = decision.atr_value
        fills: list[PaperFill] = []

        if broker.direction == 0:
            if action in ("OPEN_LONG", "OPEN_SHORT"):
                direction = 1 if action == "OPEN_LONG" else -1
                fill = broker.open_position(ts, open_p, direction, atr)
                if fill:
                    fills.append(fill)
            return fills

        if broker.direction != 0:
            if (broker.direction == 1 and action == "OPEN_SHORT") or (
                broker.direction == -1 and action == "OPEN_LONG"
            ):
                close_fill = broker.close_position(ts, open_p, "reverse_signal")
                fills.append(close_fill)
                gate.on_position_closed(close_fill.realized_pnl, ts, broker.equity())
                direction = 1 if action == "OPEN_LONG" else -1
                open_fill = broker.open_position(ts, open_p, direction, atr)
                if open_fill:
                    fills.append(open_fill)
            elif action == "FLAT":
                close_fill = broker.close_position(ts, open_p, "signal_flat")
                fills.append(close_fill)
                gate.on_position_closed(close_fill.realized_pnl, ts, broker.equity())

        return fills

    def _decide_symbol_on_bar(self, sym: str, bar: pd.Series) -> None:
        if self.strategy_mode == "cross_sectional":
            wide, syms = align_basket(self.ctx)
            if len(syms) >= 2 and not wide.empty:
                cs_feats = compute_cross_sectional_features(wide, syms)
                if sym in cs_feats and not cs_feats[sym].empty:
                    row = cs_feats[sym].iloc[-1]
                    self.pending[sym] = decide_cross_sectional(
                        row,
                        self.brokers[sym].state,
                        self.settings.risk,
                    )
        elif self.strategy_mode == "funding_squeeze":
            fdf = self.funding_store.load(sym)
            cdf = self.ctx[sym]
            if fdf is not None and not fdf.empty:
                merged = cdf.merge(fdf[["ts_ms", "funding_rate"]], on="ts_ms", how="left")
                merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0001)
                mean = merged["funding_rate"].rolling(720, min_periods=50).mean()
                std = merged["funding_rate"].rolling(720, min_periods=50).std().replace(0, 1e-6)
                merged["f_funding_zscore"] = (merged["funding_rate"] - mean) / std
                row = merged.iloc[-1]
                self.pending[sym] = decide_funding_squeeze(
                    row, self.brokers[sym].state, self.settings.risk
                )
        elif self.strategy_mode == "triple_barrier":
            frame, _ = build_feature_frame(self.ctx[sym], self.settings.features, drop_na=False)
            row = frame.iloc[-1]
            model = self.models_by_symbol.get(sym)
            if model is not None:
                f_cols = sorted(c for c in frame.columns if c.startswith("f_"))
                proba = model.predict_proba(pd.DataFrame([row[f_cols]], columns=f_cols))[0]
                self.pending[sym] = decide_triple_barrier(
                    row,
                    self.brokers[sym].state,
                    self.settings.strategy,
                    self.settings.risk,
                    proba,
                )
