"""BotRunner tests: warmup, decision->next-open execution, stops, gate
rejections, snapshot restore, kill switch, exchange mirroring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import requests

from src.config import (
    BacktestSettings,
    DataSettings,
    EnvSettings,
    ExecutionSettings,
    RiskSettings,
    Settings,
    StrategySettings,
)
from src.data_ingestion.candle_downloader import CandleStore
from src.execution.bybit_executor import BybitExecutor
from src.runner.runner import BotRunner

IV = 300_000
START = 1_700_000_000_000
N = 260  # more than warmup_bars=200


def make_frame():
    rows = [
        [START + i * IV, 100.0, 100.5, 99.5, 100.0, 10.0, 1000.0]
        for i in range(N)
    ]
    return pd.DataFrame(
        rows, columns=["ts_ms", "open", "high", "low", "close", "volume", "turnover"]
    )


class FixedModel:
    def __init__(self, proba):
        self.proba = np.array(proba)

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


class FakeClient:
    """Synthesizes future bars on demand so ticks can progress past warmup."""

    def __init__(self, df, fail=False, extra_lows=None, extra_bars=None, now_ms=None):
        self.base = df
        self.fail = fail
        self.extra_lows = extra_lows or [99.5] * 5
        self.extra_bars = extra_bars
        self.calls = 0
        self.extras: list[list] = []
        self.now_ms = now_ms if now_ms is not None else int(df["ts_ms"].iloc[-1])

    def server_time_ms(self):
        return self.now_ms

    def fetch_candles(self, symbol, interval, limit=10, start_ms=None, end_ms=None):
        self.calls += 1
        if self.fail:
            raise requests.exceptions.ConnectionError("down")
        if self.extra_bars is not None:
            self.extras = [list(b) for b in self.extra_bars]
        else:
            while len(self.extras) < len(self.extra_lows):
                if self.extras:
                    last_ts = self.extras[-1][0]
                else:
                    last_ts = int(self.base["ts_ms"].iloc[-1])
                low = self.extra_lows[len(self.extras)]
                self.extras.append([last_ts + IV, 100.0, 100.5, low, 100.0, 10.0, 1000.0])
        cols = ["ts_ms", "open", "high", "low", "close", "volume", "turnover"]
        full = pd.concat([self.base, pd.DataFrame(self.extras, columns=cols)], ignore_index=True)
        if start_ms is not None:
            full = full[full["ts_ms"] >= start_ms]
        if end_ms is not None:
            full = full[full["ts_ms"] < end_ms]
        full = full.sort_values("ts_ms").reset_index(drop=True)
        return full.tail(limit) if limit is not None else full


class FakeExecutor:
    def __init__(self, status="submitted", position=None):
        self.orders: list[tuple[str, float, bool]] = []
        self.status = status
        self.position = position
        self.setup_calls = 0

    def market_order(self, side, qty, reduce_only=False):
        self.orders.append((side, qty, reduce_only))
        return {"status": self.status, "order_id": "x", "order_link_id": "y"}

    def setup(self, leverage):
        self.setup_calls += 1

    def get_position(self):
        return self.position

    def sanitize_qty(self, qty, entry_price):
        return qty, []


def make_settings(tmp_path, **risk_kw):
    risk_kw.setdefault("max_notional_pct", 100.0)
    risk = RiskSettings(
        risk_per_trade_pct=0.5, max_notional_pct=100.0,
        stop_loss_atr_mult=2.0, take_profit_atr_mult=3.0,
        min_hold_bars=1, max_hold_bars=60, cooldown_bars=5,
    ).model_copy(update=risk_kw)
    return Settings(
        mode="paper", symbol="BTCUSDT", interval="5",
        data=DataSettings(data_dir=str(tmp_path)),
        strategy=StrategySettings(),
        risk=risk,
        execution=ExecutionSettings(taker_fee=0.001, slippage_bps=0.0),
        backtest=BacktestSettings(initial_equity=10_000.0, funding_rate=0.0001),
        env=EnvSettings(),
    )


def make_runner(tmp_path, settings, client, executor=None, **kw):
    store = CandleStore(tmp_path / "data")
    store.write(client.base, "BTCUSDT", "5")
    kw.setdefault("warmup_bars", 200)
    kw.setdefault("journal_dir", tmp_path / "runner")
    kw.setdefault("state_path", tmp_path / "runner" / "state.json")
    return BotRunner(
        settings=settings, client=client, store=store,
        model=FixedModel([0.2, 0.1, 0.7]), meta={"model_id": "t"},
        executor=executor, **kw,
    )


def journal_records(runner, type_=None):
    files = list(runner.journal_dir.glob("journal_*.jsonl"))
    assert files, "no journal file written"
    recs = [json.loads(l) for f in files for l in f.read_text().splitlines()]
    return [r for r in recs if r["type"] == type_] if type_ else recs


def test_warmup_loads_and_predecides(tmp_path):
    settings = make_settings(tmp_path)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()))
    runner.warmup()
    assert runner.last_ts == START + (N - 1) * IV  # last bar of the 260-bar frame
    assert runner.pending is not None
    assert runner.pending.action == "OPEN_LONG"


def test_tick_executes_decision_at_next_open(tmp_path):
    settings = make_settings(tmp_path)
    executor = FakeExecutor()
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()), executor=executor)
    runner.warmup()

    next_ts = runner.last_ts + IV
    result = runner.tick(now_ms=next_ts + IV)  # next bar is closed
    assert result["new_bars"] == 1
    assert runner.broker.direction == 1
    assert runner.broker.state.entry_ts_ms == next_ts

    fills = journal_records(runner, "fill")
    assert len(fills) == 1
    assert fills[0]["action"] == "OPEN_LONG"
    assert fills[0]["reason"] == "entry"
    assert runner.state_path.exists()
    assert executor.orders == [("Buy", fills[0]["qty"], False)]


def test_stop_loss_closes_position_and_updates_daily_loss(tmp_path):
    settings = make_settings(tmp_path)
    runner = make_runner(
        tmp_path, settings, FakeClient(make_frame(), extra_lows=[99.5, 95.0, 99.5, 99.5, 99.5])
    )
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)  # extra bar 0: entry
    runner.tick(now_ms=runner.last_ts + 3 * IV)  # extra bar 1: low 95 -> stop breach

    assert runner.broker.direction == 0
    fills = journal_records(runner, "fill")
    closes = [f for f in fills if f["reason"] == "stop_loss"]
    assert len(closes) == 1
    assert closes[0]["realized_pnl"] < 0
    assert runner.gate.daily_loss.day_pnl() == pytest.approx(closes[0]["realized_pnl"])
    # stop exit arms cooldown (5); the extra bar after the stop already decremented it once
    assert runner.broker.state.cooldown_bars_left == 4


def test_gate_rejection_blocks_entry_and_journals(tmp_path):
    settings = make_settings(tmp_path)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()))
    runner.warmup()
    runner.gate.daily_loss.update(-300.0, runner.last_ts, 9_700)  # over the 2% limit
    runner.tick(now_ms=runner.last_ts + 2 * IV)
    assert runner.broker.direction == 0
    rejected = journal_records(runner, "rejected")
    assert len(rejected) == 1
    assert "daily loss" in rejected[0]["reasons"]


def test_snapshot_restore_resumes_position(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(make_frame())
    runner = make_runner(tmp_path, settings, client)
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)
    assert runner.broker.direction == 1

    runner2 = make_runner(tmp_path, settings, client)
    runner2.warmup()
    assert runner2.broker.direction == 1
    assert runner2.broker.state.entry_ts_ms == runner.broker.state.entry_ts_ms
    assert runner2.broker.state.qty == pytest.approx(runner.broker.state.qty)


def test_kill_switch_aborts_on_api_error_streak(tmp_path):
    settings = make_settings(tmp_path, max_api_error_streak=2)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame(), fail=True))
    runner.warmup()
    with pytest.raises(requests.exceptions.ConnectionError):
        runner.tick()
    with pytest.raises(RuntimeError, match="kill switch tripped"):
        runner.tick()


def test_flat_signal_no_trades(tmp_path):
    settings = make_settings(tmp_path)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()))
    runner.model = FixedModel([0.3, 0.4, 0.3])  # below confidence
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)
    assert runner.broker.direction == 0
    assert journal_records(runner, "fill") == []


class GapClient:
    """Cache (base, 260 bars) lags the exchange (full, 12 more bars). `base` is
    what the store holds; `full` is what the exchange can serve, so the runner's
    first tick page starts 3 bars past last_ts and must be backfilled."""

    def __init__(self):
        base = make_frame()
        cols = ["ts_ms", "open", "high", "low", "close", "volume", "turnover"]
        extras = [[int(base["ts_ms"].iloc[-1]) + (i + 1) * IV, 100.0, 100.5, 99.5, 100.0, 10.0, 1000.0]
                  for i in range(12)]
        self.base = base
        self.full = pd.concat([base, pd.DataFrame(extras, columns=cols)], ignore_index=True)
        self.now_ms = int(base["ts_ms"].iloc[-1])  # cache end == server time during warmup

    def server_time_ms(self):
        return self.now_ms

    def fetch_candles(self, symbol, interval, limit=10, start_ms=None, end_ms=None):
        page = self.full
        if start_ms is not None:
            page = page[page["ts_ms"] >= start_ms]
        if end_ms is not None:
            page = page[page["ts_ms"] < end_ms]
        page = page.sort_values("ts_ms").reset_index(drop=True)
        return page.tail(limit) if limit is not None else page


def test_tick_backfills_gap_bars_and_persists(tmp_path):
    """Bars skipped while the bot was down must be fetched and processed, not
    jumped over (stops/funding are evaluated per bar), then persisted."""
    settings = make_settings(tmp_path)
    client = GapClient()
    runner = make_runner(tmp_path, settings, client)
    runner.warmup()
    assert runner.last_ts == int(client.base["ts_ms"].iloc[-1])

    client.now_ms = int(client.full["ts_ms"].iloc[-1])  # market moved on while the bot was down
    result = runner.tick(now_ms=client.server_time_ms())
    assert result["new_bars"] == 11  # 12 exchange bars, last one still unclosed
    marks = journal_records(runner, "mark")
    assert len(marks) == 11  # every backfilled bar produced an equity mark (no skip)
    stored = runner.store.load("BTCUSDT", "5")
    assert len(stored) == 260 + 11  # cache now holds the gap bars too


def test_reverse_rejected_when_daily_loss_limit_reached(tmp_path):
    """The closed leg's P&L is applied to the gate BEFORE the fresh leg is
    approved, so a day at the loss limit rejects the reverse and stays flat."""
    settings = make_settings(tmp_path)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()))
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)  # enters long
    assert runner.broker.direction == 1

    runner.model = FixedModel([0.65, 0.0, 0.35])  # strong short signal -> reverse
    runner.gate.daily_loss.update(-1_000.0, runner.last_ts, 9_000)  # over the 2% limit
    runner.tick(now_ms=runner.last_ts + 3 * IV)

    fills = journal_records(runner, "fill")
    assert [f["reason"] for f in fills] == ["entry", "reverse"]
    assert runner.broker.direction == 0  # old leg closed, new leg rejected
    rejected = journal_records(runner, "rejected")
    assert len(rejected) == 1


def test_failed_order_status_trips_kill_switch(tmp_path):
    settings = make_settings(tmp_path)
    executor = FakeExecutor(status="failed")
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()), executor=executor)
    runner.warmup()
    assert executor.setup_calls == 1
    with pytest.raises(RuntimeError, match="kill switch tripped"):
        runner.tick(now_ms=runner.last_ts + 2 * IV)
    assert runner.gate.kill_switch.is_tripped()


def test_qty_rejected_by_exchange_rounding_journals(tmp_path):
    """A qty that the exchange would reject (below min order qty) must not
    create a local fill."""
    settings = make_settings(tmp_path)
    executor = FakeExecutor()
    executor.sanitize_qty = lambda qty, entry: (qty, ["qty below min order qty 1"])
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()), executor=executor)
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)
    assert runner.broker.direction == 0
    rejected = journal_records(runner, "rejected")
    assert len(rejected) == 1
    assert "below min order qty" in rejected[0]["reasons"]


class JumpClient(FakeClient):
    """Next bar closes 300% above the previous close — an impossible move."""

    def __init__(self, df):
        super().__init__(df)
        jump_ts = int(df["ts_ms"].iloc[-1]) + IV
        self.extra_bars = [[jump_ts, 100.0, 500.0, 99.0, 400.0, 10.0, 1000.0]]


def test_bad_feed_trips_kill_switch_before_trading_or_persisting(tmp_path):
    settings = make_settings(tmp_path)
    client = JumpClient(make_frame())
    runner = make_runner(tmp_path, settings, client)
    runner.warmup()
    n_stored = len(runner.store.load("BTCUSDT", "5"))

    with pytest.raises(RuntimeError, match="kill switch tripped"):
        runner.tick(now_ms=runner.last_ts + 2 * IV)

    assert runner.gate.kill_switch.is_tripped()
    assert runner.broker.direction == 0  # zero bars processed: no position
    assert len(runner.store.load("BTCUSDT", "5")) == n_stored  # nothing persisted
    assert journal_records(runner, "fill") == []


def test_daily_loss_limit_survives_restart(tmp_path):
    """F3: lose past the limit, restart from the snapshot, next entry rejected."""
    settings = make_settings(tmp_path)
    client = FakeClient(make_frame(), extra_lows=[99.5, 85.0, 99.5, 99.5, 99.5])
    runner = make_runner(tmp_path, settings, client)
    runner.warmup()
    runner.tick(now_ms=runner.last_ts + 2 * IV)  # enters long
    runner.tick(now_ms=runner.last_ts + 3 * IV)  # stop breach -> losing close, flat
    assert runner.broker.direction == 0
    assert runner.gate.daily_loss.day_pnl() < 0

    runner.gate.daily_loss.update(-1_000.0, runner.last_ts, 9_000)  # now over the 2% limit
    runner._save_snapshot()

    # restart on the SAME store (it holds the persisted bars, incl. the two ticks)
    runner2 = BotRunner(
        settings=settings, client=FakeClient(make_frame()), store=runner.store,
        model=FixedModel([0.2, 0.1, 0.7]), meta={"model_id": "t"},
        journal_dir=tmp_path / "runner", state_path=tmp_path / "runner" / "state.json",
        warmup_bars=200,
    )
    runner2.warmup()
    assert not runner2.gate.daily_loss.allowed(runner2.last_ts)  # the loss came back
    assert runner2.broker.state.cooldown_bars_left > 0  # stop-out cooldown also restored
    runner2.broker.state.cooldown_bars_left = 0  # let the strategy want a fresh entry
    runner2._decide_on_last()
    runner2.tick(now_ms=runner2.last_ts + 2 * IV)
    assert runner2.broker.direction == 0  # the restored limit blocks the entry
    rejected = journal_records(runner2, "rejected")
    assert any("daily loss" in r["reasons"] for r in rejected)


def test_reconcile_mismatch_writes_tombstone(tmp_path):
    settings = make_settings(tmp_path)
    executor = FakeExecutor(position={"side": "Buy", "size": 1.0})  # exchange holds, ledger flat
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()), executor=executor)
    with pytest.raises(RuntimeError, match="kill switch"):
        runner.warmup()
    assert runner.gate.kill_switch.is_tripped()
    assert (tmp_path / "runner" / "KILL_SWITCH.json").exists()


def test_executor_api_failures_reach_gate_streak_and_trip(tmp_path):
    """A1: order-path failures must increment the runner's streak (binding works)."""
    settings = make_settings(tmp_path)

    class FailingSession:
        def switch_position_mode(self, **kw):
            return {"retCode": 0, "result": {}}

        def set_leverage(self, **kw):
            return {"retCode": 0, "result": {}}

        def get_positions(self, **kw):
            raise requests.exceptions.ConnectionError("down")

        def place_order(self, **kw):
            raise requests.exceptions.ConnectionError("down")

        def get_order_history(self, **kw):
            raise requests.exceptions.ConnectionError("down")

    executor = BybitExecutor(FailingSession(), "BTCUSDT", max_retries=0)
    runner = make_runner(tmp_path, settings, FakeClient(make_frame()), executor=executor)
    runner.warmup()  # setup ok; reconcile fetch fails -> warning, not a mismatch
    assert not runner.gate.kill_switch.is_tripped()

    for _ in range(3):
        result = executor.market_order("Buy", 0.01)
        assert result["status"] == "failed"
    assert runner.gate.kill_switch.is_tripped()  # streak reached the gate via the binding
