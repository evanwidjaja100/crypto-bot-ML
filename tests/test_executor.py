"""BybitExecutor tests against a fake signed session: idempotent placement,
retry semantics, kill-switch callbacks, position/balance parsing."""

from __future__ import annotations

import pytest
import requests

import src.execution.bybit_executor as ex_mod
from src.execution.bybit_executor import BybitExecutor


class FakeSession:
    def __init__(self):
        self.calls: dict[str, int] = {}
        self.place_failures = 0  # consecutive place_order failures
        self.history_rows: list[dict] = []
        self.positions: list[dict] = []
        self.last_kwargs: dict | None = None

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.calls[name]

    def place_order(self, **kw):
        self._count("place_order")
        self.last_kwargs = kw
        if self.calls["place_order"] <= self.place_failures:
            raise requests.exceptions.ConnectionError("boom")
        return {"retCode": 0, "result": {"orderId": "O-1"}}

    def get_order_history(self, **kw):
        self._count("get_order_history")
        self.history_kwargs = kw
        return {"retCode": 0, "result": {"list": self.history_rows}}

    def get_positions(self, **kw):
        self._count("get_positions")
        return {"retCode": 0, "result": {"list": self.positions}}

    def get_wallet_balance(self, **kw):
        self._count("get_wallet_balance")
        return {"retCode": 0, "result": {"list": [{"totalEquity": "1234.5"}]}}

    def cancel_all_orders(self, **kw):
        self._count("cancel_all_orders")
        return {"retCode": 0, "result": {}}

    def get_instruments_info(self, **kw):
        self._count("get_instruments_info")
        return {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {
                            "qtyStep": "0.1",
                            "minOrderQty": "1",
                            "minNotionalValue": "5",
                        },
                    }
                ]
            },
        }

    def switch_position_mode(self, **kw):
        self._count("switch_position_mode")
        return {"retCode": 0, "result": {}}

    def set_leverage(self, **kw):
        self._count("set_leverage")
        return {"retCode": 0, "result": {}}


def make(session, **kw):
    kw.setdefault("max_retries", 2)
    return BybitExecutor(session, "BTCUSDT", **kw)


def test_market_order_params_and_result(monkeypatch):
    monkeypatch.setattr(ex_mod.time, "sleep", lambda s: None)
    session = FakeSession()
    ex = make(session)
    result = ex.market_order("Buy", 0.001, reduce_only=True)
    assert result["status"] == "submitted"
    assert result["order_id"] == "O-1"
    kw = session.last_kwargs
    assert kw["category"] == "linear"
    assert kw["symbol"] == "BTCUSDT"
    assert kw["side"] == "Buy"
    assert kw["orderType"] == "Market"
    assert kw["qty"] == "0.001"
    assert kw["reduceOnly"] is True
    assert kw["orderLinkId"].startswith("oc-")


def test_retries_transient_then_submitted(monkeypatch):
    monkeypatch.setattr(ex_mod.time, "sleep", lambda s: None)
    session = FakeSession()
    session.place_failures = 2
    ex = make(session, max_retries=3)
    result = ex.market_order("Buy", 0.001)
    assert result["status"] == "submitted"
    assert session.calls["place_order"] == 3


def test_timeout_checks_history_no_double_place(monkeypatch):
    monkeypatch.setattr(ex_mod.time, "sleep", lambda s: None)
    session = FakeSession()
    session.place_failures = 99
    session.history_rows = [{"orderId": "O-9", "orderLinkId": "oc-x"}]
    ex = make(session, max_retries=2)
    result = ex.market_order("Buy", 0.001)
    # the exchange saw the order -> we must NOT place again
    assert result["status"] == "already_placed"
    assert result["order_id"] == "O-9"
    # one logical place attempt (3 internal retries), one history check
    assert session.calls["place_order"] == 3
    assert session.calls["get_order_history"] == 1


def test_failed_when_order_never_reached_exchange(monkeypatch):
    monkeypatch.setattr(ex_mod.time, "sleep", lambda s: None)
    session = FakeSession()
    session.place_failures = 99
    ex = make(session, max_retries=1)
    result = ex.market_order("Buy", 0.001)
    assert result["status"] == "failed"


def test_error_and_success_callbacks(monkeypatch):
    monkeypatch.setattr(ex_mod.time, "sleep", lambda s: None)
    session = FakeSession()
    errors, successes = [], []
    session.place_failures = 1
    ex = make(
        session,
        on_api_error=lambda: errors.append(1),
        on_api_success=lambda: successes.append(1),
    )
    ex.market_order("Buy", 0.001)  # a transient failure that recovers is NOT an error op
    assert errors == []
    assert len(successes) == 1

    session.place_failures = 99
    result = ex.market_order("Sell", 0.001)
    assert result["status"] == "failed"
    # every failed place attempt counts toward the kill-switch streak
    assert len(errors) == 3
    assert len(successes) >= 1  # history lookups that succeeded still count as successes


def test_get_position_parsing():
    session = FakeSession()
    ex = make(session)
    assert ex.get_position() is None
    session.positions = [
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "size": "0.5",
            "avgPrice": "100.5",
            "unrealisedPnl": "2.25",
        }
    ]
    pos = ex.get_position()
    assert pos["side"] == "Sell"
    assert pos["size"] == 0.5
    assert pos["entry_price"] == 100.5


def test_sanitize_qty_floors_to_step_and_caches_instruments():
    session = FakeSession()
    ex = make(session)
    assert ex.sanitize_qty(12.3459, 100.0) == (12.3, [])  # floored to the qtyStep
    assert ex.sanitize_qty(12.3459, 100.0) == (12.3, [])
    assert session.calls["get_instruments_info"] == 1  # cached, fetched once


def test_sanitize_qty_rejects_below_min_qty_and_min_notional():
    session = FakeSession()
    ex = make(session)
    rounded, reasons = ex.sanitize_qty(0.6, 100.0)
    assert rounded == pytest.approx(0.6)
    assert "below min order qty" in reasons[0]
    rounded, reasons = ex.sanitize_qty(1.0, 3.0)  # qty ok, notional 3.0 < 5.0
    assert "below min notional" in reasons[0]


def test_setup_switches_one_way_and_sets_leverage():
    session = FakeSession()
    ex = make(session)
    ex.setup(3)
    assert session.calls["switch_position_mode"] == 1
    assert session.calls["set_leverage"] == 1


def test_get_equity_and_cancel_all():
    session = FakeSession()
    ex = make(session)
    assert ex.get_equity() == 1234.5
    ex.cancel_all()
    assert session.calls["cancel_all_orders"] == 1
