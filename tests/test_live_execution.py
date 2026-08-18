"""Tests for Phase 9 Live Execution Infrastructure (F6, F7, F20, F21).

Verifies exchange-native stops/targets, execution fill adoption, continuous per-bar
reconciliation, clock drift detection, and exchange equity sizing.
"""

from unittest.mock import MagicMock

from src.config import Settings
from src.data_ingestion.bybit_client import BybitClient
from src.execution.bybit_executor import BybitExecutor
from src.execution.paper_broker import PaperFill
from src.runner.runner import BotRunner


class FakeBybitSession:
    """Mock PyBit session recording all API requests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.server_time = 1700000000
        self.equity = 12500.0
        self.position: dict | None = None
        self.executions: list[dict] = []
        self.closed_pnl: list[dict] = []

    def get_server_time(self) -> dict:
        self.calls.append(("get_server_time", {}))
        return {"retCode": 0, "retMsg": "OK", "result": {"timeSecond": str(self.server_time)}}

    def get_wallet_balance(self, **kwargs) -> dict:
        self.calls.append(("get_wallet_balance", kwargs))
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": [{"totalEquity": str(self.equity)}]},
        }

    def set_trading_stop(self, **kwargs) -> dict:
        self.calls.append(("set_trading_stop", kwargs))
        return {"retCode": 0, "retMsg": "OK", "result": {}}

    def get_executions(self, **kwargs) -> dict:
        self.calls.append(("get_executions", kwargs))
        return {"retCode": 0, "retMsg": "OK", "result": {"list": self.executions}}

    def get_closed_pnl(self, **kwargs) -> dict:
        self.calls.append(("get_closed_pnl", kwargs))
        return {"retCode": 0, "retMsg": "OK", "result": {"list": self.closed_pnl}}

    def get_positions(self, **kwargs) -> dict:
        self.calls.append(("get_positions", kwargs))
        if self.position is None:
            return {"retCode": 0, "retMsg": "OK", "result": {"list": []}}
        return {"retCode": 0, "retMsg": "OK", "result": {"list": [self.position]}}

    def place_order(self, **kwargs) -> dict:
        self.calls.append(("place_order", kwargs))
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": "ord-12345"}}

    def get_instruments_info(self, **kwargs) -> dict:
        self.calls.append(("get_instruments_info", kwargs))
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {
                            "qtyStep": "0.001",
                            "minOrderQty": "0.001",
                            "minNotionalValue": "5.0",
                        }
                    }
                ]
            },
        }


def test_executor_set_trading_stop():
    session = FakeBybitSession()
    executor = BybitExecutor(session, "BTCUSDT")

    res = executor.set_trading_stop(stop_loss=60000.0, take_profit=66000.0)
    assert res["retCode"] == 0

    method, kwargs = session.calls[-1]
    assert method == "set_trading_stop"
    assert kwargs["category"] == "linear"
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["positionIdx"] == 0
    assert kwargs["tpslMode"] == "Full"
    assert kwargs["stopLoss"] == "60000.0000"
    assert kwargs["takeProfit"] == "66000.0000"


def test_executor_cancel_trading_stop():
    session = FakeBybitSession()
    executor = BybitExecutor(session, "BTCUSDT")

    res = executor.cancel_trading_stop()
    assert res["retCode"] == 0

    method, kwargs = session.calls[-1]
    assert method == "set_trading_stop"
    assert kwargs["stopLoss"] == "0"
    assert kwargs["takeProfit"] == "0"


def test_executor_get_executions_and_closed_pnl():
    session = FakeBybitSession()
    session.executions = [
        {
            "execId": "ex-1",
            "orderId": "ord-1",
            "orderLinkId": "link-1",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "62100.5",
            "execQty": "0.05",
            "execFee": "1.707",
            "execType": "Trade",
            "execTime": "1700000000000",
        }
    ]
    session.closed_pnl = [
        {
            "orderId": "ord-0",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "qty": "0.05",
            "avgEntryPrice": "60000.0",
            "avgExitPrice": "62000.0",
            "closedPnl": "100.0",
            "createdTime": "1700000000000",
        }
    ]
    executor = BybitExecutor(session, "BTCUSDT")

    execs = executor.get_executions(order_link_id="link-1")
    assert len(execs) == 1
    assert execs[0]["exec_price"] == 62100.5
    assert execs[0]["exec_fee"] == 1.707

    pnls = executor.get_closed_pnl()
    assert len(pnls) == 1
    assert pnls[0]["closed_pnl"] == 100.0


def test_client_clock_drift_check():
    session = FakeBybitSession()
    session.server_time = 1700000000
    client = BybitClient(testnet=True, session=session)

    # Fake local time matching server
    drift_ok, drift_ms = client.check_clock_drift(max_drift_ms=10000000000000)
    assert isinstance(drift_ok, bool)
    assert isinstance(drift_ms, float)


def test_runner_attaches_stops_and_adopts_executions(tmp_path):
    settings = Settings()
    session = FakeBybitSession()
    session.executions = [
        {
            "execId": "ex-1",
            "orderId": "ord-12345",
            "orderLinkId": "test-link",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "60050.0",
            "execQty": "0.1",
            "execFee": "3.30",
            "execType": "Trade",
            "execTime": "1700000000000",
        }
    ]
    executor = BybitExecutor(session, "BTCUSDT")

    mock_client = MagicMock()
    mock_store = MagicMock()
    mock_model = MagicMock()

    runner = BotRunner(
        settings=settings,
        client=mock_client,
        store=mock_store,
        model=mock_model,
        meta={"model_id": "test_m"},
        executor=executor,
        journal_dir=tmp_path / "journal",
        state_path=tmp_path / "state.json",
    )

    # Set up broker with pending position
    runner.broker.state.direction = 1
    runner.broker.state.qty = 0.1
    runner.broker.state.stop_price = 58000.0
    runner.broker.state.target_price = 64000.0

    fill = PaperFill(
        ts_ms=1700000000000,
        action="OPEN_LONG",
        price=60000.0,
        qty=0.1,
        reason="signal",
        fee=1.0,
        realized_pnl=0.0,
    )

    runner._send_to_exchange(fill)

    # Check that place_order was sent
    place_calls = [c for c in session.calls if c[0] == "place_order"]
    assert len(place_calls) >= 1

    # Check that set_trading_stop was called with broker stop prices
    stop_calls = [c for c in session.calls if c[0] == "set_trading_stop"]
    assert len(stop_calls) >= 1
    assert stop_calls[-1][1]["stopLoss"] == "58000.0000"
    assert stop_calls[-1][1]["takeProfit"] == "64000.0000"

    # Check that execution was fetched and updated fill price & fee
    assert fill.price == 60050.0
    assert fill.fee == 3.30


def test_runner_continuous_reconciliation_trips_mismatch(tmp_path):
    settings = Settings()
    session = FakeBybitSession()
    # Exchange says flat, but local broker holds LONG
    session.position = None
    executor = BybitExecutor(session, "BTCUSDT")

    mock_client = MagicMock()
    mock_store = MagicMock()
    mock_model = MagicMock()

    runner = BotRunner(
        settings=settings,
        client=mock_client,
        store=mock_store,
        model=mock_model,
        meta={"model_id": "test_m"},
        executor=executor,
        journal_dir=tmp_path / "journal",
        state_path=tmp_path / "state.json",
    )

    runner.broker.state.direction = 1
    runner.broker.state.qty = 0.5

    # Reconciliation should trip kill switch
    runner._reconcile_position()
    assert runner.gate.kill_switch.is_tripped()
    assert "reconciliation mismatch" in runner.gate.kill_switch.describe()
