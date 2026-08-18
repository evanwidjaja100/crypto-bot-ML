"""Phase 10: signed-order executor for testnet/live Bybit (V5 unified trading).

Order placement is the one place where a blind retry is DANGEROUS (a market
order can fill while we are unsure of its state). Contract:
  - Every attempt carries a unique orderLinkId.
  - On a failed/unknown attempt we query get_order_history by orderLinkId.
    If the order exists there it was accepted (filled or cancelled) and we
    MUST NOT place again — we return its state.
    If it does not exist, the order never reached the exchange and we retry.
  - Kill-switch callbacks: on_api_error is called once per logical operation
    that ultimately fails, on_api_success once per successful operation.
    The runner wires these to the RiskGate.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

import requests
from pybit.exceptions import FailedRequestError

_RETRYABLE = (FailedRequestError, requests.exceptions.RequestException)

SIDE_BY_DIRECTION = {1: "Buy", -1: "Sell"}


class BybitExecutor:
    def __init__(
        self,
        session: Any,
        symbol: str,
        *,
        max_retries: int = 3,
        on_api_error: Callable[[], None] | None = None,
        on_api_success: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self.symbol = symbol
        self._max_retries = max_retries
        self._on_api_error = on_api_error
        self._on_api_success = on_api_success
        self._rng = random.Random(0)
        self._instruments: dict | None = None

    # ------------------------------------------------------------- helpers
    def _request(self, method: str, **kwargs: Any) -> dict:
        """One call attempt; raises RuntimeError on failure (no retry)."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = getattr(self._session, method)(**kwargs)
                if resp.get("retCode") != 0:
                    raise FailedRequestError(
                        request=str(kwargs),
                        message=resp.get("retMsg", "unknown error"),
                        status_code=resp.get("retCode"),
                        time=None,
                        resp_headers=None,
                    )
                self._notify_success()
                return resp
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                time.sleep((2.0**attempt) * (0.3 + self._rng.random()))
        self._notify_error()
        raise RuntimeError(f"Bybit {method} failed after {self._max_retries} retries") from last_exc

    def _notify_error(self) -> None:
        if self._on_api_error:
            self._on_api_error()

    def _notify_success(self) -> None:
        if self._on_api_success:
            self._on_api_success()

    def bind_callbacks(self, on_api_error, on_api_success) -> None:
        """Wire the risk-gate callbacks (the runner calls this after construction)."""
        self._on_api_error = on_api_error
        self._on_api_success = on_api_success

    # ------------------------------------------------------------- orders
    def market_order(self, side: str, qty: float, *, reduce_only: bool = False) -> dict:
        """Place a market order idempotently. Returns {status, order_id, order_link_id}.

        status is one of "submitted" (freshly placed), "already_placed" (a
        previous attempt reached the exchange), or "failed".
        """
        link_id = f"oc-{int(time.time() * 1000)}-{self._rng.randint(0, 999999)}"
        kwargs = dict(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=reduce_only,
            orderLinkId=link_id,
            timeInForce="IOC",
        )
        for _ in range(self._max_retries + 1):
            try:
                resp = self._request("place_order", **kwargs)
                return {
                    "status": "submitted",
                    "order_id": resp["result"]["orderId"],
                    "order_link_id": link_id,
                }
            except RuntimeError:
                # state unknown -> check whether the exchange saw the order
                try:
                    hist = self._request(
                        "get_order_history",
                        category="linear",
                        symbol=self.symbol,
                        orderLinkId=link_id,
                    )
                except RuntimeError:
                    continue  # cannot confirm; try placing again
                rows = hist.get("result", {}).get("list", [])
                if rows:
                    return {
                        "status": "already_placed",
                        "order_id": rows[0]["orderId"],
                        "order_link_id": link_id,
                    }
                # not on the exchange -> safe to place again
        return {"status": "failed", "order_id": None, "order_link_id": link_id}

    def get_position(self) -> dict | None:
        """Current linear position for the symbol, or None if flat."""
        resp = self._request(
            "get_positions",
            category="linear",
            symbol=self.symbol,
            settleCoin="USDT",
        )
        rows = resp.get("result", {}).get("list", [])
        for row in rows:
            if row.get("symbol") == self.symbol and float(row.get("size", 0) or 0) != 0:
                return {
                    "side": row.get("side"),
                    "size": float(row.get("size")),
                    "entry_price": float(row.get("avgPrice")),
                    "unrealised_pnl": float(row.get("unrealisedPnl") or 0),
                }
        return None

    def get_equity(self) -> float:
        """Unified account total equity in USDT."""
        resp = self._request("get_wallet_balance", accountType="UNIFIED")
        total = resp["result"]["list"][0].get("totalEquity")
        return float(total)

    def cancel_all(self) -> None:
        self._request("cancel_all_orders", category="linear", symbol=self.symbol)

    # ---------------------------------------------------------- instrument
    def get_instruments_info(self) -> dict:
        """Qty precision + min-order constraints for the symbol (fetched once, cached)."""
        if self._instruments is None:
            resp = self._request("get_instruments_info", category="linear", symbol=self.symbol)
            item = resp["result"]["list"][0]
            lot = item.get("lotSizeFilter", {})
            self._instruments = {
                "qty_step": float(lot.get("qtyStep", "1") or 1),
                "min_order_qty": float(lot.get("minOrderQty", "0") or 0),
                "min_notional": float(lot.get("minNotionalValue", "0") or 0),
            }
        return self._instruments

    def sanitize_qty(self, qty: float, entry_price: float) -> tuple[float, list[str]]:
        """Round qty DOWN to the symbol's qtyStep; reject below min qty / min notional.

        Returns (qty, reasons); reasons non-empty means the order must NOT be sent.
        """
        info = self.get_instruments_info()
        step = info["qty_step"]
        rounded = math.floor(qty / step + 1e-9) * step
        reasons: list[str] = []
        if rounded < info["min_order_qty"]:
            reasons.append(f"qty {rounded} below min order qty {info['min_order_qty']}")
        notional = rounded * entry_price
        if info["min_notional"] > 0 and notional < info["min_notional"]:
            reasons.append(f"notional {notional:.6f} below min notional {info['min_notional']}")
        return rounded, reasons

    def setup(self, leverage: int) -> None:
        """One-time linear-USDT perp setup: one-way position mode + leverage."""
        try:
            self._request("switch_position_mode", category="linear", symbol=self.symbol, mode=0)
        except RuntimeError:
            pass  # already one-way: bybit rejects the no-op switch; not fatal
        self._request(
            "set_leverage",
            category="linear",
            symbol=self.symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )

    # -------------------------------------------------- Phase 9 Live Execution
    def set_trading_stop(
        self,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        tp_trigger_by: str = "LastPrice",
        sl_trigger_by: str = "LastPrice",
    ) -> dict:
        """Attach exchange-native position-level Stop Loss and/or Take Profit (F6)."""
        kwargs: dict[str, Any] = {
            "category": "linear",
            "symbol": self.symbol,
            "positionIdx": 0,  # One-Way mode
            "tpslMode": "Full",
            "tpTriggerBy": tp_trigger_by,
            "slTriggerBy": sl_trigger_by,
        }
        if stop_loss is not None:
            kwargs["stopLoss"] = f"{stop_loss:.4f}"
        if take_profit is not None:
            kwargs["takeProfit"] = f"{take_profit:.4f}"

        return self._request("set_trading_stop", **kwargs)

    def cancel_trading_stop(self) -> dict:
        """Clear active exchange-native stop loss and take profit."""
        return self._request(
            "set_trading_stop",
            category="linear",
            symbol=self.symbol,
            positionIdx=0,
            stopLoss="0",
            takeProfit="0",
            tpslMode="Full",
        )

    def get_executions(self, order_link_id: str | None = None, limit: int = 10) -> list[dict]:
        """Fetch actual execution fills from the exchange (F7, F20)."""
        kwargs: dict[str, Any] = {
            "category": "linear",
            "symbol": self.symbol,
            "limit": limit,
        }
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id

        resp = self._request("get_executions", **kwargs)
        raw_list = resp.get("result", {}).get("list", [])
        executions: list[dict] = []
        for item in raw_list:
            executions.append(
                {
                    "exec_id": item.get("execId"),
                    "order_id": item.get("orderId"),
                    "order_link_id": item.get("orderLinkId"),
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "exec_price": float(item.get("execPrice", 0) or 0),
                    "exec_qty": float(item.get("execQty", 0) or 0),
                    "exec_fee": float(item.get("execFee", 0) or 0),
                    "exec_type": item.get("execType"),
                    "exec_time_ms": int(item.get("execTime", 0) or 0),
                }
            )
        return executions

    def get_closed_pnl(self, limit: int = 10) -> list[dict]:
        """Fetch realized PnL and funding deductions for closed positions (F7, F20)."""
        resp = self._request(
            "get_closed_pnl",
            category="linear",
            symbol=self.symbol,
            limit=limit,
        )
        raw_list = resp.get("result", {}).get("list", [])
        records: list[dict] = []
        for item in raw_list:
            records.append(
                {
                    "order_id": item.get("orderId"),
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "qty": float(item.get("qty", 0) or 0),
                    "entry_price": float(item.get("avgEntryPrice", 0) or 0),
                    "exit_price": float(item.get("avgExitPrice", 0) or 0),
                    "closed_pnl": float(item.get("closedPnl", 0) or 0),
                    "created_time_ms": int(item.get("createdTime", 0) or 0),
                }
            )
        return records

    def server_time_ms(self) -> int:
        """Fetch Bybit server time in milliseconds (F21)."""
        resp = self._request("get_server_time")
        # Bybit V5 returns time in seconds or ms in 'time' field
        t = resp.get("time") or resp.get("result", {}).get("timeSecond")
        if t is not None:
            val = int(t)
            return val if val > 1e11 else val * 1000
        return int(time.time() * 1000)
