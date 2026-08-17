"""Unit tests for BasketRunner and multi-symbol paper execution."""

from __future__ import annotations

import pandas as pd

from src.config import load_settings
from src.data_ingestion.candle_downloader import CandleStore
from src.data_ingestion.funding_downloader import FundingStore
from src.runner.basket_runner import BasketRunner
from tests.conftest import make_candles


class FakeClient:
    def __init__(self, candles_by_symbol: dict[str, pd.DataFrame]):
        self.candles = candles_by_symbol

    def server_time_ms(self) -> int:
        return 1_700_000_000_000 + 400 * 3600_000

    def fetch_candles(self, symbol: str, interval: str, limit: int = 10) -> pd.DataFrame:
        df = self.candles.get(symbol, pd.DataFrame())
        return df.tail(limit).reset_index(drop=True)


def test_basket_runner_warmup_and_snapshot(tmp_path):
    settings = load_settings()
    store = CandleStore(tmp_path, network="mainnet")
    funding_store = FundingStore(tmp_path, network="mainnet")

    # Generate synthetic candles for 2 symbols
    df_btc = make_candles(100, seed=1, interval_ms=3600_000)
    df_eth = make_candles(100, seed=2, interval_ms=3600_000)
    store.write(df_btc, "BTCUSDT", "60")
    store.write(df_eth, "ETHUSDT", "60")

    client = FakeClient({"BTCUSDT": df_btc, "ETHUSDT": df_eth})

    runner = BasketRunner(
        settings=settings,
        symbols=["BTCUSDT", "ETHUSDT"],
        client=client,
        store=store,
        funding_store=funding_store,
        strategy_mode="cross_sectional",
        journal_dir=tmp_path / "journal",
        state_path=tmp_path / "basket_state.json",
        warmup_bars=50,
    )

    runner.warmup()
    assert len(runner.ctx["BTCUSDT"]) == 50
    assert len(runner.ctx["ETHUSDT"]) == 50
    assert runner.brokers["BTCUSDT"].equity() == 5000.0
    assert runner.brokers["ETHUSDT"].equity() == 5000.0

    # Save snapshot and restore
    runner._save_snapshot()
    assert (tmp_path / "basket_state.json").exists()

    runner2 = BasketRunner(
        settings=settings,
        symbols=["BTCUSDT", "ETHUSDT"],
        client=client,
        store=store,
        funding_store=funding_store,
        strategy_mode="cross_sectional",
        journal_dir=tmp_path / "journal",
        state_path=tmp_path / "basket_state.json",
        warmup_bars=50,
    )
    assert runner2._restore_snapshot() is True
    assert runner2.brokers["BTCUSDT"].equity() == 5000.0


def test_basket_runner_tick_execution(tmp_path):
    settings = load_settings()
    store = CandleStore(tmp_path, network="mainnet")

    df_btc = make_candles(100, seed=1, interval_ms=3600_000)
    df_sol = make_candles(100, seed=3, drift=0.001, interval_ms=3600_000)
    store.write(df_btc, "BTCUSDT", "60")
    store.write(df_sol, "SOLUSDT", "60")

    client = FakeClient({"BTCUSDT": df_btc, "SOLUSDT": df_sol})

    runner = BasketRunner(
        settings=settings,
        symbols=["BTCUSDT", "SOLUSDT"],
        client=client,
        store=store,
        strategy_mode="cross_sectional",
        journal_dir=tmp_path / "journal",
        state_path=tmp_path / "basket_state.json",
        warmup_bars=40,
    )
    runner.warmup()

    # Create continuous new closed candle for each asset
    last_btc = df_btc.iloc[-1]
    last_sol = df_sol.iloc[-1]
    new_ts = int(last_btc["ts_ms"]) + 3600_000

    new_btc = pd.DataFrame(
        [
            {
                "ts_ms": new_ts,
                "open": float(last_btc["close"]),
                "high": float(last_btc["close"]) * 1.002,
                "low": float(last_btc["close"]) * 0.998,
                "close": float(last_btc["close"]) * 1.001,
                "volume": 10.0,
                "turnover": 1000.0,
            }
        ]
    )
    new_sol = pd.DataFrame(
        [
            {
                "ts_ms": new_ts,
                "open": float(last_sol["close"]),
                "high": float(last_sol["close"]) * 1.005,
                "low": float(last_sol["close"]) * 0.999,
                "close": float(last_sol["close"]) * 1.004,
                "volume": 20.0,
                "turnover": 1000.0,
            }
        ]
    )

    client.candles["BTCUSDT"] = pd.concat([df_btc, new_btc], ignore_index=True)
    client.candles["SOLUSDT"] = pd.concat([df_sol, new_sol], ignore_index=True)

    res = runner.tick(now_ms=new_ts + 3600_000)
    assert res["new_bars"] == 2
    assert res["total_equity"] > 0
