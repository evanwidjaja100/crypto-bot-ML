"""run_bot main() loop tests: transient failures survive, streaks halt (F5)."""
from __future__ import annotations

import pandas as pd
import requests

import scripts.run_bot as run_bot
from src.features.manifest import feature_set_id
from src.features.pipeline import build_feature_frame
from test_runner import IV, START, FixedModel, make_settings


def make_long_frame(n=2600):
    rows = [
        [START + i * IV, 100.0, 100.5, 99.5, 100.0, 10.0, 1000.0]
        for i in range(n)
    ]
    return pd.DataFrame(
        rows, columns=["ts_ms", "open", "high", "low", "close", "volume", "turnover"]
    )


class FailingClient:
    """Market data is down: every call raises."""

    def server_time_ms(self):
        raise requests.exceptions.ConnectionError("down")

    def fetch_candles(self, *a, **k):
        raise requests.exceptions.ConnectionError("down")


def _patch_main(tmp_path, monkeypatch, settings):
    """Point run_bot at a hermetic tmp pipeline with a working model."""
    df = make_long_frame()
    frame, cols = build_feature_frame(df, settings.features)
    fid = feature_set_id(settings.features.version, cols, settings.features.model_dump())
    meta = {"model_id": "t", "framework": "lgbm", "feature_set_id": fid}

    monkeypatch.setattr(run_bot, "load_settings", lambda: settings)
    monkeypatch.setattr(run_bot, "latest_model", lambda artifacts: (FixedModel([0.2, 0.1, 0.7]), meta))
    return df


def _main_args(tmp_path, *extra):
    return [
        "--journal-dir", str(tmp_path / "runner"),
        "--state-path", str(tmp_path / "runner" / "state.json"),
        *extra,
    ]


def test_once_with_failing_feed_returns_zero_not_three(tmp_path, monkeypatch):
    """One network blip must not kill the bot: --once reports and exits 0."""
    settings = make_settings(tmp_path)
    df = _patch_main(tmp_path, monkeypatch, settings)
    monkeypatch.setattr(run_bot, "BybitClient", lambda testnet: FailingClient())

    from src.data_ingestion.candle_downloader import CandleStore
    CandleStore(settings.data.data_dir).write(df, settings.symbol, settings.interval)

    assert run_bot.main(_main_args(tmp_path, "--once")) == 0


def test_api_error_streak_halts_with_exit_3(tmp_path, monkeypatch):
    """Five consecutive tick failures trip the kill switch -> exit 3."""
    settings = make_settings(tmp_path)
    df = _patch_main(tmp_path, monkeypatch, settings)
    monkeypatch.setattr(run_bot, "BybitClient", lambda testnet: FailingClient())

    from src.data_ingestion.candle_downloader import CandleStore
    CandleStore(settings.data.data_dir).write(df, settings.symbol, settings.interval)

    # 0 is falsy for --sleep-secs -> use non-zero
    assert run_bot.main(_main_args(tmp_path, "--sleep-secs", "0.001")) == 3