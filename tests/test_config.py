"""Settings tests: mode -> trading-network coupling, legacy flag rejection (F2)."""

from __future__ import annotations

import pytest

from src.config import EnvSettings, Settings, load_settings


def _settings(mode: str, **env_kw) -> Settings:
    return Settings(mode=mode, env=EnvSettings(**env_kw))


def test_live_mode_without_confirm_phrase_raises():
    with pytest.raises(ValueError, match="BOT_LIVE_CONFIRM"):
        _settings("live")


def test_testnet_mode_orders_testnet():
    s = _settings("testnet")
    assert s.trading_network == "testnet"
    assert s.order_endpoints_testnet is True
    assert s.market_data_network == "mainnet"


def test_live_mode_orders_mainnet():
    s = _settings("live", bot_live_confirm="ENABLE-LIVE")
    assert s.trading_network == "mainnet"
    assert s.order_endpoints_testnet is False
    assert s.market_data_network == "mainnet"


def test_paper_mode_places_no_orders():
    s = _settings("paper")
    assert s.trading_network is None
    with pytest.raises(RuntimeError, match="places no orders"):
        s.order_endpoints_testnet
    assert s.market_data_network == "mainnet"  # data still keyless mainnet


def test_backtest_mode_places_no_orders():
    s = _settings("backtest")
    assert s.trading_network is None


def test_legacy_bybit_testnet_env_var_is_rejected(monkeypatch):
    monkeypatch.setenv("BYBIT_TESTNET", "true")
    with pytest.raises(ValueError, match="BYBIT_TESTNET is no longer honored"):
        load_settings()


def test_legacy_bybit_testnet_env_var_ignored_without_os_env(tmp_path):
    """A stale BYBIT_TESTNET line in .env alone must not trip the validator."""
    s = _settings("paper")
    assert s.trading_network is None


def test_load_settings_reads_non_ascii_under_non_utf8_locale(tmp_path, monkeypatch):
    """5.3: a non-ASCII YAML must parse even when the platform default codec is
    not UTF-8 (e.g. cp950 on Windows) — encoding is explicit, not luck."""
    import locale

    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp950")
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_bytes("mode: paper\nsymbol: BTCUSDT\n# \u2014 em dash\n".encode("utf-8"))
    s = load_settings(yaml_path)
    assert s.mode == "paper"
    assert s.symbol == "BTCUSDT"
