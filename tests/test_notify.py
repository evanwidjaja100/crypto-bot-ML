"""Unit tests for Notifier: Discord, Telegram, and generic webhooks with safe no-op defaults."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import NotificationSettings
from src.monitoring.notify import Notifier


def test_notifier_noop_when_unconfigured():
    n = Notifier(NotificationSettings(enabled=True))
    assert not n.is_configured
    assert not n.send("Test", "No op message")
    assert not n.notify_fill("BTCUSDT", "OPEN_LONG", 50000.0, 0.1, "signal")
    assert not n.notify_exit("BTCUSDT", "CLOSE_LONG", 51000.0, 0.1, "take_profit", 100.0)
    assert not n.notify_daily_loss_warning("BTCUSDT", 1.5, 2.0)
    assert not n.notify_kill_switch("Tripped test", "2026-08-17")
    assert not n.notify_error_streak(3, "Timeout")


def test_notifier_discord_webhook():
    cfg = NotificationSettings(
        enabled=True,
        webhook_url="https://discord.com/api/webhooks/12345/abcdef",
    )
    n = Notifier(cfg)
    assert n.is_configured

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("requests.post", return_value=mock_resp) as mock_post:
        assert n.notify_fill("ETHUSDT", "OPEN_LONG", 3000.0, 1.0, "leader_breakout")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://discord.com/api/webhooks/12345/abcdef"
        assert "embeds" in kwargs["json"]
        assert kwargs["json"]["embeds"][0]["title"] == "Order Filled: ETHUSDT OPEN_LONG"


def test_notifier_telegram():
    cfg = NotificationSettings(
        enabled=True,
        telegram_bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        telegram_chat_id="987654321",
    )
    n = Notifier(cfg)
    assert n.is_configured

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        assert n.notify_exit("SOLUSDT", "CLOSE_LONG", 150.0, 10.0, "stop_loss", -25.0)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert (
            "api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/sendMessage" in args[0]
        )
        assert kwargs["json"]["chat_id"] == "987654321"


def test_notifier_exception_resilience():
    cfg = NotificationSettings(
        enabled=True,
        webhook_url="https://example.com/webhook",
    )
    n = Notifier(cfg)

    # Simulating connection error / timeout
    with patch("requests.post", side_effect=Exception("Connection timed out")):
        assert n.send("Title", "Message") is False
