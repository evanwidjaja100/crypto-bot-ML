"""Unit tests for Heartbeat monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.monitoring.heartbeat import Heartbeat


def test_heartbeat_noop_when_unconfigured():
    hb = Heartbeat("")
    assert not hb.is_configured
    assert not hb.ping()
    assert not hb.start()
    assert not hb.fail()


def test_heartbeat_ping_and_signals():
    url = "https://hc-ping.com/12345678-abcd-1234-abcd-1234567890ab"
    hb = Heartbeat(url)
    assert hb.is_configured

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        assert hb.start() is True
        assert mock_post.call_args[0][0] == f"{url}/start"

        assert hb.ping("equity=10500.00") is True
        assert mock_post.call_args[0][0] == url

        assert hb.fail("Kill switch tripped") is True
        assert mock_post.call_args[0][0] == f"{url}/fail"


def test_heartbeat_resilience_to_errors():
    hb = Heartbeat("https://hc-ping.com/test")
    with patch("requests.post", side_effect=Exception("Network error")):
        assert hb.ping() is False
        assert hb.start() is False
        assert hb.fail() is False
