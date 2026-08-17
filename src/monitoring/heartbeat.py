"""Dead-man heartbeat monitoring (Healthchecks.io / Uptime Kuma compatible).

Sends periodic liveness pings on completed candle processing cycles.
Silently no-ops if healthcheck_url is not configured.
"""

from __future__ import annotations

import logging

import requests

log = logging.getLogger("heartbeat")


class Heartbeat:
    """Sends liveness pings to an external monitoring endpoint."""

    def __init__(self, healthcheck_url: str = "", *, timeout_s: float = 3.0) -> None:
        self.url = healthcheck_url.strip()
        self.timeout_s = timeout_s

    @property
    def is_configured(self) -> bool:
        return bool(self.url)

    def ping(self, msg: str = "") -> bool:
        """Send regular liveness ping."""
        if not self.is_configured:
            return False
        try:
            resp = requests.post(
                self.url, data=msg.encode("utf-8") if msg else None, timeout=self.timeout_s
            )
            return resp.status_code == 200
        except Exception as exc:
            log.warning("heartbeat ping failed (%s): %s", self.url, exc)
            return False

    def start(self) -> bool:
        """Send job start signal (for Healthchecks.io)."""
        if not self.is_configured:
            return False
        url = f"{self.url.rstrip('/')}/start"
        try:
            resp = requests.post(url, timeout=self.timeout_s)
            return resp.status_code == 200
        except Exception as exc:
            log.warning("heartbeat start signal failed: %s", exc)
            return False

    def fail(self, error_msg: str = "") -> bool:
        """Send job failure alert signal (for Healthchecks.io)."""
        if not self.is_configured:
            return False
        url = f"{self.url.rstrip('/')}/fail"
        try:
            resp = requests.post(
                url, data=error_msg.encode("utf-8") if error_msg else None, timeout=self.timeout_s
            )
            return resp.status_code == 200
        except Exception as exc:
            log.warning("heartbeat fail signal failed: %s", exc)
            return False
