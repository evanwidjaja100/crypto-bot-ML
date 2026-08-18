"""Multi-channel notification engine (Discord, Telegram, and generic JSON webhooks).

Operates as a safe, silent no-op when unconfigured.
Guarantees zero crashes: network glitches or rate limits will never disrupt the trading runner.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..config import NotificationSettings

log = logging.getLogger("notify")

LEVEL_COLORS = {
    "INFO": 0x3498DB,  # Blue
    "SUCCESS": 0x2ECC71,  # Green
    "WARNING": 0xF39C12,  # Orange
    "CRITICAL": 0xE74C3C,  # Red
}


class Notifier:
    """Dispatches formatted alerts to webhooks and messaging services."""

    def __init__(self, cfg: NotificationSettings | None = None, *, timeout_s: float = 3.0) -> None:
        self.cfg = cfg or NotificationSettings()
        self.timeout_s = timeout_s

    @property
    def is_configured(self) -> bool:
        return bool(
            self.cfg.enabled
            and (
                self.cfg.webhook_url or (self.cfg.telegram_bot_token and self.cfg.telegram_chat_id)
            )
        )

    def send(
        self,
        title: str,
        message: str,
        level: str = "INFO",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send a notification. Returns True on success, False on no-op or failure."""
        if not self.is_configured:
            return False

        success = True
        # 1. Webhook (Discord / Slack / Generic)
        if self.cfg.webhook_url:
            success = self._send_webhook(title, message, level, metadata) and success

        # 2. Telegram Bot API
        if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
            success = self._send_telegram(title, message, level, metadata) and success

        return success

    def _send_webhook(
        self,
        title: str,
        message: str,
        level: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        url = self.cfg.webhook_url
        color = LEVEL_COLORS.get(level.upper(), 0x3498DB)

        payload: dict[str, Any]
        # Discord webhook format
        if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
            fields = [
                {"name": str(k), "value": str(v), "inline": True}
                for k, v in (metadata or {}).items()
            ]
            payload = {
                "embeds": [
                    {
                        "title": title,
                        "description": message,
                        "color": color,
                        "fields": fields[:25],
                    }
                ]
            }
        else:
            # Generic JSON webhook payload
            payload = {
                "title": title,
                "message": message,
                "level": level,
                "metadata": metadata or {},
                "ts_ms": int(time.time() * 1000),
            }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            return resp.status_code in (200, 204)
        except Exception as exc:
            log.warning("webhook dispatch failed (%s): %s", url, exc)
            return False

    def _send_telegram(
        self,
        title: str,
        message: str,
        level: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        meta_str = ""
        if metadata:
            meta_str = "\n" + "\n".join(f"• *{k}*: `{v}`" for k, v in metadata.items())
        text = f"*{title}* [{level}]\n{message}{meta_str}"

        payload = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            return resp.status_code == 200
        except Exception as exc:
            log.warning("telegram dispatch failed: %s", exc)
            return False

    # ----------------------------------------------------------- convenience
    def notify_fill(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: float,
        reason: str,
        fee: float = 0.0,
    ) -> bool:
        level = "SUCCESS" if action.startswith("OPEN") else "INFO"
        return self.send(
            title=f"Order Filled: {symbol} {action}",
            message=f"Filled {qty:.4f} {symbol} @ ${price:,.2f} ({reason})",
            level=level,
            metadata={
                "Symbol": symbol,
                "Action": action,
                "Price": f"${price:,.2f}",
                "Qty": f"{qty:.4f}",
                "Fee": f"${fee:.4f}",
                "Reason": reason,
            },
        )

    def notify_exit(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: float,
        reason: str,
        realized_pnl: float,
    ) -> bool:
        level = "SUCCESS" if realized_pnl >= 0 else "WARNING"
        return self.send(
            title=f"Position Closed: {symbol} ({reason})",
            message=f"Realized PnL: ${realized_pnl:+,.2f} ({action} {qty:.4f} @ ${price:,.2f})",
            level=level,
            metadata={
                "Symbol": symbol,
                "Realized PnL": f"${realized_pnl:+,.2f}",
                "Exit Reason": reason,
                "Price": f"${price:,.2f}",
                "Qty": f"{qty:.4f}",
            },
        )

    def notify_daily_loss_warning(
        self,
        context: str,
        current_loss_pct: float,
        max_loss_pct: float,
    ) -> bool:
        return self.send(
            title="⚠️ Daily Loss Warning",
            message=f"{context} daily loss is at {current_loss_pct:.2f}% (Limit: {max_loss_pct:.2f}%)",
            level="WARNING",
            metadata={
                "Context": context,
                "Current Loss": f"{current_loss_pct:.2f}%",
                "Daily Limit": f"{max_loss_pct:.2f}%",
            },
        )

    def notify_kill_switch(self, reason: str, tripped_at: str) -> bool:
        return self.send(
            title="🚨 CRITICAL: KILL SWITCH TRIPPED",
            message=f"Trading halted immediately: {reason}",
            level="CRITICAL",
            metadata={
                "Reason": reason,
                "Tripped At": tripped_at,
                "Action Required": "Inspect state & run reset_kill_switch.py",
            },
        )

    def notify_error_streak(self, streak: int, last_error: str) -> bool:
        return self.send(
            title="⚠️ API Error Streak",
            message=f"Bot has experienced {streak} consecutive failures. Last error: {last_error}",
            level="WARNING",
            metadata={"Consecutive Errors": streak, "Last Error": last_error[:200]},
        )

    def alert(
        self,
        title: str,
        message: str,
        level: str = "CRITICAL",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Generic high-priority alert dispatcher."""
        return self.send(title=title, message=message, level=level, metadata=metadata)
