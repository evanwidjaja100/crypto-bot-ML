"""Typed configuration: settings.yaml + environment overrides.

Fails fast on invalid config (bad mode, unknown interval, live without confirmation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_MODES = ("backtest", "paper", "testnet", "live")
ALLOWED_INTERVALS = ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")


class DataSettings(BaseModel):
    data_dir: str = "./data"
    history_days: int = 365
    chunk_days: int = 30
    page_size: int = 1000
    max_bar_move_pct: float = 25.0  # blocking bar-to-bar close-move threshold; None disables


class FeatureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "v2"  # v2: ATR normalized by close (scale-free model feature)
    rsi_period: int = 14
    atr_period: int = 14
    ema_periods: list[int] = [10, 30, 90]
    sma_ratio_periods: list[int] = [20, 50]
    return_periods: list[int] = [1, 3, 5, 10, 20]
    vol_periods: list[int] = [5, 10, 20]
    vol_zscore_period: int = 20


class LabelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: int = 5
    threshold_sigma: float = 0.5
    threshold_window: int = 100
    min_abs_threshold: float = 1e-4


class LgbmSettings(BaseModel):
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50


class WalkForwardSettings(BaseModel):
    n_splits: int = 5
    min_train_rows: int = 5000


class ModelSettings(BaseModel):
    seed: int = 42
    lgbm: LgbmSettings = Field(default_factory=LgbmSettings)
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    min_promote: dict[str, float] = Field(default_factory=lambda: {"test_accuracy": 0.34})


class RiskSettings(BaseModel):
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 1
    max_notional_pct: float = 20.0
    leverage_cap: int = 3
    max_api_error_streak: int = 5
    cooldown_bars: int = 5
    min_hold_bars: int = 3
    max_hold_bars: int = 60
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0


class ExecutionSettings(BaseModel):
    taker_fee: float = 0.00055
    maker_fee: float = 0.0002
    slippage_bps: float = 2.0
    order_timeout_s: int = 30
    max_order_retries: int = 3


class LoggingSettings(BaseModel):
    log_dir: str = "./logs"
    log_level: str = "INFO"


class LiveSettings(BaseModel):
    confirm_phrase: str = "ENABLE-LIVE"


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    bot_mode: str = ""
    bot_live_confirm: str = ""
    data_dir: str = ""
    log_level: str = ""
    bot_strategy_confidence_long: float = 0.0
    bot_strategy_confidence_short: float = 0.0
    bot_strategy_confidence_reverse: float = 0.0


class StrategySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_long: float = 0.55
    confidence_short: float = 0.55
    confidence_reverse: float = 0.60


class BacktestSettings(BaseModel):
    initial_equity: float = 10_000.0
    funding_rate: float = 0.0001


class Settings(BaseModel):
    mode: str = "paper"
    symbol: str = "BTCUSDT"
    interval: str = "5"
    data: DataSettings = Field(default_factory=DataSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    labels: LabelSettings = Field(default_factory=LabelSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)
    env: EnvSettings = Field(default_factory=EnvSettings)

    @model_validator(mode="after")
    def _validate_mode_and_interval(self) -> "Settings":
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"mode={self.mode!r} not in {ALLOWED_MODES}")
        if self.interval not in ALLOWED_INTERVALS:
            raise ValueError(f"interval={self.interval!r} not in {ALLOWED_INTERVALS}")
        if self.mode == "live" and self.env.bot_live_confirm != self.live.confirm_phrase:
            raise ValueError(
                f"live mode requires BOT_LIVE_CONFIRM={self.live.confirm_phrase!r} in .env"
            )
        return self

    def needs_credentials(self) -> bool:
        return self.mode in ("testnet", "live")

    def check_credentials(self) -> None:
        if not self.needs_credentials():
            return
        if not self.env.bybit_api_key or not self.env.bybit_api_secret:
            raise ValueError(
                f"mode={self.mode} requires BYBIT_API_KEY and BYBIT_API_SECRET in .env"
            )


def load_settings(yaml_path: str | Path = ROOT / "config" / "settings.yaml") -> Settings:
    """Load settings.yaml, overlay environment overrides, return validated Settings."""
    path = Path(yaml_path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}
    env = EnvSettings()

    if env.bot_mode:
        raw["mode"] = env.bot_mode
    if env.data_dir:
        raw.setdefault("data", {})["data_dir"] = env.data_dir
    if env.log_level:
        raw.setdefault("logging", {})["log_level"] = env.log_level
    strategy_ov = {}
    if env.bot_strategy_confidence_long > 0.0:
        strategy_ov["confidence_long"] = env.bot_strategy_confidence_long
    if env.bot_strategy_confidence_short > 0.0:
        strategy_ov["confidence_short"] = env.bot_strategy_confidence_short
    if env.bot_strategy_confidence_reverse > 0.0:
        strategy_ov["confidence_reverse"] = env.bot_strategy_confidence_reverse
    if strategy_ov:
        raw.setdefault("strategy", {}).update(strategy_ov)
    raw["env"] = env.model_dump()

    return Settings.model_validate(raw)
