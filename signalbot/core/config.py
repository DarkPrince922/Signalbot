"""Configuration loading: .env (secrets) + config.yaml (behaviour)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets and environment-specific values loaded from .env / environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_user_id: int = Field(default=0, alias="TELEGRAM_USER_ID")
    exchange_id: str = Field(default="binanceusdm", alias="EXCHANGE_ID")


class DataConfig(BaseModel):
    exchange_id: str = "binanceusdm"
    cache_dir: str = "./data_cache"


class AccountConfig(BaseModel):
    size_usdt: float = 1000.0
    risk_per_trade_pct: float = 1.0


class CostsConfig(BaseModel):
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 2.0
    include_funding: bool = False
    funding_rate_bps_per_8h: float = 1.0


class StrategyConfig(BaseModel):
    enabled: bool = True
    timeframe: str = "1h"
    params: dict = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    oos_split: float = 0.7
    fill_mode: str = "next_open"  # next_open | close
    warmup: int = 250
    max_concurrent_per_pair: int = 1


class TrackerConfig(BaseModel):
    expire_after_bars: int = 48


class EngineConfig(BaseModel):
    poll_offset_seconds: int = 15


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/signalbot.log"


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    account: AccountConfig = Field(default_factory=AccountConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    pairs: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # populated after load, not from yaml
    secrets: Secrets = Field(default_factory=Secrets)


def load_config(path: str | os.PathLike = "config.yaml") -> AppConfig:
    """Load config.yaml, overlay .env secrets, return a validated AppConfig."""
    raw: dict = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}

    secrets = Secrets()
    cfg = AppConfig(**raw)
    cfg.secrets = secrets

    # .env exchange_id wins over yaml default so deployment can switch venues
    if secrets.exchange_id:
        cfg.data.exchange_id = secrets.exchange_id
    return cfg
