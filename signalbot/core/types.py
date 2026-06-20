"""Core domain types shared across the project."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Outcome(str, Enum):
    """Lifecycle status of a virtual trade / signal."""

    OPEN = "OPEN"
    WIN = "WIN"
    LOSS = "LOSS"
    EXPIRED = "EXPIRED"


class Mode(str, Enum):
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Signal:
    """A trading signal produced by a strategy. This is a recommendation only."""

    symbol: str
    timeframe: str
    direction: SignalDirection
    entry: float
    stop_loss: float
    take_profits: list[float]
    strategy: str
    reason: str
    created_at: datetime = field(default_factory=utcnow)
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def risk_per_unit(self) -> float:
        """Absolute price distance between entry and stop."""
        return abs(self.entry - self.stop_loss)

    @property
    def rr(self) -> float | None:
        """Reward-to-risk of the first take-profit level."""
        if not self.take_profits or self.risk_per_unit == 0:
            return None
        reward = abs(self.take_profits[0] - self.entry)
        return reward / self.risk_per_unit

    def dedup_key(self) -> str:
        """Stable identifier for one signal on one candle (used for de-dup)."""
        ts = int(self.created_at.timestamp())
        return f"{self.strategy}:{self.symbol}:{self.timeframe}:{self.direction.value}:{ts}"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
