"""Abstract market-data provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Timeframe -> milliseconds. Used to detect/skip the unclosed candle.
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def timeframe_ms(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_MS[timeframe]


class DataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return OHLCV with columns timestamp, open, high, low, close, volume.

        `timestamp` is a pandas datetime (UTC). Only closed candles are returned.
        """

    @abstractmethod
    def latest_closed_candle(self, symbol: str, timeframe: str) -> pd.Series:
        """Return the most recent fully-closed candle as a Series."""
