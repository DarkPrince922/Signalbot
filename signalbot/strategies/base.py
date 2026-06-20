"""Abstract strategy base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..core.types import Signal


class Strategy(ABC):
    """Base for all plug-in strategies.

    Subclasses define `name`, `timeframe` and `default_params` and implement
    `generate`. `generate` receives a slice of OHLCV up to and including the
    last CLOSED candle and must only look at the past (no future rows).
    """

    name: str = "base"
    timeframe: str = "1h"
    default_params: dict = {}

    def __init__(self, params: dict | None = None) -> None:
        merged = dict(self.default_params)
        if params:
            merged.update(params)
        self.params = merged

    @property
    def warmup(self) -> int:
        """Minimum number of candles required before signals are meaningful."""
        return 50

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> Signal | None:
        """Return a Signal for the last row of `df`, or None.

        Must be time-causal: it is given history up to the latest closed candle
        and must never reference rows beyond the last one.
        """
