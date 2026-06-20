"""Supertrend trend-following: enter on a Supertrend flip, stop at the line."""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import supertrend as supertrend_indicator


class Supertrend(Strategy):
    name = "supertrend"
    timeframe = "4h"
    default_params = {"period": 10, "multiplier": 3.0, "rr": 2.0}

    @property
    def warmup(self) -> int:
        return self.params["period"] + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        line, direction = supertrend_indicator(
            df, self.params["period"], self.params["multiplier"]
        )
        d_now, d_prev = int(direction.iloc[-1]), int(direction.iloc[-2])
        st_now = line.iloc[-1]
        close = float(df["close"].iloc[-1])

        if pd.isna(st_now):
            return None

        flip_up = d_prev <= 0 and d_now > 0
        flip_down = d_prev >= 0 and d_now < 0
        if not (flip_up or flip_down):
            return None

        rr = self.params["rr"]
        if flip_up:
            sig_dir = SignalDirection.LONG
            stop = float(st_now)
            risk = close - stop
            if risk <= 0:
                return None
            tps = [close + rr * risk]
        else:
            sig_dir = SignalDirection.SHORT
            stop = float(st_now)
            risk = stop - close
            if risk <= 0:
                return None
            tps = [close - rr * risk]

        return Signal(
            symbol=df.attrs.get("symbol", "?"),
            timeframe=df.attrs.get("timeframe", self.timeframe),
            direction=sig_dir,
            entry=close,
            stop_loss=stop,
            take_profits=[float(t) for t in tps],
            strategy=self.name,
            reason=f"Supertrend flipped {'bullish' if flip_up else 'bearish'}",
            confidence=0.5,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"supertrend": float(st_now)},
        )
