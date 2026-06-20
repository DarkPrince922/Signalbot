"""Donchian channel breakout with an ATR-based stop."""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import atr, donchian


class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    timeframe = "4h"
    default_params = {"channel": 20, "atr_period": 14, "atr_mult": 3.0, "rr": 2.0}

    @property
    def warmup(self) -> int:
        return max(self.params["channel"], self.params["atr_period"]) + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        upper, lower = donchian(df, self.params["channel"])
        atr_series = atr(df, self.params["atr_period"])

        u = upper.iloc[-1]
        low = lower.iloc[-1]
        a = atr_series.iloc[-1]
        close = float(df["close"].iloc[-1])
        high = float(df["high"].iloc[-1])
        low_px = float(df["low"].iloc[-1])

        if pd.isna(u) or pd.isna(low) or pd.isna(a) or a <= 0:
            return None

        mult = self.params["atr_mult"]
        rr = self.params["rr"]

        broke_up = high > u
        broke_down = low_px < low

        if broke_up:
            direction = SignalDirection.LONG
            stop = close - mult * a
            risk = close - stop
            tps = [close + rr * risk]
            reason = f"Broke {self.params['channel']}-bar high {u:.4f}, ATR trail"
        elif broke_down:
            direction = SignalDirection.SHORT
            stop = close + mult * a
            risk = stop - close
            tps = [close - rr * risk]
            reason = f"Broke {self.params['channel']}-bar low {low:.4f}, ATR trail"
        else:
            return None

        return Signal(
            symbol=df.attrs.get("symbol", "?"),
            timeframe=df.attrs.get("timeframe", self.timeframe),
            direction=direction,
            entry=close,
            stop_loss=float(stop),
            take_profits=[float(t) for t in tps],
            strategy=self.name,
            reason=reason,
            confidence=0.5,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"atr": float(a)},
        )
