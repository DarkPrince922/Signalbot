"""EMA crossover strategy: fast/slow EMA cross, ATR stop, R-multiple target."""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import atr, ema


class EmaCross(Strategy):
    name = "ema_cross"
    timeframe = "1h"
    default_params = {"fast": 20, "slow": 50, "atr_period": 14, "atr_mult": 2.0, "rr": 2.0}

    @property
    def warmup(self) -> int:
        return max(self.params["slow"], self.params["atr_period"]) + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        fast = ema(df["close"], self.params["fast"])
        slow = ema(df["close"], self.params["slow"])
        atr_series = atr(df, self.params["atr_period"])

        # use the last closed candle (index -1) and the one before it
        f_now, f_prev = fast.iloc[-1], fast.iloc[-2]
        s_now, s_prev = slow.iloc[-1], slow.iloc[-2]
        a = atr_series.iloc[-1]
        close = float(df["close"].iloc[-1])

        if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(a) or a <= 0:
            return None

        cross_up = f_prev <= s_prev and f_now > s_now
        cross_down = f_prev >= s_prev and f_now < s_now
        if not (cross_up or cross_down):
            return None

        mult = self.params["atr_mult"]
        rr = self.params["rr"]
        if cross_up:
            direction = SignalDirection.LONG
            stop = close - mult * a
            risk = close - stop
            tps = [close + rr * risk]
        else:
            direction = SignalDirection.SHORT
            stop = close + mult * a
            risk = stop - close
            tps = [close - rr * risk]

        return Signal(
            symbol=df.attrs.get("symbol", "?"),
            timeframe=df.attrs.get("timeframe", self.timeframe),
            direction=direction,
            entry=close,
            stop_loss=float(stop),
            take_profits=[float(t) for t in tps],
            strategy=self.name,
            reason=f"EMA{self.params['fast']} crossed EMA{self.params['slow']} "
            f"{'up' if cross_up else 'down'}, ATR stop",
            confidence=0.5,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"atr": float(a)},
        )
