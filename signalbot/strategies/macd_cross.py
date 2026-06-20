"""MACD crossover: trade MACD line crossing its signal line, ATR stop, R target."""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import atr, macd


class MacdCross(Strategy):
    name = "macd_cross"
    timeframe = "1h"
    default_params = {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "atr_period": 14,
        "atr_mult": 2.0,
        "rr": 2.0,
    }

    @property
    def warmup(self) -> int:
        return self.params["slow"] + self.params["signal"] + self.params["atr_period"] + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        macd_line, signal_line, _ = macd(
            df["close"], self.params["fast"], self.params["slow"], self.params["signal"]
        )
        atr_series = atr(df, self.params["atr_period"])

        m_now, m_prev = macd_line.iloc[-1], macd_line.iloc[-2]
        s_now, s_prev = signal_line.iloc[-1], signal_line.iloc[-2]
        a = atr_series.iloc[-1]
        close = float(df["close"].iloc[-1])

        if pd.isna(m_prev) or pd.isna(s_prev) or pd.isna(a) or a <= 0:
            return None

        cross_up = m_prev <= s_prev and m_now > s_now
        cross_down = m_prev >= s_prev and m_now < s_now
        if not (cross_up or cross_down):
            return None

        mult = self.params["atr_mult"]
        rr = self.params["rr"]
        if cross_up:
            direction = SignalDirection.LONG
            stop = close - mult * a
            tps = [close + rr * (close - stop)]
        else:
            direction = SignalDirection.SHORT
            stop = close + mult * a
            tps = [close - rr * (stop - close)]

        return Signal(
            symbol=df.attrs.get("symbol", "?"),
            timeframe=df.attrs.get("timeframe", self.timeframe),
            direction=direction,
            entry=close,
            stop_loss=float(stop),
            take_profits=[float(t) for t in tps],
            strategy=self.name,
            reason=f"MACD line crossed signal {'up' if cross_up else 'down'}, ATR stop",
            confidence=0.5,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"macd": float(m_now), "signal": float(s_now), "atr": float(a)},
        )
