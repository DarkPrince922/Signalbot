"""Bollinger Bands mean-reversion: fade closes outside the bands.

Long when price closes back inside from below the lower band, short when it
closes back inside from above the upper band. Optional trend filter via a long
EMA. Take-profit is the middle band (capped at an R-multiple), stop by ATR.
"""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import atr, bollinger, ema


class BollingerReversion(Strategy):
    name = "bollinger_reversion"
    timeframe = "1h"
    default_params = {
        "period": 20,
        "num_std": 2.0,
        "atr_period": 14,
        "atr_mult": 1.5,
        "rr": 1.5,
        "trend_ema": 0,  # 0 disables the trend filter
    }

    @property
    def warmup(self) -> int:
        return max(self.params["period"], self.params["atr_period"], self.params["trend_ema"]) + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        upper, middle, lower = bollinger(df["close"], self.params["period"], self.params["num_std"])
        atr_series = atr(df, self.params["atr_period"])

        c_now, c_prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        u_now, l_now, m_now = upper.iloc[-1], lower.iloc[-1], middle.iloc[-1]
        u_prev, l_prev = upper.iloc[-2], lower.iloc[-2]
        a = atr_series.iloc[-1]

        if pd.isna(u_prev) or pd.isna(l_prev) or pd.isna(a) or a <= 0:
            return None

        trend_ok_long = trend_ok_short = True
        if self.params["trend_ema"]:
            t = ema(df["close"], self.params["trend_ema"]).iloc[-1]
            if pd.isna(t):
                return None
            trend_ok_long = c_now > t
            trend_ok_short = c_now < t

        # re-entry from below lower band -> long; from above upper band -> short
        long_setup = c_prev < l_prev and c_now > l_now and trend_ok_long
        short_setup = c_prev > u_prev and c_now < u_now and trend_ok_short

        mult = self.params["atr_mult"]
        rr = self.params["rr"]

        if long_setup:
            direction = SignalDirection.LONG
            stop = c_now - mult * a
            risk = c_now - stop
            tp = min(float(m_now), c_now + rr * risk)
            tps = [tp]
            reason = "Close re-entered above lower Bollinger band"
        elif short_setup:
            direction = SignalDirection.SHORT
            stop = c_now + mult * a
            risk = stop - c_now
            tp = max(float(m_now), c_now - rr * risk)
            tps = [tp]
            reason = "Close re-entered below upper Bollinger band"
        else:
            return None

        # skip degenerate targets (TP on the wrong side of entry)
        if direction == SignalDirection.LONG and tps[0] <= c_now:
            return None
        if direction == SignalDirection.SHORT and tps[0] >= c_now:
            return None

        return Signal(
            symbol=df.attrs.get("symbol", "?"),
            timeframe=df.attrs.get("timeframe", self.timeframe),
            direction=direction,
            entry=c_now,
            stop_loss=float(stop),
            take_profits=[float(t) for t in tps],
            strategy=self.name,
            reason=reason,
            confidence=0.45,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"atr": float(a), "middle": float(m_now)},
        )
