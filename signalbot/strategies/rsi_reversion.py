"""RSI mean-reversion: fade extremes, filtered by a long-term trend EMA."""
from __future__ import annotations

import pandas as pd

from ..core.types import Signal, SignalDirection
from .base import Strategy
from .indicators import atr, ema, rsi


class RsiReversion(Strategy):
    name = "rsi_reversion"
    timeframe = "1h"
    default_params = {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "trend_ema": 200,
        "atr_period": 14,
        "atr_mult": 2.0,
        "rr": 1.5,
    }

    @property
    def warmup(self) -> int:
        return max(self.params["trend_ema"], self.params["rsi_period"]) + 5

    def generate(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.warmup:
            return None

        r = rsi(df["close"], self.params["rsi_period"])
        trend = ema(df["close"], self.params["trend_ema"])
        atr_series = atr(df, self.params["atr_period"])

        r_now, r_prev = r.iloc[-1], r.iloc[-2]
        t_now = trend.iloc[-1]
        a = atr_series.iloc[-1]
        close = float(df["close"].iloc[-1])

        if pd.isna(r_prev) or pd.isna(t_now) or pd.isna(a) or a <= 0:
            return None

        os = self.params["oversold"]
        ob = self.params["overbought"]
        mult = self.params["atr_mult"]
        rr = self.params["rr"]

        # Long only above trend when RSI crosses back up out of oversold.
        long_setup = close > t_now and r_prev < os <= r_now
        # Short only below trend when RSI crosses back down out of overbought.
        short_setup = close < t_now and r_prev > ob >= r_now

        if long_setup:
            direction = SignalDirection.LONG
            stop = close - mult * a
            risk = close - stop
            tps = [close + rr * risk]
        elif short_setup:
            direction = SignalDirection.SHORT
            stop = close + mult * a
            risk = stop - close
            tps = [close - rr * risk]
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
            reason=f"RSI reverted from {'oversold' if long_setup else 'overbought'} "
            f"with trend filter (EMA{self.params['trend_ema']})",
            confidence=0.45,
            created_at=df["timestamp"].iloc[-1].to_pydatetime(),
            meta={"rsi": float(r_now), "atr": float(a)},
        )
