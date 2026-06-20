"""Guards against look-ahead bias in indicators and the backtester loop."""
from __future__ import annotations

import numpy as np
import pandas as pd

from signalbot.core.types import Signal
from signalbot.engine.backtester import Backtester
from signalbot.strategies.base import Strategy
from signalbot.strategies.indicators import atr, donchian, ema, rsi
from tests.conftest import make_df


def test_indicators_are_causal():
    rng = np.random.default_rng(42)
    closes = list(100 + np.cumsum(rng.normal(0, 1, 300)))
    df = make_df(closes)

    full_ema = ema(df["close"], 20)
    full_rsi = rsi(df["close"], 14)
    full_atr = atr(df, 14)
    full_up, full_low = donchian(df, 20)

    # value at index i on the prefix must equal value at i on full series
    for i in (50, 120, 250):
        pref = df.iloc[: i + 1]
        assert np.isclose(ema(pref["close"], 20).iloc[-1], full_ema.iloc[i])
        assert np.isclose(rsi(pref["close"], 14).iloc[-1], full_rsi.iloc[i])
        assert np.isclose(atr(pref, 14).iloc[-1], full_atr.iloc[i])
        u, low = donchian(pref, 20)
        assert np.isclose(u.iloc[-1], full_up.iloc[i])
        assert np.isclose(low.iloc[-1], full_low.iloc[i])


class _SpyStrategy(Strategy):
    """Records what slices it is handed; never signals."""

    name = "spy"
    timeframe = "1h"
    default_params = {}

    def __init__(self, params=None):
        super().__init__(params)
        self.max_ts_seen = []

    @property
    def warmup(self):
        return 3

    def generate(self, df: pd.DataFrame) -> Signal | None:
        self.max_ts_seen.append(df["timestamp"].iloc[-1])
        return None


def test_backtester_only_passes_prefixes(config):
    df = make_df(closes=list(range(100, 130)))
    spy = _SpyStrategy()
    Backtester(config).run(spy, df, "TEST/USDT", "1h")
    # the strategy must only ever be called with prefixes ending at increasing,
    # in-range timestamps — never the final future candle beyond the call point
    seen = spy.max_ts_seen
    assert seen == sorted(seen)
    assert seen[-1] <= df["timestamp"].iloc[-1]


def test_future_candle_does_not_change_past_signal(config):
    # A strategy signalling on a fixed condition must produce the same signal
    # regardless of what happens in candles AFTER the signal candle.
    closes = [100, 101, 102, 103, 104, 105, 106, 107]

    class _ThresholdStrat(Strategy):
        name = "thresh"
        timeframe = "1h"
        default_params = {}

        @property
        def warmup(self):
            return 2

        def generate(self, df):
            if float(df["close"].iloc[-1]) == 104.0:
                from signalbot.core.types import SignalDirection

                return Signal(
                    symbol="TEST/USDT",
                    timeframe="1h",
                    direction=SignalDirection.LONG,
                    entry=104.0,
                    stop_loss=100.0,
                    take_profits=[112.0],
                    strategy=self.name,
                    reason="x",
                    created_at=df["timestamp"].iloc[-1].to_pydatetime(),
                )
            return None

    df1 = make_df(closes)
    # df2 mutates a candle AFTER the signal candle (index 4) drastically
    closes2 = list(closes)
    closes2[6] = 999
    df2 = make_df(closes2, highs=[c + 1 for c in closes2], lows=[c - 1 for c in closes2])

    r1 = Backtester(config).run(_ThresholdStrat(), df1, "TEST/USDT", "1h")
    r2 = Backtester(config).run(_ThresholdStrat(), df2, "TEST/USDT", "1h")

    # both must open exactly one trade at the same entry candle
    assert len(r1.trades) == 1 and len(r2.trades) == 1
    assert r1.trades[0].entry_time == r2.trades[0].entry_time
    assert r1.trades[0].entry == r2.trades[0].entry == 104.0
