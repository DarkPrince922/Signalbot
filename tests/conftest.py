"""Shared test fixtures and synthetic data helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalbot.core.config import AppConfig


def make_df(closes, highs=None, lows=None, opens=None, start="2024-01-01", tf="1h") -> pd.DataFrame:
    closes = list(closes)
    n = len(closes)
    opens = list(opens) if opens is not None else closes
    highs = list(highs) if highs is not None else [max(o, c) for o, c in zip(opens, closes)]
    lows = list(lows) if lows is not None else [min(o, c) for o, c in zip(opens, closes)]
    ts = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n),
        }
    )
    df.attrs["symbol"] = "TEST/USDT"
    df.attrs["timeframe"] = tf
    return df


@pytest.fixture
def config() -> AppConfig:
    cfg = AppConfig()
    cfg.costs.taker_fee_bps = 0.0
    cfg.costs.slippage_bps = 0.0
    cfg.costs.include_funding = False
    cfg.account.risk_per_trade_pct = 1.0
    cfg.backtest.warmup = 3
    cfg.backtest.fill_mode = "close"
    cfg.tracker.expire_after_bars = 50
    return cfg
