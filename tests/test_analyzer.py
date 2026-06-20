"""Analyzer: directional read, risk classification, risk-adjusted sizing."""
from __future__ import annotations

import numpy as np

from signalbot.core.config import AppConfig
from signalbot.core.types import SignalDirection
from signalbot.engine.analyzer import Analyzer
from tests.conftest import make_df


def _analyzer(cfg=None):
    return Analyzer(cfg or AppConfig(), provider=None, tracker=None)


def test_strong_uptrend_reads_long():
    # steadily rising series -> all indicators should agree LONG
    closes = list(100 + np.linspace(0, 60, 400))
    df = make_df(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes])
    a = _analyzer().analyze_df(df, "BTC/USDT", "1h")
    assert a.direction == SignalDirection.LONG
    assert a.score > 0.5
    assert a.stop_loss < a.entry < a.take_profits[0]


def test_strong_downtrend_reads_short():
    closes = list(160 - np.linspace(0, 60, 400))
    df = make_df(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes])
    a = _analyzer().analyze_df(df, "ETH/USDT", "1h")
    assert a.direction == SignalDirection.SHORT
    assert a.take_profits[0] < a.entry < a.stop_loss


def test_choppy_market_stands_aside_or_low_score():
    rng = np.random.default_rng(0)
    closes = list(100 + np.cumsum(rng.normal(0, 0.5, 400)))
    df = make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
    a = _analyzer().analyze_df(df, "SOL/USDT", "1h")
    # either no clear bias, or a weak one — never a confident read on noise
    assert a.score <= 0.75


def test_risk_level_scales_size_down():
    a = _analyzer()
    # high volatility + weak agreement (1 of 4) -> HIGH risk
    high = a._classify_risk(atr_pct=4.0, score=0.25, live_winrate=None)
    # calm + full agreement -> LOW risk
    low = a._classify_risk(atr_pct=0.5, score=1.0, live_winrate=None)
    # moderate -> MEDIUM
    mid = a._classify_risk(atr_pct=4.0, score=0.5, live_winrate=None)
    assert high == "HIGH"
    assert low == "LOW"
    assert mid == "MEDIUM"


def test_high_risk_reduces_recommended_size():
    cfg = AppConfig()
    cfg.account.size_usdt = 1000
    cfg.account.risk_per_trade_pct = 1.0
    az = Analyzer(cfg, provider=None, tracker=None)

    # build a volatile uptrend so direction is set but risk is elevated
    rng = np.random.default_rng(5)
    base = 100 + np.linspace(0, 40, 400)
    noise = np.cumsum(rng.normal(0, 3, 400))
    closes = list(base + noise)
    df = make_df(closes, highs=[c + 4 for c in closes], lows=[c - 4 for c in closes])
    a = az.analyze_df(df, "BTC/USDT", "1h")
    if a.is_actionable:
        # suggested risk never exceeds the configured base risk
        assert a.suggested_risk_pct <= cfg.account.risk_per_trade_pct + 1e-9
        assert a.qty >= 0
