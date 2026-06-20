"""Market overview / movement dynamics aggregation."""
from __future__ import annotations

import numpy as np

from signalbot.core.config import AppConfig
from signalbot.core.types import SignalDirection
from signalbot.engine.analyzer import Analyzer
from tests.conftest import make_df


def _analyzer(cfg=None):
    return Analyzer(cfg or AppConfig(), provider=None, tracker=None)


def _trend_df(start, end, n=400, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(start, end, n) + rng.normal(0, noise, n)
    closes = list(base)
    return make_df(closes, highs=[c + noise + 0.5 for c in closes], lows=[c - noise - 0.5 for c in closes])


def test_change_pct_and_momentum_populated():
    az = _analyzer()
    df = _trend_df(100, 130)  # ~ +30% overall, recent window strongly up
    a = az.analyze_df(df, "BTC/USDT", "1h")
    assert a.change_pct > 0
    assert a.momentum in {"UP", "STRONG_UP"}


def test_momentum_down_label():
    az = _analyzer()
    df = _trend_df(160, 100)
    a = az.analyze_df(df, "ETH/USDT", "1h")
    assert a.change_pct < 0
    assert a.momentum in {"DOWN", "STRONG_DOWN"}


def test_overview_risk_on_when_breadth_high():
    az = _analyzer()
    analyses = [
        az.analyze_df(_trend_df(100, 140, seed=i), f"C{i}/USDT", "1h") for i in range(6)
    ]
    ov = az.aggregate_overview(analyses, "1h")
    assert ov.pairs == 6
    assert ov.bullish >= 5
    assert ov.breadth_pct >= 60
    assert ov.regime == "RISK-ON"
    assert ov.avg_change_pct > 0
    assert len(ov.leaders) == 3


def test_overview_risk_off_when_breadth_low():
    az = _analyzer()
    analyses = [
        az.analyze_df(_trend_df(160, 110, seed=i), f"C{i}/USDT", "1h") for i in range(6)
    ]
    ov = az.aggregate_overview(analyses, "1h")
    assert ov.bearish >= 5
    assert ov.regime == "RISK-OFF"
    assert ov.avg_change_pct < 0


def test_overview_picks_btc_bellwether():
    az = _analyzer()
    analyses = [
        az.analyze_df(_trend_df(100, 140, seed=1), "BTC/USDT", "1h"),
        az.analyze_df(_trend_df(100, 120, seed=2), "ETH/USDT", "1h"),
    ]
    ov = az.aggregate_overview(analyses, "1h")
    assert ov.btc is not None
    assert ov.btc.symbol == "BTC/USDT"


def test_universe_falls_back_to_pairs_when_empty():
    cfg = AppConfig()
    cfg.pairs = ["BTC/USDT", "ETH/USDT"]
    cfg.analysis.universe = []
    az = Analyzer(cfg, provider=None, tracker=None)
    assert az.universe() == ["BTC/USDT", "ETH/USDT"]

    cfg.analysis.universe = ["SOL/USDT"]
    assert az.universe() == ["SOL/USDT"]


def test_empty_overview_is_safe():
    az = _analyzer()
    ov = az.aggregate_overview([], "1h")
    assert ov.pairs == 0
    assert ov.regime == "—"
