"""Causality of new indicators and smoke tests for the new strategies."""
from __future__ import annotations

import numpy as np

from signalbot.strategies.indicators import bollinger, macd, supertrend
from signalbot.strategies.registry import build_strategy, discover_strategies
from tests.conftest import make_df


def _series(seed=7, n=400):
    rng = np.random.default_rng(seed)
    closes = list(100 + np.cumsum(rng.normal(0.03, 1.2, n)))
    return make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])


def test_new_strategies_registered():
    found = discover_strategies()
    for name in ("macd_cross", "bollinger_reversion", "supertrend"):
        assert name in found


def test_macd_and_bollinger_are_causal():
    df = _series()
    m_full, s_full, _ = macd(df["close"])
    bu_full, bm_full, bl_full = bollinger(df["close"])
    for i in (120, 250, 380):
        pref = df.iloc[: i + 1]
        m_p, s_p, _ = macd(pref["close"])
        assert np.isclose(m_p.iloc[-1], m_full.iloc[i])
        assert np.isclose(s_p.iloc[-1], s_full.iloc[i])
        bu_p, bm_p, bl_p = bollinger(pref["close"])
        assert np.isclose(bu_p.iloc[-1], bu_full.iloc[i])
        assert np.isclose(bl_p.iloc[-1], bl_full.iloc[i])


def test_supertrend_is_causal():
    df = _series()
    line_full, dir_full = supertrend(df, 10, 3.0)
    for i in (120, 250, 380):
        pref = df.iloc[: i + 1]
        line_p, dir_p = supertrend(pref, 10, 3.0)
        # the recursive value at i must match whether computed on prefix or full
        assert np.isclose(line_p.iloc[-1], line_full.iloc[i], equal_nan=True)
        assert int(dir_p.iloc[-1]) == int(dir_full.iloc[i])


def test_new_strategies_generate_valid_signals():
    # run each over a window; whenever a signal fires it must be internally
    # consistent (stop/tp on the correct side of entry)
    df = _series(seed=11, n=600)
    for name in ("macd_cross", "bollinger_reversion", "supertrend"):
        strat = build_strategy(name)
        fired = 0
        for i in range(strat.warmup, len(df)):
            sig = strat.generate(df.iloc[: i + 1])
            if sig is None:
                continue
            fired += 1
            if sig.direction.value == "LONG":
                assert sig.stop_loss < sig.entry < sig.take_profits[0]
            else:
                assert sig.take_profits[0] < sig.entry < sig.stop_loss
        # each should be able to fire at least once on a 600-bar random walk
        assert fired >= 1, f"{name} produced no signals"
