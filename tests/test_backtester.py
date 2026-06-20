"""SL/TP outcome resolution, including the both-in-one-candle case."""
from __future__ import annotations

import pandas as pd

from signalbot.core.types import SignalDirection
from signalbot.engine.backtester import Backtester
from signalbot.engine.tracker import resolve_outcome
from tests.conftest import make_df


def test_tp_then_resolves_win(config):
    bt = Backtester(config)
    # long entry 100, stop 95, take 110; a later candle reaches 110 cleanly
    df = make_df(
        closes=[100, 101, 111],
        highs=[100, 102, 112],
        lows=[100, 100, 109],
    )
    trade = bt._simulate_trade(df, SignalDirection.LONG, 100.0, 95.0, 110.0, 0, 50)
    assert trade.outcome == "WIN"
    assert trade.exit == 110.0


def test_sl_resolves_loss(config):
    bt = Backtester(config)
    df = make_df(
        closes=[100, 94, 93],
        highs=[100, 96, 95],
        lows=[100, 94, 90],
    )
    trade = bt._simulate_trade(df, SignalDirection.LONG, 100.0, 95.0, 110.0, 0, 50)
    assert trade.outcome == "LOSS"
    assert trade.exit == 95.0


def test_both_targets_same_candle_is_conservative_loss(config):
    bt = Backtester(config)
    # one candle spans both 95 (stop) and 110 (take) -> must be LOSS
    df = make_df(
        closes=[100, 105],
        highs=[100, 112],
        lows=[100, 94],
    )
    trade = bt._simulate_trade(df, SignalDirection.LONG, 100.0, 95.0, 110.0, 0, 50)
    assert trade.outcome == "LOSS"
    assert trade.exit == 95.0


def test_short_both_targets_same_candle_loss(config):
    bt = Backtester(config)
    # short entry 100, stop 105, take 90; one candle hits both -> LOSS
    df = make_df(
        closes=[100, 95],
        highs=[100, 106],
        lows=[100, 89],
    )
    trade = bt._simulate_trade(df, SignalDirection.SHORT, 100.0, 105.0, 90.0, 0, 50)
    assert trade.outcome == "LOSS"
    assert trade.exit == 105.0


def test_expire_closes_at_close(config):
    config.tracker.expire_after_bars = 2
    bt = Backtester(config)
    df = make_df(
        closes=[100, 101, 102, 103],
        highs=[100, 101, 102, 103],
        lows=[100, 100, 101, 102],
    )
    trade = bt._simulate_trade(df, SignalDirection.LONG, 100.0, 90.0, 130.0, 0, 2)
    assert trade.outcome == "EXPIRED"


def test_resolve_outcome_matches_backtester():
    # tracker uses the same conservative rule
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
            "high": [112.0],
            "low": [94.0],
            "close": [105.0],
        }
    )
    outcome, exit_price, _, bars = resolve_outcome(
        SignalDirection.LONG, 100.0, 95.0, 110.0, future, 50
    )
    assert outcome.value == "LOSS"
    assert exit_price == 95.0
    assert bars == 1
