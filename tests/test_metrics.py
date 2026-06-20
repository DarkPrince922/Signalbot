"""Metrics on a synthetic series with a known answer."""
from __future__ import annotations

from signalbot.engine.metrics import compute_metrics


def test_known_winrate_and_expectancy():
    # 3 wins of +2R, 2 losses of -1R
    trades = [
        {"r": 2.0, "return_pct": 2.0, "bars_held": 1},
        {"r": 2.0, "return_pct": 2.0, "bars_held": 1},
        {"r": 2.0, "return_pct": 2.0, "bars_held": 1},
        {"r": -1.0, "return_pct": -1.0, "bars_held": 1},
        {"r": -1.0, "return_pct": -1.0, "bars_held": 1},
    ]
    m = compute_metrics(trades, total_bars=100)
    assert m.trades == 5
    assert m.wins == 3 and m.losses == 2
    assert abs(m.winrate - 60.0) < 1e-9
    assert abs(m.avg_win_r - 2.0) < 1e-9
    assert abs(m.avg_loss_r - (-1.0)) < 1e-9
    # expectancy = (3*2 + 2*-1)/5 = 0.8 R
    assert abs(m.expectancy_r - 0.8) < 1e-9


def test_profit_factor():
    trades = [
        {"r": 1.0, "return_pct": 3.0, "bars_held": 1},
        {"r": -1.0, "return_pct": -1.0, "bars_held": 1},
    ]
    m = compute_metrics(trades)
    # gross win 3, gross loss 1 -> PF 3
    assert abs(m.profit_factor - 3.0) < 1e-9


def test_empty():
    m = compute_metrics([])
    assert m.trades == 0
    assert m.profit_factor == 0.0
