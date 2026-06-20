"""Performance metrics computed from a list of closed trades.

A trade is a dict with at least: 'r' (realized R multiple), 'return_pct'
(net return on notional), 'bars_held'. Equity is reconstructed by compounding
per-trade returns so drawdown/Sharpe reflect the actual sequence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    exposure_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_metrics(
    trades: list[dict],
    total_bars: int | None = None,
    periods_per_year: float = 365 * 24,
) -> Metrics:
    """Aggregate trades into a Metrics object.

    `total_bars` enables exposure% (sum of bars_held / total_bars).
    `periods_per_year` scales Sharpe/Sortino and CAGR when trade timing is in
    bars of one timeframe; pass the count of timeframe bars per year.
    """
    m = Metrics()
    if not trades:
        return m

    m.trades = len(trades)
    rs = [t["r"] for t in trades]
    rets = [t.get("return_pct", 0.0) for t in trades]

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    m.wins = len(wins)
    m.losses = len(losses)
    m.winrate = m.wins / m.trades * 100.0
    m.avg_win_r = _safe_mean(wins)
    m.avg_loss_r = _safe_mean(losses)
    m.expectancy_r = _safe_mean(rs)

    gross_win = sum(t.get("return_pct", 0.0) for t in trades if t.get("return_pct", 0.0) > 0)
    gross_loss = -sum(t.get("return_pct", 0.0) for t in trades if t.get("return_pct", 0.0) < 0)
    m.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # equity curve by compounding net returns
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    equity_returns: list[float] = []
    for r in rets:
        prev = equity
        equity *= 1.0 + r / 100.0
        equity_returns.append(equity / prev - 1.0)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    m.total_return_pct = (equity - 1.0) * 100.0
    m.max_drawdown_pct = max_dd * 100.0

    # Sharpe / Sortino on per-trade returns (rf = 0)
    n = len(equity_returns)
    mean_ret = _safe_mean(equity_returns)
    if n > 1:
        var = sum((x - mean_ret) ** 2 for x in equity_returns) / (n - 1)
        std = math.sqrt(var)
        downside = [x for x in equity_returns if x < 0]
        dvar = sum(x**2 for x in downside) / (n - 1) if downside else 0.0
        dstd = math.sqrt(dvar)
        # scale per-trade stats by sqrt(trades per year), approximated via
        # average trade frequency if total_bars known
        scale = math.sqrt(n) if total_bars is None else math.sqrt(
            periods_per_year / max(total_bars / n, 1e-9)
        )
        # simpler, comparable Sharpe: annualize by sqrt(number of trades)
        ann = math.sqrt(max(n, 1))
        m.sharpe = (mean_ret / std * ann) if std > 0 else 0.0
        m.sortino = (mean_ret / dstd * ann) if dstd > 0 else 0.0

    if total_bars and total_bars > 0:
        bars_held = sum(t.get("bars_held", 0) for t in trades)
        m.exposure_pct = bars_held / total_bars * 100.0
        years = total_bars / periods_per_year
        if years > 0 and equity > 0:
            m.cagr_pct = (equity ** (1.0 / years) - 1.0) * 100.0

    return m
