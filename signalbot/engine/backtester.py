"""Event-driven backtester with no look-ahead.

For each candle from `warmup` to the end we call ``strategy.generate`` on the
slice up to and including that candle (the last closed one). On a signal we
open one virtual trade and walk forward candle-by-candle to find which of SL /
TP / timeout is hit first. If both SL and TP fall inside the same candle we
resolve conservatively as a LOSS (stop assumed hit first).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.config import AppConfig
from ..core.types import Outcome, SignalDirection
from ..data.provider import timeframe_ms
from ..strategies.base import Strategy
from .metrics import Metrics, compute_metrics


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry: float
    stop: float
    take: float
    exit: float
    outcome: str
    r: float
    return_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    timeframe: str
    trades: list[Trade] = field(default_factory=list)
    metrics_all: Metrics = field(default_factory=Metrics)
    metrics_is: Metrics = field(default_factory=Metrics)
    metrics_oos: Metrics = field(default_factory=Metrics)
    equity_curve: list[float] = field(default_factory=list)
    split_index: int = 0


def _periods_per_year(timeframe: str) -> float:
    ms = timeframe_ms(timeframe)
    return (365 * 24 * 3600 * 1000) / ms


class Backtester:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.costs = config.costs
        self.bt = config.backtest

    def run(self, strategy: Strategy, df: pd.DataFrame, symbol: str, timeframe: str) -> BacktestResult:
        df = df.reset_index(drop=True).copy()
        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe

        warmup = max(self.bt.warmup, strategy.warmup)
        fill_next = self.bt.fill_mode == "next_open"
        expire_bars = self.config.tracker.expire_after_bars

        trades: list[Trade] = []
        n = len(df)
        i = warmup
        open_until = -1  # index until which we already hold a position (1 pos/pair)

        while i < n:
            if i <= open_until:
                i += 1
                continue

            sig = strategy.generate(df.iloc[: i + 1])
            if sig is None:
                i += 1
                continue

            # entry candle index and price
            if fill_next:
                entry_idx = i + 1
                if entry_idx >= n:
                    break
                entry = float(df["open"].iloc[entry_idx])
            else:
                entry_idx = i
                entry = float(df["close"].iloc[i])

            trade = self._simulate_trade(df, sig.direction, entry, sig.stop_loss,
                                         sig.take_profits[0], entry_idx, expire_bars)
            if trade is not None:
                trades.append(trade)
                # block re-entry until this trade closes (max 1 position per pair)
                open_until = entry_idx + trade.bars_held
                i = open_until + 1
            else:
                i += 1

        return self._assemble(strategy.name, symbol, timeframe, trades, n)

    def _simulate_trade(self, df, direction, entry, stop, take, entry_idx, expire_bars):
        risk = abs(entry - stop)
        if risk <= 0:
            return None

        n = len(df)
        long = direction == SignalDirection.LONG
        last_idx = min(entry_idx + expire_bars, n - 1)

        exit_price = None
        outcome = Outcome.EXPIRED
        exit_idx = last_idx

        for j in range(entry_idx, last_idx + 1):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            hit_stop = low <= stop if long else high >= stop
            hit_take = high >= take if long else low <= take

            if hit_stop and hit_take:
                # conservative: assume stop hit first
                exit_price, outcome, exit_idx = stop, Outcome.LOSS, j
                break
            if hit_stop:
                exit_price, outcome, exit_idx = stop, Outcome.LOSS, j
                break
            if hit_take:
                exit_price, outcome, exit_idx = take, Outcome.WIN, j
                break

        if exit_price is None:
            exit_price = float(df["close"].iloc[last_idx])
            outcome = Outcome.EXPIRED
            exit_idx = last_idx

        r, return_pct = self._apply_costs(long, entry, exit_price, risk, exit_idx - entry_idx + 1, df.attrs.get("timeframe", "1h"))

        return Trade(
            entry_time=df["timestamp"].iloc[entry_idx],
            exit_time=df["timestamp"].iloc[exit_idx],
            direction=direction.value,
            entry=entry,
            stop=stop,
            take=take,
            exit=exit_price,
            outcome=outcome.value,
            r=r,
            return_pct=return_pct,
            bars_held=exit_idx - entry_idx + 1,
        )

    def _apply_costs(self, long, entry, exit_price, risk, bars_held, timeframe):
        slip = self.costs.slippage_bps / 1e4
        fee = self.costs.taker_fee_bps / 1e4

        if long:
            entry_eff = entry * (1 + slip)
            exit_eff = exit_price * (1 - slip)
            pnl = exit_eff - entry_eff
        else:
            entry_eff = entry * (1 - slip)
            exit_eff = exit_price * (1 + slip)
            pnl = entry_eff - exit_eff

        fee_cost = (entry_eff + exit_eff) * fee
        pnl -= fee_cost

        if self.costs.include_funding:
            tf_hours = timeframe_ms(timeframe) / 3_600_000
            funding = self.costs.funding_rate_bps_per_8h / 1e4 * (bars_held * tf_hours / 8.0)
            pnl -= entry_eff * funding

        r = pnl / risk
        return_pct = r * self.config.account.risk_per_trade_pct
        return r, return_pct

    def _assemble(self, name, symbol, timeframe, trades, total_bars):
        ppy = _periods_per_year(timeframe)
        result = BacktestResult(strategy=name, symbol=symbol, timeframe=timeframe, trades=trades)

        dicts = [
            {"r": t.r, "return_pct": t.return_pct, "bars_held": t.bars_held} for t in trades
        ]
        result.metrics_all = compute_metrics(dicts, total_bars, ppy)

        # in-sample / out-of-sample split by trade order
        split = self.bt.oos_split
        k = int(len(trades) * split)
        result.split_index = k
        result.metrics_is = compute_metrics(dicts[:k], None, ppy)
        result.metrics_oos = compute_metrics(dicts[k:], None, ppy)

        # equity curve (risk-normalized, base 1.0)
        equity = 1.0
        curve = [equity]
        for d in dicts:
            equity *= 1.0 + d["return_pct"] / 100.0
            curve.append(equity)
        result.equity_curve = curve
        return result
