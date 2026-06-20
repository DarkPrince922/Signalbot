"""Forward tracker: follows the fate of each live signal on real future candles.

This is the honest evaluation layer — it shows whether a strategy that looked
good in backtest actually performs out-of-sample on live signals.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from ..core.config import AppConfig
from ..core.db import Database
from ..core.models import SignalRow
from ..core.types import Outcome, SignalDirection
from ..data.provider import DataProvider, timeframe_ms
from .metrics import Metrics, compute_metrics

log = logging.getLogger(__name__)


def resolve_outcome(
    direction: SignalDirection,
    entry: float,
    stop: float,
    take: float,
    future: pd.DataFrame,
    expire_after_bars: int,
) -> tuple[Outcome, float | None, datetime | None, int]:
    """Walk `future` candles (strictly after entry) and return outcome.

    Returns (outcome, exit_price, exit_time, bars_held). If neither SL nor TP is
    touched within the timeout, returns EXPIRED at the last available close.
    When both SL and TP are inside one candle, resolves conservatively as LOSS.
    """
    long = direction == SignalDirection.LONG
    bars = 0
    for _, row in future.iterrows():
        bars += 1
        high = float(row["high"])
        low = float(row["low"])
        hit_stop = low <= stop if long else high >= stop
        hit_take = high >= take if long else low <= take

        if hit_stop and hit_take:
            return Outcome.LOSS, stop, row["timestamp"].to_pydatetime(), bars
        if hit_stop:
            return Outcome.LOSS, stop, row["timestamp"].to_pydatetime(), bars
        if hit_take:
            return Outcome.WIN, take, row["timestamp"].to_pydatetime(), bars
        if bars >= expire_after_bars:
            return Outcome.EXPIRED, float(row["close"]), row["timestamp"].to_pydatetime(), bars

    return Outcome.OPEN, None, None, bars


class Tracker:
    def __init__(self, config: AppConfig, db: Database, provider: DataProvider) -> None:
        self.config = config
        self.db = db
        self.provider = provider

    def update_open_signals(self) -> list[SignalRow]:
        """Re-evaluate every OPEN live signal; persist any that closed."""
        closed: list[SignalRow] = []
        with self.db.session() as s:
            rows = s.execute(
                select(SignalRow).where(
                    SignalRow.status == Outcome.OPEN.value, SignalRow.mode == "LIVE"
                )
            ).scalars().all()

            for row in rows:
                try:
                    closed_row = self._evaluate(row)
                    if closed_row is not None:
                        closed.append(closed_row)
                except Exception as exc:  # keep going on per-signal errors
                    log.warning("tracker failed for signal %s: %s", row.id, exc)
        return closed

    def _evaluate(self, row: SignalRow) -> SignalRow | None:
        expire = self.config.tracker.expire_after_bars
        # fetch candles since the signal time (plenty for the timeout window)
        since_ms = int(row.created_at.replace(tzinfo=None).timestamp() * 1000)
        df = self.provider.fetch_ohlcv(
            row.symbol, row.timeframe, since=since_ms, limit=expire + 5
        )
        if df.empty:
            return None

        created = pd.Timestamp(row.created_at).tz_localize("UTC") if pd.Timestamp(row.created_at).tz is None else pd.Timestamp(row.created_at)
        future = df[df["timestamp"] > created].reset_index(drop=True)
        if future.empty:
            return None

        outcome, exit_price, exit_time, bars = resolve_outcome(
            SignalDirection(row.direction),
            row.entry,
            row.stop_loss,
            row.take_profits[0],
            future,
            expire,
        )
        if outcome == Outcome.OPEN:
            return None

        risk = abs(row.entry - row.stop_loss)
        if SignalDirection(row.direction) == SignalDirection.LONG:
            realized_r = (exit_price - row.entry) / risk if risk else 0.0
        else:
            realized_r = (row.entry - exit_price) / risk if risk else 0.0

        row.status = outcome.value
        row.closed_at = exit_time
        row.realized_r = realized_r
        log.info("Signal %s -> %s (R=%.2f)", row.id, outcome.value, realized_r)
        return row

    def live_metrics(self, strategy: str | None = None) -> Metrics:
        with self.db.session() as s:
            stmt = select(SignalRow).where(
                SignalRow.mode == "LIVE",
                SignalRow.status.in_([Outcome.WIN.value, Outcome.LOSS.value, Outcome.EXPIRED.value]),
            )
            if strategy:
                stmt = stmt.where(SignalRow.strategy == strategy)
            rows = s.execute(stmt).scalars().all()

        trades = [
            {"r": r.realized_r or 0.0, "return_pct": (r.realized_r or 0.0) * self.config.account.risk_per_trade_pct, "bars_held": 0}
            for r in rows
        ]
        return compute_metrics(trades)
