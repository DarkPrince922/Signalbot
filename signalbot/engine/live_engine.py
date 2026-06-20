"""Live engine: on each newly closed candle, run strategies and emit signals.

Scheduling is handled by APScheduler in the bot entrypoint; this module holds
the per-poll logic: fetch fresh data, generate signals, de-duplicate, persist,
and hand new signals to a notifier callback. Also drives the forward tracker.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from sqlalchemy import select

from ..core.config import AppConfig
from ..core.db import Database
from ..core.models import SignalRow
from ..core.types import Signal
from ..data.provider import DataProvider
from ..strategies.base import Strategy
from ..strategies.registry import build_strategy
from .tracker import Tracker

log = logging.getLogger(__name__)

NotifyFn = Callable[[Signal], Awaitable[None]]
CloseNotifyFn = Callable[[SignalRow], Awaitable[None]]


class LiveEngine:
    def __init__(self, config: AppConfig, db: Database, provider: DataProvider) -> None:
        self.config = config
        self.db = db
        self.provider = provider
        self.tracker = Tracker(config, db, provider)

    # --- runtime-editable state (persisted in settings table) ---
    def signals_enabled(self) -> bool:
        return bool(self.db.get_setting("signals_enabled", True))

    def set_signals_enabled(self, on: bool) -> None:
        self.db.set_setting("signals_enabled", on)

    def enabled_strategies(self) -> dict[str, Strategy]:
        """Build strategy instances for everything enabled (config + overrides)."""
        overrides = self.db.get_setting("strategy_enabled", {}) or {}
        result: dict[str, Strategy] = {}
        for name, sc in self.config.strategies.items():
            enabled = overrides.get(name, sc.enabled)
            if not enabled:
                continue
            strat = build_strategy(name, sc.params)
            strat.timeframe = sc.timeframe
            result[name] = strat
        return result

    def pairs(self) -> list[str]:
        return self.db.get_setting("pairs", self.config.pairs) or self.config.pairs

    def active_timeframes(self) -> set[str]:
        return {s.timeframe for s in self.enabled_strategies().values()}

    # --- core poll ---
    def poll_once(self, timeframe: str | None = None) -> list[Signal]:
        """Generate signals for all (pair x enabled strategy) and persist new ones.

        If `timeframe` is given, only strategies on that timeframe are polled.
        Returns the list of newly created signals (de-duplicated).
        """
        if not self.signals_enabled():
            return []

        new_signals: list[Signal] = []
        strategies = self.enabled_strategies()
        for symbol in self.pairs():
            for strat in strategies.values():
                if timeframe is not None and strat.timeframe != timeframe:
                    continue
                try:
                    sig = self._eval_one(symbol, strat)
                except Exception as exc:
                    log.warning("eval failed %s/%s: %s", symbol, strat.name, exc)
                    continue
                if sig is not None:
                    new_signals.append(sig)
        return new_signals

    def _eval_one(self, symbol: str, strat: Strategy) -> Signal | None:
        tf = strat.timeframe
        df = self.provider.fetch_ohlcv(symbol, tf, since=None, limit=max(300, strat.warmup + 50))
        if df.empty or len(df) < strat.warmup:
            return None
        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = tf

        sig = strat.generate(df)
        if sig is None:
            return None
        sig.symbol = symbol
        sig.timeframe = tf

        if self._persist_if_new(sig):
            return sig
        return None

    def _persist_if_new(self, sig: Signal) -> bool:
        """Insert the signal unless its dedup_key already exists. Returns inserted?"""
        key = sig.dedup_key()
        with self.db.session() as s:
            exists = s.execute(
                select(SignalRow.id).where(SignalRow.dedup_key == key)
            ).first()
            if exists:
                return False
            s.add(
                SignalRow(
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    strategy=sig.strategy,
                    direction=sig.direction.value,
                    entry=sig.entry,
                    stop_loss=sig.stop_loss,
                    take_profits=sig.take_profits,
                    confidence=sig.confidence,
                    reason=sig.reason,
                    created_at=sig.created_at.replace(tzinfo=None),
                    status="OPEN",
                    mode="LIVE",
                    dedup_key=key,
                )
            )
        return True
