"""De-duplication: the same signal on the same candle is stored only once."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select

from signalbot.core.config import AppConfig
from signalbot.core.db import Database
from signalbot.core.models import SignalRow
from signalbot.core.types import Signal, SignalDirection
from signalbot.engine.live_engine import LiveEngine


def _signal(ts):
    return Signal(
        symbol="BTC/USDT",
        timeframe="1h",
        direction=SignalDirection.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profits=[110.0],
        strategy="ema_cross",
        reason="t",
        created_at=ts,
    )


def test_persist_if_new_dedups():
    db = Database("sqlite:///:memory:")
    db.init()
    engine = LiveEngine(AppConfig(), db, provider=None)

    ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert engine._persist_if_new(_signal(ts)) is True
    # same strategy/symbol/tf/direction/candle -> duplicate, not inserted
    assert engine._persist_if_new(_signal(ts)) is False

    with db.session() as s:
        count = s.execute(select(func.count()).select_from(SignalRow)).scalar()
    assert count == 1


def test_different_candle_is_new():
    db = Database("sqlite:///:memory:")
    db.init()
    engine = LiveEngine(AppConfig(), db, provider=None)

    assert engine._persist_if_new(_signal(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)))
    assert engine._persist_if_new(_signal(datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)))

    with db.session() as s:
        count = s.execute(select(func.count()).select_from(SignalRow)).scalar()
    assert count == 2
