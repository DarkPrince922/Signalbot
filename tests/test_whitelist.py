"""Whitelist: only the configured user id is served; others are ignored."""
from __future__ import annotations

from types import SimpleNamespace

from aiogram import F


def _fake_message(user_id: int):
    return SimpleNamespace(from_user=SimpleNamespace(id=user_id))


def test_reject_filter_matches_foreign_user():
    allowed = 42
    reject = F.from_user.id != allowed  # matches => message gets ignored

    assert reject.resolve(_fake_message(999)) is True   # foreign -> rejected
    assert reject.resolve(_fake_message(42)) is False    # owner -> passes through


def test_strategies_registry_discovers_builtins():
    from signalbot.strategies.registry import discover_strategies

    found = discover_strategies()
    assert "ema_cross" in found
    assert "rsi_reversion" in found
    assert "donchian_breakout" in found
