"""Diagnostics: scan surfaces errors, and futures symbols resolve correctly."""
from __future__ import annotations

import pytest

from signalbot.core.config import AppConfig
from signalbot.engine.analyzer import Analyzer


class _RaisingProvider:
    def fetch_ohlcv(self, *a, **k):
        raise RuntimeError("exchange unreachable")

    def latest_closed_candle(self, *a, **k):
        raise RuntimeError("exchange unreachable")


def test_scan_detailed_returns_errors_instead_of_swallowing():
    az = Analyzer(AppConfig(), _RaisingProvider(), tracker=None)
    analyses, errors = az.scan_detailed(["BTC/USDT", "ETH/USDT"], "1h")
    assert analyses == []
    assert len(errors) == 2
    assert "BTC/USDT" in errors[0]
    assert "exchange unreachable" in errors[0]


def test_scan_keeps_backward_compatible_signature():
    az = Analyzer(AppConfig(), _RaisingProvider(), tracker=None)
    result = az.scan(["BTC/USDT"], "1h")
    assert result == []  # still returns a plain list


def test_futures_symbol_resolution():
    ccxt = pytest.importorskip("ccxt")  # noqa: F841
    from signalbot.data.ccxt_provider import CCXTProvider

    p = CCXTProvider("binanceusdm")
    # pretend markets are already loaded (avoid network)
    p._markets_loaded = True
    p.exchange.markets = {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}

    assert p._resolve_symbol("BTC/USDT") == "BTC/USDT:USDT"   # spot form -> linear
    assert p._resolve_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"  # already linear
    assert p._resolve_symbol("DOGE/XYZ") == "DOGE/XYZ"        # unknown -> unchanged
