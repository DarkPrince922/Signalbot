"""Lightweight technical indicators implemented on pandas/numpy.

Implemented natively (rather than via pandas-ta) so the project has no
binary indicator dependency and stays installable on modern numpy.
All functions are causal: value at index i uses only data up to i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # when avg_loss == 0 RSI is 100 by definition
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def donchian(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Return (upper, lower) Donchian channel computed on *prior* candles.

    Shifted by one so the breakout test on candle i compares against the
    channel formed by the previous `period` candles, never including i.
    """
    upper = df["high"].rolling(window=period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(window=period, min_periods=period).min().shift(1)
    return upper, lower
