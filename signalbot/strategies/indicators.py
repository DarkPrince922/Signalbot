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


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram). Causal."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, middle, lower) Bollinger bands. Causal."""
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Return (supertrend_line, direction).

    direction is +1 when the trend is up (line acts as support / lower band),
    -1 when down. Computed recursively from past and current candles only —
    no look-ahead.
    """
    atr_series = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upperband = hl2 + multiplier * atr_series
    lowerband = hl2 - multiplier * atr_series

    n = len(df)
    final_upper = [float("nan")] * n
    final_lower = [float("nan")] * n
    st = [float("nan")] * n
    direction = [1] * n

    close = df["close"].to_numpy()
    ub = upperband.to_numpy()
    lb = lowerband.to_numpy()

    for i in range(n):
        if i == 0 or np.isnan(ub[i]) or np.isnan(lb[i]):
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            st[i] = ub[i]
            direction[i] = -1
            continue

        # carry-forward bands so the trend line only loosens, never tightens
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = ub[i]
        else:
            final_upper[i] = ub[i] if (ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]

        if np.isnan(final_lower[i - 1]):
            final_lower[i] = lb[i]
        else:
            final_lower[i] = lb[i] if (lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]

        prev_st = st[i - 1]
        if np.isnan(prev_st) or prev_st == final_upper[i - 1]:
            if close[i] <= final_upper[i]:
                st[i] = final_upper[i]
                direction[i] = -1
            else:
                st[i] = final_lower[i]
                direction[i] = 1
        else:
            if close[i] >= final_lower[i]:
                st[i] = final_lower[i]
                direction[i] = 1
            else:
                st[i] = final_upper[i]
                direction[i] = -1

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)
