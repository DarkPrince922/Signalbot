"""ccxt-backed data provider with local SQLite caching and retry/backoff.

Drops the still-forming last candle so callers only ever see closed candles.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from ..core.db import Database
from ..core.models import OHLCVCache
from .provider import OHLCV_COLUMNS, DataProvider, timeframe_ms

log = logging.getLogger(__name__)

try:  # ccxt is optional at import time so the rest of the package stays usable
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


def _now_ms() -> int:
    return int(time.time() * 1000)


class CCXTProvider(DataProvider):
    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        db: Database | None = None,
        max_retries: int = 4,
        request_pause: float = 0.2,
    ) -> None:
        if ccxt is None:  # pragma: no cover
            raise RuntimeError("ccxt is not installed; cannot use CCXTProvider")
        self.exchange_id = exchange_id
        klass = getattr(ccxt, exchange_id)
        self.exchange = klass({"enableRateLimit": True})
        self.db = db
        self.max_retries = max_retries
        self.request_pause = request_pause

    # --- retry wrapper ---
    def _with_retry(self, fn, *args, **kwargs):
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # network/exchange errors
                last_exc = exc
                log.warning("ccxt call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"ccxt call failed after {self.max_retries} retries") from last_exc

    # --- raw network fetch (one page) ---
    def _fetch_raw(self, symbol: str, timeframe: str, since: int | None, limit: int) -> list[list]:
        return self._with_retry(
            self.exchange.fetch_ohlcv, symbol, timeframe, since, limit
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        tf_ms = timeframe_ms(timeframe)
        rows: list[list] = []
        cursor = since
        # paginate until we reach `limit` or run out of data
        while True:
            batch = self._fetch_raw(symbol, timeframe, cursor, min(1000, limit - len(rows)))
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1][0] + tf_ms
            time.sleep(self.request_pause)
            if len(rows) >= limit or len(batch) < 2:
                break

        df = self._rows_to_df(rows)
        df = self._drop_unclosed(df, tf_ms)
        if self.db is not None and not df.empty:
            self._cache_store(symbol, timeframe, df)
        if since is None and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    def latest_closed_candle(self, symbol: str, timeframe: str) -> pd.Series:
        df = self.fetch_ohlcv(symbol, timeframe, since=None, limit=3)
        if df.empty:
            raise RuntimeError(f"No candles returned for {symbol} {timeframe}")
        return df.iloc[-1]

    # --- helpers ---
    @staticmethod
    def _rows_to_df(rows: list[list]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[OHLCV_COLUMNS]

    @staticmethod
    def _drop_unclosed(df: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
        """Remove the last candle if its close time is in the future."""
        if df.empty:
            return df
        last_open_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)
        if last_open_ms + tf_ms > _now_ms():
            df = df.iloc[:-1].reset_index(drop=True)
        return df

    def _cache_store(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        with self.db.session() as s:
            for _, r in df.iterrows():
                ts_ms = int(r["timestamp"].timestamp() * 1000)
                existing = s.get(OHLCVCache, (symbol, timeframe, ts_ms))
                if existing is None:
                    s.add(
                        OHLCVCache(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=ts_ms,
                            open=float(r["open"]),
                            high=float(r["high"]),
                            low=float(r["low"]),
                            close=float(r["close"]),
                            volume=float(r["volume"]),
                        )
                    )
