"""On-demand market analyzer: 'where to enter right now' + risk assessment.

Unlike a strategy (which only fires on a discrete event such as a cross), the
analyzer always produces a rankable read on every pair by blending several
indicators into a directional vote, then derives an entry/stop/target plan and
lets the bot DETERMINE THE RISK of the setup (volatility + signal agreement +
the strategy stack's live track record) and scale the recommended size down
accordingly.

This is decision support, not financial advice — execution stays with the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.config import AppConfig
from ..core.types import SignalDirection
from ..data.provider import DataProvider
from ..strategies.indicators import atr, ema, macd, rsi, supertrend


@dataclass
class Analysis:
    symbol: str
    timeframe: str
    direction: SignalDirection | None  # None = no clear bias, stand aside
    score: float                        # 0..1 agreement between indicators
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profits: list[float] = field(default_factory=list)
    atr: float = 0.0
    atr_pct: float = 0.0                # ATR as % of price (volatility proxy)
    risk_level: str = "—"              # LOW | MEDIUM | HIGH
    suggested_risk_pct: float = 0.0     # risk-adjusted % of account to risk
    qty: float = 0.0
    risk_usdt: float = 0.0
    rr: float | None = None
    votes: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction is not None and self.score > 0.0


class Analyzer:
    def __init__(self, config: AppConfig, provider: DataProvider, tracker=None) -> None:
        self.config = config
        self.provider = provider
        self.tracker = tracker  # optional; used for live win-rate in risk calc
        self.cfg = config.analysis

    # --- public ---
    def analyze_pair(self, symbol: str, timeframe: str | None = None) -> Analysis:
        tf = timeframe or self.cfg.timeframe
        warmup = max(self.cfg.slow_ema, self.cfg.atr_period, self.cfg.supertrend_period) + 50
        df = self.provider.fetch_ohlcv(symbol, tf, since=None, limit=max(300, warmup))
        return self.analyze_df(df, symbol, tf)

    def scan(self, pairs: list[str], timeframe: str | None = None) -> list[Analysis]:
        out: list[Analysis] = []
        for sym in pairs:
            try:
                out.append(self.analyze_pair(sym, timeframe))
            except Exception:
                continue
        # actionable first, then by score desc
        out.sort(key=lambda a: (a.is_actionable, a.score), reverse=True)
        return out

    # --- pure core (testable without network) ---
    def analyze_df(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Analysis:
        c = self.cfg
        result = Analysis(symbol=symbol, timeframe=timeframe, direction=None, score=0.0)
        if df is None or len(df) < max(c.slow_ema, c.atr_period, c.supertrend_period) + 2:
            return result

        close = float(df["close"].iloc[-1])
        fast = ema(df["close"], c.fast_ema).iloc[-1]
        slow = ema(df["close"], c.slow_ema).iloc[-1]
        r = rsi(df["close"], c.rsi_period).iloc[-1]
        macd_line, sig_line, _ = macd(df["close"])
        m, s = macd_line.iloc[-1], sig_line.iloc[-1]
        _, st_dir = supertrend(df, c.supertrend_period, c.supertrend_mult)
        st = int(st_dir.iloc[-1])
        a = atr(df, c.atr_period).iloc[-1]

        if pd.isna(fast) or pd.isna(slow) or pd.isna(r) or pd.isna(m) or pd.isna(a) or a <= 0:
            return result

        # --- directional votes (each in {-1, 0, +1}) ---
        votes = {
            "ema_trend": 1 if fast > slow else (-1 if fast < slow else 0),
            "macd": 1 if m > s else (-1 if m < s else 0),
            "rsi": 1 if r > 55 else (-1 if r < 45 else 0),
            "supertrend": 1 if st > 0 else -1,
        }
        net = sum(votes.values())
        n = len(votes)
        result.votes = votes
        result.atr = float(a)
        result.atr_pct = float(a) / close * 100.0

        if net == 0:
            result.score = 0.0
            return result  # mixed signals -> no clear place to enter

        direction = SignalDirection.LONG if net > 0 else SignalDirection.SHORT
        score = abs(net) / n  # agreement fraction 0..1

        # --- plan ---
        mult, rr = c.atr_mult, c.rr
        if direction == SignalDirection.LONG:
            stop = close - mult * a
            risk = close - stop
            tps = [close + rr * risk, close + (rr + 1.0) * risk]
        else:
            stop = close + mult * a
            risk = stop - close
            tps = [close - rr * risk, close - (rr + 1.0) * risk]

        # --- risk determination ---
        live_wr = self._live_winrate()
        risk_level = self._classify_risk(result.atr_pct, score, live_wr)
        size_factor = {
            "LOW": c.size_factor_low,
            "MEDIUM": c.size_factor_medium,
            "HIGH": c.size_factor_high,
        }[risk_level]
        suggested_risk_pct = self.config.account.risk_per_trade_pct * size_factor
        risk_usdt = self.config.account.size_usdt * suggested_risk_pct / 100.0
        qty = risk_usdt / risk if risk > 0 else 0.0

        result.direction = direction
        result.score = score
        result.entry = close
        result.stop_loss = float(stop)
        result.take_profits = [float(t) for t in tps]
        result.risk_level = risk_level
        result.suggested_risk_pct = suggested_risk_pct
        result.risk_usdt = risk_usdt
        result.qty = qty
        result.rr = rr
        return result

    # --- helpers ---
    def _live_winrate(self) -> float | None:
        if self.tracker is None:
            return None
        try:
            m = self.tracker.live_metrics()
            return m.winrate if m.trades >= 10 else None
        except Exception:
            return None

    def _classify_risk(self, atr_pct: float, score: float, live_winrate: float | None) -> str:
        c = self.cfg
        points = 0
        # volatility
        if atr_pct >= c.vol_high_pct:
            points += 2
        elif atr_pct >= c.vol_low_pct:
            points += 1
        # signal agreement (weak agreement = riskier)
        if score < 0.5:
            points += 2
        elif score < 0.75:
            points += 1
        # live track record, if we have enough samples
        if live_winrate is not None and live_winrate < 35.0:
            points += 1

        if points <= 1:
            return "LOW"
        if points <= 3:
            return "MEDIUM"
        return "HIGH"
