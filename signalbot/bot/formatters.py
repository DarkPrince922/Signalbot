"""Message formatting for signals and reports."""
from __future__ import annotations

from ..core.config import AppConfig
from ..core.types import Signal, SignalDirection
from ..engine.analyzer import Analysis, MarketOverview
from ..engine.backtester import BacktestResult
from ..engine.metrics import Metrics

_RISK_LABEL = {"LOW": "🟩 НИЗКИЙ", "MEDIUM": "🟨 СРЕДНИЙ", "HIGH": "🟥 ВЫСОКИЙ", "—": "—"}
_VOTE_LABEL = {
    "ema_trend": "Тренд EMA",
    "macd": "MACD",
    "rsi": "RSI",
    "supertrend": "Supertrend",
}
_MOMENTUM_LABEL = {
    "STRONG_UP": "🚀 сильный рост",
    "UP": "📈 рост",
    "FLAT": "➡️ боковик",
    "DOWN": "📉 снижение",
    "STRONG_DOWN": "🔻 сильное падение",
}
_REGIME_LABEL = {
    "RISK-ON": "🟢 RISK-ON (аппетит к риску)",
    "RISK-OFF": "🔴 RISK-OFF (уход от риска)",
    "MIXED": "🟡 MIXED (разнонаправленно)",
    "—": "—",
}
_VOL_LABEL = {
    "CALM": "🟩 спокойная",
    "NORMAL": "🟨 обычная",
    "TURBULENT": "🟥 турбулентная",
    "—": "—",
}


def _fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.1f}".replace(",", " ")
    if p >= 1:
        return f"{p:,.3f}".replace(",", " ")
    return f"{p:.6f}"


def position_size(signal: Signal, account_size: float, risk_pct: float) -> tuple[float, float]:
    """Return (quantity, risk_usdt) recommendation based on stop distance."""
    risk_usdt = account_size * risk_pct / 100.0
    per_unit = signal.risk_per_unit
    qty = risk_usdt / per_unit if per_unit > 0 else 0.0
    return qty, risk_usdt


def format_signal(signal: Signal, config: AppConfig) -> str:
    arrow = "🟢 LONG" if signal.direction == SignalDirection.LONG else "🔴 SHORT"
    stop_pct = (signal.stop_loss - signal.entry) / signal.entry * 100.0
    tps = " / ".join(_fmt_price(t) for t in signal.take_profits)
    rr = signal.rr
    base = signal.symbol.split("/")[0]
    qty, risk_usdt = position_size(
        signal, config.account.size_usdt, config.account.risk_per_trade_pct
    )

    lines = [
        f"{arrow}  {signal.symbol}  ({signal.timeframe})",
        f"Стратегия: {signal.strategy}",
        f"Вход:   {_fmt_price(signal.entry)}",
        f"Стоп:   {_fmt_price(signal.stop_loss)}  ({stop_pct:+.1f}%)",
        f"Тейк:   {tps}",
        f"R:R:    1 : {rr:.1f}" if rr else "R:R:    n/a",
        f"Размер: {qty:.4f} {base}  (риск {config.account.risk_per_trade_pct:.0f}% от "
        f"{config.account.size_usdt:.0f} USDT)",
        f"Conf:   {signal.confidence:.2f}",
        f"Причина: {signal.reason}",
        f"⏱ {signal.created_at.strftime('%Y-%m-%d %H:%M')} UTC",
    ]
    return "\n".join(lines)


def format_metrics(m: Metrics, header: str = "") -> str:
    pf = "∞" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    lines = [
        header,
        f"Сделок: {m.trades}  (W:{m.wins} / L:{m.losses})",
        f"Winrate: {m.winrate:.1f}%",
        f"Avg win/loss (R): {m.avg_win_r:.2f} / {m.avg_loss_r:.2f}",
        f"Profit factor: {pf}",
        f"Expectancy: {m.expectancy_r:.3f} R",
        f"Total return: {m.total_return_pct:.2f}%",
        f"CAGR: {m.cagr_pct:.2f}%",
        f"Max DD: {m.max_drawdown_pct:.2f}%",
        f"Sharpe / Sortino: {m.sharpe:.2f} / {m.sortino:.2f}",
        f"Exposure: {m.exposure_pct:.1f}%",
    ]
    return "\n".join(x for x in lines if x)


def format_backtest(result: BacktestResult) -> str:
    head = f"📊 Backtest {result.strategy} {result.symbol} {result.timeframe}"
    parts = [
        head,
        format_metrics(result.metrics_all, "— Все сделки —"),
        "",
        format_metrics(result.metrics_is, "— In-sample —"),
        "",
        format_metrics(result.metrics_oos, "— Out-of-sample —"),
        "",
        "⚠️ Хороший бэктест ≠ будущая прибыль. Смотри форвард-трекер.",
    ]
    return "\n".join(parts)


def format_analysis(a: Analysis, config: AppConfig) -> str:
    if a.direction is None:
        votes = ", ".join(
            f"{_VOTE_LABEL.get(k, k)}:{'+' if v > 0 else ('-' if v < 0 else '0')}"
            for k, v in a.votes.items()
        )
        return (
            f"⚪ {a.symbol} ({a.timeframe}) — нет чёткого перевеса, лучше подождать.\n"
            f"Индикаторы: {votes or 'n/a'}\n"
            f"Волатильность (ATR): {a.atr_pct:.2f}% от цены"
        )

    arrow = "🟢 LONG" if a.direction == SignalDirection.LONG else "🔴 SHORT"
    stop_pct = (a.stop_loss - a.entry) / a.entry * 100.0
    tps = " / ".join(_fmt_price(t) for t in a.take_profits)
    base = a.symbol.split("/")[0]
    agree = ", ".join(
        f"{_VOTE_LABEL.get(k, k)} {'↑' if v > 0 else ('↓' if v < 0 else '·')}"
        for k, v in a.votes.items()
    )
    lines = [
        f"{arrow}  {a.symbol}  ({a.timeframe})  — анализ сейчас",
        f"Согласованность сигналов: {int(a.score * 100)}%",
        f"Динамика: {_MOMENTUM_LABEL.get(a.momentum, a.momentum)} ({a.change_pct:+.1f}% за окно)",
        f"Индикаторы: {agree}",
        "",
        f"Вход:   {_fmt_price(a.entry)}",
        f"Стоп:   {_fmt_price(a.stop_loss)}  ({stop_pct:+.1f}%)",
        f"Тейк:   {tps}",
        f"R:R:    1 : {a.rr:.1f}" if a.rr else "R:R:    n/a",
        "",
        f"⚖️ Риск сделки: {_RISK_LABEL[a.risk_level]}",
        f"Волатильность (ATR): {a.atr_pct:.2f}% от цены",
        f"Рекоменд. риск: {a.suggested_risk_pct:.2f}% "
        f"(база {config.account.risk_per_trade_pct:.1f}%, скорректировано по риску)",
        f"Размер: {a.qty:.4f} {base}  (≈ {a.risk_usdt:.1f} USDT под риском)",
        "",
        "⚠️ Это подсказка, не финансовый совет. Исполняешь сам.",
    ]
    return "\n".join(lines)


def _mover_line(a: Analysis) -> str:
    return f"  {a.symbol}: {a.change_pct:+.1f}%  {_MOMENTUM_LABEL.get(a.momentum, '')}"


def format_market_overview(ov: MarketOverview) -> str:
    if ov.pairs == 0:
        return "Нет данных для оценки рынка."

    lines = [
        f"🌐 Динамика крипторынка ({ov.timeframe}) — {ov.pairs} монет",
        "",
        f"Режим: {_REGIME_LABEL.get(ov.regime, ov.regime)}",
        f"Волатильность: {_VOL_LABEL.get(ov.volatility, ov.volatility)} "
        f"(ср. ATR {ov.avg_atr_pct:.2f}%)",
        f"Ширина рынка: {ov.breadth_pct:.0f}% в аптренде "
        f"(🟢{ov.bullish} / 🔴{ov.bearish} / ⚪{ov.neutral})",
        f"Средн. движение за окно: {ov.avg_change_pct:+.1f}%",
    ]
    if ov.btc is not None:
        lines.append(
            f"BTC (ориентир): {_MOMENTUM_LABEL.get(ov.btc.momentum, ov.btc.momentum)} "
            f"({ov.btc.change_pct:+.1f}%)"
        )
    if ov.leaders:
        lines.append("")
        lines.append("📈 Лидеры:")
        lines.extend(_mover_line(a) for a in ov.leaders)
    if ov.laggards:
        lines.append("📉 Аутсайдеры:")
        lines.extend(_mover_line(a) for a in ov.laggards)
    lines.append("")
    lines.append("⚠️ Это обзор для контекста, не финансовый совет.")
    return "\n".join(lines)
