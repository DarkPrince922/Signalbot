"""Message formatting for signals and reports."""
from __future__ import annotations

from ..core.config import AppConfig
from ..core.types import Signal, SignalDirection
from ..engine.backtester import BacktestResult
from ..engine.metrics import Metrics


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
