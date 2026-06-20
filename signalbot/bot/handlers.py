"""Telegram command handlers (aiogram 3.x). Single-user whitelist enforced."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, CallbackQuery, Message
from sqlalchemy import desc, select

from ..core.config import AppConfig
from ..core.db import Database
from ..core.models import SignalRow
from ..core.types import Signal, SignalDirection, utcnow
from ..data.provider import DataProvider, timeframe_ms
from ..engine.analyzer import Analysis, Analyzer
from ..engine.backtester import Backtester
from ..engine.live_engine import LiveEngine
from ..engine.plotting import equity_curve_png
from ..engine.tracker import Tracker
from ..strategies.registry import build_strategy, discover_strategies
from . import keyboards as kb
from .formatters import (
    format_analysis,
    format_backtest,
    format_market_overview,
    format_metrics,
    format_signal,
)

log = logging.getLogger(__name__)


@dataclass
class BotContext:
    config: AppConfig
    db: Database
    provider: DataProvider
    engine: LiveEngine
    tracker: Tracker
    analyzer: Analyzer


HELP_TEXT = (
    "📡 Signalbot — инструмент для исследования стратегий (НЕ торгует).\n\n"
    "/signals on|off — вкл/выкл уведомления\n"
    "/strategies — список стратегий и статус\n"
    "/enable <strategy> | /disable <strategy>\n"
    "/pairs | /addpair <SYM> | /removepair <SYM>\n"
    "/stats [strategy] [live|backtest] — метрики (по умолчанию live)\n"
    "/backtest <strategy> <pair> <tf> <from> <to> — прогон + equity curve\n"
    "/last [N] — последние сигналы\n"
    "/open — открытые отслеживаемые сигналы\n"
    "/menu — кнопки управления\n"
    "/scan [tf] — проанализировать пары прямо сейчас\n"
    "/market [tf] — динамика крипторынка в целом\n\n"
    "⚠️ Это анализ, а не финансовый совет."
)


def build_router(ctx: BotContext) -> Router:
    router = Router()
    allowed_id = ctx.config.secrets.telegram_user_id

    @router.message(F.from_user.id != allowed_id)
    async def reject(message: Message) -> None:
        # silently ignore anyone who is not the whitelisted user
        log.info("Ignored message from non-whitelisted user %s", message.from_user.id)

    @router.callback_query(F.from_user.id != allowed_id)
    async def reject_cb(cb: CallbackQuery) -> None:
        await cb.answer("Доступ запрещён", show_alert=False)

    @router.message(Command("start"))
    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=kb.main_menu())

    @router.message(Command("menu"))
    async def cmd_menu(message: Message) -> None:
        await message.answer("Меню управления:", reply_markup=kb.main_menu())

    @router.message(Command("signals"))
    async def cmd_signals(message: Message, command: CommandObject) -> None:
        arg = (command.args or "").strip().lower()
        if arg not in {"on", "off"}:
            await message.answer("Использование: /signals on|off")
            return
        ctx.engine.set_signals_enabled(arg == "on")
        await message.answer(f"Уведомления: {'включены' if arg == 'on' else 'выключены'}")

    @router.message(Command("strategies"))
    async def cmd_strategies(message: Message) -> None:
        available = discover_strategies()
        overrides = ctx.db.get_setting("strategy_enabled", {}) or {}
        lines = ["Стратегии:"]
        for name in sorted(available):
            sc = ctx.config.strategies.get(name)
            enabled = overrides.get(name, sc.enabled if sc else False)
            tf = sc.timeframe if sc else available[name].timeframe
            params = sc.params if sc else available[name].default_params
            status = "✅" if enabled else "⛔"
            lines.append(f"{status} {name} ({tf}) {params}")
        await message.answer("\n".join(lines))

    @router.message(Command("enable"))
    async def cmd_enable(message: Message, command: CommandObject) -> None:
        await _toggle_strategy(message, command, True)

    @router.message(Command("disable"))
    async def cmd_disable(message: Message, command: CommandObject) -> None:
        await _toggle_strategy(message, command, False)

    async def _toggle_strategy(message: Message, command: CommandObject, enable: bool) -> None:
        name = (command.args or "").strip()
        if not name or name not in discover_strategies():
            await message.answer(f"Неизвестная стратегия. Доступны: {sorted(discover_strategies())}")
            return
        overrides = ctx.db.get_setting("strategy_enabled", {}) or {}
        overrides[name] = enable
        ctx.db.set_setting("strategy_enabled", overrides)
        await message.answer(f"{name}: {'включена' if enable else 'выключена'}")

    @router.message(Command("pairs"))
    async def cmd_pairs(message: Message) -> None:
        pairs = ctx.engine.pairs()
        await message.answer("Пары: " + ", ".join(pairs))

    @router.message(Command("addpair"))
    async def cmd_addpair(message: Message, command: CommandObject) -> None:
        sym = (command.args or "").strip().upper()
        if not sym:
            await message.answer("Использование: /addpair BTC/USDT")
            return
        pairs = list(ctx.engine.pairs())
        if sym not in pairs:
            pairs.append(sym)
            ctx.db.set_setting("pairs", pairs)
        await message.answer("Пары: " + ", ".join(pairs))

    @router.message(Command("removepair"))
    async def cmd_removepair(message: Message, command: CommandObject) -> None:
        sym = (command.args or "").strip().upper()
        pairs = [p for p in ctx.engine.pairs() if p != sym]
        ctx.db.set_setting("pairs", pairs)
        await message.answer("Пары: " + (", ".join(pairs) or "—"))

    @router.message(Command("stats"))
    async def cmd_stats(message: Message, command: CommandObject) -> None:
        args = (command.args or "").split()
        strategy = None
        mode = "live"
        for a in args:
            if a.lower() in {"live", "backtest"}:
                mode = a.lower()
            else:
                strategy = a
        if mode == "live":
            m = ctx.tracker.live_metrics(strategy)
            head = f"📈 Live stats {strategy or '(все)'}"
            await message.answer(format_metrics(m, head))
        else:
            await message.answer("Для backtest используй /backtest <strategy> <pair> <tf> <from> <to>")

    @router.message(Command("last"))
    async def cmd_last(message: Message, command: CommandObject) -> None:
        n = 5
        if command.args and command.args.strip().isdigit():
            n = min(int(command.args.strip()), 30)
        with ctx.db.session() as s:
            rows = s.execute(
                select(SignalRow).order_by(desc(SignalRow.created_at)).limit(n)
            ).scalars().all()
        if not rows:
            await message.answer("Сигналов пока нет.")
            return
        lines = [f"Последние {len(rows)} сигналов:"]
        for r in rows:
            rr = f"{r.realized_r:+.2f}R" if r.realized_r is not None else "—"
            lines.append(
                f"{r.created_at:%m-%d %H:%M} {r.direction} {r.symbol} {r.strategy} "
                f"[{r.status}] {rr}"
            )
        await message.answer("\n".join(lines))

    @router.message(Command("open"))
    async def cmd_open(message: Message) -> None:
        with ctx.db.session() as s:
            rows = s.execute(
                select(SignalRow).where(SignalRow.status == "OPEN").order_by(SignalRow.created_at)
            ).scalars().all()
        if not rows:
            await message.answer("Открытых сигналов нет.")
            return
        lines = ["Открытые сигналы:"]
        for r in rows:
            lines.append(
                f"#{r.id} {r.direction} {r.symbol} {r.strategy} вход {r.entry} "
                f"стоп {r.stop_loss} тейк {r.take_profits[0]}"
            )
        await message.answer("\n".join(lines))

    @router.message(Command("backtest"))
    async def cmd_backtest(message: Message, command: CommandObject) -> None:
        parts = (command.args or "").split()
        if len(parts) < 5:
            await message.answer(
                "Использование: /backtest <strategy> <pair> <tf> <from YYYY-MM-DD> <to YYYY-MM-DD>"
            )
            return
        name, pair, tf, dfrom, dto = parts[0], parts[1].upper(), parts[2], parts[3], parts[4]
        try:
            since = int(datetime.strptime(dfrom, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            until = int(datetime.strptime(dto, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            await message.answer("Даты в формате YYYY-MM-DD")
            return

        sc = ctx.config.strategies.get(name)
        params = sc.params if sc else None
        try:
            strat = build_strategy(name, params)
        except KeyError as exc:
            await message.answer(str(exc))
            return

        await message.answer("⏳ Гружу историю и считаю...")
        limit = int((until - since) / timeframe_ms(tf)) + strat.warmup + 10
        df = ctx.provider.fetch_ohlcv(pair, tf, since=since, limit=limit)
        df = df[df["timestamp"] <= datetime.fromtimestamp(until / 1000, tz=timezone.utc)]
        if df.empty:
            await message.answer("Нет данных за период.")
            return

        result = Backtester(ctx.config).run(strat, df, pair, tf)
        await message.answer(format_backtest(result))
        png = equity_curve_png(
            result.equity_curve, f"{name} {pair} {tf}", result.split_index
        )
        await message.answer_document(
            BufferedInputFile(png, filename="equity.png"),
            caption="Equity curve (пунктир = граница IS/OOS)",
        )

    # ---------------- Analysis / "where to enter now" ----------------

    async def _do_scan() -> tuple[str, list[Analysis]]:
        pairs = ctx.engine.pairs()
        tf = ctx.config.analysis.timeframe
        analyses = ctx.analyzer.scan(pairs, tf)
        actionable = [a for a in analyses if a.is_actionable]
        header = (
            f"🔎 Анализ {tf} · {len(pairs)} пар · с перевесом: {len(actionable)}\n"
            "Выбери пару, чтобы увидеть план и риск:"
        )
        return header, analyses

    @router.message(Command("scan"))
    async def cmd_scan(message: Message, command: CommandObject) -> None:
        tf = (command.args or "").strip() or None
        if tf:
            ctx.config.analysis.timeframe = tf
        await message.answer("⏳ Анализирую рынок...")
        header, analyses = await _do_scan()
        if not analyses:
            await message.answer("Нет данных для анализа.")
            return
        await message.answer(header, reply_markup=kb.scan_results(analyses))

    @router.message(Command("market"))
    async def cmd_market(message: Message, command: CommandObject) -> None:
        tf = (command.args or "").strip() or ctx.config.analysis.timeframe
        await message.answer("⏳ Оцениваю динамику рынка...")
        ov = ctx.analyzer.market_overview(tf)
        await message.answer(format_market_overview(ov), reply_markup=kb.market_overview_kb())

    @router.callback_query(F.data == kb.CB_MARKET)
    async def cb_market(cb: CallbackQuery) -> None:
        await cb.answer("Оцениваю рынок...")
        ov = ctx.analyzer.market_overview(ctx.config.analysis.timeframe)
        await cb.message.edit_text(
            format_market_overview(ov), reply_markup=kb.market_overview_kb()
        )

    @router.callback_query(F.data == kb.CB_MENU)
    async def cb_menu(cb: CallbackQuery) -> None:
        await cb.message.edit_text("Меню управления:", reply_markup=kb.main_menu())
        await cb.answer()

    @router.callback_query(F.data == kb.CB_SCAN)
    async def cb_scan(cb: CallbackQuery) -> None:
        await cb.answer("Анализирую...")
        header, analyses = await _do_scan()
        if not analyses:
            await cb.message.edit_text("Нет данных для анализа.", reply_markup=kb.main_menu())
            return
        await cb.message.edit_text(header, reply_markup=kb.scan_results(analyses))

    @router.callback_query(F.data == kb.CB_OPEN)
    async def cb_open(cb: CallbackQuery) -> None:
        with ctx.db.session() as s:
            rows = s.execute(
                select(SignalRow).where(SignalRow.status == "OPEN").order_by(SignalRow.created_at)
            ).scalars().all()
        if not rows:
            text = "Открытых сигналов нет."
        else:
            text = "Открытые сигналы:\n" + "\n".join(
                f"#{r.id} {r.direction} {r.symbol} {r.strategy} вход {r.entry}" for r in rows
            )
        await cb.message.edit_text(text, reply_markup=kb.main_menu())
        await cb.answer()

    @router.callback_query(F.data == kb.CB_STATS)
    async def cb_stats(cb: CallbackQuery) -> None:
        m = ctx.tracker.live_metrics()
        await cb.message.edit_text(
            format_metrics(m, "📈 Live stats (все стратегии)"), reply_markup=kb.main_menu()
        )
        await cb.answer()

    @router.callback_query(F.data == kb.CB_STRATS)
    async def cb_strats(cb: CallbackQuery) -> None:
        available = discover_strategies()
        overrides = ctx.db.get_setting("strategy_enabled", {}) or {}
        lines = ["Стратегии:"]
        for name in sorted(available):
            sc = ctx.config.strategies.get(name)
            enabled = overrides.get(name, sc.enabled if sc else False)
            lines.append(f"{'✅' if enabled else '⛔'} {name}")
        await cb.message.edit_text("\n".join(lines), reply_markup=kb.main_menu())
        await cb.answer()

    @router.callback_query(F.data.startswith(f"{kb.CB_PICK}:"))
    async def cb_pick(cb: CallbackQuery) -> None:
        _, symbol, tf = cb.data.split(":", 2)
        await cb.answer("Считаю план и риск...")
        analysis = ctx.analyzer.analyze_pair(symbol, tf)
        await cb.message.edit_text(
            format_analysis(analysis, ctx.config),
            reply_markup=kb.analysis_actions(analysis),
        )

    @router.callback_query(F.data.startswith(f"{kb.CB_TRACK}:"))
    async def cb_track(cb: CallbackQuery) -> None:
        _, symbol, tf = cb.data.split(":", 2)
        analysis = ctx.analyzer.analyze_pair(symbol, tf)
        if not analysis.is_actionable:
            await cb.answer("Сейчас нет перевеса — нечего отслеживать.", show_alert=True)
            return
        sig = _analysis_to_signal(analysis)
        if ctx.engine._persist_if_new(sig):
            await cb.answer("✅ Добавлено в отслеживание")
            await cb.message.edit_text(
                format_analysis(analysis, ctx.config) + "\n\n✅ Сигнал отслеживается трекером.",
                reply_markup=kb.main_menu(),
            )
        else:
            await cb.answer("Уже отслеживается", show_alert=True)

    return router


def _analysis_to_signal(a: Analysis) -> Signal:
    """Turn a chosen analysis into a virtual signal for the forward tracker."""
    return Signal(
        symbol=a.symbol,
        timeframe=a.timeframe,
        direction=a.direction,
        entry=a.entry,
        stop_loss=a.stop_loss,
        take_profits=a.take_profits,
        strategy="manual_analysis",
        reason=f"Ручной выбор по анализу (риск {a.risk_level}, score {int(a.score * 100)}%)",
        confidence=a.score,
        created_at=utcnow(),
        meta={"risk_level": a.risk_level, "atr_pct": a.atr_pct},
    )
