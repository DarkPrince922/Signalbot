"""aiogram entrypoint: wires config, DB, provider, engine, scheduler and bot."""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
from pathlib import Path

from aiogram import Bot, Dispatcher
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..core.config import AppConfig, load_config
from ..core.db import Database
from ..data.ccxt_provider import CCXTProvider
from ..engine.analyzer import Analyzer
from ..engine.live_engine import LiveEngine
from ..engine.tracker import Tracker
from .formatters import format_signal
from .handlers import BotContext, build_router

log = logging.getLogger(__name__)


def setup_logging(config: AppConfig) -> None:
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = config.logging.file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(fh)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _cron_for_timeframe(tf: str, offset_seconds: int) -> CronTrigger:
    """Map a timeframe to a cron trigger that fires just after candle close."""
    sec = offset_seconds % 60
    mapping = {
        "1m": dict(second=sec),
        "5m": dict(minute="*/5", second=sec),
        "15m": dict(minute="*/15", second=sec),
        "30m": dict(minute="*/30", second=sec),
        "1h": dict(minute=0, second=sec),
        "2h": dict(hour="*/2", minute=0, second=sec),
        "4h": dict(hour="*/4", minute=0, second=sec),
        "6h": dict(hour="*/6", minute=0, second=sec),
        "12h": dict(hour="*/12", minute=0, second=sec),
        "1d": dict(hour=0, minute=0, second=sec),
    }
    return CronTrigger(timezone="UTC", **mapping.get(tf, dict(minute=0, second=sec)))


async def run() -> None:
    config = load_config()
    setup_logging(config)

    if not config.secrets.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (see .env.example)")

    db = Database()
    db.init()
    provider = CCXTProvider(config.data.exchange_id, db=db)
    engine = LiveEngine(config, db, provider)
    tracker = Tracker(config, db, provider)
    analyzer = Analyzer(config, provider, tracker)

    bot = Bot(token=config.secrets.telegram_bot_token)
    dp = Dispatcher()
    ctx = BotContext(
        config=config, db=db, provider=provider, engine=engine, tracker=tracker, analyzer=analyzer
    )
    dp.include_router(build_router(ctx))

    async def notify(signal) -> None:
        try:
            await bot.send_message(config.secrets.telegram_user_id, format_signal(signal, config))
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to send signal: %s", exc)

    async def poll_timeframe(tf: str) -> None:
        log.info("Polling timeframe %s", tf)
        # tracker first so closed signals settle on the freshest candles
        for closed in tracker.update_open_signals():
            try:
                await bot.send_message(
                    config.secrets.telegram_user_id,
                    f"Сигнал #{closed.id} {closed.symbol} {closed.strategy} закрыт: "
                    f"{closed.status} ({closed.realized_r:+.2f}R)",
                )
            except Exception as exc:  # pragma: no cover
                log.warning("close notify failed: %s", exc)
        for sig in engine.poll_once(timeframe=tf):
            await notify(sig)

    scheduler = AsyncIOScheduler(timezone="UTC")
    for tf in sorted(engine.active_timeframes()):
        scheduler.add_job(
            poll_timeframe,
            _cron_for_timeframe(tf, config.engine.poll_offset_seconds),
            args=[tf],
            id=f"poll_{tf}",
            max_instances=1,
            coalesce=True,
        )
    scheduler.start()

    log.info(
        "Signalbot started. exchange=%s pairs=%s strategies=%s timeframes=%s",
        config.data.exchange_id,
        engine.pairs(),
        sorted(engine.enabled_strategies()),
        sorted(engine.active_timeframes()),
    )

    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
