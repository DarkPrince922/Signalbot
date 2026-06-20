"""Inline keyboard builders for the bot UI."""
from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from ..core.types import SignalDirection
from ..engine.analyzer import Analysis

# callback_data prefixes
CB_SCAN = "scan"
CB_PICK = "pick"      # pick:<symbol>:<tf>
CB_TRACK = "track"    # track:<symbol>:<tf>
CB_OPEN = "open"
CB_STATS = "stats"
CB_STRATS = "strats"
CB_MENU = "menu"


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔎 Анализ сейчас", callback_data=CB_SCAN)],
            [
                InlineKeyboardButton(text="📂 Открытые", callback_data=CB_OPEN),
                InlineKeyboardButton(text="📈 Stats", callback_data=CB_STATS),
            ],
            [InlineKeyboardButton(text="⚙️ Стратегии", callback_data=CB_STRATS)],
        ]
    )


def _arrow(direction: SignalDirection | None) -> str:
    if direction == SignalDirection.LONG:
        return "🟢"
    if direction == SignalDirection.SHORT:
        return "🔴"
    return "⚪"


_RISK_EMOJI = {"LOW": "🟩", "MEDIUM": "🟨", "HIGH": "🟥", "—": "⬜"}


def scan_results(analyses: list[Analysis]) -> InlineKeyboardMarkup:
    """One button per pair: arrow + symbol + direction + score + risk."""
    rows: list[list[InlineKeyboardButton]] = []
    for a in analyses:
        if a.direction is None:
            label = f"⚪ {a.symbol}: нет перевеса"
            cb = f"{CB_PICK}:{a.symbol}:{a.timeframe}"
        else:
            label = (
                f"{_arrow(a.direction)} {a.symbol} {a.direction.value} "
                f"· {int(a.score * 100)}% · риск {_RISK_EMOJI[a.risk_level]}"
            )
            cb = f"{CB_PICK}:{a.symbol}:{a.timeframe}"
        rows.append([InlineKeyboardButton(text=label, callback_data=cb)])
    rows.append([InlineKeyboardButton(text="🔄 Обновить", callback_data=CB_SCAN)])
    rows.append([InlineKeyboardButton(text="⬅️ Меню", callback_data=CB_MENU)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def analysis_actions(analysis: Analysis) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if analysis.is_actionable:
        rows.append(
            [
                InlineKeyboardButton(
                    text="✅ Отслеживать этот сигнал",
                    callback_data=f"{CB_TRACK}:{analysis.symbol}:{analysis.timeframe}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(text="⬅️ К списку", callback_data=CB_SCAN),
            InlineKeyboardButton(text="🏠 Меню", callback_data=CB_MENU),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
