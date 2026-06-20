"""Equity-curve rendering for backtest reports."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def equity_curve_png(equity: list[float], title: str, split_index: int | None = None) -> bytes:
    """Render an equity curve to PNG bytes. split_index marks IS/OOS boundary."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(equity, color="#1f77b4", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (base 1.0)")
    ax.grid(True, alpha=0.3)
    if split_index is not None and 0 < split_index < len(equity):
        ax.axvline(split_index, color="#d62728", linestyle="--", alpha=0.7, label="IS / OOS")
        ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
