"""Auto-discovery of Strategy subclasses in this package.

Drop a new file in strategies/ with a Strategy subclass and it is picked up
automatically — no core code changes required.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

from .base import Strategy


def discover_strategies() -> dict[str, type[Strategy]]:
    """Import every module in this package and collect Strategy subclasses."""
    found: dict[str, type[Strategy]] = {}
    package = importlib.import_module(__package__)
    for mod_info in pkgutil.iter_modules(package.__path__):
        if mod_info.name in {"base", "registry", "indicators"}:
            continue
        module = importlib.import_module(f"{__package__}.{mod_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy:
                if obj.__module__ == module.__name__:
                    found[obj.name] = obj
    return found


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    registry = discover_strategies()
    if name not in registry:
        raise KeyError(f"Unknown strategy: {name}. Available: {sorted(registry)}")
    return registry[name](params)
