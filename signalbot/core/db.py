"""Database engine, session factory and runtime settings helpers."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Setting


class Database:
    def __init__(self, url: str = "sqlite:///signalbot.db") -> None:
        if url.startswith("sqlite:///"):
            db_path = url.replace("sqlite:///", "", 1)
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def init(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --- key/value settings, editable at runtime from the bot ---
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            row = s.get(Setting, key)
            if row is None:
                return default
            try:
                return json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                return row.value

    def set_setting(self, key: str, value: Any) -> None:
        with self.session() as s:
            row = s.get(Setting, key)
            payload = json.dumps(value)
            if row is None:
                s.add(Setting(key=key, value=payload))
            else:
                row.value = payload

    def all_settings(self) -> dict[str, Any]:
        with self.session() as s:
            rows = s.execute(select(Setting)).scalars().all()
            out: dict[str, Any] = {}
            for r in rows:
                try:
                    out[r.key] = json.loads(r.value)
                except (json.JSONDecodeError, TypeError):
                    out[r.key] = r.value
            return out
