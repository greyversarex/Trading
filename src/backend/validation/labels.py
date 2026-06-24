"""Хранилище ручной разметки паттернов (SQLite).

Таблица ``validation_labels`` будет использоваться позже для обучения и
обратной связи ML. На этапе 0.2 хранилище создаётся и предоставляет базовый
CRUD; разметка может быть пустой — метрики при этом считаются предварительными.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, List, Optional

DEFAULT_DB_PATH = os.path.join("data", "validation.db")


class LabelStore:
    """Простой доступ к таблице размеченных событий в SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    start_time INTEGER NOT NULL,
                    end_time INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def add_label(
        self,
        symbol: str,
        timeframe: str,
        pattern_type: str,
        start_time: int,
        end_time: int,
        label: str,
    ) -> int:
        """Добавляет размеченное событие и возвращает его id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO validation_labels
                    (symbol, timeframe, pattern_type, start_time, end_time, label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    pattern_type,
                    int(start_time),
                    int(end_time),
                    label,
                    time.time(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_labels(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> List[Dict]:
        """Возвращает разметку с опциональной фильтрацией по символу/ТФ."""
        query = "SELECT * FROM validation_labels"
        conditions: List[str] = []
        params: List = []
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe is not None:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        """Полностью очищает таблицу разметки (для тестов)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM validation_labels")
            conn.commit()
