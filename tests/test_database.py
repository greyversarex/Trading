"""Тесты слоя базы данных (SQLite через aiosqlite).

Проверяют инициализацию схемы, персистентность структур/матчей и корректность
механизма обратной связи, включая самодостаточные записи фидбэка (без match_id).
Каждый тест использует изолированную временную БД, чтобы не затрагивать данные
приложения.
"""

import asyncio
import os
import tempfile

import numpy as np

from src.backend.database import Database
from src.backend.structure_extractor import (
    StructureFeatures,
    StructureType,
    PivotPoint,
)
from src.backend.similarity_matcher import MatchResult


def _make_features() -> StructureFeatures:
    return StructureFeatures(
        pivot_points=[
            PivotPoint(index=0, value=1.0, is_high=False, relative_position=0.0),
            PivotPoint(index=5, value=2.0, is_high=True, relative_position=0.5),
        ],
        normalized_line=np.linspace(0.0, 1.0, 10),
        pivot_sequence=[0.0, 1.0],
        relative_distances=[0.5],
        trend_direction=1.0,
        volatility=0.2,
        compression_ratio=0.3,
        structure_type=StructureType.DOUBLE_TOP,
        feature_vector=np.array([0.1, 0.2, 0.3]),
    )


def _make_match(symbol: str = "BTC") -> MatchResult:
    return MatchResult(
        symbol=symbol,
        timeframe="1h",
        similarity_score=87.5,
        structure_type=StructureType.DOUBLE_TOP,
        timestamp="2026-07-04T00:00:00",
        is_mirrored=False,
        normalized_line=[0.0, 0.5, 1.0],
    )


def _fresh_db() -> Database:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    db = Database(db_path=path)
    asyncio.run(db.initialize())
    return db


def _cleanup(db: Database):
    if os.path.exists(db.db_path):
        os.remove(db.db_path)


def test_initialize_creates_schema():
    """initialize() создаёт все таблицы; повторный вызов идемпотентен."""
    db = _fresh_db()
    try:
        assert os.path.exists(db.db_path)
        # повторная инициализация не должна падать
        asyncio.run(db.initialize())
        structures = asyncio.run(db.get_all_structures())
        assert structures == []
    finally:
        _cleanup(db)


def test_save_and_get_structure_roundtrip():
    """Сохранённая структура читается обратно с сохранением полей."""
    db = _fresh_db()
    try:
        sid = asyncio.run(db.save_structure("тест", _make_features()))
        assert isinstance(sid, int) and sid > 0

        loaded = asyncio.run(db.get_structure(sid))
        assert loaded is not None
        assert loaded["name"] == "тест"
        assert loaded["structure_type"] == StructureType.DOUBLE_TOP.value
        assert loaded["trend_direction"] == 1.0

        all_structs = asyncio.run(db.get_all_structures())
        assert len(all_structs) == 1
    finally:
        _cleanup(db)


def test_get_missing_structure_returns_none():
    db = _fresh_db()
    try:
        assert asyncio.run(db.get_structure(999)) is None
    finally:
        _cleanup(db)


def test_save_and_get_matches():
    """Матчи персистятся и извлекаются по structure_id."""
    db = _fresh_db()
    try:
        sid = asyncio.run(db.save_structure("s", _make_features()))
        mid = asyncio.run(db.save_match(sid, _make_match("ETH")))
        assert isinstance(mid, int) and mid > 0

        matches = asyncio.run(db.get_matches(sid))
        assert len(matches) == 1
        assert matches[0]["symbol"] == "ETH"
        assert matches[0]["similarity_score"] == 87.5
        assert matches[0]["is_mirrored"] == 0
    finally:
        _cleanup(db)


def test_feedback_stats_from_matches():
    """get_feedback_stats агрегирует релевантность по матчам структуры."""
    db = _fresh_db()
    try:
        sid = asyncio.run(db.save_structure("s", _make_features()))
        mid1 = asyncio.run(db.save_match(sid, _make_match("A")))
        mid2 = asyncio.run(db.save_match(sid, _make_match("B")))

        asyncio.run(db.save_feedback(is_relevant=True, match_id=mid1))
        asyncio.run(db.save_feedback(is_relevant=False, match_id=mid2))

        stats = asyncio.run(db.get_feedback_stats(sid))
        assert stats["relevant"] == 1
        assert stats["irrelevant"] == 1
    finally:
        _cleanup(db)


def test_self_contained_feedback_without_match_id():
    """Фидбэк без match_id (скан по типу) сохраняет свои метаданные и features."""
    db = _fresh_db()
    try:
        asyncio.run(
            db.save_feedback(
                is_relevant=True,
                symbol="SOL",
                timeframe="4h",
                structure_type=StructureType.TRIANGLE.value,
                features={"score": 0.9, "vol": 0.4},
            )
        )
        all_fb = asyncio.run(db.get_all_feedback())
        assert len(all_fb) == 1
        rec = all_fb[0]
        assert rec["is_relevant"] is True
        assert rec["symbol"] == "SOL"
        assert rec["timeframe"] == "4h"
        assert rec["structure_type"] == StructureType.TRIANGLE.value
        assert rec["features"] == {"score": 0.9, "vol": 0.4}
    finally:
        _cleanup(db)


def test_delete_structure_cascades():
    """delete_structure удаляет структуру и связанные матчи/фидбэк."""
    db = _fresh_db()
    try:
        sid = asyncio.run(db.save_structure("s", _make_features()))
        mid = asyncio.run(db.save_match(sid, _make_match()))
        asyncio.run(db.save_feedback(is_relevant=True, match_id=mid))

        asyncio.run(db.delete_structure(sid))

        assert asyncio.run(db.get_structure(sid)) is None
        assert asyncio.run(db.get_matches(sid)) == []
        stats = asyncio.run(db.get_feedback_stats(sid))
        assert stats["relevant"] == 0
        assert stats["irrelevant"] == 0
    finally:
        _cleanup(db)
