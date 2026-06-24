"""Тесты каузальной (неперерисовывающей) детекции паттернов.

Проверяем два ключевых инварианта для T1.1:

1. ПОДТВЕРЖДЕНИЕ: чёткий синтетический паттерн подтверждается (``is_confirmed``)
   на баре пробоя, который наступает СТРОГО ПОСЛЕ бара-кандидата.
2. ОТСУТСТВИЕ ПЕРЕРИСОВКИ (no repaint): если подать только данные ДО бара
   подтверждения, паттерн НЕ должен подтверждаться. Это гарантирует, что
   сигнал не появляется задним числом.

Развороты (double top/bottom) и каналы стабильно классифицируются базовым
классификатором, поэтому проверяются через полный конвейер
``extract_features_causal``. Голова-плечи и треугольники синтетически плохо
распознаются базовым классификатором, поэтому их семейство подтверждения
проверяется напрямую через построители кандидатов
(``_build_candidate`` + ``_confirm_candidate``).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.structure_extractor import StructureExtractor, StructureType
from src.backend.binance_scanner import CandleData


@pytest.fixture(scope="module")
def ext():
    return StructureExtractor()


def _mk(closes):
    """Строит список свечей из массива цен закрытия."""
    return [
        CandleData(
            open_time=i * 60000,
            open=float(c),
            high=float(c) * 1.001,
            low=float(c) * 0.999,
            close=float(c),
            volume=1000.0,
            close_time=i * 60000 + 59999,
        )
        for i, c in enumerate(closes)
    ]


def _cat(*parts):
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def _build_and_confirm(ext, st, seg):
    """Прямое построение каузального кандидата нужного типа и его подтверждение."""
    candles = _mk(seg)
    closes = np.array([c.close for c in candles], dtype=float)
    norm = ext.normalize_line(closes)
    norm_pivots = ext.detect_pivots(norm)
    raw_pivots = ext._map_pivots_to_raw(norm_pivots, closes)
    cand = ext._build_candidate(st, closes, raw_pivots, candles)
    if cand is None:
        return None, None
    res = ext._confirm_candidate(cand, candles)
    return cand, res


# --- Синтетические паттерны ------------------------------------------------

def _double_top():
    return _cat(
        np.linspace(100, 120, 15),
        np.linspace(120, 110, 8),
        np.linspace(110, 120, 8),
        np.linspace(120, 104, 12),
    )


def _double_bottom():
    return _cat(
        np.linspace(120, 100, 15),
        np.linspace(100, 110, 8),
        np.linspace(110, 100, 8),
        np.linspace(100, 116, 12),
    )


def _channel_up():
    return _cat(
        np.linspace(100, 112, 6),
        np.linspace(112, 106, 4),
        np.linspace(106, 118, 6),
        np.linspace(118, 112, 4),
        np.linspace(112, 124, 6),
        np.linspace(124, 118, 4),
        np.linspace(118, 130, 6),
    )


def _channel_down():
    return _cat(
        np.linspace(130, 118, 6),
        np.linspace(118, 124, 4),
        np.linspace(124, 112, 6),
        np.linspace(112, 118, 4),
        np.linspace(118, 106, 6),
        np.linspace(106, 112, 4),
        np.linspace(112, 100, 6),
    )


def _head_shoulders():
    return _cat(
        np.linspace(95, 115, 8),
        np.linspace(115, 101, 4),
        np.linspace(101, 130, 9),
        np.linspace(130, 101, 9),
        np.linspace(101, 115, 4),
        np.linspace(115, 92, 9),
    )


def _inv_head_shoulders():
    return _cat(
        np.linspace(125, 105, 8),
        np.linspace(105, 119, 4),
        np.linspace(119, 90, 9),
        np.linspace(90, 119, 9),
        np.linspace(119, 105, 4),
        np.linspace(105, 128, 9),
    )


def _ascending_triangle():
    return _cat(
        np.linspace(100, 120, 8),
        np.linspace(120, 108, 5),
        np.linspace(108, 120, 6),
        np.linspace(120, 112, 5),
        np.linspace(112, 120, 5),
        np.linspace(120, 134, 8),
    )


def _descending_triangle():
    return _cat(
        np.linspace(120, 100, 8),
        np.linspace(100, 112, 5),
        np.linspace(112, 100, 6),
        np.linspace(100, 108, 5),
        np.linspace(108, 100, 5),
        np.linspace(100, 86, 8),
    )


# --- Полный конвейер: развороты и каналы -----------------------------------

PIPELINE_CASES = [
    ("double_top", _double_top, "double_top"),
    ("double_bottom", _double_bottom, "double_bottom"),
    ("channel_up", _channel_up, "channel_up"),
    ("channel_down", _channel_down, "channel_down"),
]


@pytest.mark.parametrize("name,gen,expected_type", PIPELINE_CASES)
def test_pipeline_confirms_after_candidate(ext, name, gen, expected_type):
    """Полный конвейер подтверждает паттерн на баре пробоя ПОСЛЕ кандидата."""
    seg = gen()
    f = ext.extract_features_causal(_mk(seg))
    assert f is not None, f"{name}: features is None"
    assert f.structure_type.value == expected_type, (
        f"{name}: classified as {f.structure_type.value}, expected {expected_type}"
    )
    assert f.is_confirmed is True, f"{name}: not confirmed"
    assert f.is_invalidated is False, f"{name}: unexpectedly invalidated"
    assert f.candidate_index >= 0, f"{name}: no candidate index"
    assert f.confirmation_index > f.candidate_index, (
        f"{name}: confirmation_index={f.confirmation_index} must be after "
        f"candidate_index={f.candidate_index}"
    )
    assert f.is_pattern_active is True, f"{name}: pattern should be active"


@pytest.mark.parametrize("name,gen,expected_type", PIPELINE_CASES)
def test_pipeline_no_repaint(ext, name, gen, expected_type):
    """Усечение данных до бара подтверждения убирает подтверждение (no repaint)."""
    seg = gen()
    f = ext.extract_features_causal(_mk(seg))
    assert f is not None and f.is_confirmed, f"{name}: precondition failed"

    truncated = seg[: f.confirmation_index]
    f2 = ext.extract_features_causal(_mk(truncated))
    confirmed = bool(f2 and f2.is_confirmed)
    assert confirmed is False, (
        f"{name}: repaint detected — pattern confirmed on data truncated "
        f"before the breakout bar (index {f.confirmation_index})"
    )


@pytest.mark.parametrize("name,gen,expected_type", PIPELINE_CASES)
def test_pipeline_confidence_synced_to_candidate(ext, name, gen, expected_type):
    """Регрессия: после переопределения типа каузальным кандидатом поля
    ``pattern_confidence`` и ``detected_patterns`` синхронизированы с кандидатом,
    а не остаются от легаси-классификатора (иначе оценка совпадения искажается)."""
    seg = gen()
    f = ext.extract_features_causal(_mk(seg))
    assert f is not None and f.is_confirmed, f"{name}: precondition failed"
    tv = f.structure_type.value
    assert f.pattern_confidence > 0.0, f"{name}: pattern_confidence not set"
    assert tv in f.detected_patterns, (
        f"{name}: detected_patterns missing causal type {tv}"
    )
    assert f.detected_patterns[tv] == pytest.approx(f.pattern_confidence), (
        f"{name}: detected_patterns[{tv}]={f.detected_patterns[tv]} != "
        f"pattern_confidence={f.pattern_confidence}"
    )


# --- Прямые построители: голова-плечи и треугольники -----------------------

BUILDER_CASES = [
    ("head_shoulders", StructureType.HEAD_SHOULDERS, _head_shoulders, "down"),
    ("inv_head_shoulders", StructureType.INV_HEAD_SHOULDERS, _inv_head_shoulders, "up"),
    ("ascending_triangle", StructureType.ASCENDING_TRIANGLE, _ascending_triangle, "up"),
    ("descending_triangle", StructureType.DESCENDING_TRIANGLE, _descending_triangle, "down"),
]


@pytest.mark.parametrize("name,st,gen,direction", BUILDER_CASES)
def test_builder_confirms_after_candidate(ext, name, st, gen, direction):
    """Семейство подтверждения H&S/треугольников срабатывает на пробое после кандидата."""
    seg = gen()
    cand, res = _build_and_confirm(ext, st, seg)
    assert cand is not None, f"{name}: candidate not built"
    assert cand.structure_type == st, f"{name}: wrong candidate type {cand.structure_type}"
    assert cand.direction == direction, (
        f"{name}: direction={cand.direction}, expected {direction}"
    )
    assert res is not None and res.confirmed is True, f"{name}: not confirmed"
    assert res.confirmation_index > cand.candidate_index, (
        f"{name}: confirmation_index={res.confirmation_index} must be after "
        f"candidate_index={cand.candidate_index}"
    )


@pytest.mark.parametrize("name,st,gen,direction", BUILDER_CASES)
def test_builder_no_repaint(ext, name, st, gen, direction):
    """Усечение до бара подтверждения убирает подтверждение для H&S/треугольников."""
    seg = gen()
    cand, res = _build_and_confirm(ext, st, seg)
    assert cand is not None and res is not None and res.confirmed, (
        f"{name}: precondition failed"
    )

    truncated = seg[: res.confirmation_index]
    cand2, res2 = _build_and_confirm(ext, st, truncated)
    confirmed = bool(res2 and res2.confirmed)
    assert confirmed is False, (
        f"{name}: repaint detected — confirmed on data truncated before the "
        f"breakout bar (index {res.confirmation_index})"
    )


def test_insufficient_data_returns_none(ext):
    """Слишком короткий ряд свечей не даёт каузальных признаков."""
    seg = np.linspace(100, 110, 5)
    assert ext.extract_features_causal(_mk(seg)) is None
