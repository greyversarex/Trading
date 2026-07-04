"""Тесты устаревания трендов: тренд помечается неактивным, если цена
развернулась против его направления."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.structure_extractor import StructureExtractor, StructureType


@pytest.fixture(scope="module")
def ext():
    return StructureExtractor()


def _cat(*parts):
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def _classify(ext, closes):
    norm = ext.normalize_line(np.asarray(closes, dtype=float))
    pivots = ext.detect_pivots(norm)
    trend = ext.calculate_trend(norm)
    compression = ext.calculate_compression(norm, pivots)
    return ext.classify_structure(trend, compression, pivots, norm)


def test_trend_still_active_up_intact(ext):
    line = ext.normalize_line(np.linspace(100, 140, 60))
    assert ext._trend_still_active(line, "up", float(np.max(line) - np.min(line)))


def test_trend_still_active_up_reversed(ext):
    line = ext.normalize_line(_cat(np.linspace(100, 140, 30), np.linspace(140, 108, 30)))
    assert not ext._trend_still_active(line, "up", float(np.max(line) - np.min(line)))


def test_trend_still_active_down_reversed(ext):
    line = ext.normalize_line(_cat(np.linspace(140, 100, 30), np.linspace(100, 132, 30)))
    assert not ext._trend_still_active(line, "down", float(np.max(line) - np.min(line)))


def test_pullback_uptrend_marked_inactive(ext):
    # Восходящий тренд с сильным откатом в хвосте (как в примере NEAR):
    # рост до пика, затем падение — всё ещё нетто-положительный, но развернулся.
    closes = _cat(np.linspace(100, 160, 60), np.linspace(160, 130, 40))
    st, conf, patterns, is_active, freshness = _classify(ext, closes)
    assert st == StructureType.TREND_UP
    assert is_active is False
    assert freshness < 1.0
    # Устаревший тренд не должен просачиваться во вторичные detected_patterns
    # (иначе он снова попадёт в type-scan через secondary_match).
    assert StructureType.TREND_UP.value not in patterns


def test_intact_uptrend_stays_active(ext):
    closes = np.linspace(100, 150, 90)
    st, conf, patterns, is_active, freshness = _classify(ext, closes)
    assert st == StructureType.TREND_UP
    assert is_active is True
    assert freshness == 1.0
