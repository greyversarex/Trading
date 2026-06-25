"""Тесты детекторов недостающих паттернов (Phase 2.2).

Проверяются пять новых структур: triple_top, triple_bottom, cup_and_handle,
pennant, rounding_bottom. Для каждой:

1. КАУЗАЛЬНАЯ ДЕТЕКЦИЯ: на синтетическом образце с внедрённым паттерном
   ``extract_features_causal`` помечает паттерн как подтверждённый и кладёт его
   в ``detected_patterns``.
2. ОТСУТСТВИЕ ПЕРЕРИСОВКИ: на данных, обрезанных до бара подтверждения, паттерн
   не подтверждается.
3. TYPE_SCAN: неперерисовывающий ``extract_features`` выдаёт паттерн в
   ``detected_patterns`` (мульти-лейбл), что и использует режим сканирования по
   типу.

Дополнительно — регистрация новых типов в ``similarity_matcher`` и отсутствие
массовых ложных срабатываний на чистом шуме.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.structure_extractor import StructureExtractor, StructureType
from src.backend.synthetic import SyntheticChartGenerator
from src.backend import similarity_matcher as sm


NEW_TYPES = [
    StructureType.TRIPLE_TOP,
    StructureType.TRIPLE_BOTTOM,
    StructureType.CUP_AND_HANDLE,
    StructureType.PENNANT,
    StructureType.ROUNDING_BOTTOM,
]


@pytest.fixture(scope="module")
def ext():
    return StructureExtractor()


@pytest.fixture(scope="module")
def gen():
    return SyntheticChartGenerator()


def _build(gen, label, seed):
    """Случайное блуждание с внедрённым паттерном ``label``."""
    walk = gen.generate_random_walk(n=100, start_price=100.0, volatility=0.005, seed=seed)
    builders = {
        "triple_top": lambda c: gen.inject_triple_top(c, 12, 78),
        "triple_bottom": lambda c: gen.inject_triple_bottom(c, 12, 78),
        "cup_and_handle": lambda c: gen.inject_cup_and_handle(c, 12, 64, 82),
        "pennant": lambda c: gen.inject_pennant(c, 12, 36, 80),
        "rounding_bottom": lambda c: gen.inject_rounding_bottom(c, 12, 80),
    }
    return builders[label](walk)


PATTERN_LABELS = [
    "triple_top",
    "triple_bottom",
    "cup_and_handle",
    "pennant",
    "rounding_bottom",
]


# --- 1. Каузальная детекция -------------------------------------------------

@pytest.mark.parametrize("label", PATTERN_LABELS)
def test_causal_detects_and_confirms(ext, gen, label):
    """Хотя бы на одном из нескольких сидов паттерн каузально подтверждается."""
    confirmed_any = False
    for seed in range(8):
        candles = _build(gen, label, seed)
        f = ext.extract_features_causal(candles)
        if f is None:
            continue
        if label in (f.detected_patterns or {}) and f.is_confirmed:
            confirmed_any = True
            break
    assert confirmed_any, f"{label}: не подтверждён каузально ни на одном сиде"


@pytest.mark.parametrize("label", PATTERN_LABELS)
def test_causal_detected_patterns_contains_label(ext, gen, label):
    """Подтверждённый паттерн обязательно присутствует в detected_patterns."""
    found = False
    for seed in range(8):
        candles = _build(gen, label, seed)
        f = ext.extract_features_causal(candles)
        if f and f.is_confirmed and f.structure_type.value == label:
            assert label in (f.detected_patterns or {})
            found = True
            break
    assert found, f"{label}: не выбран как первичный тип ни на одном сиде"


# --- 2. Отсутствие перерисовки ---------------------------------------------

@pytest.mark.parametrize("label", PATTERN_LABELS)
def test_no_repaint_before_confirmation(ext, gen, label):
    """Обрезка до бара подтверждения не должна давать подтверждение."""
    for seed in range(8):
        candles = _build(gen, label, seed)
        f = ext.extract_features_causal(candles)
        if not (f and f.is_confirmed and f.structure_type.value == label):
            continue
        ci = f.confirmation_index
        if ci <= 1:
            continue
        truncated = candles[:ci]
        f2 = ext.extract_features_causal(truncated)
        same = bool(f2 and f2.is_confirmed and f2.structure_type.value == label)
        assert not same, (
            f"{label}: перерисовка — паттерн подтверждён на данных до бара пробоя"
        )
        return
    pytest.skip(f"{label}: не нашёлся подтверждённый образец для проверки")


# --- 3. type_scan: мульти-лейбл в неперерисовывающем extract_features --------

@pytest.mark.parametrize("label", PATTERN_LABELS)
def test_type_scan_detected_patterns(ext, gen, label):
    """Неперерисовывающий extract_features кладёт паттерн в detected_patterns."""
    found = False
    for seed in range(10):
        candles = _build(gen, label, seed)
        closes = np.array([c.close for c in candles], dtype=float)
        f = ext.extract_features(closes)
        if f and label in (f.detected_patterns or {}):
            found = True
            break
    assert found, f"{label}: отсутствует в detected_patterns (type_scan) на всех сидах"


# --- 4. Регистрация в similarity_matcher ------------------------------------

@pytest.mark.parametrize("st", NEW_TYPES)
def test_registered_in_similarity_sets(st):
    """Каждый новый тип относится к REVERSAL или CONSOLIDATION."""
    assert st in sm.REVERSAL_TYPES or st in sm.CONSOLIDATION_TYPES


@pytest.mark.parametrize("st", NEW_TYPES)
def test_registered_min_confidence(st):
    assert st in sm.MIN_CONFIDENCE_THRESHOLDS


def test_triple_opposite_pair_registered():
    pair = (StructureType.TRIPLE_TOP, StructureType.TRIPLE_BOTTOM)
    rev = (StructureType.TRIPLE_BOTTOM, StructureType.TRIPLE_TOP)
    assert pair in sm.OPPOSITE_PAIRS or rev in sm.OPPOSITE_PAIRS


def test_matcher_handles_new_types(ext, gen):
    """Матчер сравнивает два образца с новыми типами без ошибок и в диапазоне."""
    matcher = sm.SimilarityMatcher()
    a = ext.extract_features(np.array([c.close for c in _build(gen, "triple_top", 1)]))
    b = ext.extract_features(np.array([c.close for c in _build(gen, "triple_top", 2)]))
    assert a is not None and b is not None
    score, mirrored = matcher.calculate_similarity(a, b)
    assert 0.0 <= float(score) <= 100.0
    assert isinstance(mirrored, bool)


# --- 5. Ложные срабатывания на чистом шуме ----------------------------------

def test_low_false_positive_on_noise(ext, gen):
    """Доля каузальных подтверждений новых паттернов на чистом шуме умеренная."""
    new_values = {t.value for t in NEW_TYPES}
    confirmed = 0
    total = 40
    for seed in range(total):
        walk = gen.generate_random_walk(n=100, start_price=100.0, volatility=0.012, seed=1000 + seed)
        f = ext.extract_features_causal(walk)
        if f and f.is_confirmed and f.structure_type.value in new_values:
            confirmed += 1
    assert confirmed / total <= 0.5, (
        f"слишком много ложных подтверждений новых паттернов на шуме: {confirmed}/{total}"
    )


def test_enum_values_stable():
    """Строковые значения новых типов стабильны (контракт REST/WS)."""
    assert StructureType.TRIPLE_TOP.value == "triple_top"
    assert StructureType.TRIPLE_BOTTOM.value == "triple_bottom"
    assert StructureType.CUP_AND_HANDLE.value == "cup_and_handle"
    assert StructureType.PENNANT.value == "pennant"
    assert StructureType.ROUNDING_BOTTOM.value == "rounding_bottom"


# --- 6. Phase 3.4: повышение recall (edge cases) ---------------------------

def test_phase34_config_knobs_present():
    """Новые тюнинговые параметры объявлены и имеют разумные значения."""
    from src.backend.config import CONFIG
    assert CONFIG.pattern.triple_top_asymmetry_tolerance > CONFIG.pattern.triple_top_tolerance
    assert (CONFIG.pattern.cup_and_handle_relaxed_handle_retrace_max
            > CONFIG.pattern.cup_and_handle_handle_retrace_max)


def test_cup_without_handle_detected(ext, gen):
    """«Чаша без ручки» (fallback 3.4): U-образная чаша с восстановлением к кромке
    и пробоем вверх, но БЕЗ отката-ручки (область ручки выровнена по кромке).
    Хотя бы на одном из сидов попадает в detected_patterns через fallback."""
    found = False
    for seed in range(16):
        walk = gen.generate_random_walk(n=100, start_price=100.0, volatility=0.005, seed=seed)
        candles = gen.inject_cup_and_handle(walk, 12, 64, 82)
        closes = np.array([c.close for c in candles], dtype=float)
        closes[64:82] = closes[12]  # убираем ручку: плоско на уровне кромки
        f = ext.extract_features(closes)
        if f and "cup_and_handle" in (f.detected_patterns or {}):
            found = True
            break
    assert found, "чаша без ручки не распознана ни на одном сиде (fallback не сработал)"


def test_triple_top_asymmetric_peaks_detected(ext, gen):
    """Тройная вершина с НЕравными по высоте пиками (повышенный peak_noise) и
    чётким пробоем нэклайна распознаётся благодаря ослабленному допуску
    асимметрии (3.4). Достаточно одного сида из набора."""
    found = False
    for seed in range(16):
        walk = gen.generate_random_walk(n=100, start_price=100.0, volatility=0.005, seed=seed)
        candles = gen.inject_triple_top(walk, 12, 78, peak_noise=0.01)
        closes = np.array([c.close for c in candles], dtype=float)
        f = ext.extract_features(closes)
        if f and "triple_top" in (f.detected_patterns or {}):
            found = True
            break
    assert found, "асимметричная тройная вершина не распознана ни на одном сиде"
