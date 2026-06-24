"""Тесты ATR-адаптивных порогов детекции (T1.2).

Проверяем три свойства:

1. ХЕЛПЕРЫ ВОЛАТИЛЬНОСТИ: ``average_true_range`` и ``compute_volatility_scale``
   считают корректные значения по сырым high/low/close и устойчивы к краевым
   случаям (пустой ряд, один бар).
2. ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ: паттерн с разными по высоте вершинами, отвергаемый
   ФИКСИРОВАННЫМ допуском (``volatility_scale = 0``), обнаруживается при
   ATR-адаптивном допуске (большой ``volatility_scale``).
3. НИЗКАЯ ВОЛАТИЛЬНОСТЬ: та же форма с близкими вершинами не теряется при низкой
   волатильности — адаптивный путь не делает пороги строже фиксированных.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.structure_extractor import (
    StructureExtractor,
    StructureType,
    average_true_range,
    compute_volatility_scale,
)
from src.backend.binance_scanner import CandleData


@pytest.fixture(scope="module")
def ext():
    return StructureExtractor()


def _cat(*parts):
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def _mk(closes, half_range_frac=0.001):
    """Свечи из цен закрытия; ``half_range_frac`` задаёт ширину intrabar high/low.

    Малое значение → низкая волатильность (узкие свечи); большое → высокая
    (широкие свечи), что поднимает ATR, не меняя сами цены закрытия.
    """
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        w = c * half_range_frac
        out.append(
            CandleData(
                open_time=i * 60000,
                open=c,
                high=c + w,
                low=c - w,
                close=c,
                volume=1000.0,
                close_time=i * 60000 + 59999,
            )
        )
    return out


def _pivots(ext, closes):
    norm = ext.normalize_line(closes)
    norm_pivots = ext.detect_pivots(norm)
    return ext._map_pivots_to_raw(norm_pivots, closes)


def _double_top_uneven(p2=120.8):
    """Двойная вершина с НЕРАВНЫМИ вершинами (120 и p2)."""
    return _cat(
        np.linspace(100, 120, 15),
        np.linspace(120, 110, 8),
        np.linspace(110, p2, 8),
        np.linspace(p2, 104, 12),
    )


def _double_top_even():
    """Двойная вершина с РАВНЫМИ вершинами (120 и 120)."""
    return _cat(
        np.linspace(100, 120, 15),
        np.linspace(120, 110, 8),
        np.linspace(110, 120, 8),
        np.linspace(120, 104, 12),
    )


# --- 1. Хелперы волатильности ----------------------------------------------

def test_atr_basic():
    closes = np.array([10, 11, 12, 11, 13], dtype=float)
    highs = closes + 1.0
    lows = closes - 1.0
    atr = average_true_range(highs, lows, closes, period=14)
    assert atr > 0.0
    # TR каждого бара тут не меньше high-low = 2.0
    assert atr >= 2.0


def test_atr_edge_cases():
    assert average_true_range(np.array([]), np.array([]), np.array([])) == 0.0
    one = np.array([10.0])
    assert average_true_range(one + 1, one - 1, one) == pytest.approx(2.0)


def test_volatility_scale_floor():
    # Плавный рост на МНОГО баров → крошечный шаг и ATR << price_range * 0.01,
    # поэтому base unit ограничивается снизу полом price_range * 0.01.
    closes = np.linspace(100, 120, 2001)  # шаг ≈ 0.01
    highs = closes + 1e-6
    lows = closes - 1e-6
    price_range = float(closes.max() - closes.min())
    atr = average_true_range(highs, lows, closes)
    assert atr < price_range * 0.01, "ATR должен быть ниже пола для этого кейса"
    vs = compute_volatility_scale(highs, lows, closes)
    assert vs == pytest.approx(price_range * 0.01, rel=1e-3)


def test_volatility_scale_uses_atr_when_high():
    closes = np.linspace(100, 120, 50)
    highs = closes + 5.0
    lows = closes - 5.0
    atr = average_true_range(highs, lows, closes)
    vs = compute_volatility_scale(highs, lows, closes)
    assert vs == pytest.approx(atr)
    assert vs > (closes.max() - closes.min()) * 0.01


# --- 2. Высокая волатильность: адаптивный допуск находит паттерн ------------

def test_uneven_double_top_rejected_by_fixed_detected_by_adaptive(ext):
    seg = _double_top_uneven(p2=120.8)
    candles = _mk(seg, half_range_frac=0.05)  # широкие свечи → высокий ATR
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    pivots = _pivots(ext, closes)

    price_range = float(closes.max() - closes.min())
    # Вершины различаются сильнее фиксированного допуска, но в пределах ATR.
    fixed_tol = 0.03 * price_range
    assert 0.8 > fixed_tol, "тест-кейс должен превышать фиксированный допуск"

    vol_scale = compute_volatility_scale(highs, lows, closes)
    assert vol_scale > 1.6, f"ожидался высокий ATR, получено {vol_scale}"

    # Фиксированный путь (volatility_scale=0) отвергает паттерн.
    cand_fixed = ext._candidate_double_top(closes, pivots, volatility_scale=0.0)
    assert cand_fixed is None, "фиксированный допуск не должен находить неравные вершины"

    # ATR-адаптивный путь находит паттерн.
    cand_adaptive = ext._candidate_double_top(closes, pivots, volatility_scale=vol_scale)
    assert cand_adaptive is not None, "адаптивный допуск должен находить паттерн"
    assert cand_adaptive.structure_type == StructureType.DOUBLE_TOP


# --- 3. Низкая волатильность: форма не теряется -----------------------------

def test_even_double_top_not_lost_at_low_volatility(ext):
    seg = _double_top_even()
    candles = _mk(seg, half_range_frac=0.001)  # узкие свечи → низкий ATR
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    pivots = _pivots(ext, closes)

    vol_scale = compute_volatility_scale(highs, lows, closes)
    cand = ext._candidate_double_top(closes, pivots, volatility_scale=vol_scale)
    assert cand is not None, "чёткая двойная вершина не должна теряться при низкой волатильности"
    assert cand.structure_type == StructureType.DOUBLE_TOP


def test_pipeline_still_confirms_with_adaptive(ext):
    """Полный каузальный конвейер по-прежнему подтверждает чёткий паттерн."""
    seg = _double_top_even()
    f = ext.extract_features_causal(_mk(seg, half_range_frac=0.01))
    assert f is not None
    assert f.structure_type == StructureType.DOUBLE_TOP
    assert f.is_confirmed is True
    assert f.confirmation_index > f.candidate_index
