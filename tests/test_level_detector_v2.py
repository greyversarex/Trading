"""Тесты для LevelDetectorV2 (T1.3).

Проверяют:
- структуру результата (support_levels / resistance_levels / trendlines),
- корректность диапазонов силы и числа касаний,
- детекцию горизонтальных зон на осциллирующем ряду,
- детекцию диагональных трендовых линий,
- профиль объёма (POC) и круглые числа,
- дедупликацию близких уровней,
- детерминизм и устойчивость к коротким рядам.
"""

import numpy as np
import pytest

from src.backend.level_detector_v2 import LevelDetectorV2


@pytest.fixture
def detector():
    return LevelDetectorV2()


def _oscillating_series(low=100.0, high=110.0, cycles=6, per_leg=12, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    closes = []
    for _ in range(cycles):
        closes += list(np.linspace(low, high, per_leg))
        closes += list(np.linspace(high, low, per_leg))
    closes = np.array(closes) + rng.normal(0, noise, len(closes))
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.abs(rng.normal(1000, 100, len(closes)))
    return highs, lows, closes, volumes


def test_returns_correct_keys(detector):
    highs, lows, closes, volumes = _oscillating_series()
    res = detector.detect_levels(highs, lows, closes, volumes)
    assert set(res.keys()) == {"support_levels", "resistance_levels", "trendlines"}
    assert isinstance(res["support_levels"], list)
    assert isinstance(res["resistance_levels"], list)
    assert isinstance(res["trendlines"], list)


def test_short_series_returns_empty(detector):
    closes = np.linspace(100, 105, 10)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.ones(10) * 1000
    res = detector.detect_levels(highs, lows, closes, volumes)
    assert res["support_levels"] == []
    assert res["resistance_levels"] == []
    assert res["trendlines"] == []


def test_mismatched_lengths_returns_empty(detector):
    closes = np.linspace(100, 110, 50)
    highs = closes + 0.5
    lows = closes[:40] - 0.5  # неверная длина
    res = detector.detect_levels(highs, lows, closes, None)
    assert res["support_levels"] == []
    assert res["resistance_levels"] == []


def test_strength_and_touches_ranges(detector):
    highs, lows, closes, volumes = _oscillating_series()
    res = detector.detect_levels(highs, lows, closes, volumes)
    all_levels = res["support_levels"] + res["resistance_levels"]
    assert len(all_levels) > 0
    for lv in all_levels:
        assert 0.0 <= lv["strength"] <= 1.0
        assert lv["num_touches"] >= 1
        assert lv["type"] in ("support", "resistance")
        assert lv["first_touch_time"] <= lv["last_touch_time"]
    for tl in res["trendlines"]:
        assert 0.0 <= tl["strength"] <= 1.0
        assert len(tl["touches"]) >= detector.cfg.min_touches


def test_detects_horizontal_zones(detector):
    highs, lows, closes, volumes = _oscillating_series(low=100.0, high=110.0)
    res = detector.detect_levels(highs, lows, closes, volumes)
    # Должна найтись поддержка около 100 и сопротивление около 110.
    sup_prices = [lv["price"] for lv in res["support_levels"]]
    res_prices = [lv["price"] for lv in res["resistance_levels"]]
    assert any(abs(p - 100.0) < 3.0 for p in sup_prices), sup_prices
    assert any(abs(p - 110.0) < 3.0 for p in res_prices), res_prices


def test_detects_support_trendline_uptrend(detector):
    # Восходящий канал: цены растут, дно поднимается линейно.
    rng = np.random.default_rng(1)
    n = 120
    base = np.linspace(100, 130, n)
    osc = 4 * np.sin(np.linspace(0, 10 * np.pi, n))
    closes = base + osc + rng.normal(0, 0.3, n)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.abs(rng.normal(1000, 80, n))
    res = detector.detect_levels(highs, lows, closes, volumes)
    sup_lines = [tl for tl in res["trendlines"] if tl["type"] == "support_trendline"]
    assert len(sup_lines) >= 1
    assert sup_lines[0]["slope"] > 0


def test_volume_profile_poc_detected(detector):
    # Большой объём сосредоточен в узкой ценовой зоне -> там POC.
    rng = np.random.default_rng(2)
    n = 120
    closes = np.concatenate([
        np.full(60, 105.0) + rng.normal(0, 0.3, 60),  # консолидация (высокий объём)
        np.linspace(105, 120, 60) + rng.normal(0, 0.3, 60),
    ])
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.concatenate([
        np.full(60, 5000.0),  # высокий объём в консолидации
        np.full(60, 500.0),
    ])
    res = detector.detect_levels(highs, lows, closes, volumes)
    sources = [lv["source"] for lv in res["support_levels"] + res["resistance_levels"]]
    assert any(s.startswith("volume_") for s in sources), sources


def test_round_numbers_detected(detector):
    highs, lows, closes, volumes = _oscillating_series(low=100.0, high=110.0)
    res = detector.detect_levels(highs, lows, closes, volumes)
    sources = [lv["source"] for lv in res["support_levels"] + res["resistance_levels"]]
    assert any(s == "round_number" for s in sources), sources


def test_deduplication_no_near_duplicates(detector):
    highs, lows, closes, volumes = _oscillating_series()
    vol_scale = None
    from src.backend.structure_extractor import compute_volatility_scale
    vol_scale = compute_volatility_scale(highs, lows, closes)
    res = detector.detect_levels(highs, lows, closes, volumes)
    for levels in (res["support_levels"], res["resistance_levels"]):
        prices = sorted(lv["price"] for lv in levels)
        for a, b in zip(prices, prices[1:]):
            assert abs(a - b) >= 0.3 * vol_scale - 1e-6, (a, b)


def test_deterministic(detector):
    highs, lows, closes, volumes = _oscillating_series()
    r1 = detector.detect_levels(highs, lows, closes, volumes)
    r2 = detector.detect_levels(highs, lows, closes, volumes)
    assert r1 == r2


def test_times_used_for_touch_times(detector):
    highs, lows, closes, volumes = _oscillating_series()
    n = len(closes)
    times = [1_700_000_000 + i * 3600 for i in range(n)]
    res = detector.detect_levels(highs, lows, closes, volumes, times=times)
    all_levels = res["support_levels"] + res["resistance_levels"]
    assert len(all_levels) > 0
    for lv in all_levels:
        assert lv["first_touch_time"] >= times[0]
        assert lv["last_touch_time"] <= times[-1]
