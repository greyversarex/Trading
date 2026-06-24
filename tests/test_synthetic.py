"""Тесты синтетического генератора паттернов (Phase 2.1).

Проверяем, что:
1. Каждый генератор паттернов возвращает достаточный по длине ряд свечей.
2. Форма паттерна узнаваема статистически (например, у triple top три самые
   высокие вершины близки по уровню и разнесены по времени).
3. Чистый шум не образует случайно сильный паттерн (санити-проверка).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.synthetic import SyntheticChartGenerator
from src.backend.binance_scanner import CandleData


@pytest.fixture(scope="module")
def gen():
    return SyntheticChartGenerator()


def _closes(candles):
    return np.array([c.close for c in candles], dtype=float)


def _ohlc_valid(candles) -> bool:
    """OHLC согласован: high >= max(open,close), low <= min(open,close)."""
    for c in candles:
        if c.high < max(c.open, c.close) - 1e-6:
            return False
        if c.low > min(c.open, c.close) + 1e-6:
            return False
        if c.volume <= 0:
            return False
    return True


def test_random_walk_length_and_ohlc(gen):
    candles = gen.generate_random_walk(n=100, seed=1)
    assert len(candles) == 100
    assert all(isinstance(c, CandleData) for c in candles)
    assert _ohlc_valid(candles)


@pytest.mark.parametrize("builder_name,args", [
    ("inject_triple_top", (12, 78)),
    ("inject_triple_bottom", (12, 78)),
    ("inject_rounding_bottom", (12, 80)),
])
def test_two_arg_injectors_produce_enough_candles(gen, builder_name, args):
    walk = gen.generate_random_walk(n=100, seed=7)
    out = getattr(gen, builder_name)(walk, *args)
    assert len(out) >= 20
    assert _ohlc_valid(out)


def test_cup_and_handle_and_pennant_lengths(gen):
    walk = gen.generate_random_walk(n=100, seed=8)
    cup = gen.inject_cup_and_handle(walk, 12, 64, 82)
    assert len(cup) >= 20 and _ohlc_valid(cup)

    walk2 = gen.generate_random_walk(n=100, seed=9)
    pen = gen.inject_pennant(walk2, 12, 36, 80)
    assert len(pen) >= 20 and _ohlc_valid(pen)


def test_triple_top_three_peaks_similar_and_spaced(gen):
    """Три самые высокие вершины в зоне паттерна близки (<2%) и разнесены."""
    walk = gen.generate_random_walk(n=100, seed=11)
    out = gen.inject_triple_top(walk, 12, 78)
    closes = _closes(out)[12:78]
    # локальные максимумы внутри зоны
    peaks = [i for i in range(1, len(closes) - 1)
             if closes[i] >= closes[i - 1] and closes[i] >= closes[i + 1]]
    peak_vals = sorted(((closes[i], i) for i in peaks), reverse=True)[:3]
    assert len(peak_vals) == 3, "ожидались три вершины"
    vals = [v for v, _ in peak_vals]
    spread = (max(vals) - min(vals)) / max(vals)
    assert spread < 0.02, f"вершины различаются на {spread:.3f} (>2%)"
    idxs = sorted(i for _, i in peak_vals)
    gaps = np.diff(idxs)
    assert all(g >= 5 for g in gaps), f"вершины слишком близко: {idxs}"


def test_triple_bottom_three_troughs_similar(gen):
    walk = gen.generate_random_walk(n=100, seed=12)
    out = gen.inject_triple_bottom(walk, 12, 78)
    closes = _closes(out)[12:78]
    troughs = [i for i in range(1, len(closes) - 1)
               if closes[i] <= closes[i - 1] and closes[i] <= closes[i + 1]]
    trough_vals = sorted(((closes[i], i) for i in troughs))[:3]
    assert len(trough_vals) == 3
    vals = [v for v, _ in trough_vals]
    spread = (max(vals) - min(vals)) / max(vals)
    assert spread < 0.02


def test_cup_is_u_shaped(gen):
    """Дно чаши заметно ниже краёв (U-образность)."""
    walk = gen.generate_random_walk(n=100, seed=13)
    out = gen.inject_cup_and_handle(walk, 12, 64, 82)
    closes = _closes(out)
    rim_left = closes[12]
    cup_bottom = np.min(closes[12:64])
    assert cup_bottom < rim_left * 0.97, "чаша недостаточно глубокая"


def test_rounding_bottom_smooth_min_in_middle(gen):
    walk = gen.generate_random_walk(n=100, seed=14)
    out = gen.inject_rounding_bottom(walk, 12, 80)
    closes = _closes(out)[12:80]
    min_idx = int(np.argmin(closes))
    rel = min_idx / len(closes)
    assert 0.25 < rel < 0.75, "минимум округлого дна должен быть в середине"


def test_noise_has_no_strong_triple_top(gen):
    """Санити: чистый шум редко даёт три почти равные вершины подряд."""
    flagged = 0
    for s in range(30):
        walk = gen.generate_random_walk(n=100, volatility=0.012, seed=100 + s)
        closes = _closes(walk)
        peaks = [i for i in range(1, len(closes) - 1)
                 if closes[i] >= closes[i - 1] and closes[i] >= closes[i + 1]]
        if len(peaks) >= 3:
            vals = sorted((closes[i] for i in peaks), reverse=True)[:3]
            spread = (max(vals) - min(vals)) / max(vals)
            if spread < 0.005:
                flagged += 1
    assert flagged <= 3, f"слишком много шумовых рядов похожи на triple top: {flagged}/30"


def test_labeled_dataset_structure_and_balance(gen):
    ds = gen.generate_labeled_dataset(n_per_class=5, n_noise=10, seed=3)
    assert len(ds) == 5 * 5 + 10
    labels = {}
    for s in ds:
        assert set(s.keys()) == {"candles", "label", "target_confirmed"}
        assert isinstance(s["candles"], list) and len(s["candles"]) == 100
        assert set(s["candles"][0].keys()) == {
            "open_time", "open", "high", "low", "close", "volume", "close_time"
        }
        labels[s["label"]] = labels.get(s["label"], 0) + 1
        if s["label"] == "noise":
            assert s["target_confirmed"] is False
        else:
            assert s["target_confirmed"] is True
    assert labels["noise"] == 10
    for lbl in ("triple_top", "triple_bottom", "cup_and_handle", "pennant", "rounding_bottom"):
        assert labels[lbl] == 5


def test_candle_dict_roundtrip(gen):
    walk = gen.generate_random_walk(n=10, seed=5)
    d = gen._candle_to_dict(walk[0])
    c2 = gen.candle_from_dict(d)
    assert c2 == walk[0]
