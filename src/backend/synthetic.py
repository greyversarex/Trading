"""Генератор синтетических OHLCV-рядов с инъекцией графических паттернов.

Используется для:
  * тестирования детекторов паттернов на «идеальных, но зашумлённых» формах;
  * генерации размеченного датасета для обучения ML-фильтра релевантности
    (Phase 2.3) и для осмысленной валидации (Phase 2.5).

Все ряды строятся на базе ``CandleData`` из ``binance_scanner`` — той же
структуры, что и реальные данные Binance, поэтому синтетика проходит через
тот же конвейер детекции (``extract_features_causal``), что и живой рынок.

Каждый паттерн формируется как кусочно-гладкая форма цены закрытия с
добавлением шума, после чего по ряду закрытий достраивается согласованный
OHLC (open = предыдущий close, тени вокруг тела). В конце инъекции
присутствует «пробой» (выход цены из паттерна), чтобы каузальный детектор мог
подтвердить структуру.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np

from .binance_scanner import CandleData


_MINUTE_MS = 60_000


class SyntheticChartGenerator:
    """Строит случайные блуждания и внедряет в них графические паттерны."""

    def __init__(self, base_volume: float = 1000.0):
        self.base_volume = float(base_volume)

    # ------------------------------------------------------------------
    # Базовое случайное блуждание
    # ------------------------------------------------------------------
    def generate_random_walk(
        self,
        n: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.01,
        seed: Optional[int] = None,
    ) -> List[CandleData]:
        """Случайное блуждание с реалистичным OHLCV.

        ``volatility`` — относительное стандартное отклонение шага доходности.
        """
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0, volatility, size=n)
        closes = start_price * np.cumprod(1.0 + steps)
        closes = np.maximum(closes, 1e-6)
        return self._candles_from_closes(closes, rng)

    # ------------------------------------------------------------------
    # Инъекции паттернов
    # ------------------------------------------------------------------
    def inject_triple_top(
        self,
        candles: List[CandleData],
        start_idx: int,
        end_idx: int,
        peak_noise: float = 0.005,
    ) -> List[CandleData]:
        """Три вершины примерно на одном уровне + пробой нэклайна вниз."""
        closes = self._closes(candles)
        s, e = self._clip_range(start_idx, end_idx, len(closes), min_len=21)
        if s is None:
            return candles
        rng = np.random.default_rng(int(abs(closes[s]) * 1000) % (2**32))
        base = float(closes[s])
        peak = base * 1.10
        trough = base * 1.03
        seg = self._three_extrema(e - s, base, peak, trough, peak_noise, rng, top=True)
        closes[s:e] = seg
        self._inject_breakout(closes, e, base * 1.00, direction="down", rng=rng)
        return self._candles_from_closes(closes, rng, src=candles)

    def inject_triple_bottom(
        self,
        candles: List[CandleData],
        start_idx: int,
        end_idx: int,
        trough_noise: float = 0.005,
    ) -> List[CandleData]:
        """Три донышка примерно на одном уровне + пробой нэклайна вверх."""
        closes = self._closes(candles)
        s, e = self._clip_range(start_idx, end_idx, len(closes), min_len=21)
        if s is None:
            return candles
        rng = np.random.default_rng(int(abs(closes[s]) * 1000) % (2**32))
        base = float(closes[s])
        bottom = base * 0.90
        peak = base * 0.97
        seg = self._three_extrema(e - s, base, bottom, peak, trough_noise, rng, top=False)
        closes[s:e] = seg
        self._inject_breakout(closes, e, base * 1.00, direction="up", rng=rng)
        return self._candles_from_closes(closes, rng, src=candles)

    def inject_cup_and_handle(
        self,
        candles: List[CandleData],
        cup_start: int,
        cup_end: int,
        handle_end: int,
    ) -> List[CandleData]:
        """U-образная чаша, затем небольшой откат-ручка и пробой вверх."""
        closes = self._closes(candles)
        n = len(closes)
        if not (0 <= cup_start < cup_end < handle_end <= n - 2):
            return candles
        rng = np.random.default_rng(int(abs(closes[cup_start]) * 1000) % (2**32))
        rim = float(closes[cup_start])
        depth = rim * 0.12
        cup_len = cup_end - cup_start
        t = np.linspace(0.0, np.pi, cup_len)
        cup = rim - depth * np.sin(t)  # гладкая U-образная чаша
        cup += rng.normal(0.0, rim * 0.002, size=cup_len)
        closes[cup_start:cup_end] = cup
        # Ручка: небольшой откат (<= ~35% глубины чаши) и консолидация.
        handle_len = handle_end - cup_end
        retr = depth * 0.30
        h = np.linspace(0.0, np.pi, handle_len)
        handle = rim - retr * np.sin(h)
        handle += rng.normal(0.0, rim * 0.0015, size=handle_len)
        closes[cup_end:handle_end] = handle
        self._inject_breakout(closes, handle_end, rim, direction="up", rng=rng)
        return self._candles_from_closes(closes, rng, src=candles)

    def inject_pennant(
        self,
        candles: List[CandleData],
        pole_start: int,
        pole_end: int,
        pennant_end: int,
    ) -> List[CandleData]:
        """Сильный импульс (полюс) + сходящийся симметричный треугольник + пробой."""
        closes = self._closes(candles)
        n = len(closes)
        if not (0 <= pole_start < pole_end < pennant_end <= n - 2):
            return candles
        rng = np.random.default_rng(int(abs(closes[pole_start]) * 1000) % (2**32))
        base = float(closes[pole_start])
        top = base * 1.25  # мощный полюс
        pole_len = pole_end - pole_start
        closes[pole_start:pole_end] = np.linspace(base, top, pole_len) + \
            rng.normal(0.0, base * 0.002, size=pole_len)
        # Вымпел: затухающие колебания вокруг вершины полюса (схождение).
        pen_len = pennant_end - pole_end
        amp = (top - base) * 0.18
        decay = np.linspace(1.0, 0.1, pen_len)
        osc = np.sin(np.linspace(0.0, 3.0 * np.pi, pen_len)) * amp * decay
        closes[pole_end:pennant_end] = top - amp * 0.3 + osc + \
            rng.normal(0.0, base * 0.0015, size=pen_len)
        self._inject_breakout(closes, pennant_end, top, direction="up", rng=rng)
        return self._candles_from_closes(closes, rng, src=candles)

    def inject_rounding_bottom(
        self,
        candles: List[CandleData],
        start_idx: int,
        end_idx: int,
    ) -> List[CandleData]:
        """Гладкое U-образное округлое донышко (парабола)."""
        closes = self._closes(candles)
        s, e = self._clip_range(start_idx, end_idx, len(closes), min_len=21)
        if s is None:
            return candles
        rng = np.random.default_rng(int(abs(closes[s]) * 1000) % (2**32))
        rim = float(closes[s])
        depth = rim * 0.15
        length = e - s
        x = np.linspace(-1.0, 1.0, length)
        curve = rim - depth * (1.0 - x ** 2)  # парабола: края у rim, центр глубже
        curve += rng.normal(0.0, rim * 0.0015, size=length)
        closes[s:e] = curve
        self._inject_breakout(closes, e, rim, direction="up", rng=rng)
        return self._candles_from_closes(closes, rng, src=candles)

    # ------------------------------------------------------------------
    # Размеченный датасет
    # ------------------------------------------------------------------
    def generate_labeled_dataset(
        self,
        n_per_class: int = 100,
        n_noise: int = 500,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Возвращает список словарей ``{candles, label, target_confirmed}``.

        Классы паттернов помечаются ``target_confirmed=True``, чистый шум —
        ``False``. Каждый образец — это случайное блуждание длиной 100 с (для
        паттернов) внедрённой формой, размещённой так, чтобы после неё оставался
        запас баров для пробоя/подтверждения.
        """
        master = np.random.default_rng(seed)
        samples: List[Dict[str, Any]] = []

        pattern_builders = {
            "triple_top": lambda c: self.inject_triple_top(c, 12, 78),
            "triple_bottom": lambda c: self.inject_triple_bottom(c, 12, 78),
            "cup_and_handle": lambda c: self.inject_cup_and_handle(c, 12, 64, 82),
            "pennant": lambda c: self.inject_pennant(c, 12, 36, 80),
            "rounding_bottom": lambda c: self.inject_rounding_bottom(c, 12, 80),
        }

        for label, builder in pattern_builders.items():
            for _ in range(n_per_class):
                sub_seed = int(master.integers(0, 2**31 - 1))
                walk = self.generate_random_walk(
                    n=100, start_price=100.0, volatility=0.006, seed=sub_seed
                )
                candles = builder(walk)
                samples.append({
                    "candles": [self._candle_to_dict(c) for c in candles],
                    "label": label,
                    "target_confirmed": True,
                })

        for _ in range(n_noise):
            sub_seed = int(master.integers(0, 2**31 - 1))
            walk = self.generate_random_walk(
                n=100, start_price=100.0, volatility=0.012, seed=sub_seed
            )
            samples.append({
                "candles": [self._candle_to_dict(c) for c in walk],
                "label": "noise",
                "target_confirmed": False,
            })

        master.shuffle(samples)
        return samples

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    @staticmethod
    def _closes(candles: List[CandleData]) -> np.ndarray:
        return np.array([c.close for c in candles], dtype=float)

    @staticmethod
    def _clip_range(start_idx: int, end_idx: int, n: int, min_len: int = 21):
        s = max(0, int(start_idx))
        e = min(n - 2, int(end_idx))  # оставить минимум 2 бара под пробой
        if e - s < min_len:
            return None, None
        return s, e

    @staticmethod
    def _three_extrema(length, base, extreme, mid, noise, rng, top: bool) -> np.ndarray:
        """Строит форму с тремя экстремумами на уровне ``extreme`` и двумя
        промежуточными откатами к ``mid``. ``top=True`` — вершины (triple top),
        иначе донышки (triple bottom). Возвращает массив длины ``length``."""
        anchors_x = np.linspace(0, length - 1, 7)  # base, E, mid, E, mid, E, mid->base
        e = extreme * (1.0 + rng.normal(0.0, noise * 0.5, size=3))
        anchors_y = np.array([
            base,
            e[0],
            mid,
            e[1],
            mid,
            e[2],
            base if top else base,
        ], dtype=float)
        xs = np.arange(length)
        seg = np.interp(xs, anchors_x, anchors_y)
        seg += rng.normal(0.0, base * 0.0015, size=length)
        return seg

    @staticmethod
    def _inject_breakout(closes: np.ndarray, from_idx: int, level: float,
                         direction: str, rng) -> None:
        """Двигает оставшиеся бары за уровень ``level`` (пробой паттерна)."""
        n = len(closes)
        if from_idx >= n:
            return
        length = n - from_idx
        start = float(closes[from_idx - 1]) if from_idx > 0 else float(closes[0])
        target = level * (0.94 if direction == "down" else 1.06)
        ramp = np.linspace(start, target, length)
        ramp += rng.normal(0.0, abs(start) * 0.002, size=length)
        closes[from_idx:] = ramp

    def _candles_from_closes(self, closes: np.ndarray, rng,
                             src: Optional[List[CandleData]] = None) -> List[CandleData]:
        """Достраивает согласованный OHLCV по ряду цен закрытия."""
        closes = np.maximum(np.asarray(closes, dtype=float), 1e-6)
        n = len(closes)
        out: List[CandleData] = []
        for i in range(n):
            close = float(closes[i])
            op = float(closes[i - 1]) if i > 0 else close
            hi = max(op, close) * (1.0 + abs(rng.normal(0.0, 0.0015)))
            lo = min(op, close) * (1.0 - abs(rng.normal(0.0, 0.0015)))
            vol = self.base_volume * (1.0 + abs(rng.normal(0.0, 0.35)))
            if src is not None and i < len(src):
                open_time = src[i].open_time
                close_time = src[i].close_time
            else:
                open_time = i * _MINUTE_MS
                close_time = i * _MINUTE_MS + _MINUTE_MS - 1
            out.append(CandleData(
                open_time=open_time,
                open=op,
                high=hi,
                low=lo,
                close=close,
                volume=vol,
                close_time=close_time,
            ))
        return out

    @staticmethod
    def _candle_to_dict(c: CandleData) -> Dict[str, float]:
        return {
            "open_time": c.open_time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "close_time": c.close_time,
        }

    @staticmethod
    def candle_from_dict(d: Dict[str, Any]) -> CandleData:
        return CandleData(
            open_time=int(d["open_time"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d["volume"]),
            close_time=int(d["close_time"]),
        )
