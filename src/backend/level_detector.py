import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TouchPoint:
    candle_index: int
    price: float
    deviation: float
    is_high: bool


@dataclass
class DetectedLevel:
    symbol: str
    timeframe: str
    line_type: str
    slope: float
    intercept: float
    anchor_start: Tuple[int, float]
    anchor_end: Tuple[int, float]
    touches: List[TouchPoint]
    touch_count: int
    avg_deviation_pct: float
    max_deviation_pct: float
    quality_score: float
    coverage: float
    price_at_last: float


class LevelDetector:

    def __init__(self, deviation_pct: float = 0.15, min_touches: int = 3):
        self.deviation_pct = deviation_pct
        self.min_touches = min_touches

    def detect_levels(
        self,
        symbol: str,
        timeframe: str,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        min_touches: int = None,
        max_levels: int = 10,
    ) -> List[DetectedLevel]:
        if len(highs) < 20:
            return []

        if min_touches is None:
            min_touches = self.min_touches

        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        c = np.array(closes, dtype=float)
        n = len(h)

        high_pivots = self._find_pivots(h, is_high=True)
        low_pivots = self._find_pivots(l, is_high=False)

        levels: List[DetectedLevel] = []

        res_levels = self._find_lines(high_pivots, h, l, c, symbol, timeframe, "resistance", min_touches)
        sup_levels = self._find_lines(low_pivots, h, l, c, symbol, timeframe, "support", min_touches)

        levels.extend(res_levels)
        levels.extend(sup_levels)

        levels = self._deduplicate(levels, n)
        levels.sort(key=lambda lv: lv.quality_score, reverse=True)

        return levels[:max_levels]

    def _find_pivots(self, data: np.ndarray, is_high: bool) -> List[Tuple[int, float]]:
        n = len(data)
        if n < 5:
            return []

        pivots = []
        order = max(2, n // 20)

        for i in range(order, n - order):
            if is_high:
                window = data[i - order: i + order + 1]
                if data[i] == np.max(window):
                    pivots.append((i, float(data[i])))
            else:
                window = data[i - order: i + order + 1]
                if data[i] == np.min(window):
                    pivots.append((i, float(data[i])))

        return pivots

    def _find_lines(
        self,
        pivots: List[Tuple[int, float]],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        symbol: str,
        timeframe: str,
        line_type: str,
        min_touches: int,
    ) -> List[DetectedLevel]:
        if len(pivots) < 2:
            return []

        n = len(highs)
        price_range = float(np.max(highs) - np.min(lows))
        if price_range <= 0:
            return []

        tolerance = price_range * (self.deviation_pct / 100.0)

        candidates: List[DetectedLevel] = []

        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                idx_a, val_a = pivots[i]
                idx_b, val_b = pivots[j]

                dx = idx_b - idx_a
                if dx == 0:
                    continue

                slope = (val_b - val_a) / dx
                intercept = val_a - slope * idx_a

                touches = self._count_touches(
                    slope, intercept, highs, lows, closes,
                    tolerance, line_type
                )

                if len(touches) < min_touches:
                    continue

                avg_dev = np.mean([t.deviation for t in touches])
                max_dev = np.max([t.deviation for t in touches])
                avg_dev_pct = (avg_dev / price_range) * 100
                max_dev_pct = (max_dev / price_range) * 100

                indices = [t.candle_index for t in touches]
                coverage = (max(indices) - min(indices)) / max(n - 1, 1)

                spacing_score = self._spacing_score(indices, n)

                quality = (
                    len(touches) * 20
                    + (1.0 - avg_dev_pct / self.deviation_pct) * 30
                    + coverage * 25
                    + spacing_score * 25
                )
                quality = max(0, min(100, quality))

                line_val_last = slope * (n - 1) + intercept
                anchor_start = (int(idx_a), float(val_a))
                anchor_end = (int(idx_b), float(val_b))

                candidates.append(DetectedLevel(
                    symbol=symbol,
                    timeframe=timeframe,
                    line_type=line_type,
                    slope=slope,
                    intercept=intercept,
                    anchor_start=anchor_start,
                    anchor_end=anchor_end,
                    touches=touches,
                    touch_count=len(touches),
                    avg_deviation_pct=round(avg_dev_pct, 4),
                    max_deviation_pct=round(max_dev_pct, 4),
                    quality_score=round(quality, 2),
                    coverage=round(coverage, 3),
                    price_at_last=round(line_val_last, 8),
                ))

        return candidates

    def _count_touches(
        self,
        slope: float,
        intercept: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        tolerance: float,
        line_type: str,
    ) -> List[TouchPoint]:
        n = len(highs)
        touches: List[TouchPoint] = []
        min_gap = max(2, n // 20)

        for i in range(n):
            line_val = slope * i + intercept

            if line_type == "resistance":
                price = float(highs[i])
                is_h = True
            else:
                price = float(lows[i])
                is_h = False

            dev = abs(price - line_val)

            if dev <= tolerance:
                if touches and (i - touches[-1].candle_index) < min_gap:
                    if dev < touches[-1].deviation:
                        touches[-1] = TouchPoint(
                            candle_index=i,
                            price=price,
                            deviation=dev,
                            is_high=is_h,
                        )
                    continue

                touches.append(TouchPoint(
                    candle_index=i,
                    price=price,
                    deviation=dev,
                    is_high=is_h,
                ))

        return touches

    def _spacing_score(self, indices: List[int], total_len: int) -> float:
        if len(indices) < 2:
            return 0.0

        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        if not gaps:
            return 0.0

        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        if avg_gap == 0:
            return 0.0

        cv = std_gap / avg_gap
        uniformity = 1.0 / (1.0 + cv)
        return uniformity

    def _deduplicate(self, levels: List[DetectedLevel], n: int) -> List[DetectedLevel]:
        if not levels:
            return []

        levels.sort(key=lambda lv: lv.quality_score, reverse=True)
        kept: List[DetectedLevel] = []

        for lv in levels:
            is_dup = False
            for k in kept:
                if lv.line_type != k.line_type:
                    continue

                mid = n // 2
                val_lv = lv.slope * mid + lv.intercept
                val_k = k.slope * mid + k.intercept

                price_range = abs(k.intercept) + 1e-10
                if abs(val_lv - val_k) / price_range < 0.005:
                    slope_diff = abs(lv.slope - k.slope)
                    if slope_diff < 0.01:
                        is_dup = True
                        break

            if not is_dup:
                kept.append(lv)

        return kept
