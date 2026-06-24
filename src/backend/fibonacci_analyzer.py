import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

from .config import CONFIG


class FiboLevel(str, Enum):
    LEVEL_0 = "0.0"
    LEVEL_236 = "23.6"
    LEVEL_382 = "38.2"
    LEVEL_500 = "50.0"
    LEVEL_618 = "61.8"
    LEVEL_786 = "78.6"
    LEVEL_1000 = "100.0"
    LEVEL_1272 = "127.2"
    LEVEL_1618 = "161.8"
    LEVEL_2618 = "261.8"


FIBO_RATIOS = {
    FiboLevel.LEVEL_0: 0.0,
    FiboLevel.LEVEL_236: 0.236,
    FiboLevel.LEVEL_382: 0.382,
    FiboLevel.LEVEL_500: 0.500,
    FiboLevel.LEVEL_618: 0.618,
    FiboLevel.LEVEL_786: 0.786,
    FiboLevel.LEVEL_1000: 1.0,
    FiboLevel.LEVEL_1272: 1.272,
    FiboLevel.LEVEL_1618: 1.618,
    FiboLevel.LEVEL_2618: 2.618,
}

FIBO_NAMES = {
    FiboLevel.LEVEL_0: "0%",
    FiboLevel.LEVEL_236: "23.6%",
    FiboLevel.LEVEL_382: "38.2%",
    FiboLevel.LEVEL_500: "50%",
    FiboLevel.LEVEL_618: "61.8%",
    FiboLevel.LEVEL_786: "78.6%",
    FiboLevel.LEVEL_1000: "100%",
    FiboLevel.LEVEL_1272: "127.2%",
    FiboLevel.LEVEL_1618: "161.8%",
    FiboLevel.LEVEL_2618: "261.8%",
}


@dataclass
class FiboTouch:
    level: FiboLevel
    price: float
    index: int
    deviation_pct: float
    is_bounce: bool


@dataclass
class FiboResult:
    symbol: str
    timeframe: str
    swing_high: float
    swing_low: float
    swing_high_idx: int
    swing_low_idx: int
    is_uptrend: bool
    levels: Dict[str, float]
    touches: List[FiboTouch]
    total_touches: int
    unique_levels_touched: int
    quality_score: float
    best_level: str
    best_level_touches: int
    normalized_line: List[float]
    timestamp: str = ""
    price_change_24h: float = 0.0


class FibonacciAnalyzer:

    def __init__(self, touch_tolerance: float = None, min_swing_pct: float = None):
        self.touch_tolerance = touch_tolerance if touch_tolerance is not None else CONFIG.fibo.touch_tolerance
        self.min_swing_pct = min_swing_pct if min_swing_pct is not None else CONFIG.fibo.min_swing_pct

    def find_swing_points(self, closes: List[float], highs: List[float], lows: List[float]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        if len(closes) < 20:
            return None, None, None, None

        n = len(closes)
        lookback = max(20, n // 3)

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        offset = n - lookback

        abs_high_idx = int(np.argmax(recent_highs)) + offset
        abs_low_idx = int(np.argmin(recent_lows)) + offset

        high_val = highs[abs_high_idx]
        low_val = lows[abs_low_idx]

        swing_range = (high_val - low_val) / high_val if high_val > 0 else 0
        if swing_range < self.min_swing_pct:
            return None, None, None, None

        if abs_high_idx < abs_low_idx:
            return abs_high_idx, abs_low_idx, None, None
        else:
            return None, None, abs_low_idx, abs_high_idx

    def find_major_swings(self, closes: List[float], highs: List[float], lows: List[float]) -> List[dict]:
        if len(closes) < 30:
            return []

        n = len(closes)
        results = []

        window_sizes = [n // 2, n // 3, n // 4]
        for ws in window_sizes:
            if ws < 20:
                continue
            for start in range(max(0, n - ws * 2), n - ws + 1, ws // 2):
                end = min(start + ws, n)
                seg_h = highs[start:end]
                seg_l = lows[start:end]
                if len(seg_h) < 15:
                    continue

                hi_idx = int(np.argmax(seg_h)) + start
                lo_idx = int(np.argmin(seg_l)) + start

                hi_val = highs[hi_idx]
                lo_val = lows[lo_idx]
                rng = (hi_val - lo_val) / hi_val if hi_val > 0 else 0
                if rng < self.min_swing_pct:
                    continue

                is_uptrend = lo_idx < hi_idx
                results.append({
                    "swing_high_idx": hi_idx,
                    "swing_low_idx": lo_idx,
                    "swing_high": hi_val,
                    "swing_low": lo_val,
                    "is_uptrend": is_uptrend,
                    "range_pct": rng
                })

        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: -x["range_pct"]):
            key = (r["swing_high_idx"], r["swing_low_idx"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
                if len(unique) >= 3:
                    break

        return unique

    def calculate_fibo_levels(self, swing_high: float, swing_low: float, is_uptrend: bool) -> Dict[str, float]:
        levels = {}
        diff = swing_high - swing_low

        for level, ratio in FIBO_RATIOS.items():
            if ratio <= 1.0:
                if is_uptrend:
                    price = swing_high - diff * ratio
                else:
                    price = swing_low + diff * ratio
            else:
                if is_uptrend:
                    price = swing_low + diff * ratio
                else:
                    price = swing_high - diff * ratio
            levels[FIBO_NAMES[level]] = round(price, 8)

        return levels

    def detect_touches(self, closes: List[float], highs: List[float], lows: List[float],
                       levels: Dict[str, float], start_idx: int = 0) -> List[FiboTouch]:
        touches = []
        key_levels = ["23.6%", "38.2%", "50%", "61.8%", "78.6%", "127.2%", "161.8%", "261.8%"]

        for level_name in key_levels:
            if level_name not in levels:
                continue
            level_price = levels[level_name]
            level_enum = None
            for le, name in FIBO_NAMES.items():
                if name == level_name:
                    level_enum = le
                    break

            if level_enum is None:
                continue

            for i in range(start_idx, len(closes)):
                h = highs[i]
                l = lows[i]
                c = closes[i]

                for price in [h, l, c]:
                    if level_price == 0:
                        continue
                    deviation = abs(price - level_price) / level_price
                    if deviation <= self.touch_tolerance:
                        is_bounce = False
                        if i + 2 < len(closes):
                            future_move = abs(closes[min(i + 2, len(closes) - 1)] - price)
                            expected_move = level_price * self.touch_tolerance * 2
                            if future_move > expected_move:
                                is_bounce = True

                        touches.append(FiboTouch(
                            level=level_enum,
                            price=price,
                            index=i,
                            deviation_pct=round(deviation * 100, 3),
                            is_bounce=is_bounce
                        ))
                        break

        filtered = []
        seen_idx_level = set()
        for t in touches:
            key = (t.index, t.level)
            if key not in seen_idx_level:
                seen_idx_level.add(key)
                filtered.append(t)

        return filtered

    def score_quality(self, touches: List[FiboTouch], total_candles: int) -> float:
        if not touches:
            return 0.0

        unique_levels = set(t.level for t in touches)
        total_touches = len(touches)

        bounces = sum(1 for t in touches if t.is_bounce)
        bounce_ratio = bounces / total_touches if total_touches > 0 else 0

        key_levels_hit = sum(1 for l in [FiboLevel.LEVEL_382, FiboLevel.LEVEL_500, FiboLevel.LEVEL_618]
                            if l in unique_levels)

        avg_deviation = np.mean([t.deviation_pct for t in touches]) if touches else 1.0
        precision_score = max(0, 1.0 - avg_deviation / 0.3)

        touch_score = min(1.0, total_touches / 5)
        level_score = min(1.0, len(unique_levels) / 3)
        key_score = key_score = key_levels_hit / 3.0

        quality = (
            touch_score * 0.25 +
            level_score * 0.20 +
            bounce_ratio * 0.20 +
            precision_score * 0.15 +
            key_score * 0.20
        )

        return round(min(100, quality * 100), 1)

    def analyze(self, symbol: str, timeframe: str, closes: List[float],
                highs: List[float], lows: List[float],
                min_touches: int = 2, min_quality: float = 30.0) -> List[FiboResult]:
        if len(closes) < 20:
            return []

        results = []
        swings = self.find_major_swings(closes, highs, lows)

        if not swings:
            down_hi_idx, down_lo_idx, up_lo_idx, up_hi_idx = self.find_swing_points(closes, highs, lows)
            if down_hi_idx is not None and down_lo_idx is not None:
                swings.append({
                    "swing_high_idx": down_hi_idx,
                    "swing_low_idx": down_lo_idx,
                    "swing_high": highs[down_hi_idx],
                    "swing_low": lows[down_lo_idx],
                    "is_uptrend": False,
                    "range_pct": (highs[down_hi_idx] - lows[down_lo_idx]) / highs[down_hi_idx]
                })
            if up_lo_idx is not None and up_hi_idx is not None:
                swings.append({
                    "swing_high_idx": up_hi_idx,
                    "swing_low_idx": up_lo_idx,
                    "swing_high": highs[up_hi_idx],
                    "swing_low": lows[up_lo_idx],
                    "is_uptrend": True,
                    "range_pct": (highs[up_hi_idx] - lows[up_lo_idx]) / highs[up_hi_idx]
                })

        for swing in swings:
            levels = self.calculate_fibo_levels(
                swing["swing_high"], swing["swing_low"], swing["is_uptrend"]
            )

            retracement_start = min(swing["swing_high_idx"], swing["swing_low_idx"])
            touches = self.detect_touches(closes, highs, lows, levels, start_idx=retracement_start)

            if len(touches) < min_touches:
                continue

            quality = self.score_quality(touches, len(closes))
            if quality < min_quality:
                continue

            level_counts = {}
            for t in touches:
                name = FIBO_NAMES[t.level]
                level_counts[name] = level_counts.get(name, 0) + 1

            best_level = max(level_counts, key=level_counts.get) if level_counts else ""
            best_level_touches = level_counts.get(best_level, 0)

            mn = min(closes)
            mx = max(closes)
            rng = mx - mn if mx > mn else 1
            normalized = [(v - mn) / rng for v in closes[-50:]] if len(closes) >= 50 else [(v - mn) / rng for v in closes]

            results.append(FiboResult(
                symbol=symbol,
                timeframe=timeframe,
                swing_high=swing["swing_high"],
                swing_low=swing["swing_low"],
                swing_high_idx=swing["swing_high_idx"],
                swing_low_idx=swing["swing_low_idx"],
                is_uptrend=swing["is_uptrend"],
                levels=levels,
                touches=touches,
                total_touches=len(touches),
                unique_levels_touched=len(set(t.level for t in touches)),
                quality_score=quality,
                best_level=best_level,
                best_level_touches=best_level_touches,
                normalized_line=normalized
            ))

        results.sort(key=lambda r: -r.quality_score)
        return results[:1]
