import numpy as np
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class StructureType(str, Enum):
    COMPRESSION = "compression"
    ACCUMULATION = "accumulation"
    TRIANGLE = "triangle"
    RANGE = "range"
    RETEST = "retest"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    IMPULSE_UP = "impulse_up"
    IMPULSE_DOWN = "impulse_down"
    BREAKOUT = "breakout"
    SQUEEZE_UP = "squeeze_up"
    SQUEEZE_DOWN = "squeeze_down"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inv_head_shoulders"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    UNKNOWN = "unknown"


@dataclass
class PivotPoint:
    index: int
    value: float
    is_high: bool
    relative_position: float
    prominence: float = 0.0
    confidence: float = 0.0
    
    def __lt__(self, other):
        if not isinstance(other, PivotPoint):
            return NotImplemented
        return self.index < other.index
    
    def __le__(self, other):
        if not isinstance(other, PivotPoint):
            return NotImplemented
        return self.index <= other.index
    
    def __gt__(self, other):
        if not isinstance(other, PivotPoint):
            return NotImplemented
        return self.index > other.index
    
    def __ge__(self, other):
        if not isinstance(other, PivotPoint):
            return NotImplemented
        return self.index >= other.index


@dataclass
class StructureFeatures:
    pivot_points: List[PivotPoint]
    normalized_line: np.ndarray
    pivot_sequence: List[float]
    relative_distances: List[float]
    trend_direction: float
    volatility: float
    compression_ratio: float
    structure_type: StructureType
    feature_vector: np.ndarray
    quality_score: float = 0.0
    pattern_confidence: float = 0.0
    pivot_slopes: List[float] = field(default_factory=list)
    pivot_angles: List[float] = field(default_factory=list)
    symmetry_score: float = 0.0
    convergence_rate: float = 0.0
    breakout_strength: float = 0.0
    avg_pivot_confidence: float = 0.0
    trend_consistency: float = 0.0
    detected_patterns: Dict[str, float] = field(default_factory=dict)
    is_pattern_active: bool = True
    pattern_freshness: float = 1.0
    volume_confirmation: float = 0.5


class StructureExtractor:
    """Extracts structural features from price lines for comparison."""
    
    def __init__(self, num_pivots: int = 10, resample_points: int = 100):
        self.num_pivots = num_pivots
        self.resample_points = resample_points
        self.min_quality = 0.15
    
    def normalize_line(self, line: np.ndarray) -> np.ndarray:
        if len(line) == 0:
            return np.array([])
        
        x_original = np.linspace(0, 1, len(line))
        x_target = np.linspace(0, 1, self.resample_points)
        resampled = np.interp(x_target, x_original, line)
        
        min_val = np.min(resampled)
        max_val = np.max(resampled)
        range_val = max_val - min_val
        
        if range_val > 0:
            normalized = (resampled - min_val) / range_val
        else:
            normalized = np.zeros_like(resampled)
        
        return normalized
    
    def _adaptive_smoothing(self, line: np.ndarray) -> np.ndarray:
        n = len(line)
        if n < 11:
            return line.copy()
        
        window1 = min(5, n - 1)
        if window1 % 2 == 0:
            window1 -= 1
        if window1 < 3:
            window1 = 3
        
        poly_order1 = min(2, window1 - 1)
        
        try:
            smoothed = savgol_filter(line, window1, poly_order1)
        except:
            smoothed = line.copy()
        
        diffs = np.abs(np.diff(smoothed))
        mean_diff = np.mean(diffs)
        volatility = np.std(diffs)
        
        if mean_diff > 0:
            noise_ratio = volatility / mean_diff
        else:
            noise_ratio = 0
        
        if noise_ratio > 1.5:
            if noise_ratio > 2.0:
                window2 = min(11, n - 1)
            else:
                window2 = min(9, n - 1)
            
            if window2 % 2 == 0:
                window2 -= 1
            if window2 < 3:
                window2 = 3
            
            poly_order2 = min(2, window2 - 1)
            
            try:
                smoothed = savgol_filter(smoothed, window2, poly_order2)
            except:
                pass
        
        return smoothed
    
    def detect_pivots(self, line: np.ndarray, order: int = None) -> List[PivotPoint]:
        if len(line) < 11:
            return []
        
        if order is None:
            order = self._adaptive_order(line)
        
        zigzag_pivots = self._zigzag_pivots(line)
        
        smoothed = self._adaptive_smoothing(line)
        multi_scale_pivots = self._multi_scale_pivots(smoothed, line, adaptive_order=order)
        
        if len(zigzag_pivots) >= 3:
            pivots = list(zigzag_pivots)
            min_dist = max(3, len(line) // 25)
            price_range = np.max(line) - np.min(line)
            diff_noise = np.std(np.diff(line)) / (price_range + 1e-10) if price_range > 0 else 0
            prominence_threshold = max(0.08, min(0.25, 0.12 + diff_noise * 2.0))
            for mp in multi_scale_pivots:
                if mp.prominence > prominence_threshold:
                    too_close = False
                    for zp in pivots:
                        if abs(mp.index - zp.index) < min_dist:
                            too_close = True
                            break
                    if not too_close:
                        pivots.append(mp)
            pivots.sort(key=lambda p: p.index)
        else:
            pivots = multi_scale_pivots if len(multi_scale_pivots) >= 3 else self._fallback_pivots(line)
        
        pivots = self._calculate_prominence(pivots, line)
        pivots = self._filter_redundant_pivots(pivots, line)
        pivots = self._calculate_confidence(pivots, line, multi_scale_pivots)
        
        if len(pivots) > self.num_pivots:
            pivots = self._select_important_pivots(pivots, line)
        
        return pivots
    
    def _zigzag_pivots(self, line: np.ndarray) -> List[PivotPoint]:
        n = len(line)
        if n < 11:
            return []
        
        window = max(5, n // 10)
        abs_diffs = np.abs(np.diff(line))
        if len(abs_diffs) < window:
            atr = np.mean(abs_diffs)
        else:
            rolling_sum = np.convolve(abs_diffs, np.ones(window), mode='valid')
            atr = np.mean(rolling_sum / window)
        
        diff_std = np.std(abs_diffs) if len(abs_diffs) > 1 else 0
        noise_ratio = diff_std / (atr + 1e-10)
        atr_mult = min(3.5, max(1.5, 1.5 + noise_ratio * 0.8))
        min_swing = max(atr * atr_mult, 0.02)
        min_spacing = max(3, n // 20)
        
        pivots = []
        last_high_idx = 0
        last_low_idx = 0
        last_high_val = line[0]
        last_low_val = line[0]
        direction = 0
        
        for i in range(1, n):
            if direction >= 0:
                if line[i] > last_high_val:
                    last_high_idx = i
                    last_high_val = line[i]
                elif last_high_val - line[i] >= min_swing:
                    if direction == 0 and i > min_spacing:
                        if last_high_idx >= min_spacing or not pivots:
                            pivots.append(PivotPoint(
                                index=last_high_idx,
                                value=line[last_high_idx],
                                is_high=True,
                                relative_position=last_high_idx / n
                            ))
                    elif direction > 0:
                        if not pivots or (last_high_idx - pivots[-1].index) >= min_spacing:
                            pivots.append(PivotPoint(
                                index=last_high_idx,
                                value=line[last_high_idx],
                                is_high=True,
                                relative_position=last_high_idx / n
                            ))
                    direction = -1
                    last_low_idx = i
                    last_low_val = line[i]
            
            if direction <= 0:
                if line[i] < last_low_val:
                    last_low_idx = i
                    last_low_val = line[i]
                elif line[i] - last_low_val >= min_swing:
                    if direction == 0 and i > min_spacing:
                        if last_low_idx >= min_spacing or not pivots:
                            pivots.append(PivotPoint(
                                index=last_low_idx,
                                value=line[last_low_idx],
                                is_high=False,
                                relative_position=last_low_idx / n
                            ))
                    elif direction < 0:
                        if not pivots or (last_low_idx - pivots[-1].index) >= min_spacing:
                            pivots.append(PivotPoint(
                                index=last_low_idx,
                                value=line[last_low_idx],
                                is_high=False,
                                relative_position=last_low_idx / n
                            ))
                    direction = 1
                    last_high_idx = i
                    last_high_val = line[i]
        
        pivots.sort(key=lambda p: p.index)
        return pivots
    
    def _calculate_confidence(self, pivots: List[PivotPoint], line: np.ndarray, 
                              multi_scale_pivots: List[PivotPoint]) -> List[PivotPoint]:
        if not pivots or len(line) == 0:
            return pivots
        
        line_range = np.max(line) - np.min(line)
        if line_range == 0:
            return pivots
        
        ms_indices = set(mp.index for mp in multi_scale_pivots)
        
        for i, p in enumerate(pivots):
            prominence_score = min(1.0, p.prominence * 2.0)
            
            confirmation_count = 0
            for ms_idx in ms_indices:
                if abs(ms_idx - p.index) <= 3:
                    confirmation_count += 1
            confirmation_score = min(1.0, confirmation_count / 3.0)
            
            if i > 0 and i < len(pivots) - 1:
                left_dist = pivots[i].index - pivots[i - 1].index
                right_dist = pivots[i + 1].index - pivots[i].index
                min_dist = min(left_dist, right_dist)
            elif i > 0:
                min_dist = pivots[i].index - pivots[i - 1].index
            elif i < len(pivots) - 1:
                min_dist = pivots[i + 1].index - pivots[i].index
            else:
                min_dist = len(line)
            distance_score = min(1.0, min_dist / (len(line) * 0.15))
            
            ctx_start = max(0, p.index - len(line) // 10)
            ctx_end = min(len(line), p.index + len(line) // 10)
            local_segment = line[ctx_start:ctx_end]
            if len(local_segment) > 0:
                local_range = np.max(local_segment) - np.min(local_segment)
                amplitude_score = min(1.0, (local_range / line_range) * 1.5)
            else:
                amplitude_score = 0.0
            
            p.confidence = (
                prominence_score * 0.35 +
                confirmation_score * 0.25 +
                distance_score * 0.20 +
                amplitude_score * 0.20
            )
        
        return pivots
    
    def _adaptive_order(self, line: np.ndarray) -> int:
        diffs = np.abs(np.diff(line))
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        if max_diff > mean_diff * 8:
            return 2
        elif max_diff > mean_diff * 4:
            return 3
        else:
            return max(3, min(5, len(line) // 20))
    
    def _multi_scale_pivots(self, smoothed: np.ndarray, original: np.ndarray, adaptive_order: int = 3) -> List[PivotPoint]:
        all_pivots = []
        seen_indices = set()
        
        base_orders = [2, 3, 5]
        if adaptive_order not in base_orders:
            base_orders.append(adaptive_order)
            base_orders.sort()
        orders = base_orders
        
        for order in orders:
            if len(smoothed) < order * 2 + 1:
                continue
            
            highs = argrelextrema(smoothed, np.greater, order=order)[0]
            lows = argrelextrema(smoothed, np.less, order=order)[0]
            
            for idx in highs:
                if not any(abs(idx - s) <= 2 for s in seen_indices):
                    all_pivots.append(PivotPoint(
                        index=idx,
                        value=original[idx],
                        is_high=True,
                        relative_position=idx / len(original)
                    ))
                    seen_indices.add(idx)
            
            for idx in lows:
                if not any(abs(idx - s) <= 2 for s in seen_indices):
                    all_pivots.append(PivotPoint(
                        index=idx,
                        value=original[idx],
                        is_high=False,
                        relative_position=idx / len(original)
                    ))
                    seen_indices.add(idx)
        
        all_pivots.sort(key=lambda p: p.index)
        return all_pivots
    
    def _fallback_pivots(self, line: np.ndarray) -> List[PivotPoint]:
        n = len(line)
        pivots = []
        
        segment_size = max(5, n // 8)
        
        for start in range(0, n - segment_size + 1, segment_size):
            end = min(start + segment_size, n)
            segment = line[start:end]
            
            max_idx = start + np.argmax(segment)
            min_idx = start + np.argmin(segment)
            
            if line[max_idx] > np.mean(line):
                pivots.append(PivotPoint(
                    index=max_idx,
                    value=line[max_idx],
                    is_high=True,
                    relative_position=max_idx / n
                ))
            
            if line[min_idx] < np.mean(line):
                pivots.append(PivotPoint(
                    index=min_idx,
                    value=line[min_idx],
                    is_high=False,
                    relative_position=min_idx / n
                ))
        
        pivots.sort(key=lambda p: p.index)
        return self._filter_redundant_pivots(pivots, line)
    
    def _calculate_prominence(self, pivots: List[PivotPoint], line: np.ndarray) -> List[PivotPoint]:
        if not pivots or len(line) == 0:
            return pivots
        
        line_range = np.max(line) - np.min(line)
        if line_range == 0:
            return pivots
        
        for i, p in enumerate(pivots):
            left_bound = pivots[i - 1].index if i > 0 else 0
            right_bound = pivots[i + 1].index if i < len(pivots) - 1 else len(line) - 1
            
            segment = line[left_bound:right_bound + 1]
            
            if p.is_high:
                ref_val = np.min(segment)
                p.prominence = (p.value - ref_val) / line_range
            else:
                ref_val = np.max(segment)
                p.prominence = (ref_val - p.value) / line_range
        
        return pivots
    
    def _filter_redundant_pivots(self, pivots: List[PivotPoint], line: np.ndarray) -> List[PivotPoint]:
        if len(pivots) <= 2:
            return pivots
        
        line_range = np.max(line) - np.min(line)
        abs_diffs = np.abs(np.diff(line))
        atr = np.mean(abs_diffs) if len(abs_diffs) > 0 else 0.0
        min_amplitude = max(0.04, atr * 1.5)
        
        filtered = [pivots[0]]
        
        for i in range(1, len(pivots)):
            prev = filtered[-1]
            curr = pivots[i]
            
            if curr.is_high == prev.is_high:
                if curr.is_high and curr.value >= prev.value:
                    filtered[-1] = curr
                elif not curr.is_high and curr.value <= prev.value:
                    filtered[-1] = curr
            else:
                value_diff = abs(curr.value - prev.value)
                if line_range > 0 and value_diff / line_range >= min_amplitude:
                    filtered.append(curr)
        
        alternated = [filtered[0]]
        for i in range(1, len(filtered)):
            if filtered[i].is_high == alternated[-1].is_high:
                if filtered[i].is_high:
                    if filtered[i].value > alternated[-1].value:
                        alternated[-1] = filtered[i]
                else:
                    if filtered[i].value < alternated[-1].value:
                        alternated[-1] = filtered[i]
            else:
                alternated.append(filtered[i])
        
        return alternated
    
    def _select_important_pivots(self, pivots: List[PivotPoint], line: np.ndarray) -> List[PivotPoint]:
        if len(pivots) <= self.num_pivots:
            return pivots
        
        line_range = np.max(line) - np.min(line)
        if line_range == 0:
            return pivots[:self.num_pivots]
        
        scores = []
        for i, p in enumerate(pivots):
            prominence = p.prominence if p.prominence > 0 else abs(p.value - np.mean(line)) / line_range
            
            position_weight = 1.0
            if p.relative_position > 0.7:
                position_weight = 1.5
            elif p.relative_position < 0.1:
                position_weight = 1.3
            
            neighbor_diff = 0
            if i > 0:
                neighbor_diff = max(neighbor_diff, abs(p.value - pivots[i-1].value) / line_range)
            if i < len(pivots) - 1:
                neighbor_diff = max(neighbor_diff, abs(p.value - pivots[i+1].value) / line_range)
            
            score = (prominence * 0.5 + neighbor_diff * 0.3 + 0.2 * position_weight) * position_weight
            scores.append((score, p))
        
        scores.sort(reverse=True)
        selected = [s[1] for s in scores[:self.num_pivots]]
        selected.sort(key=lambda p: p.index)
        
        return selected
    
    def calculate_pivot_sequence(self, pivots: List[PivotPoint], line: np.ndarray) -> List[float]:
        if not pivots or len(line) == 0:
            return []
        
        min_val = np.min(line)
        max_val = np.max(line)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.5] * len(pivots)
        
        return [(p.value - min_val) / range_val for p in pivots]
    
    def calculate_relative_distances(self, pivots: List[PivotPoint]) -> List[float]:
        if len(pivots) < 2:
            return []
        
        distances = []
        for i in range(1, len(pivots)):
            dx = pivots[i].relative_position - pivots[i-1].relative_position
            dy = pivots[i].value - pivots[i-1].value
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        total = sum(distances) if distances else 1
        return [d / total for d in distances] if total > 0 else distances
    
    def calculate_trend(self, line: np.ndarray) -> float:
        if len(line) < 2:
            return 0.0
        
        x = np.arange(len(line))
        slope, _, r_value, _, _ = linregress(x, line)
        
        normalized_slope = np.tanh(slope * len(line) / (np.max(line) - np.min(line) + 1e-10))
        return normalized_slope * abs(r_value)
    
    def calculate_volatility(self, line: np.ndarray) -> float:
        if len(line) < 2:
            return 0.0
        
        returns = np.diff(line) / (line[:-1] + 1e-10)
        return float(np.std(returns))
    
    def calculate_compression(self, line: np.ndarray, pivots: List[PivotPoint]) -> float:
        if len(line) < 20:
            return 0.0
        
        mid = len(line) // 2
        first_half = line[:mid]
        second_half = line[mid:]
        
        vol_first = np.std(first_half) if len(first_half) > 1 else 0
        vol_second = np.std(second_half) if len(second_half) > 1 else 0
        
        if vol_first == 0:
            return 0.0
        
        return 1.0 - (vol_second / vol_first) if vol_first > vol_second else -(vol_second / vol_first - 1.0)
    
    def calculate_quality_score(self, line: np.ndarray, pivots: List[PivotPoint]) -> float:
        if len(line) < 10 or len(pivots) < 2:
            return 0.0
        
        line_range = np.max(line) - np.min(line)
        if line_range < 1e-10:
            return 0.0
        
        pivot_count_score = min(1.0, len(pivots) / 4.0)
        
        alternation_count = 0
        for i in range(1, len(pivots)):
            if pivots[i].is_high != pivots[i-1].is_high:
                alternation_count += 1
        alternation_score = alternation_count / max(1, len(pivots) - 1)
        
        avg_prominence = np.mean([p.prominence for p in pivots if p.prominence > 0]) if any(p.prominence > 0 for p in pivots) else 0
        prominence_score = min(1.0, avg_prominence * 5)
        
        diffs = np.abs(np.diff(line))
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        noise_ratio = std_diff / (mean_diff + 1e-10)
        noise_score = max(0, 1.0 - noise_ratio / 3.0)
        
        quality = (
            pivot_count_score * 0.25 +
            alternation_score * 0.30 +
            prominence_score * 0.25 +
            noise_score * 0.20
        )
        
        return float(quality)

    def _detect_impulse(self, line: np.ndarray) -> Tuple[bool, str, float]:
        n = len(line)
        if n < 15:
            return False, "", 0.0

        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, "", 0.0

        best_confidence = 0.0
        best_direction = ""

        segment_size = max(3, n // 5)
        for start in range(n - segment_size):
            end = start + segment_size
            seg_move = line[end] - line[start]
            seg_abs = abs(seg_move)
            move_pct = seg_abs / price_range

            if move_pct < 0.55:
                continue

            before = line[:start] if start > 3 else None
            after = line[end:] if end < n - 3 else None

            calm_before = True
            if before is not None and len(before) > 3:
                before_range = (np.max(before) - np.min(before)) / price_range
                calm_before = before_range < 0.35

            calm_after = True
            if after is not None and len(after) > 3:
                after_range = (np.max(after) - np.min(after)) / price_range
                calm_after = after_range < 0.35

            if not (calm_before or calm_after):
                continue

            seg_diffs = np.diff(line[start:end+1])
            if seg_move > 0:
                directional_ratio = np.sum(seg_diffs > 0) / len(seg_diffs)
            else:
                directional_ratio = np.sum(seg_diffs < 0) / len(seg_diffs)

            if directional_ratio < 0.6:
                continue

            conf = move_pct * 0.5 + directional_ratio * 0.3 + 0.2
            conf = min(1.0, conf)
            if conf > best_confidence:
                best_confidence = conf
                best_direction = "up" if seg_move > 0 else "down"

        return best_confidence > 0.55, best_direction, best_confidence

    def _detect_double_top(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, float, bool]:
        highs = [p for p in pivots if p.is_high]
        if len(highs) < 2:
            return False, 0.0, True
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, 0.0, True
        best_conf = 0.0
        best_neckline = None
        best_top2_idx = 0
        n = len(line)
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                diff = abs(highs[i].value - highs[j].value) / price_range
                if diff < 0.03:
                    between_lows = [p for p in pivots if not p.is_high and highs[i].index < p.index < highs[j].index]
                    if between_lows:
                        dip = min(p.value for p in between_lows)
                        dip_depth = (highs[i].value - dip) / price_range
                        if dip_depth > 0.12:
                            conf = (1.0 - diff / 0.03) * 0.5 + min(1.0, dip_depth / 0.3) * 0.5
                            if conf > best_conf:
                                best_conf = conf
                                best_neckline = dip
                                best_top2_idx = highs[j].index
        if best_conf <= 0.4:
            return False, 0.0, True
        is_active = True
        if best_neckline is not None:
            tail_start = max(best_top2_idx, int(n * 0.8))
            tail = line[tail_start:]
            if len(tail) > 2:
                below_neckline = np.sum(tail < best_neckline - price_range * 0.02)
                if below_neckline > len(tail) * 0.5:
                    is_active = False
        return best_conf > 0.4, best_conf, is_active

    def _detect_double_bottom(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, float, bool]:
        lows = [p for p in pivots if not p.is_high]
        if len(lows) < 2:
            return False, 0.0, True
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, 0.0, True
        best_conf = 0.0
        best_neckline = None
        best_bot2_idx = 0
        n = len(line)
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                diff = abs(lows[i].value - lows[j].value) / price_range
                if diff < 0.03:
                    between_highs = [p for p in pivots if p.is_high and lows[i].index < p.index < lows[j].index]
                    if between_highs:
                        peak = max(p.value for p in between_highs)
                        peak_height = (peak - lows[i].value) / price_range
                        if peak_height > 0.12:
                            conf = (1.0 - diff / 0.03) * 0.5 + min(1.0, peak_height / 0.3) * 0.5
                            if conf > best_conf:
                                best_conf = conf
                                best_neckline = peak
                                best_bot2_idx = lows[j].index
        if best_conf <= 0.4:
            return False, 0.0, True
        is_active = True
        if best_neckline is not None:
            tail_start = max(best_bot2_idx, int(n * 0.8))
            tail = line[tail_start:]
            if len(tail) > 2:
                above_neckline = np.sum(tail > best_neckline + price_range * 0.02)
                if above_neckline > len(tail) * 0.5:
                    is_active = False
        return best_conf > 0.4, best_conf, is_active

    def _detect_head_shoulders(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, float, bool]:
        highs = [p for p in pivots if p.is_high]
        if len(highs) < 3:
            return False, 0.0, True
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, 0.0, True
        best_conf = 0.0
        best_neckline = None
        best_right_idx = 0
        n = len(line)
        for i in range(len(highs) - 2):
            left = highs[i].value
            head = highs[i + 1].value
            right = highs[i + 2].value
            if head > left and head > right:
                shoulder_diff = abs(left - right) / price_range
                head_prominence = (head - max(left, right)) / price_range
                if shoulder_diff < 0.18 and head_prominence > 0.06:
                    neckline_lows = [p for p in pivots if not p.is_high and highs[i].index < p.index < highs[i+2].index]
                    neckline_bonus = 0.2 if len(neckline_lows) >= 2 else 0.0
                    conf = (1.0 - shoulder_diff / 0.18) * 0.3 + min(1.0, head_prominence / 0.15) * 0.4 + neckline_bonus + 0.1
                    shoulder_avg = (left + right) / 2
                    shoulder_head_ratio = shoulder_avg / head if head > 0 else 0
                    if shoulder_head_ratio < 0.4 or shoulder_head_ratio > 0.95:
                        conf *= 0.6
                    time_span = highs[i + 2].index - highs[i].index
                    left_span = highs[i + 1].index - highs[i].index
                    right_span = highs[i + 2].index - highs[i + 1].index
                    if time_span > 0:
                        symmetry_ratio = min(left_span, right_span) / max(left_span, right_span) if max(left_span, right_span) > 0 else 0
                        if symmetry_ratio < 0.3:
                            conf *= 0.7
                    if conf > best_conf:
                        best_conf = conf
                        if neckline_lows:
                            best_neckline = np.mean([p.value for p in neckline_lows])
                        best_right_idx = highs[i + 2].index
        if best_conf <= 0.4:
            return False, 0.0, True
        is_active = True
        if best_neckline is not None:
            tail_start = max(best_right_idx, int(n * 0.8))
            tail = line[tail_start:]
            if len(tail) > 2:
                below_neckline = np.sum(tail < best_neckline - price_range * 0.02)
                if below_neckline > len(tail) * 0.5:
                    is_active = False
        return best_conf > 0.4, best_conf, is_active

    def _detect_inv_head_shoulders(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, float, bool]:
        lows = [p for p in pivots if not p.is_high]
        if len(lows) < 3:
            return False, 0.0, True
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, 0.0, True
        best_conf = 0.0
        best_neckline = None
        best_right_idx = 0
        n = len(line)
        for i in range(len(lows) - 2):
            left = lows[i].value
            head = lows[i + 1].value
            right = lows[i + 2].value
            if head < left and head < right:
                shoulder_diff = abs(left - right) / price_range
                head_prominence = (min(left, right) - head) / price_range
                if shoulder_diff < 0.18 and head_prominence > 0.06:
                    neckline_highs = [p for p in pivots if p.is_high and lows[i].index < p.index < lows[i+2].index]
                    neckline_bonus = 0.2 if len(neckline_highs) >= 2 else 0.0
                    conf = (1.0 - shoulder_diff / 0.18) * 0.3 + min(1.0, head_prominence / 0.15) * 0.4 + neckline_bonus + 0.1
                    shoulder_avg = (left + right) / 2
                    if neckline_highs:
                        neckline_val = np.mean([p.value for p in neckline_highs])
                        head_depth = neckline_val - head
                        shoulder_depth = neckline_val - shoulder_avg
                        if head_depth > 0:
                            depth_ratio = shoulder_depth / head_depth
                            if depth_ratio < 0.15 or depth_ratio > 0.85:
                                conf *= 0.6
                    time_span = lows[i + 2].index - lows[i].index
                    left_span = lows[i + 1].index - lows[i].index
                    right_span = lows[i + 2].index - lows[i + 1].index
                    if time_span > 0:
                        symmetry_ratio = min(left_span, right_span) / max(left_span, right_span) if max(left_span, right_span) > 0 else 0
                        if symmetry_ratio < 0.3:
                            conf *= 0.7
                    if conf > best_conf:
                        best_conf = conf
                        if neckline_highs:
                            best_neckline = np.mean([p.value for p in neckline_highs])
                        best_right_idx = lows[i + 2].index
        if best_conf <= 0.4:
            return False, 0.0, True
        is_active = True
        if best_neckline is not None:
            tail_start = max(best_right_idx, int(n * 0.8))
            tail = line[tail_start:]
            if len(tail) > 2:
                above_neckline = np.sum(tail > best_neckline + price_range * 0.02)
                if above_neckline > len(tail) * 0.5:
                    is_active = False
        return best_conf > 0.4, best_conf, is_active

    def _detect_flag(self, line: np.ndarray, pivots: List[PivotPoint]) -> Tuple[bool, str, float, bool]:
        n = len(line)
        if n < 20:
            return False, "", 0.0, True
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, "", 0.0, True
        
        best_conf = 0.0
        best_dir = ""
        best_flag_high = 0.0
        best_flag_low = 0.0
        best_flag_end = n
        
        for split_pct in [0.3, 0.4, 0.5]:
            split = int(n * split_pct)
            pole = line[:split]
            flag_part = line[split:]
            pole_move = pole[-1] - pole[0]
            pole_ratio = abs(pole_move) / price_range
            if pole_ratio < 0.35:
                continue
            flag_range = np.max(flag_part) - np.min(flag_part)
            flag_ratio = flag_range / price_range
            if flag_ratio > 0.45:
                continue
            flag_trend = flag_part[-1] - flag_part[0]
            
            if pole_move > 0 and flag_trend <= 0:
                conf = pole_ratio * 0.5 + (1.0 - flag_ratio) * 0.3 + 0.2
                if conf > best_conf:
                    best_conf = conf
                    best_dir = "bull"
                    best_flag_high = np.max(flag_part)
                    best_flag_low = np.min(flag_part)
                    best_flag_end = split + len(flag_part)
            if pole_move < 0 and flag_trend >= 0:
                conf = pole_ratio * 0.5 + (1.0 - flag_ratio) * 0.3 + 0.2
                if conf > best_conf:
                    best_conf = conf
                    best_dir = "bear"
                    best_flag_high = np.max(flag_part)
                    best_flag_low = np.min(flag_part)
                    best_flag_end = split + len(flag_part)
        
        if best_conf <= 0.5:
            return False, "", 0.0, True
        flag_range = best_flag_high - best_flag_low
        if flag_range > 0:
            pole_range = price_range - flag_range
            if pole_range < flag_range * 1.5:
                best_conf *= 0.7
        is_active = True
        tail = line[int(n * 0.85):]
        if len(tail) > 2:
            if best_dir == "bull" and np.mean(tail) > best_flag_high + price_range * 0.03:
                is_active = False
            elif best_dir == "bear" and np.mean(tail) < best_flag_low - price_range * 0.03:
                is_active = False
        return True, best_dir, best_conf, is_active

    def _detect_wedge(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, str, float, bool]:
        high_pivots = [p for p in pivots if p.is_high]
        low_pivots = [p for p in pivots if not p.is_high]
        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return False, "", 0.0, True
        n = len(line)
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, "", 0.0, True

        h_indices = np.array([p.index for p in high_pivots], dtype=float)
        h_values = np.array([p.value for p in high_pivots])
        l_indices = np.array([p.index for p in low_pivots], dtype=float)
        l_values = np.array([p.value for p in low_pivots])

        h_slope_raw, h_int, h_r, _, _ = linregress(h_indices, h_values)
        l_slope_raw, l_int, l_r, _, _ = linregress(l_indices, l_values)

        both_up = h_slope_raw > 0 and l_slope_raw > 0
        both_down = h_slope_raw < 0 and l_slope_raw < 0

        if not (both_up or both_down):
            return False, "", 0.0, True

        first_idx = pivots[0].index
        last_idx = max(pivots[-1].index, n - 1)
        spread_start = (h_int + h_slope_raw * first_idx) - (l_int + l_slope_raw * first_idx)
        spread_end = (h_int + h_slope_raw * last_idx) - (l_int + l_slope_raw * last_idx)

        if spread_start <= 0 or spread_end <= 0:
            return False, "", 0.0, True

        convergence = 1.0 - (spread_end / spread_start)
        if convergence < 0.15:
            return False, "", 0.0, True

        h_residuals = np.abs(h_values - (h_int + h_slope_raw * h_indices))
        l_residuals = np.abs(l_values - (l_int + l_slope_raw * l_indices))
        fit_score = max(0, 1.0 - (np.mean(h_residuals) + np.mean(l_residuals)) / (2 * price_range))

        if fit_score < 0.7:
            return False, "", 0.0, True

        conf = convergence * 0.4 + fit_score * 0.4 + 0.2
        conf = min(1.0, conf)

        is_active = True
        tail_start = int(n * 0.85)
        breakout_count = 0
        total_tail = n - tail_start
        for i in range(tail_start, n):
            upper = h_int + h_slope_raw * i
            lower = l_int + l_slope_raw * i
            margin = price_range * 0.02
            if line[i] > upper + margin or line[i] < lower - margin:
                breakout_count += 1
        if total_tail > 0 and breakout_count > total_tail * 0.5:
            is_active = False

        if both_up:
            return True, "rising", conf, is_active
        else:
            return True, "falling", conf, is_active

    def _detect_squeeze(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, str, float]:
        if len(pivots) < 4:
            return False, "", 0.0

        highs = [p.value for p in pivots if p.is_high]
        lows = [p.value for p in pivots if not p.is_high]

        if len(highs) < 2 or len(lows) < 2:
            return False, "", 0.0

        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, "", 0.0

        overall_move = abs(line[-1] - line[0]) / price_range
        if overall_move > 0.4:
            return False, "", 0.0

        high_range = (max(highs) - min(highs)) / price_range
        low_range = (max(lows) - min(lows)) / price_range
        high_trend = (highs[-1] - highs[0]) / price_range
        low_trend = (lows[-1] - lows[0]) / price_range

        if high_range < 0.2 and low_trend > 0.1:
            conf = (1.0 - high_range / 0.2) * 0.4 + min(1.0, low_trend / 0.25) * 0.4 + 0.2
            return True, "up", conf

        if low_range < 0.2 and high_trend < -0.1:
            conf = (1.0 - low_range / 0.2) * 0.4 + min(1.0, abs(high_trend) / 0.25) * 0.4 + 0.2
            return True, "down", conf

        return False, "", 0.0

    def _detect_triangle(self, pivots: List[PivotPoint], line: np.ndarray, compression: float) -> Tuple[bool, float, bool]:
        high_pivots = [p for p in pivots if p.is_high]
        low_pivots = [p for p in pivots if not p.is_high]
        n = len(line)

        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return False, 0.0, True

        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, 0.0, True

        overall_trend = abs(line[-1] - line[0]) / price_range
        if overall_trend > 0.35:
            return False, 0.0, True

        mid_idx = n // 2
        first_half_range = np.max(line[:mid_idx]) - np.min(line[:mid_idx]) if mid_idx > 0 else 0
        second_half_range = np.max(line[mid_idx:]) - np.min(line[mid_idx:]) if mid_idx < n else 0
        if first_half_range > 0 and second_half_range > 0:
            half_ratio = max(first_half_range, second_half_range) / min(first_half_range, second_half_range)
            if half_ratio > 3.0:
                return False, 0.0, True

        h_values_raw = np.array([p.value for p in high_pivots])
        l_values_raw = np.array([p.value for p in low_pivots])

        if len(h_values_raw) >= 3:
            h_max = np.max(h_values_raw)
            h_others = np.sort(h_values_raw)[-2]
            if (h_max - h_others) / price_range > 0.4:
                return False, 0.0, True

        if len(l_values_raw) >= 3:
            l_min = np.min(l_values_raw)
            l_others = np.sort(l_values_raw)[1]
            if (l_others - l_min) / price_range > 0.4:
                return False, 0.0, True

        h_indices = np.array([p.index for p in high_pivots], dtype=float)
        h_values = h_values_raw
        l_indices = np.array([p.index for p in low_pivots], dtype=float)
        l_values = l_values_raw

        if len(h_indices) >= 2:
            h_slope, h_intercept, h_r, _, _ = linregress(h_indices, h_values)
        else:
            return False, 0.0, True
        if len(l_indices) >= 2:
            l_slope, l_intercept, l_r, _, _ = linregress(l_indices, l_values)
        else:
            return False, 0.0, True

        if len(h_values) >= 3:
            h_diffs = np.diff(h_values)
            h_consistent = np.sum(h_diffs < 0) / len(h_diffs)
            if h_consistent < 0.5 and h_slope < 0:
                return False, 0.0, True

        if len(l_values) >= 3:
            l_diffs = np.diff(l_values)
            l_consistent = np.sum(l_diffs > 0) / len(l_diffs)
            if l_consistent < 0.5 and l_slope > 0:
                return False, 0.0, True

        converging = h_slope < 0 and l_slope > 0
        descending = h_slope < 0 and abs(l_slope) < abs(h_slope) * 0.3
        ascending = l_slope > 0 and abs(h_slope) < abs(l_slope) * 0.3

        if not (converging or descending or ascending):
            return False, 0.0, True

        first_idx = pivots[0].index
        last_idx = max(pivots[-1].index, n - 1)

        spread_start = (h_intercept + h_slope * first_idx) - (l_intercept + l_slope * first_idx)
        spread_end = (h_intercept + h_slope * last_idx) - (l_intercept + l_slope * last_idx)

        if spread_start <= 0:
            return False, 0.0, True

        convergence_ratio = 1.0 - (spread_end / spread_start) if spread_start > 0 else 0
        if convergence_ratio < 0.15:
            return False, 0.0, True
        if convergence_ratio > 0.95:
            return False, 0.0, True

        if spread_end < 0:
            return False, 0.0, True

        pattern_span = last_idx - first_idx
        if pattern_span > 0:
            norm_conv_rate = convergence_ratio / (pattern_span / n) if n > 0 else 0
            if norm_conv_rate > 3.0:
                return False, 0.0, True

        h_residuals = np.abs(h_values - (h_intercept + h_slope * h_indices))
        l_residuals = np.abs(l_values - (l_intercept + l_slope * l_indices))
        h_fit = 1.0 - np.mean(h_residuals) / price_range
        l_fit = 1.0 - np.mean(l_residuals) / price_range
        fit_score = max(0, (h_fit + l_fit) / 2)

        if fit_score < 0.75:
            return False, 0.0, True

        h_r2 = h_r ** 2 if len(h_indices) >= 2 else 0
        l_r2 = l_r ** 2 if len(l_indices) >= 2 else 0
        if converging and (h_r2 < 0.3 or l_r2 < 0.3):
            return False, 0.0, True
        if descending and h_r2 < 0.4:
            return False, 0.0, True
        if ascending and l_r2 < 0.4:
            return False, 0.0, True

        pivot_count_score = min(1.0, (len(high_pivots) + len(low_pivots)) / 6.0)

        conf = convergence_ratio * 0.35 + fit_score * 0.35 + pivot_count_score * 0.2 + max(0, compression) * 0.1
        conf = min(1.0, conf)

        if conf <= 0.45:
            return False, 0.0, True

        is_active = True
        tail_start = int(n * 0.85)
        breakout_count = 0
        total_tail = n - tail_start
        for i in range(tail_start, n):
            upper = h_intercept + h_slope * i
            lower = l_intercept + l_slope * i
            margin = price_range * 0.02
            if line[i] > upper + margin or line[i] < lower - margin:
                breakout_count += 1
        if total_tail > 0 and breakout_count > total_tail * 0.5:
            is_active = False

        return True, conf, is_active

    def _detect_trend(self, pivots: List[PivotPoint], line: np.ndarray, trend_slope: float) -> Tuple[bool, str, float]:
        n = len(line)
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, "", 0.0

        overall_move = (line[-1] - line[0]) / price_range

        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if not p.is_high]

        if len(highs) < 2 or len(lows) < 2:
            if abs(overall_move) > 0.4 and abs(trend_slope) > 0.3:
                direction = "up" if trend_slope > 0 else "down"
                conf = min(1.0, abs(overall_move) * 0.5 + abs(trend_slope) * 0.5)
                return True, direction, conf
            return False, "", 0.0

        hh_count = sum(1 for i in range(1, len(highs)) if highs[i].value > highs[i-1].value)
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i].value > lows[i-1].value)
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i].value < highs[i-1].value)
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i].value < lows[i-1].value)

        hh_ratio = hh_count / (len(highs) - 1)
        hl_ratio = hl_count / (len(lows) - 1)
        lh_ratio = lh_count / (len(highs) - 1)
        ll_ratio = ll_count / (len(lows) - 1)

        x = np.arange(n, dtype=float)
        slope, intercept, r_value, _, _ = linregress(x, line)
        r_sq = r_value ** 2

        if overall_move > 0.15 and (hh_ratio >= 0.5 or hl_ratio >= 0.5):
            pivot_score = (hh_ratio + hl_ratio) / 2
            move_score = min(1.0, abs(overall_move))
            conf = move_score * 0.35 + pivot_score * 0.35 + r_sq * 0.3
            if conf > 0.35:
                return True, "up", min(1.0, conf)

        if overall_move < -0.15 and (lh_ratio >= 0.5 or ll_ratio >= 0.5):
            pivot_score = (lh_ratio + ll_ratio) / 2
            move_score = min(1.0, abs(overall_move))
            conf = move_score * 0.35 + pivot_score * 0.35 + r_sq * 0.3
            if conf > 0.35:
                return True, "down", min(1.0, conf)

        if abs(overall_move) > 0.55 and r_sq > 0.5:
            direction = "up" if overall_move > 0 else "down"
            conf = min(1.0, abs(overall_move) * 0.5 + r_sq * 0.5)
            return True, direction, conf

        return False, "", 0.0
    
    def calculate_pivot_slopes(self, pivots: List[PivotPoint], line: np.ndarray) -> List[float]:
        if len(pivots) < 2:
            return []
        slopes = []
        n = len(line)
        for i in range(1, len(pivots)):
            dx = (pivots[i].index - pivots[i-1].index) / max(1, n)
            dy = pivots[i].value - pivots[i-1].value
            slope = dy / (dx + 1e-10)
            slopes.append(float(np.tanh(slope)))
        return slopes

    def calculate_pivot_angles(self, pivots: List[PivotPoint], line: np.ndarray) -> List[float]:
        if len(pivots) < 3:
            return []
        angles = []
        n = len(line)
        price_range = np.max(line) - np.min(line) if len(line) > 0 else 1.0
        for i in range(1, len(pivots) - 1):
            dx1 = (pivots[i].index - pivots[i-1].index) / max(1, n)
            dy1 = (pivots[i].value - pivots[i-1].value) / (price_range + 1e-10)
            dx2 = (pivots[i+1].index - pivots[i].index) / max(1, n)
            dy2 = (pivots[i+1].value - pivots[i].value) / (price_range + 1e-10)
            
            dot = dx1 * dx2 + dy1 * dy2
            cross = dx1 * dy2 - dy1 * dx2
            angle = np.arctan2(cross, dot)
            angles.append(float(angle / np.pi))
        return angles

    def calculate_symmetry(self, pivots: List[PivotPoint], line: np.ndarray) -> float:
        if len(pivots) < 4:
            return 0.0
        n = len(pivots)
        mid = n // 2
        left_half = pivots[:mid]
        right_half = pivots[mid:]
        right_reversed = list(reversed(right_half))
        
        price_range = np.max(line) - np.min(line) if len(line) > 0 else 1.0
        if price_range == 0:
            return 0.0
        
        compare_len = min(len(left_half), len(right_reversed))
        if compare_len == 0:
            return 0.0
        
        symmetry_sum = 0.0
        for i in range(compare_len):
            val_diff = abs(left_half[i].value - right_reversed[i].value) / price_range
            type_match = 1.0 if left_half[i].is_high == right_reversed[i].is_high else 0.0
            symmetry_sum += (1.0 - min(1.0, val_diff)) * 0.5 + type_match * 0.5
        
        return symmetry_sum / compare_len

    def calculate_convergence_rate(self, pivots: List[PivotPoint], line: np.ndarray) -> float:
        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if not p.is_high]
        if len(highs) < 2 or len(lows) < 2:
            return 0.0
        
        price_range = np.max(line) - np.min(line) if len(line) > 0 else 1.0
        if price_range == 0:
            return 0.0
        
        spread_start = abs(highs[0].value - lows[0].value)
        spread_end = abs(highs[-1].value - lows[-1].value)
        
        if spread_start == 0:
            return 0.0
        
        rate = (spread_start - spread_end) / spread_start
        return float(np.clip(rate, -1.0, 1.0))

    def calculate_breakout_strength(self, line: np.ndarray) -> float:
        n = len(line)
        if n < 15:
            return 0.0
        
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return 0.0
        
        base_portion = line[:int(n * 0.75)]
        end_portion = line[int(n * 0.75):]
        
        if len(base_portion) < 2 or len(end_portion) < 2:
            return 0.0
        
        base_range = np.max(base_portion) - np.min(base_portion)
        base_volatility = np.std(np.diff(base_portion))
        
        end_move = abs(end_portion[-1] - end_portion[0]) / price_range
        end_volatility = np.std(np.diff(end_portion))
        
        volatility_expansion = (end_volatility / (base_volatility + 1e-10))
        range_break = max(0, (end_portion[-1] - np.max(base_portion)) / price_range) + \
                      max(0, (np.min(base_portion) - end_portion[-1]) / price_range)
        
        strength = min(1.0, end_move * 0.3 + min(3.0, volatility_expansion) / 3.0 * 0.3 + range_break * 0.4)
        return float(strength)

    def calculate_volume_confirmation(self, volumes: np.ndarray, pivots: List[PivotPoint], 
                                      line: np.ndarray) -> float:
        if volumes is None or len(volumes) < 10:
            return 0.5
        avg_vol = np.mean(volumes)
        if avg_vol == 0:
            return 0.5
        n = len(volumes)
        line_len = len(line)
        pivot_vols = []
        for p in pivots:
            if line_len > 0 and n != line_len:
                mapped_idx = int(p.index * (n - 1) / max(1, line_len - 1))
            else:
                mapped_idx = p.index
            mapped_idx = min(max(0, mapped_idx), n - 1)
            pivot_vols.append(volumes[mapped_idx])
        pivot_vol_ratio = np.mean(pivot_vols) / avg_vol if pivot_vols else 1.0
        first_half_vol = np.mean(volumes[:n//2])
        second_half_vol = np.mean(volumes[n//2:])
        vol_trend = second_half_vol / first_half_vol if first_half_vol > 0 else 1.0
        tail_start = int(n * 0.85)
        tail_vol = np.mean(volumes[tail_start:]) if tail_start < n else avg_vol
        tail_ratio = tail_vol / avg_vol
        score = (
            min(1.5, pivot_vol_ratio) / 1.5 * 0.3 +
            min(1.5, vol_trend) / 1.5 * 0.3 +
            min(2.0, tail_ratio) / 2.0 * 0.4
        )
        return float(min(1.0, score))

    def calculate_trend_consistency(self, pivots: List[PivotPoint], line: np.ndarray) -> float:
        if len(pivots) < 3:
            return 0.0
        
        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if not p.is_high]
        
        hh_score = 0.0
        if len(highs) >= 2:
            hh_count = sum(1 for i in range(1, len(highs)) if highs[i].value > highs[i-1].value)
            ll_count = sum(1 for i in range(1, len(highs)) if highs[i].value < highs[i-1].value)
            total = len(highs) - 1
            hh_score = max(hh_count, ll_count) / total if total > 0 else 0.0
        
        hl_score = 0.0
        if len(lows) >= 2:
            hl_count = sum(1 for i in range(1, len(lows)) if lows[i].value > lows[i-1].value)
            ll_count = sum(1 for i in range(1, len(lows)) if lows[i].value < lows[i-1].value)
            total = len(lows) - 1
            hl_score = max(hl_count, ll_count) / total if total > 0 else 0.0
        
        return float((hh_score + hl_score) / 2)

    def classify_structure(self, trend: float, compression: float, 
                          pivots: List[PivotPoint], line: np.ndarray) -> Tuple[StructureType, float, Dict[str, float], bool, float]:
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return StructureType.UNKNOWN, 0.0, {}, True, 1.0

        overall_move = abs(line[-1] - line[0]) / price_range

        trend_detected, trend_dir, trend_conf = self._detect_trend(pivots, line, trend)
        is_impulse, impulse_dir, impulse_conf = self._detect_impulse(line)

        candidates = []
        active_map = {}

        if is_impulse and impulse_conf > 0.55:
            st = StructureType.IMPULSE_UP if impulse_dir == "up" else StructureType.IMPULSE_DOWN
            candidates.append((st, impulse_conf))
            active_map[st] = True

        if trend_detected and trend_conf > 0.35:
            st = StructureType.TREND_UP if trend_dir == "up" else StructureType.TREND_DOWN
            candidates.append((st, trend_conf))
            active_map[st] = True

        n = len(line)
        if n > 15:
            flat_portion = line[:int(n * 0.65)]
            spike_portion = line[int(n * 0.65):]
            if len(flat_portion) > 3 and len(spike_portion) > 3:
                flat_range = np.max(flat_portion) - np.min(flat_portion)
                spike_range = np.max(spike_portion) - np.min(spike_portion)
                if price_range > 0 and flat_range / price_range < 0.25 and spike_range / price_range > 0.55:
                    breakout_conf = (1.0 - flat_range / price_range) * 0.5 + spike_range / price_range * 0.5
                    if breakout_conf > 0.6:
                        candidates.append((StructureType.BREAKOUT, breakout_conf))
                        active_map[StructureType.BREAKOUT] = True

        is_flag, flag_dir, flag_conf, flag_active = self._detect_flag(line, pivots)
        if is_flag:
            st = StructureType.BULL_FLAG if flag_dir == "bull" else StructureType.BEAR_FLAG
            candidates.append((st, flag_conf))
            active_map[st] = flag_active

        is_hs, hs_conf, hs_active = self._detect_head_shoulders(pivots, line)
        if is_hs:
            candidates.append((StructureType.HEAD_SHOULDERS, hs_conf))
            active_map[StructureType.HEAD_SHOULDERS] = hs_active

        is_ihs, ihs_conf, ihs_active = self._detect_inv_head_shoulders(pivots, line)
        if is_ihs:
            candidates.append((StructureType.INV_HEAD_SHOULDERS, ihs_conf))
            active_map[StructureType.INV_HEAD_SHOULDERS] = ihs_active

        is_dt, dt_conf, dt_active = self._detect_double_top(pivots, line)
        if is_dt:
            candidates.append((StructureType.DOUBLE_TOP, dt_conf))
            active_map[StructureType.DOUBLE_TOP] = dt_active

        is_db, db_conf, db_active = self._detect_double_bottom(pivots, line)
        if is_db:
            candidates.append((StructureType.DOUBLE_BOTTOM, db_conf))
            active_map[StructureType.DOUBLE_BOTTOM] = db_active

        is_wedge, wedge_dir, wedge_conf, wedge_active = self._detect_wedge(pivots, line)
        if is_wedge:
            st = StructureType.RISING_WEDGE if wedge_dir == "rising" else StructureType.FALLING_WEDGE
            candidates.append((st, wedge_conf))
            active_map[st] = wedge_active

        is_squeeze, squeeze_dir, squeeze_conf = self._detect_squeeze(pivots, line)
        if is_squeeze:
            st = StructureType.SQUEEZE_UP if squeeze_dir == "up" else StructureType.SQUEEZE_DOWN
            candidates.append((st, squeeze_conf))
            active_map[st] = True

        is_triangle, triangle_conf, tri_active = self._detect_triangle(pivots, line, compression)
        if is_triangle:
            candidates.append((StructureType.TRIANGLE, triangle_conf))
            active_map[StructureType.TRIANGLE] = tri_active

        if abs(trend) < 0.1 and overall_move < 0.2:
            highs = [p.value for p in pivots if p.is_high]
            lows = [p.value for p in pivots if not p.is_high]
            if len(highs) >= 2 and len(lows) >= 2:
                high_range = (max(highs) - min(highs)) / price_range
                low_range = (max(lows) - min(lows)) / price_range
                if high_range < 0.15 and low_range < 0.15:
                    range_conf = (1.0 - high_range) * 0.3 + (1.0 - low_range) * 0.3 + (1.0 - abs(trend)) * 0.4
                    candidates.append((StructureType.RANGE, range_conf))
                    active_map[StructureType.RANGE] = True

        if compression > 0.35 and overall_move < 0.3:
            candidates.append((StructureType.COMPRESSION, min(1.0, compression * 0.7)))
            active_map[StructureType.COMPRESSION] = True

        if abs(trend) < 0.12 and len(pivots) >= 4 and overall_move < 0.2:
            candidates.append((StructureType.ACCUMULATION, 0.3))
            active_map[StructureType.ACCUMULATION] = True

        if not candidates:
            return StructureType.UNKNOWN, 0.0, {}, True, 1.0

        active_candidates = [(st, conf) for st, conf in candidates if active_map.get(st, True)]
        stale_candidates = [(st, conf) for st, conf in candidates if not active_map.get(st, True)]

        all_patterns: Dict[str, float] = {}
        for st, conf in candidates:
            key = st.value
            if key not in all_patterns or conf > all_patterns[key]:
                all_patterns[key] = round(float(conf), 3)

        if active_candidates:
            active_candidates.sort(key=lambda x: x[1], reverse=True)
            best_type, best_conf = active_candidates[0]
            is_active = True
            freshness = 1.0
        elif stale_candidates:
            stale_candidates.sort(key=lambda x: x[1], reverse=True)
            best_type, best_conf = stale_candidates[0]
            best_conf *= 0.5
            is_active = False
            freshness = 0.3
        else:
            return StructureType.UNKNOWN, 0.0, {}, True, 1.0

        working_candidates = active_candidates if active_candidates else stale_candidates
        if len(working_candidates) > 1:
            second_conf = working_candidates[1][1]
            if best_conf - second_conf < 0.08:
                best_conf *= 0.85

        return best_type, float(best_conf), all_patterns, is_active, freshness
    
    def create_feature_vector(self, normalized_line: np.ndarray, 
                             pivot_sequence: List[float],
                             relative_distances: List[float],
                             trend: float, volatility: float,
                             compression: float) -> np.ndarray:
        line_features = normalized_line[:self.resample_points] if len(normalized_line) >= self.resample_points else np.pad(normalized_line, (0, self.resample_points - len(normalized_line)))
        
        pivot_features = np.zeros(self.num_pivots)
        for i, val in enumerate(pivot_sequence[:self.num_pivots]):
            pivot_features[i] = val
        
        distance_features = np.zeros(self.num_pivots - 1)
        for i, val in enumerate(relative_distances[:self.num_pivots - 1]):
            distance_features[i] = val
        
        scalar_features = np.array([trend, volatility, compression])
        
        feature_vector = np.concatenate([
            line_features * 0.35,
            pivot_features * 0.30,
            distance_features * 0.20,
            scalar_features * 0.15
        ])
        
        return feature_vector
    
    def extract_features(self, line: np.ndarray, volumes: np.ndarray = None) -> Optional[StructureFeatures]:
        if len(line) < 10:
            return None
        
        raw_range = np.max(line) - np.min(line)
        if raw_range < 1e-10:
            return None
        
        normalized_line = self.normalize_line(line)
        pivots = self.detect_pivots(normalized_line)
        
        quality = self.calculate_quality_score(normalized_line, pivots)
        if quality < self.min_quality:
            return None
        
        pivot_sequence = self.calculate_pivot_sequence(pivots, normalized_line)
        relative_distances = self.calculate_relative_distances(pivots)
        trend = self.calculate_trend(normalized_line)
        volatility = self.calculate_volatility(normalized_line)
        compression = self.calculate_compression(normalized_line, pivots)
        structure_type, pattern_conf, detected_patterns, is_active, freshness = self.classify_structure(trend, compression, pivots, normalized_line)
        
        pivot_slopes = self.calculate_pivot_slopes(pivots, normalized_line)
        pivot_angles = self.calculate_pivot_angles(pivots, normalized_line)
        symmetry = self.calculate_symmetry(pivots, normalized_line)
        convergence = self.calculate_convergence_rate(pivots, normalized_line)
        breakout_str = self.calculate_breakout_strength(normalized_line)
        trend_consist = self.calculate_trend_consistency(pivots, normalized_line)
        avg_conf = float(np.mean([p.confidence for p in pivots])) if pivots else 0.0
        
        vol_conf = 0.5
        if volumes is not None and len(volumes) >= 10:
            norm_volumes = np.array(volumes[-len(normalized_line):]) if len(volumes) > len(normalized_line) else np.array(volumes)
            vol_conf = self.calculate_volume_confirmation(norm_volumes, pivots, normalized_line)
        
        feature_vector = self.create_feature_vector(
            normalized_line, pivot_sequence, relative_distances,
            trend, volatility, compression
        )
        
        return StructureFeatures(
            pivot_points=pivots,
            normalized_line=normalized_line,
            pivot_sequence=pivot_sequence,
            relative_distances=relative_distances,
            trend_direction=trend,
            volatility=volatility,
            compression_ratio=compression,
            structure_type=structure_type,
            feature_vector=feature_vector,
            quality_score=quality,
            pattern_confidence=pattern_conf,
            pivot_slopes=pivot_slopes,
            pivot_angles=pivot_angles,
            symmetry_score=symmetry,
            convergence_rate=convergence,
            breakout_strength=breakout_str,
            avg_pivot_confidence=avg_conf,
            trend_consistency=trend_consist,
            detected_patterns=detected_patterns,
            is_pattern_active=is_active,
            pattern_freshness=freshness,
            volume_confirmation=vol_conf
        )
    
    def extract_from_candles(self, closes: List[float], volumes: List[float] = None) -> Optional[StructureFeatures]:
        line = np.array(closes)
        vol_arr = np.array(volumes) if volumes else None
        return self.extract_features(line, vol_arr)
    
    def features_from_dict(self, data: dict) -> StructureFeatures:
        pivot_points = []
        for p in data.get("pivot_points", []):
            pivot_points.append(PivotPoint(
                index=p["index"],
                value=p["value"],
                is_high=p["is_high"],
                relative_position=p.get("relative_position", p["index"] / 100)
            ))
        
        normalized_line = np.array(data.get("normalized_line", []))
        pivot_sequence = data.get("pivot_sequence", [])
        relative_distances = data.get("relative_distances", [])
        trend_direction = data.get("trend_direction", 0.0)
        volatility = data.get("volatility", 0.0)
        compression_ratio = data.get("compression_ratio", 0.0)
        
        structure_type_str = data.get("structure_type", "range")
        try:
            structure_type = StructureType(structure_type_str) if isinstance(structure_type_str, str) else structure_type_str
        except ValueError:
            structure_type = StructureType.UNKNOWN
        
        feature_vector = np.array(data.get("feature_vector", []))
        if len(feature_vector) == 0:
            feature_vector = self.create_feature_vector(
                normalized_line, pivot_sequence, relative_distances,
                trend_direction, volatility, compression_ratio
            )
        
        return StructureFeatures(
            pivot_points=pivot_points,
            normalized_line=normalized_line,
            pivot_sequence=pivot_sequence,
            relative_distances=relative_distances,
            trend_direction=trend_direction,
            volatility=volatility,
            compression_ratio=compression_ratio,
            structure_type=structure_type,
            feature_vector=feature_vector,
            quality_score=data.get("quality_score", 0.5),
            pattern_confidence=data.get("pattern_confidence", 0.5),
            pivot_slopes=data.get("pivot_slopes", []),
            pivot_angles=data.get("pivot_angles", []),
            symmetry_score=data.get("symmetry_score", 0.0),
            convergence_rate=data.get("convergence_rate", 0.0),
            breakout_strength=data.get("breakout_strength", 0.0),
            avg_pivot_confidence=data.get("avg_pivot_confidence", 0.0),
            trend_consistency=data.get("trend_consistency", 0.0),
            detected_patterns=data.get("detected_patterns", {}),
            is_pattern_active=data.get("is_pattern_active", True),
            pattern_freshness=data.get("pattern_freshness", 1.0),
            volume_confirmation=data.get("volume_confirmation", 0.5)
        )
