import numpy as np
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress
from typing import List, Tuple, Dict, Optional
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


class StructureExtractor:
    """Extracts structural features from price lines for comparison."""
    
    def __init__(self, num_pivots: int = 10, resample_points: int = 100):
        self.num_pivots = num_pivots
        self.resample_points = resample_points
    
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
    
    def detect_pivots(self, line: np.ndarray, order: int = None) -> List[PivotPoint]:
        if len(line) < 11:
            return []
        
        if order is None:
            order = self._adaptive_order(line)
        
        window_length = min(7, len(line) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            smoothed = savgol_filter(line, window_length, 2)
        else:
            smoothed = line
        
        pivots = self._multi_scale_pivots(smoothed, line, adaptive_order=order)
        
        if len(pivots) < 3:
            pivots = self._fallback_pivots(line)
        
        pivots = self._filter_redundant_pivots(pivots, line)
        
        if len(pivots) > self.num_pivots:
            pivots = self._select_important_pivots(pivots, line)
        
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
    
    def _filter_redundant_pivots(self, pivots: List[PivotPoint], line: np.ndarray) -> List[PivotPoint]:
        if len(pivots) <= 2:
            return pivots
        
        filtered = [pivots[0]]
        
        for i in range(1, len(pivots)):
            prev = filtered[-1]
            curr = pivots[i]
            
            if curr.is_high == prev.is_high:
                if curr.is_high and curr.value > prev.value:
                    filtered[-1] = curr
                elif not curr.is_high and curr.value < prev.value:
                    filtered[-1] = curr
            else:
                value_diff = abs(curr.value - prev.value)
                line_range = np.max(line) - np.min(line)
                if line_range > 0 and value_diff / line_range > 0.03:
                    filtered.append(curr)
        
        return filtered
    
    def _select_important_pivots(self, pivots: List[PivotPoint], line: np.ndarray) -> List[PivotPoint]:
        if len(pivots) <= self.num_pivots:
            return pivots
        
        line_range = np.max(line) - np.min(line)
        if line_range == 0:
            return pivots[:self.num_pivots]
        
        scores = []
        for i, p in enumerate(pivots):
            prominence = abs(p.value - np.mean(line)) / line_range
            
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
            
            score = (prominence * 0.4 + neighbor_diff * 0.4) * position_weight
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
        return np.std(returns)
    
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
    
    def _detect_impulse(self, line: np.ndarray) -> Tuple[bool, str]:
        n = len(line)
        if n < 10:
            return False, ""
        
        segment_size = max(3, n // 5)
        
        diffs = np.diff(line)
        abs_diffs = np.abs(diffs)
        mean_move = np.mean(abs_diffs)
        
        for i in range(n - segment_size):
            segment_move = abs(line[i + segment_size] - line[i])
            segment_time = segment_size / n
            
            avg_move_before = np.mean(abs_diffs[:max(1, i)]) if i > 0 else mean_move
            
            if avg_move_before > 0:
                impulse_ratio = (segment_move / segment_size) / (avg_move_before + 1e-10)
            else:
                impulse_ratio = 0
            
            price_range = np.max(line) - np.min(line)
            if price_range > 0 and segment_move / price_range > 0.5 and impulse_ratio > 3:
                direction = "up" if line[i + segment_size] > line[i] else "down"
                return True, direction
        
        last_quarter = line[int(n * 0.75):]
        first_three_quarters = line[:int(n * 0.75)]
        
        if len(last_quarter) > 1 and len(first_three_quarters) > 1:
            last_range = np.max(last_quarter) - np.min(last_quarter)
            first_range = np.max(first_three_quarters) - np.min(first_three_quarters)
            price_range = np.max(line) - np.min(line)
            
            if price_range > 0 and last_range / price_range > 0.6:
                last_trend = last_quarter[-1] - last_quarter[0]
                if abs(last_trend) / price_range > 0.4:
                    direction = "up" if last_trend > 0 else "down"
                    return True, direction
        
        return False, ""
    
    def _detect_double_top(self, pivots: List[PivotPoint], line: np.ndarray) -> bool:
        highs = [p for p in pivots if p.is_high]
        if len(highs) < 2:
            return False
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                diff = abs(highs[i].value - highs[j].value) / price_range
                if diff < 0.08:
                    between_lows = [p for p in pivots if not p.is_high and highs[i].index < p.index < highs[j].index]
                    if between_lows:
                        dip = min(p.value for p in between_lows)
                        dip_depth = (highs[i].value - dip) / price_range
                        if dip_depth > 0.15:
                            return True
        return False

    def _detect_double_bottom(self, pivots: List[PivotPoint], line: np.ndarray) -> bool:
        lows = [p for p in pivots if not p.is_high]
        if len(lows) < 2:
            return False
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                diff = abs(lows[i].value - lows[j].value) / price_range
                if diff < 0.08:
                    between_highs = [p for p in pivots if p.is_high and lows[i].index < p.index < lows[j].index]
                    if between_highs:
                        peak = max(p.value for p in between_highs)
                        peak_height = (peak - lows[i].value) / price_range
                        if peak_height > 0.15:
                            return True
        return False

    def _detect_head_shoulders(self, pivots: List[PivotPoint], line: np.ndarray) -> bool:
        highs = [p for p in pivots if p.is_high]
        if len(highs) < 3:
            return False
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False
        for i in range(len(highs) - 2):
            left = highs[i].value
            head = highs[i + 1].value
            right = highs[i + 2].value
            if head > left and head > right:
                shoulder_diff = abs(left - right) / price_range
                head_prominence = (head - max(left, right)) / price_range
                if shoulder_diff < 0.15 and head_prominence > 0.08:
                    return True
        return False

    def _detect_inv_head_shoulders(self, pivots: List[PivotPoint], line: np.ndarray) -> bool:
        lows = [p for p in pivots if not p.is_high]
        if len(lows) < 3:
            return False
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False
        for i in range(len(lows) - 2):
            left = lows[i].value
            head = lows[i + 1].value
            right = lows[i + 2].value
            if head < left and head < right:
                shoulder_diff = abs(left - right) / price_range
                head_prominence = (min(left, right) - head) / price_range
                if shoulder_diff < 0.15 and head_prominence > 0.08:
                    return True
        return False

    def _detect_flag(self, line: np.ndarray, pivots: List[PivotPoint]) -> Tuple[bool, str]:
        n = len(line)
        if n < 20:
            return False, ""
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, ""
        split = int(n * 0.4)
        pole = line[:split]
        flag_part = line[split:]
        pole_move = pole[-1] - pole[0]
        pole_ratio = abs(pole_move) / price_range
        if pole_ratio < 0.4:
            return False, ""
        flag_range = np.max(flag_part) - np.min(flag_part)
        flag_ratio = flag_range / price_range
        if flag_ratio > 0.4:
            return False, ""
        flag_trend = flag_part[-1] - flag_part[0]
        if pole_move > 0 and flag_trend <= 0:
            return True, "bull"
        if pole_move < 0 and flag_trend >= 0:
            return True, "bear"
        return False, ""

    def _detect_wedge(self, pivots: List[PivotPoint], line: np.ndarray) -> Tuple[bool, str]:
        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if not p.is_high]
        if len(highs) < 2 or len(lows) < 2:
            return False, ""
        price_range = np.max(line) - np.min(line)
        if price_range == 0:
            return False, ""
        high_slope = (highs[-1].value - highs[0].value) / (highs[-1].index - highs[0].index + 1e-10)
        low_slope = (lows[-1].value - lows[0].value) / (lows[-1].index - lows[0].index + 1e-10)
        spread_start = abs(highs[0].value - lows[0].value) if highs and lows else 0
        spread_end = abs(highs[-1].value - lows[-1].value) if highs and lows else 0
        if spread_start == 0:
            return False, ""
        converging = spread_end < spread_start * 0.7
        if not converging:
            return False, ""
        if high_slope > 0 and low_slope > 0:
            return True, "rising"
        if high_slope < 0 and low_slope < 0:
            return True, "falling"
        return False, ""

    def classify_structure(self, trend: float, compression: float, 
                          pivots: List[PivotPoint], line: np.ndarray) -> StructureType:
        is_impulse, impulse_dir = self._detect_impulse(line)
        if is_impulse:
            if impulse_dir == "up":
                return StructureType.IMPULSE_UP
            else:
                return StructureType.IMPULSE_DOWN
        
        n = len(line)
        if n > 10:
            flat_portion = line[:int(n * 0.7)]
            spike_portion = line[int(n * 0.7):]
            if len(flat_portion) > 1 and len(spike_portion) > 1:
                flat_range = np.max(flat_portion) - np.min(flat_portion)
                spike_range = np.max(spike_portion) - np.min(spike_portion)
                total_range = np.max(line) - np.min(line)
                
                if total_range > 0 and flat_range / total_range < 0.3 and spike_range / total_range > 0.5:
                    return StructureType.BREAKOUT
        
        is_flag, flag_dir = self._detect_flag(line, pivots)
        if is_flag:
            if flag_dir == "bull":
                return StructureType.BULL_FLAG
            else:
                return StructureType.BEAR_FLAG
        
        if self._detect_head_shoulders(pivots, line):
            return StructureType.HEAD_SHOULDERS
        if self._detect_inv_head_shoulders(pivots, line):
            return StructureType.INV_HEAD_SHOULDERS
        
        if self._detect_double_top(pivots, line):
            return StructureType.DOUBLE_TOP
        if self._detect_double_bottom(pivots, line):
            return StructureType.DOUBLE_BOTTOM
        
        is_wedge, wedge_dir = self._detect_wedge(pivots, line)
        if is_wedge:
            if wedge_dir == "rising":
                return StructureType.RISING_WEDGE
            else:
                return StructureType.FALLING_WEDGE
        
        if len(pivots) >= 4:
            highs = [p.value for p in pivots if p.is_high]
            lows = [p.value for p in pivots if not p.is_high]
            
            if len(highs) >= 2 and len(lows) >= 2:
                price_range = max(line) - min(line)
                if price_range > 0:
                    high_range = (max(highs) - min(highs)) / price_range
                    low_range = (max(lows) - min(lows)) / price_range
                    high_trend = (highs[-1] - highs[0]) / price_range
                    low_trend = (lows[-1] - lows[0]) / price_range
                    
                    if high_range < 0.25 and low_trend > 0.1:
                        return StructureType.SQUEEZE_UP
                    
                    if low_range < 0.25 and high_trend < -0.1:
                        return StructureType.SQUEEZE_DOWN
        
        if compression > 0.3:
            if len(pivots) >= 4:
                highs = [p.value for p in pivots if p.is_high]
                lows = [p.value for p in pivots if not p.is_high]
                
                if highs and lows:
                    high_trend = (highs[-1] - highs[0]) if len(highs) > 1 else 0
                    low_trend = (lows[-1] - lows[0]) if len(lows) > 1 else 0
                    
                    if high_trend < 0 and low_trend > 0:
                        return StructureType.TRIANGLE
            
            return StructureType.COMPRESSION
        
        if abs(trend) < 0.1:
            highs = [p.value for p in pivots if p.is_high]
            lows = [p.value for p in pivots if not p.is_high]
            
            if highs and lows:
                high_range = max(highs) - min(highs) if len(highs) > 1 else 0
                low_range = max(lows) - min(lows) if len(lows) > 1 else 0
                price_range = max(line) - min(line)
                
                if price_range > 0 and (high_range / price_range < 0.2) and (low_range / price_range < 0.2):
                    return StructureType.RANGE
        
        if len(pivots) >= 3:
            recent_pivots = pivots[-3:]
            if len(recent_pivots) >= 2:
                if any(p.is_high for p in recent_pivots[:-1]) and not recent_pivots[-1].is_high:
                    return StructureType.RETEST
        
        if trend > 0.2:
            return StructureType.TREND_UP
        elif trend < -0.2:
            return StructureType.TREND_DOWN
        
        if abs(trend) < 0.15 and len(pivots) >= 3:
            return StructureType.ACCUMULATION
        
        return StructureType.UNKNOWN
    
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
            line_features * 0.4,
            pivot_features * 0.3,
            distance_features * 0.2,
            scalar_features * 0.1
        ])
        
        return feature_vector
    
    def extract_features(self, line: np.ndarray) -> Optional[StructureFeatures]:
        if len(line) < 10:
            return None
        
        normalized_line = self.normalize_line(line)
        pivots = self.detect_pivots(normalized_line)
        pivot_sequence = self.calculate_pivot_sequence(pivots, normalized_line)
        relative_distances = self.calculate_relative_distances(pivots)
        trend = self.calculate_trend(normalized_line)
        volatility = self.calculate_volatility(normalized_line)
        compression = self.calculate_compression(normalized_line, pivots)
        structure_type = self.classify_structure(trend, compression, pivots, normalized_line)
        
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
            feature_vector=feature_vector
        )
    
    def extract_from_candles(self, closes: List[float]) -> Optional[StructureFeatures]:
        line = np.array(closes)
        return self.extract_features(line)
    
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
            feature_vector=feature_vector
        )
