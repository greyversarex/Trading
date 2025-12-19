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
        """Normalize line to be scale-invariant and volatility-invariant."""
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
    
    def detect_pivots(self, line: np.ndarray, order: int = 5) -> List[PivotPoint]:
        """Detect pivot points (local highs and lows) in the price line."""
        if len(line) < order * 2 + 1:
            return []
        
        window_length = min(11, len(line) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            smoothed = savgol_filter(line, window_length, 2)
        else:
            smoothed = line
        
        highs = argrelextrema(smoothed, np.greater, order=order)[0]
        lows = argrelextrema(smoothed, np.less, order=order)[0]
        
        pivots = []
        
        for idx in highs:
            pivots.append(PivotPoint(
                index=idx,
                value=line[idx],
                is_high=True,
                relative_position=idx / len(line)
            ))
        
        for idx in lows:
            pivots.append(PivotPoint(
                index=idx,
                value=line[idx],
                is_high=False,
                relative_position=idx / len(line)
            ))
        
        pivots.sort(key=lambda p: p.index)
        
        if len(pivots) > self.num_pivots:
            scores = []
            for p in pivots:
                prominence = abs(p.value - np.mean(line))
                scores.append((prominence, p))
            scores.sort(reverse=True)
            pivots = [s[1] for s in scores[:self.num_pivots]]
            pivots.sort(key=lambda p: p.index)
        
        return pivots
    
    def calculate_pivot_sequence(self, pivots: List[PivotPoint], line: np.ndarray) -> List[float]:
        """Calculate normalized sequence of pivot values."""
        if not pivots or len(line) == 0:
            return []
        
        min_val = np.min(line)
        max_val = np.max(line)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.5] * len(pivots)
        
        return [(p.value - min_val) / range_val for p in pivots]
    
    def calculate_relative_distances(self, pivots: List[PivotPoint]) -> List[float]:
        """Calculate relative distances between consecutive pivots."""
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
        """Calculate overall trend direction (-1 to 1)."""
        if len(line) < 2:
            return 0.0
        
        x = np.arange(len(line))
        slope, _, r_value, _, _ = linregress(x, line)
        
        normalized_slope = np.tanh(slope * len(line) / (np.max(line) - np.min(line) + 1e-10))
        return normalized_slope * abs(r_value)
    
    def calculate_volatility(self, line: np.ndarray) -> float:
        """Calculate normalized volatility."""
        if len(line) < 2:
            return 0.0
        
        returns = np.diff(line) / (line[:-1] + 1e-10)
        return np.std(returns)
    
    def calculate_compression(self, line: np.ndarray, pivots: List[PivotPoint]) -> float:
        """Calculate compression ratio (how much volatility decreases over time)."""
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
    
    def classify_structure(self, trend: float, compression: float, 
                          pivots: List[PivotPoint], line: np.ndarray) -> StructureType:
        """Classify the structure type based on features."""
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
        """Create a fixed-size feature vector for similarity comparison."""
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
        """Extract all structural features from a price line."""
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
        """Extract features from candle close prices."""
        line = np.array(closes)
        return self.extract_features(line)
