import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .structure_extractor import StructureFeatures, StructureType


@dataclass
class MatchResult:
    symbol: str
    timeframe: str
    similarity_score: float
    structure_type: StructureType
    timestamp: str
    is_mirrored: bool
    normalized_line: List[float]
    pattern_time: Optional[int] = None


class SimilarityMatcher:
    """Compares structural features between price patterns."""
    
    def __init__(self):
        self.line_weight = 0.30
        self.pivot_weight = 0.25
        self.distance_weight = 0.15
        self.shape_weight = 0.30
    
    def mirror_features(self, features: StructureFeatures) -> StructureFeatures:
        """Create mirrored version of features (bullish <-> bearish)."""
        mirrored_line = 1.0 - features.normalized_line
        mirrored_pivots = [1.0 - v for v in features.pivot_sequence]
        
        mirrored_feature_vector = features.feature_vector.copy()
        line_section = len(features.normalized_line)
        mirrored_feature_vector[:line_section] = (1.0 - features.normalized_line) * 0.35
        
        mirrored_type = features.structure_type
        type_mirror_map = {
            StructureType.IMPULSE_UP: StructureType.IMPULSE_DOWN,
            StructureType.IMPULSE_DOWN: StructureType.IMPULSE_UP,
            StructureType.TREND_UP: StructureType.TREND_DOWN,
            StructureType.TREND_DOWN: StructureType.TREND_UP,
            StructureType.SQUEEZE_UP: StructureType.SQUEEZE_DOWN,
            StructureType.SQUEEZE_DOWN: StructureType.SQUEEZE_UP,
            StructureType.DOUBLE_TOP: StructureType.DOUBLE_BOTTOM,
            StructureType.DOUBLE_BOTTOM: StructureType.DOUBLE_TOP,
            StructureType.HEAD_SHOULDERS: StructureType.INV_HEAD_SHOULDERS,
            StructureType.INV_HEAD_SHOULDERS: StructureType.HEAD_SHOULDERS,
            StructureType.BULL_FLAG: StructureType.BEAR_FLAG,
            StructureType.BEAR_FLAG: StructureType.BULL_FLAG,
            StructureType.RISING_WEDGE: StructureType.FALLING_WEDGE,
            StructureType.FALLING_WEDGE: StructureType.RISING_WEDGE,
        }
        mirrored_type = type_mirror_map.get(features.structure_type, features.structure_type)
        
        return StructureFeatures(
            pivot_points=features.pivot_points,
            normalized_line=mirrored_line,
            pivot_sequence=mirrored_pivots,
            relative_distances=features.relative_distances,
            trend_direction=-features.trend_direction,
            volatility=features.volatility,
            compression_ratio=features.compression_ratio,
            structure_type=mirrored_type,
            feature_vector=mirrored_feature_vector,
            quality_score=features.quality_score
        )
    
    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray, window: int = 10) -> float:
        n, m = len(s1), len(s2)
        if n == 0 or m == 0:
            return 1.0
        
        w = max(window, abs(n - m))
        
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        
        for i in range(1, n + 1):
            j_start = max(1, i - w)
            j_end = min(m, i + w)
            for j in range(j_start, j_end + 1):
                cost = (s1[i - 1] - s2[j - 1]) ** 2
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        return float(np.sqrt(dtw[n, m] / max(n, m)))
    
    def compare_lines(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """Compare two normalized price lines using multiple methods."""
        if len(line1) != len(line2):
            target_len = min(len(line1), len(line2))
            line1 = np.interp(np.linspace(0, 1, target_len), 
                             np.linspace(0, 1, len(line1)), line1)
            line2 = np.interp(np.linspace(0, 1, target_len), 
                             np.linspace(0, 1, len(line2)), line2)
        
        correlation = 0.0
        try:
            corr, _ = pearsonr(line1, line2)
            if not np.isnan(corr):
                correlation = (corr + 1) / 2
        except:
            pass
        
        mse = np.mean((line1 - line2) ** 2)
        mse_similarity = 1.0 / (1.0 + mse * 10)
        
        cosine_sim = 0.0
        try:
            cosine_sim = 1.0 - cosine(line1, line2) if np.any(line1) and np.any(line2) else 0.0
            if np.isnan(cosine_sim):
                cosine_sim = 0.0
        except:
            pass
        
        down1 = line1[::4] if len(line1) >= 20 else line1[::2]
        down2 = line2[::4] if len(line2) >= 20 else line2[::2]
        dtw_dist = self._dtw_distance(down1, down2, window=max(3, len(down1) // 5))
        dtw_sim = 1.0 / (1.0 + dtw_dist * 5)
        
        return 0.25 * correlation + 0.20 * mse_similarity + 0.20 * max(0, cosine_sim) + 0.35 * dtw_sim
    
    def compare_pivots(self, pivots1: List[float], pivots2: List[float]) -> float:
        """Compare pivot point sequences."""
        if not pivots1 or not pivots2:
            return 0.3
        
        len1, len2 = len(pivots1), len(pivots2)
        target_len = max(len1, len2)
        
        if len1 < target_len:
            pivots1 = list(np.interp(np.linspace(0, 1, target_len),
                                    np.linspace(0, 1, len1), pivots1))
        if len2 < target_len:
            pivots2 = list(np.interp(np.linspace(0, 1, target_len),
                                    np.linspace(0, 1, len2), pivots2))
        
        p1, p2 = np.array(pivots1), np.array(pivots2)
        
        mse = np.mean((p1 - p2) ** 2)
        mse_sim = 1.0 / (1.0 + mse * 8)
        
        if len(p1) >= 3:
            d1 = np.diff(p1)
            d2 = np.diff(p2)
            direction_match = np.mean(np.sign(d1) == np.sign(d2))
        else:
            direction_match = 0.5
        
        return 0.6 * mse_sim + 0.4 * direction_match
    
    def compare_distances(self, dist1: List[float], dist2: List[float]) -> float:
        """Compare relative distance patterns between pivots."""
        if not dist1 or not dist2:
            return 0.3
        
        len1, len2 = len(dist1), len(dist2)
        target_len = max(len1, len2)
        
        if len1 < target_len:
            dist1 = list(np.interp(np.linspace(0, 1, target_len),
                                  np.linspace(0, 1, len1), dist1))
        if len2 < target_len:
            dist2 = list(np.interp(np.linspace(0, 1, target_len),
                                  np.linspace(0, 1, len2), dist2))
        
        d1, d2 = np.array(dist1), np.array(dist2)
        
        mse = np.mean((d1 - d2) ** 2)
        return 1.0 / (1.0 + mse * 8)
    
    def compare_shape(self, features1: StructureFeatures, 
                     features2: StructureFeatures) -> float:
        """Compare overall shape characteristics with strong type matching."""
        trend_diff = abs(features1.trend_direction - features2.trend_direction)
        trend_sim = 1.0 - min(trend_diff, 2.0) / 2.0
        
        vol_ratio = min(features1.volatility, features2.volatility) / (
            max(features1.volatility, features2.volatility) + 1e-10)
        
        comp_diff = abs(features1.compression_ratio - features2.compression_ratio)
        comp_sim = 1.0 - min(comp_diff, 2.0) / 2.0
        
        DIRECTIONAL_TYPES = {
            StructureType.IMPULSE_UP, StructureType.IMPULSE_DOWN,
            StructureType.TREND_UP, StructureType.TREND_DOWN,
            StructureType.BULL_FLAG, StructureType.BEAR_FLAG,
        }
        REVERSAL_TYPES = {
            StructureType.DOUBLE_TOP, StructureType.DOUBLE_BOTTOM,
            StructureType.HEAD_SHOULDERS, StructureType.INV_HEAD_SHOULDERS,
        }
        CONSOLIDATION_TYPES = {
            StructureType.COMPRESSION, StructureType.TRIANGLE,
            StructureType.RANGE, StructureType.ACCUMULATION,
            StructureType.SQUEEZE_UP, StructureType.SQUEEZE_DOWN,
            StructureType.RISING_WEDGE, StructureType.FALLING_WEDGE,
        }
        
        t1, t2 = features1.structure_type, features2.structure_type
        
        if t1 == t2:
            type_sim = 1.0
        elif self._same_category(t1, t2, DIRECTIONAL_TYPES, REVERSAL_TYPES, CONSOLIDATION_TYPES):
            type_sim = 0.6
        else:
            type_sim = 0.2
        
        return 0.20 * trend_sim + 0.15 * vol_ratio + 0.20 * comp_sim + 0.45 * type_sim
    
    def _same_category(self, t1: StructureType, t2: StructureType, *categories) -> bool:
        for category in categories:
            if t1 in category and t2 in category:
                return True
        return False
    
    def calculate_similarity(self, reference: StructureFeatures, 
                           candidate: StructureFeatures,
                           check_mirror: bool = True) -> Tuple[float, bool]:
        """Calculate overall similarity between two structures."""
        line_sim = self.compare_lines(reference.normalized_line, candidate.normalized_line)
        pivot_sim = self.compare_pivots(reference.pivot_sequence, candidate.pivot_sequence)
        dist_sim = self.compare_distances(reference.relative_distances, candidate.relative_distances)
        shape_sim = self.compare_shape(reference, candidate)
        
        normal_score = (
            self.line_weight * line_sim +
            self.pivot_weight * pivot_sim +
            self.distance_weight * dist_sim +
            self.shape_weight * shape_sim
        )
        
        mirror_score = 0.0
        if check_mirror:
            mirrored = self.mirror_features(candidate)
            
            m_line_sim = self.compare_lines(reference.normalized_line, mirrored.normalized_line)
            m_pivot_sim = self.compare_pivots(reference.pivot_sequence, mirrored.pivot_sequence)
            m_dist_sim = self.compare_distances(reference.relative_distances, mirrored.relative_distances)
            m_shape_sim = self.compare_shape(reference, mirrored)
            
            mirror_score = (
                self.line_weight * m_line_sim +
                self.pivot_weight * m_pivot_sim +
                self.distance_weight * m_dist_sim +
                self.shape_weight * m_shape_sim
            )
        
        if mirror_score > normal_score:
            return mirror_score * 100, True
        return normal_score * 100, False
    
    def find_matches(self, reference: StructureFeatures,
                    candidates: list,
                    threshold: float = 50.0) -> List[MatchResult]:
        """Find all candidates above similarity threshold with quality filtering and dedup."""
        raw_matches = []
        
        for item in candidates:
            symbol, timeframe, features, timestamp = item[0], item[1], item[2], item[3]
            candle_time = item[4] if len(item) > 4 else None
            
            if features is None:
                continue
            
            if hasattr(features, 'quality_score') and features.quality_score < 0.2:
                continue
            
            score, is_mirrored = self.calculate_similarity(reference, features)
            
            if score >= threshold:
                raw_matches.append(MatchResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    similarity_score=round(score, 2),
                    structure_type=features.structure_type,
                    timestamp=timestamp,
                    is_mirrored=is_mirrored,
                    normalized_line=features.normalized_line.tolist(),
                    pattern_time=candle_time
                ))
        
        matches = self._deduplicate_matches(raw_matches)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Keep best match per symbol, allowing different timeframes only if significantly different."""
        best_per_symbol = {}
        
        for m in matches:
            key = m.symbol
            if key not in best_per_symbol:
                best_per_symbol[key] = [m]
            else:
                existing = best_per_symbol[key]
                dominated = False
                for ex in existing:
                    if ex.timeframe == m.timeframe:
                        if m.similarity_score > ex.similarity_score:
                            existing.remove(ex)
                            existing.append(m)
                        dominated = True
                        break
                    if abs(m.similarity_score - ex.similarity_score) < 5.0:
                        dominated = True
                        break
                if not dominated:
                    existing.append(m)
        
        result = []
        for symbol_matches in best_per_symbol.values():
            result.extend(symbol_matches)
        
        return result
