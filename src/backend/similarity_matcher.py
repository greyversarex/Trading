import numpy as np
from scipy.spatial.distance import cosine, euclidean
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


class SimilarityMatcher:
    """Compares structural features between price patterns."""
    
    def __init__(self):
        self.line_weight = 0.35
        self.pivot_weight = 0.25
        self.distance_weight = 0.20
        self.shape_weight = 0.20
    
    def mirror_features(self, features: StructureFeatures) -> StructureFeatures:
        """Create mirrored version of features (bullish <-> bearish)."""
        mirrored_line = 1.0 - features.normalized_line
        mirrored_pivots = [1.0 - v for v in features.pivot_sequence]
        
        mirrored_feature_vector = features.feature_vector.copy()
        line_section = len(features.normalized_line)
        mirrored_feature_vector[:line_section] = (1.0 - features.normalized_line) * 0.4
        
        return StructureFeatures(
            pivot_points=features.pivot_points,
            normalized_line=mirrored_line,
            pivot_sequence=mirrored_pivots,
            relative_distances=features.relative_distances,
            trend_direction=-features.trend_direction,
            volatility=features.volatility,
            compression_ratio=features.compression_ratio,
            structure_type=features.structure_type,
            feature_vector=mirrored_feature_vector
        )
    
    def compare_lines(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """Compare two normalized price lines."""
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
        
        cosine_sim = 1.0 - cosine(line1, line2) if np.any(line1) and np.any(line2) else 0.0
        
        return 0.4 * correlation + 0.3 * mse_similarity + 0.3 * max(0, cosine_sim)
    
    def compare_pivots(self, pivots1: List[float], pivots2: List[float]) -> float:
        """Compare pivot point sequences."""
        if not pivots1 or not pivots2:
            return 0.5
        
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
        return 1.0 / (1.0 + mse * 5)
    
    def compare_distances(self, dist1: List[float], dist2: List[float]) -> float:
        """Compare relative distance patterns between pivots."""
        if not dist1 or not dist2:
            return 0.5
        
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
        return 1.0 / (1.0 + mse * 5)
    
    def compare_shape(self, features1: StructureFeatures, 
                     features2: StructureFeatures) -> float:
        """Compare overall shape characteristics."""
        trend_diff = abs(features1.trend_direction - features2.trend_direction)
        trend_sim = 1.0 - min(trend_diff, 2.0) / 2.0
        
        vol_ratio = min(features1.volatility, features2.volatility) / (
            max(features1.volatility, features2.volatility) + 1e-10)
        
        comp_diff = abs(features1.compression_ratio - features2.compression_ratio)
        comp_sim = 1.0 - min(comp_diff, 2.0) / 2.0
        
        type_sim = 1.0 if features1.structure_type == features2.structure_type else 0.5
        
        return 0.3 * trend_sim + 0.2 * vol_ratio + 0.2 * comp_sim + 0.3 * type_sim
    
    def calculate_similarity(self, reference: StructureFeatures, 
                           candidate: StructureFeatures,
                           check_mirror: bool = True) -> Tuple[float, bool]:
        """
        Calculate overall similarity between two structures.
        Returns (similarity_score, is_mirrored).
        """
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
                    candidates: List[Tuple[str, str, StructureFeatures, str]],
                    threshold: float = 50.0) -> List[MatchResult]:
        """
        Find all candidates above similarity threshold.
        candidates: List of (symbol, timeframe, features, timestamp)
        """
        matches = []
        
        for symbol, timeframe, features, timestamp in candidates:
            if features is None:
                continue
            
            score, is_mirrored = self.calculate_similarity(reference, features)
            
            if score >= threshold:
                matches.append(MatchResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    similarity_score=round(score, 2),
                    structure_type=features.structure_type,
                    timestamp=timestamp,
                    is_mirrored=is_mirrored,
                    normalized_line=features.normalized_line.tolist()
                ))
        
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches
