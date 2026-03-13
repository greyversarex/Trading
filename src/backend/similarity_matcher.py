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

OPPOSITE_PAIRS = {
    (StructureType.TREND_UP, StructureType.TREND_DOWN),
    (StructureType.IMPULSE_UP, StructureType.IMPULSE_DOWN),
    (StructureType.BULL_FLAG, StructureType.BEAR_FLAG),
    (StructureType.DOUBLE_TOP, StructureType.DOUBLE_BOTTOM),
    (StructureType.HEAD_SHOULDERS, StructureType.INV_HEAD_SHOULDERS),
    (StructureType.SQUEEZE_UP, StructureType.SQUEEZE_DOWN),
    (StructureType.RISING_WEDGE, StructureType.FALLING_WEDGE),
}

MIN_CONFIDENCE_THRESHOLDS = {
    StructureType.IMPULSE_UP: 0.55,
    StructureType.IMPULSE_DOWN: 0.55,
    StructureType.DOUBLE_TOP: 0.45,
    StructureType.DOUBLE_BOTTOM: 0.45,
    StructureType.HEAD_SHOULDERS: 0.50,
    StructureType.INV_HEAD_SHOULDERS: 0.50,
    StructureType.BULL_FLAG: 0.50,
    StructureType.BEAR_FLAG: 0.50,
    StructureType.RISING_WEDGE: 0.45,
    StructureType.FALLING_WEDGE: 0.45,
    StructureType.TRIANGLE: 0.40,
    StructureType.COMPRESSION: 0.35,
    StructureType.SQUEEZE_UP: 0.40,
    StructureType.SQUEEZE_DOWN: 0.40,
    StructureType.TREND_UP: 0.45,
    StructureType.TREND_DOWN: 0.45,
    StructureType.BREAKOUT: 0.50,
    StructureType.RANGE: 0.35,
    StructureType.ACCUMULATION: 0.30,
    StructureType.RETEST: 0.35,
}


class SimilarityMatcher:
    def __init__(self):
        self.line_weight = 0.25
        self.pivot_weight = 0.20
        self.distance_weight = 0.10
        self.shape_weight = 0.25
        self.slope_weight = 0.10
        self.geometry_weight = 0.10

    def mirror_features(self, features: StructureFeatures) -> StructureFeatures:
        mirrored_line = 1.0 - features.normalized_line
        mirrored_pivots = [1.0 - v for v in features.pivot_sequence]

        mirrored_feature_vector = features.feature_vector.copy()
        line_section = len(features.normalized_line)
        mirrored_feature_vector[:line_section] = (1.0 - features.normalized_line) * 0.35

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
        mirrored_slopes = [-s for s in features.pivot_slopes] if features.pivot_slopes else []
        mirrored_angles = [-a for a in features.pivot_angles] if features.pivot_angles else []

        mirrored_patterns = {}
        dp = getattr(features, 'detected_patterns', {}) or {}
        for k, v in dp.items():
            mirrored_patterns[k] = v

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
            quality_score=features.quality_score,
            pattern_confidence=features.pattern_confidence,
            pivot_slopes=mirrored_slopes,
            pivot_angles=mirrored_angles,
            symmetry_score=features.symmetry_score,
            convergence_rate=features.convergence_rate,
            breakout_strength=features.breakout_strength,
            avg_pivot_confidence=features.avg_pivot_confidence,
            trend_consistency=features.trend_consistency,
            detected_patterns=mirrored_patterns,
            is_pattern_active=features.is_pattern_active,
            pattern_freshness=features.pattern_freshness,
            volume_confirmation=features.volume_confirmation
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

        segments = 4
        seg_len = len(line1) // segments
        segment_corrs = []
        for i in range(segments):
            s1 = line1[i*seg_len:(i+1)*seg_len]
            s2 = line2[i*seg_len:(i+1)*seg_len]
            if len(s1) >= 3:
                try:
                    sc, _ = pearsonr(s1, s2)
                    if not np.isnan(sc):
                        segment_corrs.append((sc + 1) / 2)
                except:
                    pass
        segment_sim = np.mean(segment_corrs) if segment_corrs else correlation

        return 0.20 * correlation + 0.15 * mse_similarity + 0.15 * max(0, cosine_sim) + 0.30 * dtw_sim + 0.20 * segment_sim

    def compare_pivots(self, pivots1: List[float], pivots2: List[float]) -> float:
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

            magnitude_ratios = []
            for a, b in zip(np.abs(d1), np.abs(d2)):
                if max(a, b) > 0.01:
                    magnitude_ratios.append(min(a, b) / (max(a, b) + 1e-10))
            magnitude_sim = np.mean(magnitude_ratios) if magnitude_ratios else 0.5
        else:
            direction_match = 0.5
            magnitude_sim = 0.5

        return 0.40 * mse_sim + 0.35 * direction_match + 0.25 * magnitude_sim

    def compare_distances(self, dist1: List[float], dist2: List[float]) -> float:
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

    def compare_slopes(self, features1: StructureFeatures,
                      features2: StructureFeatures) -> float:
        slopes1 = features1.pivot_slopes
        slopes2 = features2.pivot_slopes
        angles1 = features1.pivot_angles
        angles2 = features2.pivot_angles

        slope_sim = 0.5
        if slopes1 and slopes2:
            len1, len2 = len(slopes1), len(slopes2)
            target_len = max(len1, len2)
            s1 = list(np.interp(np.linspace(0, 1, target_len),
                               np.linspace(0, 1, len1), slopes1)) if len1 < target_len else slopes1
            s2 = list(np.interp(np.linspace(0, 1, target_len),
                               np.linspace(0, 1, len2), slopes2)) if len2 < target_len else slopes2
            s1, s2 = np.array(s1[:target_len]), np.array(s2[:target_len])
            mse = np.mean((s1 - s2) ** 2)
            slope_sim = 1.0 / (1.0 + mse * 5)

        angle_sim = 0.5
        if angles1 and angles2:
            len1, len2 = len(angles1), len(angles2)
            target_len = max(len1, len2)
            a1 = list(np.interp(np.linspace(0, 1, target_len),
                               np.linspace(0, 1, len1), angles1)) if len1 < target_len else angles1
            a2 = list(np.interp(np.linspace(0, 1, target_len),
                               np.linspace(0, 1, len2), angles2)) if len2 < target_len else angles2
            a1, a2 = np.array(a1[:target_len]), np.array(a2[:target_len])
            mse = np.mean((a1 - a2) ** 2)
            angle_sim = 1.0 / (1.0 + mse * 4)

        return 0.5 * slope_sim + 0.5 * angle_sim

    def compare_geometry(self, features1: StructureFeatures,
                        features2: StructureFeatures) -> float:
        sym_diff = abs(features1.symmetry_score - features2.symmetry_score)
        sym_sim = 1.0 - min(1.0, sym_diff)

        conv_diff = abs(features1.convergence_rate - features2.convergence_rate)
        conv_sim = 1.0 - min(1.0, conv_diff)

        breakout_diff = abs(features1.breakout_strength - features2.breakout_strength)
        breakout_sim = 1.0 - min(1.0, breakout_diff)

        trend_diff = abs(features1.trend_consistency - features2.trend_consistency)
        trend_sim = 1.0 - min(1.0, trend_diff)

        return 0.25 * sym_sim + 0.25 * conv_sim + 0.25 * breakout_sim + 0.25 * trend_sim

    def _pattern_overlap_bonus(self, features1: StructureFeatures,
                              features2: StructureFeatures) -> float:
        p1 = getattr(features1, 'detected_patterns', {}) or {}
        p2 = getattr(features2, 'detected_patterns', {}) or {}
        if not p1 or not p2:
            return 0.0
        shared = set(p1.keys()) & set(p2.keys())
        if not shared:
            return 0.0
        best_overlap = max(min(p1[k], p2[k]) for k in shared)
        return min(0.3, best_overlap * 0.4)

    def compare_shape(self, features1: StructureFeatures,
                     features2: StructureFeatures) -> float:
        trend_diff = abs(features1.trend_direction - features2.trend_direction)
        trend_sim = 1.0 - min(trend_diff, 2.0) / 2.0

        vol_ratio = min(features1.volatility, features2.volatility) / (
            max(features1.volatility, features2.volatility) + 1e-10)

        comp_diff = abs(features1.compression_ratio - features2.compression_ratio)
        comp_sim = 1.0 - min(comp_diff, 2.0) / 2.0

        t1, t2 = features1.structure_type, features2.structure_type

        if t1 == t2:
            type_sim = 1.0
        elif self._same_category(t1, t2):
            type_sim = 0.60
        elif self._are_opposites(t1, t2):
            type_sim = 0.15
        else:
            type_sim = 0.35

        overlap = self._pattern_overlap_bonus(features1, features2)
        type_sim = min(1.0, type_sim + overlap)

        return 0.20 * trend_sim + 0.10 * vol_ratio + 0.20 * comp_sim + 0.50 * type_sim

    def _same_category(self, t1: StructureType, t2: StructureType) -> bool:
        for category in [DIRECTIONAL_TYPES, REVERSAL_TYPES, CONSOLIDATION_TYPES]:
            if t1 in category and t2 in category:
                return True
        return False

    def _are_opposites(self, t1: StructureType, t2: StructureType) -> bool:
        pair = (t1, t2)
        reverse = (t2, t1)
        return pair in OPPOSITE_PAIRS or reverse in OPPOSITE_PAIRS

    def _calculate_type_gate(self, features1: StructureFeatures,
                            features2: StructureFeatures) -> float:
        t1, t2 = features1.structure_type, features2.structure_type

        if t1 == t2:
            return 1.0

        p1 = getattr(features1, 'detected_patterns', {}) or {}
        p2 = getattr(features2, 'detected_patterns', {}) or {}
        shared = set(p1.keys()) & set(p2.keys()) if p1 and p2 else set()
        if shared:
            return 0.95

        if self._are_opposites(t1, t2):
            return 0.55

        if self._same_category(t1, t2):
            return 0.90

        if t1 == StructureType.UNKNOWN or t2 == StructureType.UNKNOWN:
            return 0.90

        return 0.75

    def _confidence_multiplier(self, features: StructureFeatures) -> float:
        conf = getattr(features, 'pattern_confidence', 0.5)
        min_conf = MIN_CONFIDENCE_THRESHOLDS.get(features.structure_type, 0.30)

        if conf < min_conf * 0.5:
            return 0.6
        if conf < min_conf:
            return 0.8
        return 1.0

    def calculate_similarity(self, reference: StructureFeatures,
                           candidate: StructureFeatures,
                           check_mirror: bool = True) -> Tuple[float, bool]:
        normal_score = self._compute_score(reference, candidate)
        normal_score = self._apply_gates(normal_score, reference, candidate)

        mirror_score = 0.0
        if check_mirror:
            mirrored = self.mirror_features(candidate)
            raw_mirror = self._compute_score(reference, mirrored)
            raw_mirror = self._apply_gates(raw_mirror, reference, mirrored)

            t1 = reference.structure_type
            t2 = candidate.structure_type
            if self._are_opposites(t1, t2):
                raw_mirror *= 0.85
            elif t1 in CONSOLIDATION_TYPES and t2 in CONSOLIDATION_TYPES:
                raw_mirror *= 0.95
            else:
                raw_mirror *= 0.90

            mirror_score = raw_mirror

        if mirror_score > normal_score:
            return mirror_score * 100, True
        return normal_score * 100, False

    def _compute_score(self, ref: StructureFeatures, cand: StructureFeatures) -> float:
        line_sim = self.compare_lines(ref.normalized_line, cand.normalized_line)
        pivot_sim = self.compare_pivots(ref.pivot_sequence, cand.pivot_sequence)
        dist_sim = self.compare_distances(ref.relative_distances, cand.relative_distances)
        shape_sim = self.compare_shape(ref, cand)
        slope_sim = self.compare_slopes(ref, cand)
        geometry_sim = self.compare_geometry(ref, cand)

        score = (
            self.line_weight * line_sim +
            self.pivot_weight * pivot_sim +
            self.distance_weight * dist_sim +
            self.shape_weight * shape_sim +
            self.slope_weight * slope_sim +
            self.geometry_weight * geometry_sim
        )

        return score

    def _apply_gates(self, score: float, ref: StructureFeatures,
                    cand: StructureFeatures) -> float:
        type_gate = self._calculate_type_gate(ref, cand)
        score *= type_gate

        ref_mult = self._confidence_multiplier(ref)
        cand_mult = self._confidence_multiplier(cand)
        score *= min(ref_mult, cand_mult)

        ref_quality = getattr(ref, 'quality_score', 0.5)
        cand_quality = getattr(cand, 'quality_score', 0.5)
        avg_quality = (ref_quality + cand_quality) / 2
        if avg_quality < 0.3:
            score *= 0.8
        elif avg_quality < 0.4:
            score *= 0.9

        ref_conf = getattr(ref, 'avg_pivot_confidence', 0.5)
        cand_conf = getattr(cand, 'avg_pivot_confidence', 0.5)
        avg_piv_conf = (ref_conf + cand_conf) / 2
        if avg_piv_conf > 0.6:
            score *= 1.05
        elif avg_piv_conf < 0.3:
            score *= 0.92

        cand_freshness = getattr(cand, 'pattern_freshness', 1.0)
        if cand_freshness < 0.5:
            score *= 0.6 + cand_freshness * 0.8

        cand_vol = getattr(cand, 'volume_confirmation', 0.5)
        if cand_vol > 0.7:
            score *= 1.0 + (cand_vol - 0.7) * 0.3
        elif cand_vol < 0.3:
            score *= 0.85

        return min(1.0, score)

    def find_matches(self, reference: StructureFeatures,
                    candidates: list,
                    threshold: float = 50.0) -> List[MatchResult]:
        raw_matches = []

        for item in candidates:
            symbol, timeframe, features, timestamp = item[0], item[1], item[2], item[3]
            candle_time = item[4] if len(item) > 4 else None

            if features is None:
                continue

            if hasattr(features, 'quality_score') and features.quality_score < 0.2:
                continue

            if hasattr(features, 'is_pattern_active') and not features.is_pattern_active:
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
        matches = self._apply_mtf_bonus(matches)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches

    def _apply_mtf_bonus(self, matches: List[MatchResult]) -> List[MatchResult]:
        symbol_tfs = {}
        for m in matches:
            base_tf = m.timeframe.split("_w")[0] if "_w" in m.timeframe else m.timeframe
            if m.symbol not in symbol_tfs:
                symbol_tfs[m.symbol] = set()
            symbol_tfs[m.symbol].add(base_tf)
        for m in matches:
            base_tf = m.timeframe.split("_w")[0] if "_w" in m.timeframe else m.timeframe
            tf_count = len(symbol_tfs.get(m.symbol, set()))
            if tf_count >= 3:
                m.similarity_score = min(99.99, m.similarity_score * 1.08)
            elif tf_count >= 2:
                m.similarity_score = min(99.99, m.similarity_score * 1.04)
            m.similarity_score = round(m.similarity_score, 2)
        return matches

    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        best_per_symbol = {}

        for m in matches:
            key = m.symbol
            base_tf = m.timeframe.split("_w")[0] if "_w" in m.timeframe else m.timeframe

            if key not in best_per_symbol:
                best_per_symbol[key] = [m]
            else:
                existing = best_per_symbol[key]
                dominated = False
                for idx, ex in enumerate(existing):
                    ex_base_tf = ex.timeframe.split("_w")[0] if "_w" in ex.timeframe else ex.timeframe
                    if ex_base_tf == base_tf:
                        if m.similarity_score > ex.similarity_score:
                            existing[idx] = m
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
