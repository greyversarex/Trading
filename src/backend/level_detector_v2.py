"""Робастная детекция уровней и трендовых линий (T1.3).

``LevelDetectorV2`` строит:
- горизонтальные зоны поддержки/сопротивления через DBSCAN-кластеризацию цен
  пивотов (eps масштабируется волатильностью),
- диагональные трендовые линии через RANSAC / Theil-Sen регрессию по пивотам,
- уровни профиля объёма (POC и границы зоны стоимости),
- круглые числа рядом с текущей ценой,
с последующей дедупликацией близких уровней (порог — доля volatility_scale).

Все пороги масштабируются ``volatility_scale`` (см. ``compute_volatility_scale``),
а не фиксированы на нормализованной шкале. Старый ``LevelDetector`` оставлен без
изменений для обратной совместимости; V2 используется новыми режимами сканирования.
"""

import math
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from .config import CONFIG
from .structure_extractor import compute_volatility_scale

try:
    from sklearn.cluster import DBSCAN
    from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - sklearn всегда установлен в проекте
    _SKLEARN_AVAILABLE = False


class LevelDetectorV2:
    """Детектор уровней второго поколения (DBSCAN + RANSAC + volume profile)."""

    def __init__(self):
        self.cfg = CONFIG.level

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def detect_levels(
        self,
        highs,
        lows,
        closes,
        volumes,
        times: Optional[List[int]] = None,
        order: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Возвращает словарь с ключами ``support_levels``,
        ``resistance_levels`` и ``trendlines``.

        ``times`` — необязательные временные метки баров (по умолчанию индексы),
        используются как ``first_touch_time`` / ``last_touch_time``.
        """
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)
        empty = {"support_levels": [], "resistance_levels": [], "trendlines": []}
        n = len(c)
        if n < 20 or len(h) != n or len(l) != n:
            return empty

        if volumes is None:
            v = np.zeros(n, dtype=float)
        else:
            v = np.asarray(volumes, dtype=float)
            if len(v) != n:
                v = np.zeros(n, dtype=float)

        if times is None:
            times = list(range(n))
        times = [int(t) for t in times]

        vol_scale = compute_volatility_scale(h, l, c)
        if vol_scale <= 0:
            return empty

        if order is None:
            order = max(2, n // 20)

        high_pivots = self._find_pivots(h, order, is_high=True)
        low_pivots = self._find_pivots(l, order, is_high=False)

        current_price = float(c[-1])
        mean_vol = float(np.mean(v)) if np.any(v > 0) else 0.0

        # 1. Горизонтальные зоны через DBSCAN.
        resistance = self._dbscan_zones(high_pivots, "resistance", v, times, vol_scale, n, mean_vol)
        support = self._dbscan_zones(low_pivots, "support", v, times, vol_scale, n, mean_vol)

        existing_prices = [lv["price"] for lv in resistance + support]

        # 2. Профиль объёма: POC и границы зоны стоимости.
        vp_levels = self._volume_profile_levels(
            c, v, vol_scale, current_price, existing_prices, times, n, mean_vol
        )
        for lv in vp_levels:
            (resistance if lv["type"] == "resistance" else support).append(lv)
            existing_prices.append(lv["price"])

        # 3. Круглые числа рядом с текущей ценой.
        rn_levels = self._round_number_levels(
            h, l, c, v, vol_scale, current_price, existing_prices, times, n, mean_vol
        )
        for lv in rn_levels:
            (resistance if lv["type"] == "resistance" else support).append(lv)

        # 4. Дедупликация близких уровней (суммирование силы).
        resistance = self._deduplicate(resistance, vol_scale)
        support = self._deduplicate(support, vol_scale)

        resistance.sort(key=lambda d: d["strength"], reverse=True)
        support.sort(key=lambda d: d["strength"], reverse=True)

        # 5. Диагональные трендовые линии.
        trendlines = []
        sup_line = self._fit_trendline(low_pivots, "support_trendline", vol_scale, n)
        res_line = self._fit_trendline(high_pivots, "resistance_trendline", vol_scale, n)
        if sup_line:
            trendlines.append(sup_line)
        if res_line:
            trendlines.append(res_line)
        # Параллельность (канал) — относительная разница наклонов.
        if sup_line and res_line:
            channel = self._check_parallel(sup_line, res_line, vol_scale)
            for tl in trendlines:
                tl["is_channel"] = channel

        return {
            "support_levels": support,
            "resistance_levels": resistance,
            "trendlines": trendlines,
        }

    # ------------------------------------------------------------------
    # Пивоты
    # ------------------------------------------------------------------
    def _find_pivots(self, data: np.ndarray, order: int, is_high: bool) -> List[Tuple[int, float]]:
        n = len(data)
        if n < 2 * order + 1:
            return []
        pivots: List[Tuple[int, float]] = []
        for i in range(order, n - order):
            window = data[i - order: i + order + 1]
            if is_high and data[i] == np.max(window):
                pivots.append((i, float(data[i])))
            elif not is_high and data[i] == np.min(window):
                pivots.append((i, float(data[i])))
        return pivots

    # ------------------------------------------------------------------
    # Горизонтальные зоны (DBSCAN)
    # ------------------------------------------------------------------
    def _dbscan_zones(self, pivots, level_type, volumes, times, vol_scale, n, mean_vol):
        if len(pivots) < self.cfg.dbscan_min_samples or not _SKLEARN_AVAILABLE:
            return []
        prices = np.array([p[1] for p in pivots]).reshape(-1, 1)
        idxs = np.array([p[0] for p in pivots])
        eps = max(self.cfg.dbscan_eps_factor * vol_scale, 1e-9)
        labels = DBSCAN(eps=eps, min_samples=self.cfg.dbscan_min_samples).fit_predict(prices)

        levels = []
        for lab in set(labels):
            if lab == -1:
                continue
            mask = labels == lab
            cluster_prices = prices[mask].flatten()
            cluster_idxs = idxs[mask]
            num_touches = int(len(cluster_idxs))
            price = float(np.median(cluster_prices))
            touch_times = sorted(times[i] for i in cluster_idxs)
            strength = self._strength(num_touches, cluster_idxs, volumes, n, mean_vol)
            levels.append({
                "price": round(price, 8),
                "strength": strength,
                "num_touches": num_touches,
                "first_touch_time": touch_times[0],
                "last_touch_time": touch_times[-1],
                "type": level_type,
                "source": "dbscan",
            })
        return levels

    # ------------------------------------------------------------------
    # Профиль объёма (POC / Value Area)
    # ------------------------------------------------------------------
    def _volume_profile_levels(self, closes, volumes, vol_scale, current_price,
                               existing_prices, times, n, mean_vol):
        if not self.cfg.use_volume_profile or not np.any(volumes > 0):
            return []
        lo, hi = float(np.min(closes)), float(np.max(closes))
        if hi <= lo:
            return []
        bins = max(2, self.cfg.volume_profile_bins)
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bin_idx = np.clip(np.digitize(closes, edges) - 1, 0, bins - 1)
        hist = np.zeros(bins)
        for i, b in enumerate(bin_idx):
            hist[b] += volumes[i]
        if hist.sum() <= 0:
            return []

        poc_bin = int(np.argmax(hist))
        total = hist.sum()
        target = 0.7 * total
        acc = hist[poc_bin]
        lo_b = hi_b = poc_bin
        while acc < target and (lo_b > 0 or hi_b < bins - 1):
            left = hist[lo_b - 1] if lo_b > 0 else -1.0
            right = hist[hi_b + 1] if hi_b < bins - 1 else -1.0
            if right >= left:
                hi_b += 1
                acc += hist[hi_b]
            else:
                lo_b -= 1
                acc += hist[lo_b]

        candidates = [
            (float(centers[poc_bin]), "poc"),
            (float(centers[hi_b]), "vah"),
            (float(centers[lo_b]), "val"),
        ]
        out = []
        seen_prices = list(existing_prices)
        for price, tag in candidates:
            if any(abs(price - ep) < 0.3 * vol_scale for ep in seen_prices):
                continue
            touch_idxs = self._touch_indices(price, closes, vol_scale * 0.5)
            num_touches = len(touch_idxs)
            level_type = "support" if price <= current_price else "resistance"
            strength = self._strength(max(1, num_touches), touch_idxs, volumes, n, mean_vol, base=0.25)
            touch_times = sorted(times[i] for i in touch_idxs) if touch_idxs else [times[-1]]
            out.append({
                "price": round(price, 8),
                "strength": strength,
                "num_touches": num_touches,
                "first_touch_time": touch_times[0],
                "last_touch_time": touch_times[-1],
                "type": level_type,
                "source": f"volume_{tag}",
            })
            seen_prices.append(price)
        return out

    # ------------------------------------------------------------------
    # Круглые числа
    # ------------------------------------------------------------------
    def _round_number_levels(self, highs, lows, closes, volumes, vol_scale,
                             current_price, existing_prices, times, n, mean_vol):
        if not self.cfg.use_round_numbers:
            return []
        step = self.cfg.round_number_step_factor * vol_scale
        if step <= 0:
            return []
        nice = self._nice_number(step)
        if nice <= 0:
            return []
        base = round(current_price / nice) * nice
        out = []
        seen_prices = list(existing_prices)
        for k in range(-3, 4):
            price = base + k * nice
            if price <= 0:
                continue
            if any(abs(price - ep) < 0.3 * vol_scale for ep in seen_prices):
                continue
            touch_idxs = self._touch_indices_hl(price, highs, lows, vol_scale * 0.5)
            if len(touch_idxs) < 1:
                continue
            level_type = "support" if price <= current_price else "resistance"
            strength = self._strength(len(touch_idxs), touch_idxs, volumes, n, mean_vol, base=0.2)
            touch_times = sorted(times[i] for i in touch_idxs)
            out.append({
                "price": round(float(price), 8),
                "strength": strength,
                "num_touches": len(touch_idxs),
                "first_touch_time": touch_times[0],
                "last_touch_time": touch_times[-1],
                "type": level_type,
                "source": "round_number",
            })
            seen_prices.append(price)
        return out

    # ------------------------------------------------------------------
    # Диагональные трендовые линии (RANSAC / Theil-Sen)
    # ------------------------------------------------------------------
    def _fit_trendline(self, pivots, line_type, vol_scale, n):
        if len(pivots) < self.cfg.min_touches or not _SKLEARN_AVAILABLE:
            return None
        X = np.array([p[0] for p in pivots], dtype=float).reshape(-1, 1)
        y = np.array([p[1] for p in pivots], dtype=float)
        residual_threshold = max(self.cfg.ransac_residual_threshold_factor * vol_scale, 1e-9)

        slope = intercept = None
        inlier_mask = None
        if self.cfg.use_ransac:
            try:
                model = RANSACRegressor(
                    min_samples=max(2, self.cfg.ransac_min_samples),
                    residual_threshold=residual_threshold,
                    random_state=42,
                )
                model.fit(X, y)
                slope = float(model.estimator_.coef_[0])
                intercept = float(model.estimator_.intercept_)
                inlier_mask = model.inlier_mask_
            except Exception:
                slope = None
        if slope is None:
            try:
                model = TheilSenRegressor(random_state=42)
                model.fit(X, y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
                pred = slope * X.flatten() + intercept
                inlier_mask = np.abs(y - pred) <= residual_threshold
            except Exception:
                return None

        inlier_idxs = [int(pivots[i][0]) for i in range(len(pivots)) if inlier_mask[i]]
        if len(inlier_idxs) < self.cfg.min_touches:
            return None

        start_idx = min(inlier_idxs)
        end_idx = max(inlier_idxs)
        coverage = (end_idx - start_idx) / max(n - 1, 1)
        strength = float(np.clip(0.5 * min(1.0, len(inlier_idxs) / 5.0) + 0.5 * coverage, 0.0, 1.0))

        return {
            "slope": round(slope, 10),
            "intercept": round(intercept, 8),
            "type": line_type,
            "strength": round(strength, 4),
            "touches": inlier_idxs,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
        }

    def _check_parallel(self, line_a, line_b, vol_scale) -> bool:
        """Канал: относительная разница наклонов опорной/сопротивления мала."""
        sa, sb = line_a["slope"], line_b["slope"]
        denom = max(abs(sa), abs(sb), 1e-9)
        return abs(sa - sb) / denom <= CONFIG.pattern.channel_parallel_tolerance

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------
    def _strength(self, num_touches, touch_idxs, volumes, n, mean_vol, base: float = 0.0) -> float:
        size_score = min(1.0, num_touches / 5.0)
        last_touch = max(touch_idxs) if len(touch_idxs) else 0
        recency = last_touch / max(n - 1, 1)
        if mean_vol > 0 and len(touch_idxs):
            touch_vol = float(np.mean([volumes[i] for i in touch_idxs]))
            vol_score = min(1.0, touch_vol / (mean_vol + 1e-9))
        else:
            vol_score = 0.0
        strength = base + 0.5 * size_score + 0.2 * recency + 0.3 * vol_score
        return round(float(np.clip(strength, 0.0, 1.0)), 4)

    def _touch_indices(self, price, series, tol) -> List[int]:
        return [int(i) for i in range(len(series)) if abs(series[i] - price) <= tol]

    def _touch_indices_hl(self, price, highs, lows, tol) -> List[int]:
        out = []
        for i in range(len(highs)):
            if abs(highs[i] - price) <= tol or abs(lows[i] - price) <= tol:
                out.append(int(i))
        return out

    def _nice_number(self, x: float) -> float:
        if x <= 0:
            return 0.0
        exp = math.floor(math.log10(x))
        frac = x / (10 ** exp)
        if frac < 1.5:
            nice = 1.0
        elif frac < 3.0:
            nice = 2.0
        elif frac < 7.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exp)

    def _deduplicate(self, levels, vol_scale):
        if not levels:
            return []
        levels = sorted(levels, key=lambda d: d["strength"], reverse=True)
        kept = []
        for lv in levels:
            merged = False
            for k in kept:
                if k["type"] != lv["type"]:
                    continue
                if abs(k["price"] - lv["price"]) < 0.3 * vol_scale:
                    k["strength"] = round(min(1.0, k["strength"] + lv["strength"]), 4)
                    k["num_touches"] = max(k["num_touches"], lv["num_touches"])
                    k["first_touch_time"] = min(k["first_touch_time"], lv["first_touch_time"])
                    k["last_touch_time"] = max(k["last_touch_time"], lv["last_touch_time"])
                    merged = True
                    break
            if not merged:
                kept.append(lv)
        return kept
