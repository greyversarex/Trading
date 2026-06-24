"""Валидационный/бэктест харнесс.

Прогоняет детектор структуры по историческим данным «бар за баром»
(скользящее окно), собирает обнаруженные сигналы и считает метрики качества
и перерисовки. Опирается на ``CONFIG.validation`` для значений по умолчанию.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import CONFIG
from .history import fetch_history_async
from .labels import LabelStore
from .metrics import (
    DetectedEvent,
    average_latency,
    precision_recall_f1,
    repaint_rate,
)


class ValidationRunner:
    """Запускает валидацию детектора на исторических данных."""

    def __init__(
        self, label_store: Optional[LabelStore] = None, window: Optional[int] = None
    ):
        self.label_store = label_store if label_store is not None else LabelStore()
        # окно анализа: по умолчанию стандартный лимит свечей сканера
        self.window = window if window is not None else CONFIG.data.default_limit

    async def run_pattern_validation(
        self, symbol: str, timeframe: str, detector, limit: int = 500
    ) -> Dict:
        """Прогоняет детектор по истории символа/таймфрейма бар за баром.

        Скользящее окно длиной ``self.window`` сдвигается на один бар за шаг.
        На каждом шаге фиксируется обнаруженный паттерн (если есть). Затем
        считаются метрики против разметки и оценка перерисовки между шагами.
        """
        candles = await fetch_history_async(symbol, timeframe, limit)
        result: Dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "num_candles": len(candles),
            "events": [],
            "per_pattern": {},
            "repaint_rate": 0.0,
            "average_latency": None,
            "metrics": {},
            "data_available": len(candles) > 0,
        }
        if len(candles) < self.window + 1:
            return result

        events: List[DetectedEvent] = []
        active_series: List[List[str]] = []

        for i in range(self.window, len(candles) + 1):
            window_candles = candles[i - self.window : i]
            closes = np.array([c.close for c in window_candles], dtype=float)
            volumes = np.array([c.volume for c in window_candles], dtype=float)
            features = detector.extract_features(closes, volumes)

            step_active: List[str] = []
            if features is not None:
                ptype = features.structure_type.value
                confirmed = bool(getattr(features, "is_pattern_active", True))
                last = window_candles[-1]
                candidate_index = getattr(features, "candidate_index", i - 1)
                confirmation_index = getattr(
                    features, "confirmation_index", i - 1 if confirmed else -1
                )
                events.append(
                    DetectedEvent(
                        pattern_type=ptype,
                        index=i - 1,
                        time=last.open_time,
                        confirmed=confirmed,
                        candidate_index=candidate_index,
                        confirmation_index=confirmation_index,
                    )
                )
                if confirmed:
                    step_active.append(ptype)
            active_series.append(step_active)

        labeled = self.label_store.get_labels(symbol, timeframe)
        metrics = precision_recall_f1(events, labeled)
        rrate = repaint_rate(active_series)
        latency = average_latency(events)

        per_pattern: Dict[str, Dict] = {}
        for ev in events:
            stats = per_pattern.setdefault(ev.pattern_type, {"total": 0, "confirmed": 0})
            stats["total"] += 1
            if ev.confirmed:
                stats["confirmed"] += 1

        result.update(
            {
                "events": [
                    {
                        "pattern_type": ev.pattern_type,
                        "index": ev.index,
                        "time": ev.time,
                        "confirmed": ev.confirmed,
                    }
                    for ev in events
                ],
                "per_pattern": per_pattern,
                "repaint_rate": rrate,
                "average_latency": latency,
                "metrics": metrics,
            }
        )
        return result

    async def run_multi_symbol(
        self,
        detector,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        """Прогоняет валидацию по набору символов и таймфреймов из CONFIG."""
        symbols = symbols if symbols is not None else CONFIG.validation.test_symbols
        timeframes = (
            timeframes if timeframes is not None else CONFIG.validation.test_timeframes
        )
        limit = limit if limit is not None else CONFIG.validation.max_history_limit

        runs: List[Dict] = []
        for symbol in symbols:
            for timeframe in timeframes:
                res = await self.run_pattern_validation(
                    symbol, timeframe, detector, limit
                )
                runs.append(res)

        return self._aggregate(runs)

    def _aggregate(self, runs: List[Dict]) -> Dict:
        """Сводит результаты прогонов в агрегированные метрики."""
        by_pattern: Dict[str, Dict] = {}
        by_timeframe: Dict[str, Dict] = {}

        for run in runs:
            tf = run["timeframe"]
            tf_stats = by_timeframe.setdefault(
                tf, {"total": 0, "confirmed": 0, "repaint_rates": []}
            )
            for ptype, stats in run["per_pattern"].items():
                p = by_pattern.setdefault(ptype, {"total": 0, "confirmed": 0})
                p["total"] += stats["total"]
                p["confirmed"] += stats["confirmed"]
                tf_stats["total"] += stats["total"]
                tf_stats["confirmed"] += stats["confirmed"]
            if run["data_available"]:
                tf_stats["repaint_rates"].append(run["repaint_rate"])

        for tf, stats in by_timeframe.items():
            rates = stats.pop("repaint_rates")
            stats["avg_repaint_rate"] = (
                round(sum(rates) / len(rates), 4) if rates else 0.0
            )

        total_tp = sum(r["metrics"].get("tp", 0) for r in runs if r["metrics"])
        total_fp = sum(r["metrics"].get("fp", 0) for r in runs if r["metrics"])
        total_fn = sum(r["metrics"].get("fn", 0) for r in runs if r["metrics"])
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        total_events = sum(len(r["events"]) for r in runs)
        runs_with_data = sum(1 for r in runs if r["data_available"])

        return {
            "summary": {
                "num_runs": len(runs),
                "runs_with_data": runs_with_data,
                "total_events": total_events,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "by_pattern": by_pattern,
            "by_timeframe": by_timeframe,
            "runs": runs,
        }
