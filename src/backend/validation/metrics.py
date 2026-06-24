"""Метрики качества детекции паттернов для валидационного харнесса.

Содержит расчёт precision/recall/F1 относительно размеченных событий,
оценку перерисовки (``repaint_rate``) и среднюю задержку подтверждения.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DetectedEvent:
    """Одно обнаруженное событие (паттерн) в конкретной точке истории."""

    pattern_type: str
    index: int
    time: int
    confirmed: bool = False
    candidate_index: int = -1
    confirmation_index: int = -1


def precision_recall_f1(
    detected: List[DetectedEvent],
    labeled: List[dict],
    time_tolerance: int = 3,
) -> Dict[str, float]:
    """precision/recall/F1 для обнаруженных событий против разметки.

    Совпадением считается событие того же ``pattern_type``, чей ``index``
    попадает в окно ±``time_tolerance`` баров от размеченного интервала.
    При отсутствии разметки recall/F1 не определены и возвращаются как 0.0.
    """
    tp = 0
    matched_labels = set()
    for ev in detected:
        for li, lab in enumerate(labeled):
            if li in matched_labels:
                continue
            if lab.get("pattern_type") != ev.pattern_type:
                continue
            start = lab.get("start_index", lab.get("start_time", 0))
            end = lab.get("end_index", lab.get("end_time", start))
            if start - time_tolerance <= ev.index <= end + time_tolerance:
                tp += 1
                matched_labels.add(li)
                break

    fp = len(detected) - tp
    fn = len(labeled) - len(matched_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_labeled": len(labeled),
    }


def repaint_rate(series: List[List[str]]) -> float:
    """Доля сигналов, исчезающих на следующем шаге сканирования.

    ``series[t]`` — список активных типов паттернов на шаге ``t`` (скользящее
    окно). Сигнал считается «перерисованным», если присутствует на шаге ``t``,
    но отсутствует на ``t+1``. Значение в диапазоне [0, 1]; чем меньше, тем
    стабильнее (менее склонен к перерисовке) детектор.
    """
    total = 0
    disappeared = 0
    for t in range(len(series) - 1):
        current = set(series[t])
        nxt = set(series[t + 1])
        for sig in current:
            total += 1
            if sig not in nxt:
                disappeared += 1
    if total == 0:
        return 0.0
    return round(disappeared / total, 4)


def average_latency(events: List[DetectedEvent]) -> Optional[float]:
    """Средняя задержка (в барах) от кандидата до подтверждения.

    Учитываются только события с заданным ``confirmation_index`` >= 0.
    Возвращает ``None``, если подтверждённых событий нет (например, до
    появления каузальной детекции в фазе 1.1).
    """
    latencies = [
        ev.confirmation_index - ev.candidate_index
        for ev in events
        if ev.confirmation_index >= 0 and ev.candidate_index >= 0
    ]
    if not latencies:
        return None
    return round(sum(latencies) / len(latencies), 4)
