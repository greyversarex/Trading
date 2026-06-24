"""Пакет валидации/бэктеста детекторов структур.

Содержит загрузку исторических данных, метрики качества, хранилище ручной
разметки и раннер для прогона детектора по истории «бар за баром».
"""
from .history import fetch_history, fetch_history_async
from .labels import LabelStore
from .metrics import DetectedEvent, average_latency, precision_recall_f1, repaint_rate
from .runner import ValidationRunner

__all__ = [
    "fetch_history",
    "fetch_history_async",
    "LabelStore",
    "DetectedEvent",
    "average_latency",
    "precision_recall_f1",
    "repaint_rate",
    "ValidationRunner",
]
