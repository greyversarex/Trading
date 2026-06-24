"""Извлечение числовых признаков для ML-фильтра релевантности.

Признаки строятся из уже вычисленного объекта :class:`StructureFeatures`
(результат геометрической детекции) и истории свечей. Никаких сетевых вызовов
и тяжёлых вычислений — только агрегирование готовых полей и простая статистика
по свечам, чтобы фильтр можно было применять в реальном времени к каждому
эмитируемому матчу.

Порядок признаков фиксирован (:data:`FEATURE_ORDER`), чтобы вектор всегда
совпадал между обучением, сохранением модели и инференсом.
"""

from typing import Dict, List, Optional, Any

import numpy as np

from ..structure_extractor import (
    StructureFeatures,
    StructureType,
    compute_volatility_scale,
)


# Числовые (скалярные) признаки в стабильном порядке.
NUMERIC_FEATURES: List[str] = [
    "pattern_confidence",
    "quality_score",
    "symmetry_score",
    "convergence_rate",
    "breakout_strength",
    "avg_pivot_confidence",
    "trend_consistency",
    "volume_confirmation",
    "pattern_freshness",
    "is_confirmed",
    "timeframe_minutes",
    "n_pivots",
    "volatility_scale",
    "price_range",
    "relative_volume",
    "lag_to_confirmation",
]

# Полный словарь типов структур для one-hot кодирования.
STRUCTURE_TYPE_VOCAB: List[str] = [t.value for t in StructureType]

# Имена one-hot колонок.
ONEHOT_FEATURES: List[str] = [f"type_{name}" for name in STRUCTURE_TYPE_VOCAB]

# Итоговый порядок всех признаков (числовые + one-hot).
FEATURE_ORDER: List[str] = NUMERIC_FEATURES + ONEHOT_FEATURES


def feature_names() -> List[str]:
    """Возвращает стабильный список имён признаков (копию)."""
    return list(FEATURE_ORDER)


def _candle_attr(candle: Any, name: str) -> float:
    """Достаёт поле свечи как из dataclass ``CandleData``, так и из dict."""
    if isinstance(candle, dict):
        return float(candle.get(name, 0.0))
    return float(getattr(candle, name, 0.0))


def extract_ml_features(
    structure_features: StructureFeatures,
    candidate_index: int,
    confirmation_index: int,
    candle_history: List[Any],
    timeframe_minutes: float = 0.0,
) -> Dict[str, float]:
    """Строит словарь признаков для одного матча.

    Параметры
    ---------
    structure_features:
        Результат геометрической детекции (каузальной или нормализованной).
    candidate_index / confirmation_index:
        Индексы баров формирования и подтверждения паттерна (``-1`` если нет).
    candle_history:
        Список свечей (``CandleData`` либо dict) исходного ряда.
    timeframe_minutes:
        Минуты таймфрейма (1/5/15/60/240/1440); 0 если неизвестно.
    """
    sf = structure_features

    # --- Поля, напрямую доступные из StructureFeatures ---
    feats: Dict[str, float] = {
        "pattern_confidence": float(sf.pattern_confidence),
        "quality_score": float(sf.quality_score),
        "symmetry_score": float(sf.symmetry_score),
        "convergence_rate": float(sf.convergence_rate),
        "breakout_strength": float(sf.breakout_strength),
        "avg_pivot_confidence": float(sf.avg_pivot_confidence),
        "trend_consistency": float(sf.trend_consistency),
        "volume_confirmation": float(sf.volume_confirmation),
        "pattern_freshness": float(sf.pattern_freshness),
        "is_confirmed": 1.0 if sf.is_confirmed else 0.0,
        "timeframe_minutes": float(timeframe_minutes),
        "n_pivots": float(len(sf.pivot_points)),
    }

    # --- Статистика по свечам ---
    highs = np.array([_candle_attr(c, "high") for c in candle_history], dtype=float)
    lows = np.array([_candle_attr(c, "low") for c in candle_history], dtype=float)
    closes = np.array([_candle_attr(c, "close") for c in candle_history], dtype=float)
    volumes = np.array([_candle_attr(c, "volume") for c in candle_history], dtype=float)

    if closes.size > 0:
        mean_price = float(np.mean(closes)) or 1.0
        feats["volatility_scale"] = float(
            compute_volatility_scale(highs, lows, closes)
        )
        feats["price_range"] = float((np.max(closes) - np.min(closes)) / mean_price)
    else:
        feats["volatility_scale"] = 0.0
        feats["price_range"] = 0.0

    if volumes.size > 0 and float(np.mean(volumes)) > 0:
        recent = volumes[-10:] if volumes.size >= 10 else volumes
        feats["relative_volume"] = float(np.mean(recent) / np.mean(volumes))
    else:
        feats["relative_volume"] = 1.0

    # --- Лаг до подтверждения ---
    if confirmation_index is not None and confirmation_index >= 0 and candidate_index >= 0:
        feats["lag_to_confirmation"] = float(confirmation_index - candidate_index)
    else:
        feats["lag_to_confirmation"] = 0.0

    # --- One-hot структуры ---
    type_value = sf.structure_type.value if sf.structure_type is not None else "unknown"
    for name in STRUCTURE_TYPE_VOCAB:
        feats[f"type_{name}"] = 1.0 if name == type_value else 0.0

    return feats


def features_to_vector(feats: Dict[str, float]) -> List[float]:
    """Преобразует словарь признаков в вектор в порядке :data:`FEATURE_ORDER`."""
    return [float(feats.get(name, 0.0)) for name in FEATURE_ORDER]
