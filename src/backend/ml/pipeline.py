"""Удобный оркестратор инференса ML-фильтра для использования в main.py.

Инкапсулирует загрузку модели (с «мягким» поведением, если файла нет) и
скоринг готовых ``StructureFeatures``. При отсутствии модели возвращает
``ml_score=1.0`` — система ведёт себя как до внедрения ML (обратная совместимость).
"""

import os
from typing import Dict, List, Optional, Any

from ..structure_extractor import StructureFeatures
from .features import extract_ml_features
from .model import RelevanceClassifier, sklearn_available


class MLPipeline:
    """Загружает модель релевантности и считает ``ml_score`` для матчей."""

    def __init__(self, model_path: str = "data/ml_relevance_model.pkl"):
        self.model_path = model_path
        self.model: Optional[RelevanceClassifier] = None
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Пытается загрузить модель. Возвращает True при успехе."""
        self.model = None
        if not sklearn_available():
            return False
        if not os.path.exists(self.model_path):
            return False
        try:
            self.model = RelevanceClassifier.load(self.model_path)
            return True
        except Exception:
            self.model = None
            return False

    @property
    def model_exists(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    def score_features(
        self,
        structure_features: StructureFeatures,
        candle_history: List[Any],
        timeframe_minutes: float = 0.0,
    ) -> float:
        """Считает ``ml_score`` [0,1] для одного матча.

        Если модель не загружена — возвращает 1.0 (бэквард-совместимость).
        """
        if self.model is None:
            return 1.0
        feats = extract_ml_features(
            structure_features,
            candidate_index=structure_features.candidate_index,
            confirmation_index=structure_features.confirmation_index,
            candle_history=candle_history,
            timeframe_minutes=timeframe_minutes,
        )
        try:
            return self.model.predict_one(feats)
        except Exception:
            return 1.0

    def status(self) -> Dict[str, Any]:
        """Сводка состояния модели для эндпоинта /api/ml-status."""
        if self.model is None:
            return {
                "model_exists": False,
                "last_trained_at": None,
                "n_samples": 0,
                "accuracy": None,
                "top_features": [],
            }
        meta = self.model.metadata or {}
        importances = self.model.feature_importances()
        top_features = list(importances.items())[:10]
        return {
            "model_exists": True,
            "last_trained_at": meta.get("trained_at"),
            "n_samples": meta.get("n_samples", 0),
            "n_feedback": meta.get("n_feedback", 0),
            "accuracy": meta.get("accuracy"),
            "top_features": [{"name": n, "importance": i} for n, i in top_features],
        }
