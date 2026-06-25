"""Классификатор релевантности матчей (обёртка над RandomForest).

Предсказывает вероятность того, что обнаруженный паттерн релевантен (класс 1)
против ложного срабатывания (класс 0). Модель сохраняется/загружается целиком
вместе с порядком признаков и метаданными обучения, чтобы инференс был
воспроизводим независимо от окружения.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np

try:  # scikit-learn обязателен для обучения/инференса
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - окружение без sklearn
    RandomForestClassifier = None  # type: ignore
    joblib = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from .features import FEATURE_ORDER, features_to_vector


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "Для ML-фильтра требуется scikit-learn (и joblib). "
            "Установите 'scikit-learn', чтобы обучать или загружать модель."
        )


class RelevanceClassifier:
    """RandomForest-классификатор релевантности паттернов."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        _require_sklearn()
        self.feature_names: List[str] = list(FEATURE_ORDER)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.metadata: Dict[str, Any] = {
            "trained_at": None,
            "n_samples": 0,
            "n_feedback": 0,
            "accuracy": None,
        }
        self._fitted = False

    # ------------------------------------------------------------------
    # Обучение / инференс
    # ------------------------------------------------------------------
    def _to_matrix(self, X: List[Any]) -> np.ndarray:
        """Принимает список dict-признаков либо готовых векторов."""
        rows: List[List[float]] = []
        for item in X:
            if isinstance(item, dict):
                rows.append(features_to_vector(item))
            else:
                rows.append([float(v) for v in item])
        return np.array(rows, dtype=float)

    def fit(
        self,
        X: List[Any],
        y: List[int],
        sample_weight: Optional[List[float]] = None,
    ) -> "RelevanceClassifier":
        _require_sklearn()
        mat = self._to_matrix(X)
        labels = np.array(y, dtype=int)
        weights = np.array(sample_weight, dtype=float) if sample_weight is not None else None
        self.model.fit(mat, labels, sample_weight=weights)
        self._fitted = True
        self.metadata["trained_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["n_samples"] = int(mat.shape[0])
        return self

    def predict_proba(self, X: List[Any]) -> np.ndarray:
        """Возвращает вероятность класса 1 (релевантно) для каждой строки."""
        _require_sklearn()
        if not self._fitted:
            raise RuntimeError("Модель не обучена: вызовите fit() или load().")
        mat = self._to_matrix(X)
        proba = self.model.predict_proba(mat)
        # индекс колонки класса 1
        classes = list(self.model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        # вырожденный случай: единственный класс
        return proba[:, -1]

    def predict_one(self, feats: Dict[str, float]) -> float:
        """Удобный скоринг одного словаря признаков -> float [0,1]."""
        return float(self.predict_proba([feats])[0])

    def feature_importances(self) -> Dict[str, float]:
        if not self._fitted:
            return {}
        imps = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, imps), key=lambda kv: kv[1], reverse=True
            )
        }

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        _require_sklearn()
        if not self._fitted:
            raise RuntimeError("Нечего сохранять: модель не обучена.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "RelevanceClassifier":
        _require_sklearn()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        payload = joblib.load(path)
        inst = cls()
        inst.model = payload["model"]
        inst.feature_names = payload.get("feature_names", list(FEATURE_ORDER))
        inst.metadata = payload.get("metadata", {})
        inst._fitted = True
        return inst


def sklearn_available() -> bool:
    """Доступен ли scikit-learn в окружении."""
    return _SKLEARN_AVAILABLE
