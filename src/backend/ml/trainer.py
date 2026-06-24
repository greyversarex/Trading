"""Сборка обучающего набора и обучение классификатора релевантности.

Логика разметки (синтетика):
  * Образец прогоняется через ``extract_features_causal`` -> ``StructureFeatures``.
  * Метка ``1`` (релевантно), если образец — паттерн-класс, детектор ПОДТВЕРДИЛ
    паттерн и обнаруженный тип совпадает с заявленным классом.
  * Метка ``0`` (ложное), если это шум ЛИБО детектор выдал на образце «не тот»
    паттерн / не подтвердил его. То есть классификатор учится отделять реальные
    подтверждённые паттерны от ложных срабатываний.

Обратная связь пользователя (таблица ``feedback``) добавляется как
дополнительные примеры с весом ``ml_feedback_weight``: ``is_relevant=true`` ->
метка 1, ``false`` -> метка 0. Запись учитывается только если для неё удаётся
восстановить признаки (присутствует поле ``features``); иначе она пропускается,
чтобы не выдумывать данные.
"""

from typing import Dict, List, Optional, Any, Tuple

from ..config import CONFIG
from ..structure_extractor import StructureExtractor
from ..synthetic import SyntheticChartGenerator
from .features import extract_ml_features
from .model import RelevanceClassifier


class ModelTrainer:
    """Готовит данные и обучает :class:`RelevanceClassifier`."""

    def __init__(
        self,
        model_path: str = "data/ml_relevance_model.pkl",
        extractor: Optional[StructureExtractor] = None,
    ):
        self.model_path = model_path
        self.extractor = extractor or StructureExtractor()

    # ------------------------------------------------------------------
    # Построение обучающего набора
    # ------------------------------------------------------------------
    def build_training_data(
        self,
        synthetic_dataset: List[Dict[str, Any]],
        feedback_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, float]], List[int], List[float]]:
        """Возвращает ``(X, y, sample_weight)``.

        ``X`` — список словарей признаков, ``y`` — метки 0/1,
        ``sample_weight`` — веса (1.0 для синтетики, ``ml_feedback_weight`` для
        пользовательской обратной связи).
        """
        X: List[Dict[str, float]] = []
        y: List[int] = []
        weights: List[float] = []

        for sample in synthetic_dataset:
            candles_raw = sample.get("candles", [])
            if not candles_raw:
                continue
            candles = [SyntheticChartGenerator.candle_from_dict(c) for c in candles_raw]
            sf = self.extractor.extract_features_causal(candles)
            if sf is None:
                # детектор ничего не выделил — это не «матч», пропускаем
                continue

            intended = sample.get("label", "noise")
            detected = sf.structure_type.value if sf.structure_type is not None else "unknown"

            is_relevant = (
                intended != "noise"
                and sf.is_confirmed
                and detected == intended
            )
            label = 1 if is_relevant else 0

            feats = extract_ml_features(
                sf,
                candidate_index=sf.candidate_index,
                confirmation_index=sf.confirmation_index,
                candle_history=candles,
                timeframe_minutes=0.0,
            )
            X.append(feats)
            y.append(label)
            weights.append(1.0)

        # --- Обратная связь пользователя ---
        if feedback_records:
            fb_weight = float(CONFIG.ml.ml_feedback_weight)
            for rec in feedback_records:
                feats = rec.get("features")
                if not feats:
                    # без восстановимых признаков — не выдумываем
                    continue
                label = 1 if rec.get("is_relevant") else 0
                X.append(feats)
                y.append(label)
                weights.append(fb_weight)

        return X, y, weights

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def train(
        self,
        synthetic_dataset: List[Dict[str, Any]],
        feedback_records: Optional[List[Dict[str, Any]]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Обучает модель и возвращает метрики.

        Возвращает словарь: ``accuracy``, ``report`` (classification_report),
        ``n_samples``, ``n_positive``, ``n_negative``, ``feature_importances``.
        """
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        X, y, weights = self.build_training_data(synthetic_dataset, feedback_records)

        n_samples = len(X)
        n_positive = sum(1 for v in y if v == 1)
        n_negative = n_samples - n_positive
        min_samples = int(CONFIG.ml.ml_min_training_samples)

        result: Dict[str, Any] = {
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "trained": False,
        }

        # Нужны оба класса и достаточный объём данных.
        if n_samples < min_samples or n_positive == 0 or n_negative == 0:
            result["reason"] = (
                f"Недостаточно данных для обучения: {n_samples} образцов "
                f"(нужно >= {min_samples}), положительных={n_positive}, "
                f"отрицательных={n_negative}."
            )
            return result

        clf = RelevanceClassifier()

        # Разбиваем для честной оценки точности.
        stratify = y if (n_positive >= 2 and n_negative >= 2) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )
        clf.fit(X_tr, y_tr)

        proba_te = clf.predict_proba(X_te)
        pred_te = [1 if p >= 0.5 else 0 for p in proba_te]
        accuracy = float(accuracy_score(y_te, pred_te))
        report = classification_report(y_te, pred_te, zero_division=0)

        # Переобучаем на всех данных перед сохранением (больше сигнала).
        clf.fit(X, y, sample_weight=weights)
        clf.metadata["accuracy"] = accuracy

        if save:
            clf.save(self.model_path)

        result.update({
            "trained": True,
            "accuracy": accuracy,
            "report": report,
            "feature_importances": clf.feature_importances(),
            "model_path": self.model_path,
        })
        self._last_model = clf
        return result

    def retrain(
        self,
        synthetic_dataset: List[Dict[str, Any]],
        feedback_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Полностью пересобирает набор и переобучает модель, перезаписывая файл."""
        import os

        if os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
            except OSError:
                pass
        return self.train(synthetic_dataset, feedback_records, save=True)
