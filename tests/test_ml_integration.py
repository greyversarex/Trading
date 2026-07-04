"""Интеграционные тесты ML-фильтра в сканировании (фаза 2.4).

Проверяют:
  * обратную совместимость — без модели ``ml_score=1.0`` и матчи не отсекаются;
  * фильтрацию слабых матчей при наличии модели и высоком пороге;
  * эндпоинты ``/api/retrain-ml`` и ``/api/ml-status`` (вызов функций напрямую,
    без HTTP-клиента, чтобы не тянуть лишние зависимости).
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend import main as app_main
from src.backend.ml.pipeline import MLPipeline
from src.backend.ml.model import sklearn_available
from src.backend.ml.features import FEATURE_ORDER


# ---------------------------------------------------------------------------
# Хелперы интеграции
# ---------------------------------------------------------------------------
def test_tf_minutes_mapping():
    assert app_main._tf_minutes("1m") == 1
    assert app_main._tf_minutes("1h") == 60
    assert app_main._tf_minutes("4h") == 240
    assert app_main._tf_minutes("1d") == 1440
    # оконные таймфреймы сводятся к базе
    assert app_main._tf_minutes("15m_w30") == 15
    assert app_main._tf_minutes("unknown_tf") == 0


def test_backward_compat_no_model(tmp_path):
    """Без модели pipeline возвращает 1.0 и не существует как модель."""
    pipe = MLPipeline(model_path=str(tmp_path / "no_model.pkl"))
    assert not pipe.model_exists

    class _SF:
        candidate_index = -1
        confirmation_index = -1

    # даже при отсутствии валидных признаков score == 1.0 (мягкое поведение)
    score = pipe.score_features(_SF(), [], timeframe_minutes=60.0)
    assert score == 1.0


def test_gate_emits_when_no_model():
    """Контракт гейта: без модели матч НЕ отсекается (model_exists=False)."""
    pipe = MLPipeline(model_path="data/__definitely_absent__.pkl")
    ml_score = 0.01  # даже очень низкий
    # условие отсечения в main: model_exists and ml_score < threshold
    should_filter = pipe.model_exists and ml_score < 0.5
    assert should_filter is False


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_low_confidence_filtered_with_model(tmp_path):
    """С обученной моделью слабый (нулевой) вектор признаков скорится низко."""
    from src.backend.synthetic import SyntheticChartGenerator
    from src.backend.ml.trainer import ModelTrainer
    from src.backend.ml.model import RelevanceClassifier

    gen = SyntheticChartGenerator()
    dataset = gen.generate_labeled_dataset(n_per_class=50, n_noise=120, seed=11)
    model_path = str(tmp_path / "integ.pkl")
    trainer = ModelTrainer(model_path=model_path)
    res = trainer.train(dataset, save=True)
    assert res["trained"]

    clf = RelevanceClassifier.load(model_path)
    # «пустой» матч: нет подтверждения, нулевые уверенности
    weak = {name: 0.0 for name in FEATURE_ORDER}
    weak_score = clf.predict_one(weak)
    # при пороге 0.5 такой матч должен отсекаться
    assert weak_score < 0.5
    # гейт: модель есть и score ниже порога -> отсекаем
    assert (True and weak_score < 0.5) is True


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_retrain_endpoint_returns_metrics():
    """POST /api/retrain-ml (функция) возвращает success и метрики."""
    import json as _json

    from src.backend.synthetic import SyntheticChartGenerator

    # БД нужна для загрузки обратной связи
    asyncio.run(app_main.database.initialize())

    # Эндпоинт читает синтетический датасет из фиксированного пути. Делаем тест
    # самодостаточным: генерируем детерминированный датасет, а существующий файл
    # сохраняем и восстанавливаем, чтобы не затирать пользовательские данные.
    path = app_main.SYNTHETIC_DATASET_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backup = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            backup = f.read()

    gen = SyntheticChartGenerator()
    dataset = gen.generate_labeled_dataset(n_per_class=50, n_noise=120, seed=11)

    try:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(dataset, f)

        app_main._retrain_in_progress = False
        result = asyncio.run(app_main.retrain_ml())
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result.get("trained") is True
        assert result.get("accuracy", 0) > 0.7
        # после переобучения модель загружена в пайплайн
        assert app_main.ml_pipeline.model_exists
    finally:
        if backup is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(backup)
        elif os.path.exists(path):
            os.remove(path)


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_ml_status_endpoint():
    """GET /api/ml-status (функция) отражает наличие модели и порог."""
    status = asyncio.run(app_main.ml_status())
    assert "model_exists" in status
    assert "ml_score_threshold" in status
    assert status["ml_score_threshold"] == app_main.ml_score_threshold
    assert "retrain_in_progress" in status
    if status["model_exists"]:
        assert "top_features" in status
        assert isinstance(status["top_features"], list)
