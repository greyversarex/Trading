"""Тесты ML-фильтра релевантности (фаза 2.3).

Покрывают: извлечение признаков на синтетике, обучение модели и достижение
точности > 0.7 на синтетическом датасете, сохранение/загрузку модели, скоринг
через MLPipeline и мягкое поведение при отсутствии модели.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.synthetic import SyntheticChartGenerator
from src.backend.structure_extractor import StructureExtractor
from src.backend.ml.features import (
    extract_ml_features,
    feature_names,
    features_to_vector,
    FEATURE_ORDER,
)
from src.backend.ml.model import RelevanceClassifier, sklearn_available
from src.backend.ml.trainer import ModelTrainer
from src.backend.ml.pipeline import MLPipeline


@pytest.fixture(scope="module")
def dataset():
    gen = SyntheticChartGenerator()
    # компактный, но достаточный для обучения набор
    return gen.generate_labeled_dataset(n_per_class=50, n_noise=120, seed=7)


@pytest.fixture(scope="module")
def extractor():
    return StructureExtractor()


# ---------------------------------------------------------------------------
# Извлечение признаков
# ---------------------------------------------------------------------------
def test_feature_order_stable_and_unique():
    names = feature_names()
    assert names == FEATURE_ORDER
    assert len(names) == len(set(names))
    # числовые + one-hot
    assert "pattern_confidence" in names
    assert any(n.startswith("type_") for n in names)


def test_extract_features_returns_full_vector(dataset, extractor):
    sample = next(s for s in dataset if s["label"] != "noise")
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    assert sf is not None
    feats = extract_ml_features(
        sf,
        candidate_index=sf.candidate_index,
        confirmation_index=sf.confirmation_index,
        candle_history=candles,
        timeframe_minutes=60.0,
    )
    # все имена присутствуют и значения числовые/конечные
    for name in FEATURE_ORDER:
        assert name in feats
        assert isinstance(feats[name], float)
        assert np.isfinite(feats[name])
    assert feats["timeframe_minutes"] == 60.0


def test_features_to_vector_matches_order(dataset, extractor):
    sample = dataset[0]
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("детектор не выделил структуру на этом образце")
    feats = extract_ml_features(sf, sf.candidate_index, sf.confirmation_index, candles)
    vec = features_to_vector(feats)
    assert len(vec) == len(FEATURE_ORDER)
    assert all(isinstance(v, float) for v in vec)


def test_onehot_exactly_one_active(dataset, extractor):
    sample = dataset[0]
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("нет структуры")
    feats = extract_ml_features(sf, sf.candidate_index, sf.confirmation_index, candles)
    onehot_sum = sum(v for k, v in feats.items() if k.startswith("type_"))
    assert onehot_sum == 1.0


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def test_build_training_data_has_both_classes(dataset):
    trainer = ModelTrainer()
    X, y, w = trainer.build_training_data(dataset)
    assert len(X) == len(y) == len(w)
    assert len(X) > 0
    assert set(y) == {0, 1}, "ожидаются оба класса в обучающем наборе"


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_train_accuracy_above_threshold(dataset, tmp_path):
    model_path = str(tmp_path / "model.pkl")
    trainer = ModelTrainer(model_path=model_path)
    result = trainer.train(dataset, save=True)
    assert result["trained"], result.get("reason")
    assert result["accuracy"] > 0.7, f"низкая точность: {result['accuracy']}"
    assert os.path.exists(model_path)
    assert result["feature_importances"]


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_save_load_roundtrip(dataset, tmp_path):
    model_path = str(tmp_path / "rt.pkl")
    trainer = ModelTrainer(model_path=model_path)
    result = trainer.train(dataset, save=True)
    assert result["trained"]

    loaded = RelevanceClassifier.load(model_path)
    # одинаковые предсказания на одном входе
    X, y, _ = trainer.build_training_data(dataset)
    sample = X[0]
    p1 = loaded.predict_one(sample)
    p2 = loaded.predict_one(sample)
    assert 0.0 <= p1 <= 1.0
    assert p1 == p2


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_predict_proba_range(dataset, tmp_path):
    model_path = str(tmp_path / "p.pkl")
    trainer = ModelTrainer(model_path=model_path)
    trainer.train(dataset, save=True)
    clf = RelevanceClassifier.load(model_path)
    X, _, _ = trainer.build_training_data(dataset)
    proba = clf.predict_proba(X[:20])
    assert proba.shape[0] == 20
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_pipeline_no_model_returns_one(tmp_path, dataset, extractor):
    pipe = MLPipeline(model_path=str(tmp_path / "absent.pkl"))
    assert not pipe.model_exists
    sample = dataset[0]
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("нет структуры")
    score = pipe.score_features(sf, candles, timeframe_minutes=15.0)
    assert score == 1.0
    status = pipe.status()
    assert status["model_exists"] is False


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_pipeline_with_model_scores(dataset, extractor, tmp_path):
    model_path = str(tmp_path / "pipe.pkl")
    trainer = ModelTrainer(model_path=model_path)
    trainer.train(dataset, save=True)

    pipe = MLPipeline(model_path=model_path)
    assert pipe.model_exists
    # подтверждённый паттерн должен скориться валидным числом [0,1]
    sample = next(s for s in dataset if s["label"] != "noise")
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("нет структуры")
    score = pipe.score_features(sf, candles, timeframe_minutes=60.0)
    assert 0.0 <= score <= 1.0
    status = pipe.status()
    assert status["model_exists"] is True
    assert len(status["top_features"]) > 0


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_feedback_records_with_features_added(dataset, tmp_path, extractor):
    """Обратная связь с готовыми признаками увеличивает обучающий набор."""
    trainer = ModelTrainer(model_path=str(tmp_path / "fb.pkl"))
    base_X, _, _ = trainer.build_training_data(dataset)

    sample = dataset[0]
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("нет структуры")
    feats = extract_ml_features(sf, sf.candidate_index, sf.confirmation_index, candles)
    feedback = [
        {"is_relevant": True, "features": feats},
        {"is_relevant": False, "features": feats},
        {"is_relevant": True},  # без признаков — должна быть пропущена
    ]
    X2, y2, w2 = trainer.build_training_data(dataset, feedback_records=feedback)
    assert len(X2) == len(base_X) + 2


@pytest.mark.skipif(not sklearn_available(), reason="нет scikit-learn")
def test_trainer_records_n_feedback(dataset, tmp_path, extractor):
    """train() записывает число валидных записей обратной связи в метаданные."""
    trainer = ModelTrainer(model_path=str(tmp_path / "nfb.pkl"))
    sample = dataset[0]
    candles = [SyntheticChartGenerator.candle_from_dict(c) for c in sample["candles"]]
    sf = extractor.extract_features_causal(candles)
    if sf is None:
        pytest.skip("нет структуры")
    feats = extract_ml_features(sf, sf.candidate_index, sf.confirmation_index, candles)
    feedback = [
        {"is_relevant": True, "features": feats},
        {"is_relevant": False, "features": feats},
        {"is_relevant": True},  # без признаков — не считается
    ]
    result = trainer.train(dataset, feedback_records=feedback, save=True)
    assert result["trained"] is True
    assert result["n_feedback"] == 2

    pipe = MLPipeline(model_path=str(tmp_path / "nfb.pkl"))
    assert pipe.status()["n_feedback"] == 2


def test_hard_filter_inactive_without_feedback(monkeypatch):
    """Жёсткий ML-фильтр выключен, пока модели нет или мало обратной связи,
    и включается после накопления достаточного числа записей."""
    import src.backend.main as m

    monkeypatch.setattr(m.CONFIG.ml, "ml_hard_filter_min_feedback", 20)

    # модель отсутствует -> фильтр выключен
    monkeypatch.setattr(m.ml_pipeline, "model", None, raising=False)
    assert m._ml_hard_filter_active() is False

    class _Fake:
        def __init__(self, n):
            self.metadata = {"n_feedback": n}

    # модель есть, но мало обратной связи -> выключен
    monkeypatch.setattr(m.ml_pipeline, "model", _Fake(5), raising=False)
    assert m._ml_hard_filter_active() is False

    # достаточно обратной связи -> включён
    monkeypatch.setattr(m.ml_pipeline, "model", _Fake(20), raising=False)
    assert m._ml_hard_filter_active() is True


def test_hard_filter_tolerates_invalid_metadata(monkeypatch):
    """Битая/устаревшая метадата (нет ключа или нечисловое значение) не должна
    ронять обработчик скана — фильтр считается выключенным."""
    import src.backend.main as m

    monkeypatch.setattr(m.CONFIG.ml, "ml_hard_filter_min_feedback", 20)

    class _Model:
        def __init__(self, meta):
            self.metadata = meta

    # legacy-модель без ключа n_feedback
    monkeypatch.setattr(m.ml_pipeline, "model", _Model({}), raising=False)
    assert m._ml_hard_filter_active() is False

    # нечисловое значение -> безопасный fallback в 0
    for bad in (None, "abc", object()):
        monkeypatch.setattr(m.ml_pipeline, "model", _Model({"n_feedback": bad}), raising=False)
        assert m._ml_hard_filter_active() is False
