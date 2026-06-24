"""Тесты синтетической валидации (фаза 2.5).

Проверяют контракт ``ValidationRunner.run_synthetic_validation``: структуру
отчёта, корректность TPR/FPR, per-pattern метрик и гистограммы ml_score.
Реальная детекция не вызывается напрямую — используется лёгкий стаб-детектор,
чтобы тесты были детерминированы и быстры.
"""
from __future__ import annotations

from src.backend.structure_extractor import StructureType
from src.backend.synthetic import SyntheticChartGenerator
from src.backend.validation.runner import ValidationRunner


class _Features:
    """Минимальный объект признаков для стаба детектора."""

    def __init__(self, type_value: str, confirmed: bool):
        self.structure_type = type(
            "_T", (), {"value": type_value}
        )()
        self.is_confirmed = confirmed
        self.candidate_index = 0
        self.confirmation_index = 1 if confirmed else -1


class _StubDetector:
    """Детектор, возвращающий заранее заданный результат по индексу образца."""

    def __init__(self, outcomes):
        # outcomes: list of (type_value | None, confirmed)
        self._outcomes = outcomes
        self._i = 0

    def extract_features_causal(self, candles):
        type_value, confirmed = self._outcomes[self._i]
        self._i += 1
        if type_value is None:
            return None
        return _Features(type_value, confirmed)


def _make_sample(label: str):
    gen = SyntheticChartGenerator()
    candles = gen.generate_random_walk(n=30, seed=1)
    return {
        "candles": [SyntheticChartGenerator._candle_to_dict(c) for c in candles],
        "label": label,
        "target_confirmed": label != "noise",
    }


def _dataset():
    # 2 паттерна triple_top, 2 шума
    return [
        _make_sample("triple_top"),
        _make_sample("triple_top"),
        _make_sample("noise"),
        _make_sample("noise"),
    ]


def test_report_structure_keys():
    dataset = _dataset()
    detector = _StubDetector(
        [("triple_top", True), ("triple_top", True), (None, False), (None, False)]
    )
    runner = ValidationRunner()
    report = runner.run_synthetic_validation(detector, dataset)

    for key in (
        "n_samples",
        "n_pattern",
        "n_noise",
        "ml_filter_used",
        "raw",
        "ml",
        "ml_score_distribution",
    ):
        assert key in report
    assert report["n_pattern"] == 2
    assert report["n_noise"] == 2
    assert report["ml_filter_used"] is False  # без pipeline


def test_perfect_detection_no_false_positives():
    dataset = _dataset()
    # Оба паттерна распознаны верно, на шуме детектор молчит
    detector = _StubDetector(
        [("triple_top", True), ("triple_top", True), (None, False), (None, False)]
    )
    runner = ValidationRunner()
    report = runner.run_synthetic_validation(detector, dataset)

    raw = report["raw"]
    assert raw["true_positive_rate"] == 1.0
    assert raw["false_positive_rate"] == 0.0
    tt = raw["by_pattern"]["triple_top"]
    assert tt["tp"] == 2 and tt["fp"] == 0 and tt["fn"] == 0
    assert tt["precision"] == 1.0 and tt["recall"] == 1.0


def test_false_positive_on_noise_counts():
    dataset = _dataset()
    # На втором шуме детектор ошибочно подтверждает паттерн
    detector = _StubDetector(
        [("triple_top", True), (None, False), ("triple_top", True), (None, False)]
    )
    runner = ValidationRunner()
    report = runner.run_synthetic_validation(detector, dataset)

    raw = report["raw"]
    # один из двух паттернов пропущен -> TPR = 0.5
    assert raw["true_positive_rate"] == 0.5
    # одно ложное срабатывание на шуме из двух -> FPR = 0.5
    assert raw["false_positive_rate"] == 0.5
    tt = raw["by_pattern"]["triple_top"]
    assert tt["fp"] == 1  # сработал на шуме
    assert tt["fn"] == 1  # пропустил один истинный


def test_ml_distribution_present_without_model():
    dataset = _dataset()
    detector = _StubDetector(
        [("triple_top", True), ("triple_top", True), (None, False), (None, False)]
    )
    runner = ValidationRunner()
    report = runner.run_synthetic_validation(detector, dataset)

    dist = report["ml_score_distribution"]
    assert len(dist["bins"]) == 10
    assert len(dist["counts"]) == 10
    # без модели оценки не считаются
    assert dist["n_scored"] == 0
