# Отчёт по Фазе 2 — Синтетика, новые паттерны, ML-фильтр и валидация

## Цель фазы

Повысить точность сканера структур за счёт:
1. Генерации размеченных синтетических данных (надёжная «правда» для валидации
   и обучения).
2. Добавления недостающих детекторов паттернов.
3. Внедрения ML-фильтра релевантности для отсечения ложных срабатываний.
4. Интеграции ML-фильтра в реальное сканирование.
5. Реальной валидации на размеченной синтетике с измеримыми метриками.

Все изменения обратно совместимы: контракты REST/WS не сломаны (добавлены только
новые поля), `python run.py` запускается после каждого под-этапа, каждый под-этап
зафиксирован в `UPGRADE_LOG.md`.

## Что сделано по под-этапам

### 2.1 — Генератор синтетических паттернов
- `src/backend/synthetic.py` (`SyntheticChartGenerator`): случайное блуждание с
  реалистичным OHLCV и инъекторы паттернов (triple top/bottom, cup-and-handle,
  pennant, rounding bottom), `generate_labeled_dataset`.
- `scripts/generate_synthetic.py` -> `data/synthetic_dataset.json`.
- Тесты: `tests/test_synthetic.py`.

### 2.2 — Недостающие детекторы паттернов
- Новые `StructureType`: triple_top, triple_bottom, cup_and_handle, pennant,
  rounding_bottom.
- Кандидат-билдеры + диспетчеризация + приоритет в `extract_features_causal`
  (специфичные паттерны пробуются первыми, первый подтверждённый побеждает,
  фаллбэк на базовый кандидат — сохранение существующего поведения).
- Регистрация в `similarity_matcher` (REVERSAL/CONSOLIDATION, OPPOSITE_PAIRS,
  type_mirror_map, MIN_CONFIDENCE_THRESHOLDS) и в `classify_structure`
  (`detected_patterns`).
- Тесты: `tests/test_missing_patterns.py`.

### 2.3 — Признаки и классификатор ML
- `src/backend/ml/{features,model,trainer,pipeline}.py`.
- `CONFIG.ml` (`MLConfig`): `ml_min_training_samples`, `ml_feedback_weight`,
  `ml_retrain_interval_hours`, `ml_score_threshold`, `ml_model_path`.
- `database.get_all_feedback()` для подмешивания обратной связи.
- `scripts/train_ml.py` -> `data/ml_relevance_model.pkl` (accuracy ≈ 0.99).
- Тесты: `tests/test_ml_filter.py`.

### 2.4 — Интеграция ML-фильтра в сканирование
- `main.py`: глобальный `ml_pipeline`, расчёт `ml_score` для матчей режимов
  causal (live + initial) и type_scan, добавление поля `ml_score` в WS-сообщения,
  фильтрация по `ml_score_threshold` (только при наличии модели), фоновое
  авто-переобучение, эндпоинты `POST /api/retrain-ml` и `GET /api/ml-status`.
- Обратная совместимость: без модели `ml_score = 1.0`, фильтрация не применяется.
- Тесты: `tests/test_ml_integration.py`.

### 2.5 — Реальная валидация на синтетике
- `ValidationRunner.run_synthetic_validation(...)`: per-pattern
  precision/recall/F1, общие `true_positive_rate` / `false_positive_rate`,
  гистограмма `ml_score_distribution`, сравнение режимов raw vs ML.
- `scripts/validate.py --synthetic` -> `data/validation_synthetic.json`.
- Тесты: `tests/test_synthetic_validation.py`.

## Результаты валидации

Датасет: **600 образцов** (300 паттернов / 300 шума). Модель ML: accuracy ≈ 0.99.
Порог `ml_score` = 0.5.

| Режим | true_positive_rate | false_positive_rate |
|-------|--------------------|---------------------|
| Только геометрия | 73.67% | 43.67% |
| Геометрия + ML-фильтр | **73.67%** | **0.00%** |

Per-pattern (геометрия + ML-фильтр):

| Паттерн | Precision | Recall | F1 |
|---------|-----------|--------|----|
| pennant | 100.00% | 100.00% | 100.00% |
| rounding_bottom | 100.00% | 86.67% | 92.86% |
| triple_bottom | 100.00% | 78.33% | 87.85% |
| triple_top | 100.00% | 60.00% | 75.00% |
| cup_and_handle | 100.00% | 43.33% | 60.47% |

**Вывод:** ML-фильтр убирает все ложные срабатывания на шуме (FPR с 43.67% до
0.00%), не теряя ни одного истинного паттерна (TPR неизменен — 73.67%).
Precision по всем типам поднимается до 100%. Распределение `ml_score` чётко
бимодальное (масса у 0.0–0.1 и 0.9–1.0), то есть модель уверенно разделяет
релевантные и нерелевантные матчи.

Recall остаётся ограничен геометрией детектора (особенно cup_and_handle и
triple_top) — это направление для дальнейшего улучшения детекторов в следующих
фазах, а не задача ML-фильтра (его роль — отсекать ложные срабатывания).

## Тесты

Полный набор: **106 тестов проходят** (`python -m pytest -q`).

- `test_synthetic.py` — генератор синтетики.
- `test_missing_patterns.py` — новые детекторы.
- `test_ml_filter.py` — признаки/модель/пайплайн ML.
- `test_ml_integration.py` — интеграция ML в сканирование и эндпоинты.
- `test_synthetic_validation.py` — синтетическая валидация (TPR/FPR/гистограмма).
- (ранее) `test_causal_detection.py`, `test_atr_adaptive.py`,
  `test_level_detector_v2.py`.

## Совместимость и эксплуатация

- REST/WS контракты не сломаны — добавлены только новые поля (`ml_score`).
- Без обученной модели система работает как раньше (`ml_score = 1.0`).
- Переобучение: вручную (`POST /api/retrain-ml`, `scripts/train_ml.py`) или
  автоматически по интервалу `CONFIG.ml.ml_retrain_interval_hours`.
- Артефакты: `data/synthetic_dataset.json`, `data/ml_relevance_model.pkl`,
  `data/validation_synthetic.json`.
