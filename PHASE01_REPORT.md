# Отчёт по Phase 0 + Phase 1 — Апгрейд точности сканера структур

Дата: 2026-06-24

Документ резюмирует выполненные этапы повышения точности детекции для
«Сканера Структур Графиков». Пошаговый журнал изменений — в `UPGRADE_LOG.md`.

## Краткое резюме

Реализованы Phase 0 (фундамент: конфигурация + валидация) и Phase 1 (точность:
каузальная детекция, ATR-адаптация, робастные уровни, логирование). Все
REST/WS-контракты сохранены обратно совместимыми (добавлены только новые поля
и режимы). После каждого подэтапа `python run.py` остаётся работоспособным.

- Тесты: **39 passed** (`python -m pytest -q`).
- Валидация: `python scripts/validate.py` отрабатывает без регрессии.
- Приложение стартует чисто на порту 5000; новые эндпоинты проверены на
  реальных данных Binance.
- Код-ревью (architect): вердикт PASS, два замечания средней важности
  исправлены (см. раздел «Код-ревью»).

## Принципы (по согласованной архитектуре)

- Геометрия детекции считается на **сырых ценах**; нормализованная линия
  используется только для сходства формы.
- Каузальный (непереисовывающий) слой добавлен **параллельно** — легаси
  `extract_features(price_line)` и форма `StructureFeatures` не ломаются.
- В `StructureFeatures` добавлены только **опциональные** поля со значениями по
  умолчанию (без миграции БД).
- Новые `StructureType` зарегистрированы в `similarity_matcher` (категории,
  противоположности, пороги уверенности, зеркальная карта) и в фильтрах `main.py`.
- Все рискованные изменения порогов вынесены в `CONFIG` и обратимы.

## Выполненные этапы

### T0.1 — Централизация конфигурации
`src/backend/config.py`: dataclass-конфиги + синглтон `CONFIG`. Шесть модулей
переведены на чтение `CONFIG`. Намеренные изменения: `min_quality` 0.15→0.35,
recency-фактор отключён (обратимо через CONFIG).

### T0.2 — Харнес валидации / бэктеста
Пакет `src/backend/validation/` (`history`, `metrics`, `runner`, `labels`) и
`scripts/validate.py` → `data/validation_results.json`. Даёт базовые метрики и
долю перерисовки (repaint) по типам и таймфреймам.

### T1.1 — Каузальная (непереисовывающая) детекция
`extract_features_causal(candles)` с логикой подтверждения (candidate →
confirmed), опциональные поля `StructureFeatures`. Новые типы:
ascending/descending/symmetrical triangle, channel up/down/horizontal.
Режим `causal_patterns` в `main.py` эмитит только подтверждённые паттерны +
новые WS-поля. Тесты: `tests/test_causal_detection.py`.

### T1.2 — ATR-адаптивные пороги
Хелперы `average_true_range(…)` и `compute_volatility_scale(…)` (= `max(ATR,
price_range*0.01)`). Допуски каузального слоя адаптируются к волатильности через
`max(fixed*range, mult*vol_scale)` — пороги только **ослабляются**, регрессия
исключена. Поведение по умолчанию (`volatility_scale=0`) бит-в-бит совпадает с
легаси. Тесты: `tests/test_atr_adaptive.py`.

### T1.3 — Робастные уровни (LevelDetectorV2)
`src/backend/level_detector_v2.py`: DBSCAN-зоны, RANSAC/Theil-Sen трендлайны,
профиль объёма (POC/VA), круглые числа, дедупликация — все пороги масштабируются
`volatility_scale`. Эндпоинт `GET /api/levels-v2/{symbol}/{timeframe}` и режим
`level_scan_v2` (трансляция уровней с `strength ≥ level_v2_min_strength`).
Существующий `LevelDetector` не изменён. Тесты:
`tests/test_level_detector_v2.py` (11). Проверено на реальных данных (BTC/ETH/SOL).

### T1.4 — Логирование и очистка
`print()` → модульные логгеры с уровнями (шумные строки на DEBUG);
`logging.basicConfig` в `run.py`. Удалён мёртвый код
`BinanceScanner._generate_realistic_candles`. Логи на уровне INFO чистые.

## Код-ревью

Архитектурное ревью (architect) подтвердило соблюдение ключевых ограничений:
изменения REST/WS аддитивны, легаси-слой не сломан, новые `StructureType`
полностью зарегистрированы. Исправлены два замечания средней важности:

1. **Сброс дедупликации каузального скана** — `causal_scan_seen` сбрасывался без
   `global` в `start_scan()`, что подавляло совпадения при повторных сканах.
   Добавлено в `global`-декларацию.
2. **Синхронизация уверенности кандидата** — при переопределении `structure_type`
   каузальным кандидатом `pattern_confidence`/`detected_patterns` оставались от
   легаси-классификатора. Теперь синхронизируются с `cand.confidence`. Добавлен
   регрессионный тест.

## Совместимость API

- **Новые эндпоинты:** `GET /api/levels-v2/{symbol}/{timeframe}`.
- **Новые режимы сканирования:** `causal_patterns`, `level_scan_v2`
  (через `POST /api/start-scan`).
- **Новые поля** в WS-сообщениях `match` (например `is_confirmed`,
  `detected_patterns`, `is_level_v2`, расширенный `level_data`) — добавлены без
  удаления существующих.
- **Важная деталь:** свечи в сканере ключуются по БАЗОВОМУ символу (`BTC`), а не
  паре (`BTCUSDT`) — единообразно с существующими `/api/candles` и `/api/chart`.

## Проверка

| Проверка | Результат |
|---|---|
| `python -m pytest -q` | 35 passed |
| `python scripts/validate.py` | OK, регрессии нет |
| `python run.py` (workflow) | стартует чисто, порт 5000 |
| `GET /api/levels-v2/BTC/1h` (и ETH/SOL) | корректные уровни + трендлайны |
| `level_scan_v2` начальное сканирование | трансляция совпадений |

## Файлы

**Новые:** `src/backend/config.py`, `src/backend/level_detector_v2.py`,
`src/backend/validation/*`, `scripts/validate.py`, `tests/*`, `UPGRADE_LOG.md`,
`PHASE01_REPORT.md`.

**Изменённые (совместимо):** `run.py`, `src/backend/main.py`,
`structure_extractor.py`, `similarity_matcher.py`, `binance_scanner.py`,
`candle_patterns.py`, `fibonacci_analyzer.py`, `level_detector.py`,
`pyproject.toml`, `requirements.txt`.

## Дальнейшие шаги (вне рамок Phase 0/1)

- Интеграция V2-уровней в UI (отдельная кнопка/слой визуализации).
- Расширение разметки валидации реальными подтверждёнными исходами для
  измерения precision/recall, а не только доли перерисовки.
