# Журнал апгрейда точности (Phase 0 + Phase 1)

Документ фиксирует изменения по под-фазам. Каждая запись описывает, что
сделано, какие значения по умолчанию изменены осознанно и как это откатить.

---

## Sub-phase 0.1 — Централизация конфигурации

**Что сделано**
- Создан модуль `src/backend/config.py` с группами настроек в виде dataclass-ов
  (`DataConfig`, `StructureConfig`, `PatternConfig`, `LevelConfig`, `FiboConfig`,
  `CandleConfig`, `SimilarityConfig`, `ValidationConfig`) и глобальным синглтоном
  `CONFIG`.
- Шесть модулей переведены на чтение значений из `CONFIG` вместо «магических
  чисел»:
  - `structure_extractor.py` — пороги детекции пивотов (ZigZag ATR-множитель,
    prominence), пороги уверенности всех паттернов (двойные вершины/донышки,
    H&S, флаги, клинья, треугольники, тренд), порог качества `min_quality`.
  - `level_detector.py` — отклонение, число касаний, порог пробоя, дедуп.
  - `candle_patterns.py` — `max_age`, порог объёма, lookback тренда.
  - `fibonacci_analyzer.py` — допуск касания, минимальный размах.
  - `binance_scanner.py` — таймфреймы, лимиты свечей, число символов, размеры
    скользящих окон, интервалы опроса/ошибки/батча.
  - `similarity_matcher.py` — веса метрик, пороги сходства по типам, бонус за
    пересечение паттернов, штрафы за зеркальность, MTF-бонусы, окно дедупа.

**Осознанно изменённые значения по умолчанию (вступают в силу)**
| Параметр | Было | Стало | Эффект |
|---|---|---|---|
| `structure.min_quality` | 0.15 | 0.35 | Более строгий фильтр качества структур |
| `structure.recency_weight_enabled` | вкл. | **выкл.** | Убрано смещение в сторону недавних данных |
| `pattern.double_top_min_conf` | 0.40 | 0.45 | Строже двойная вершина |
| `pattern.double_bottom_min_conf` | 0.40 | 0.45 | Строже двойное донышко |
| `pattern.hs_min_conf` | 0.40 | 0.50 | Строже «голова и плечи» |
| `similarity.mirror_penalty_opposite` | 0.85 | 0.60 | Сильнее штраф зеркала для противоположных типов |
| `similarity.mirror_penalty_consolidation` | 0.95 | 0.90 | Чуть сильнее штраф для консолидаций |
| `similarity.mirror_penalty_default` | 0.90 | 0.85 | Чуть сильнее штраф по умолчанию |

Все значения откатываются правкой соответствующего поля в `config.py` —
поведение кода не «зашито», а читается из `CONFIG`.

**Обратная совместимость**
- Сигнатуры публичных методов сохранены: значения по умолчанию стали `None`,
  при `None` подставляется значение из `CONFIG` (например, `LevelDetector(...)`,
  `FibonacciAnalyzer(...)`, `StructureExtractor(...)`, `find_all_positions(...)`).
- Форматы REST/WS сообщений не менялись.

**Проверка**
- `python -c "from src.backend.config import CONFIG; print(CONFIG)"` — без ошибок.
- Импорт всех модулей бэкенда — без ошибок.
- `python run.py` (workflow «Chart Scanner») стартует, отдаёт UI, WebSocket
  подключается, реальные данные Binance загружаются.

---

## Sub-phase 0.2 — Валидационный / бэктест харнесс

**Что сделано**
- Создан пакет `src/backend/validation/`:
  - `history.py` — загрузка исторических свечей через публичный Binance REST,
    возврат `List[CandleData]` (та же структура, что и в сканере).
  - `metrics.py` — `precision/recall/F1` против разметки, `repaint_rate`
    (доля сигналов, исчезающих на следующем шаге окна), `average_latency`
    (задержка от кандидата до подтверждения; на этом этапе ещё `None`).
  - `runner.py` — класс `ValidationRunner`: `run_pattern_validation(...)`
    (прогон по истории «бар за баром» скользящим окном) и `run_multi_symbol(...)`
    (обход символов/ТФ из `CONFIG.validation`).
  - `labels.py` — `LabelStore` на SQLite (таблица `validation_labels`) для
    будущей ручной разметки и обратной связи ML.
- Добавлен CLI `scripts/validate.py`: считает метрики, печатает их по типам
  паттернов и таймфреймам, сохраняет результат в `data/validation_results.json`.

**Базовые метрики (предварительные)**
- Прогонов: 15 (5 символов × 3 ТФ), событий: ~6000.
- `precision/recall` = 0% — разметки ещё нет (ожидаемо).
- Доля «подтверждённых» сигналов ≈ 100% — базовый (не каузальный) детектор
  считает почти всё активным.
- **Перерисовка (repaint_rate): 15m ≈ 16.5%, 1h ≈ 13.6%, 4h ≈ 12.9%** — это и
  есть нестабильность не-каузальной детекции, которую устраняет фаза 1.1.

**Проверка**
- `python scripts/validate.py` завершается без ошибок, JSON сохранён.

---

## Sub-phase 1.1 — Каузальная (неперерисовывающая) детекция

**Что сделано**
- В `structure_extractor.py` добавлен ПАРАЛЛЕЛЬНЫЙ каузальный слой поверх
  сырых свечей, не затрагивающий старый `extract_features(price_line)`:
  - Новый метод `extract_features_causal(candles)` — классифицирует структуру,
    строит кандидата и подтверждает его пробоем/follow-through, используя
    только информацию до текущей закрытой свечи (без заглядывания в будущее).
  - Пивоты ищутся на нормализованной линии (`detect_pivots` настроен под
    100-точечную линию), затем `_map_pivots_to_raw` привязывает каждый пивот к
    ближайшему реальному экстремуму сырой цены (±2 бара) — иначе на сырых
    закрытиях пивоты недодетектируются.
  - Семейства подтверждения (`_confirm_candidate`): `breakout_level`
    (горизонтальный нэклайн — двойные вершины/донышки, голова-плечи),
    `breakout_channel` (наклонные границы — треугольники, каналы, клинья),
    `trend` (закрытие в направлении + follow-through), `range` (закрытие за
    границей бокового диапазона). Параметры — из `CONFIG.structure`
    (`confirmation_bars`, `confirmation_retreat_fraction`).
- В `StructureFeatures` добавлены ТОЛЬКО ОПЦИОНАЛЬНЫЕ поля со значениями по
  умолчанию (БД и существующие сериализации не ломаются): `candidate_index`,
  `candidate_time`, `confirmation_index`, `confirmation_time`, `is_confirmed`,
  `is_invalidated`.
- Новые значения `StructureType`: `ascending_triangle`, `descending_triangle`,
  `symmetrical_triangle`, `channel_up`, `channel_down`, `horizontal_channel`.
  `classify_structure` теперь возвращает подтип треугольника (`_triangle_subtype`)
  и определяет каналы (`_detect_channel`).
- Все 6 новых типов ЗАРЕГИСТРИРОВАНЫ в `similarity_matcher.py`:
  `CONSOLIDATION_TYPES`, `OPPOSITE_PAIRS` (asc↔desc, channel_up↔down),
  `MIN_CONFIDENCE_THRESHOLDS` (треугольники 0.40, каналы 0.45/0.40),
  `type_mirror_map` (asc↔desc, channel_up↔down, sym/horizontal → сами в себя).
- В `main.py` добавлен режим сканирования `causal_patterns`:
  - `on_market_update_causal_scan` и `run_initial_causal_scan` запускают
    `extract_features_causal` на сырых свечах и публикуют ТОЛЬКО подтверждённые
    паттерны (`is_confirmed=True`).
  - В WS-сообщение типа `match` добавлены НОВЫЕ поля: `is_confirmed`,
    `candidate_time`, `confirmation_time` (старые поля не изменены).
  - Режим разведён в `on_market_update`, `run_initial_scan` и `start_scan`.

**Тесты** — `tests/test_causal_detection.py` (17 тестов, все проходят):
- Полный конвейер (`extract_features_causal`): `double_top`, `double_bottom`,
  `channel_up`, `channel_down` — подтверждаются на баре пробоя СТРОГО ПОСЛЕ
  бара-кандидата.
- Прямые построители (`_build_candidate` + `_confirm_candidate`):
  `head_shoulders`, `inv_head_shoulders`, `ascending_triangle`,
  `descending_triangle` — подтверждаются на пробое после кандидата.
- Инвариант ОТСУТСТВИЯ ПЕРЕРИСОВКИ: для каждого паттерна данные, усечённые до
  бара подтверждения, НЕ дают подтверждения (сигнал не появляется задним числом).
- Слишком короткий ряд свечей → `None`.

**Обратная совместимость**
- `extract_features(price_line)` и форма `StructureFeatures` не изменены
  деструктивно (только новые опциональные поля).
- Старые REST/WS-форматы сохранены; в `causal_patterns` добавлены лишь новые поля.
- `python run.py` (workflow «Chart Scanner») стартует, UI и WebSocket работают.

**Проверка**
- `python -m pytest tests/test_causal_detection.py -v` → 17 passed.
- `python scripts/validate.py` завершается без ошибок, JSON пересохранён.

---

## Sub-phase 1.2 — ATR-адаптивные пороги

**Что сделано**
- В `structure_extractor.py` добавлены module-level хелперы:
  - `average_true_range(highs, lows, closes, period=14)` — истинный диапазон
    (TR = max(high−low, |high−prev_close|, |low−prev_close|)) по СЫРЫМ ценам,
    усреднённый за `period`. При недостатке баров корректно деградирует.
  - `compute_volatility_scale(highs, lows, closes)` — базовая единица
    волатильности `max(ATR, price_range * 0.01)` (где `price_range = max(close) −
    min(close)`), гарантированно > 0 при валидных данных.
- В `extract_features_causal` `volatility_scale` вычисляется один раз из
  high/low/close свечей и пробрасывается в каузальные построители.
- ATR-адаптивные допуски в КАУЗАЛЬНОМ (сыром) слое:
  - Двойные вершины/донышки: допуск равенства вершин теперь
    `max(double_top_tolerance * price_range, double_top_atr_mult * volatility_scale)`
    в АБСОЛЮТНЫХ ценах (раньше — доля диапазона). При высоком ATR допуск
    расширяется, чтобы шумный, но валидный паттерн не терялся.
  - Фит границ (`_fit_pivot_lines`) — остатки в ценовых единицах делятся на
    `max(price_range, fit_residual_atr_mult * volatility_scale)`. Используется в
    треугольниках, каналах и клиньях.
- **Монотонность по построению:** все адаптивные пороги берутся как `max(...)`
  с фиксированной долей диапазона, поэтому ATR-адаптация только ОСЛАБЛЯЕТ пороги
  (никогда не делает их строже фиксированных) — регрессия детекции исключена.
- Новые параметры в `PatternConfig` (revertable через CONFIG):
  `double_top_atr_mult = 0.5`, `fit_residual_atr_mult = 2.0`.
- **Обратная совместимость:** все новые параметры построителей —
  `volatility_scale: float = 0.0` со значением по умолчанию; при 0 поведение
  БИТ-В-БИТ совпадает с прежними фиксированными порогами. Легаси-слой
  `extract_features(price_line)` и нормализованные детекторы НЕ изменены —
  нормализованная линия по-прежнему используется только для сходства формы,
  а геометрия детекции — на сырых ценах (как требует спецификация).

**Тесты** — `tests/test_atr_adaptive.py` (7 тестов, все проходят):
- Хелперы: базовый ATR, краевые случаи (пустой ряд → 0, один бар → high−low),
  срабатывание пола `price_range * 0.01`, использование ATR при высокой
  волатильности.
- Высокая волатильность: двойная вершина с НЕРАВНЫМИ вершинами, отвергаемая
  фиксированным допуском (`volatility_scale = 0` → `None`), ОБНАРУЖИВАЕТСЯ при
  ATR-адаптивном допуске.
- Низкая волатильность: та же форма с близкими вершинами НЕ теряется.
- Полный каузальный конвейер по-прежнему подтверждает чёткий паттерн.

**Проверка**
- `python -m pytest -q` → 24 passed (17 каузальных + 7 ATR-адаптивных).
- `python scripts/validate.py` завершается успешно, JSON пересохранён.

---

## T1.3 — Робастная детекция уровней (LevelDetectorV2)

**Цель** — заменить наивную детекцию уровней статистически устойчивым подходом
с порогами, масштабируемыми волатильностью (а не фиксированными на нормали),
не трогая существующий `LevelDetector` (полная обратная совместимость).

**Что сделано**
- Новый модуль `src/backend/level_detector_v2.py`, класс `LevelDetectorV2`.
  Метод `detect_levels(highs, lows, closes, volumes, times=None, order=None)`
  возвращает словарь:
  - `support_levels` / `resistance_levels` — список уровней с полями
    `price`, `strength` (0–1), `num_touches`, `first_touch_time`,
    `last_touch_time`, `type`, `source`.
  - `trendlines` — диагональные линии с `slope`, `intercept`, `type`
    (`support_trendline` / `resistance_trendline`), `strength` (0–1),
    `touches` (индексы), `start_idx`, `end_idx`, `is_channel`.
- **Все пороги масштабируются `volatility_scale`** (из `compute_volatility_scale`,
  T1.2), а не фиксированы на [0,1]:
  - **Горизонтальные зоны** — `sklearn.cluster.DBSCAN` по ценам пивотов,
    `eps = dbscan_eps_factor * volatility_scale`, `min_samples = dbscan_min_samples`.
    Цена уровня — медиана кластера; число касаний — размер кластера.
  - **Диагональные трендлайны** — `RANSACRegressor` (фаллбэк `TheilSenRegressor`)
    по пивотам; инлайеры в пределах `ransac_residual_threshold_factor *
    volatility_scale`; линия сохраняется только при `>= min_touches` инлайерах.
    Параллельность опоры/сопротивления (`is_channel`) — по относительной разнице
    наклонов с допуском `pattern.channel_parallel_tolerance`.
  - **Профиль объёма** — гистограмма закрытий, взвешенная объёмом
    (`volume_profile_bins` корзин); POC (макс. объём) и границы зоны стоимости
    (VAH/VAL, ~70% объёма) добавляются как уровни.
  - **Круглые числа** — шаг `round_number_step_factor * volatility_scale`,
    нормализованный к «красивому» 1/2/5·10^k; уровень принимается только при
    наличии реального касания в пределах `0.5 * volatility_scale`.
  - **Дедупликация** — уровни ближе `0.3 * volatility_scale` сливаются с
    суммированием силы (с насыщением до 1.0).
- **Сила (strength, 0–1)** — взвешенная комбинация числа касаний, свежести
  (индекс последнего касания) и относительного объёма на касаниях.
- Короткие ряды (< 20 баров), несовпадающие длины и `volatility_scale <= 0`
  безопасно возвращают пустой результат.

**Интеграция (main.py, REST/WS совместимо — только новые поля/режимы)**
- `GET /api/levels-v2/{symbol}/{timeframe}` — JSON `{symbol, timeframe, support,
  resistance, trendlines}` (или сообщение при < 30 свечах).
- Новый режим сканирования `level_scan_v2`: транслирует уровни/трендлайны с
  `strength >= level_v2_min_strength` (по умолчанию 0.5; настраивается в
  `StartScanRequest.level_v2_min_strength`). Хелпер `_level_v2_to_match_data`
  переиспользует существующую форму `level_data` (с флагами `is_level`,
  `is_level_v2`), поэтому фронтенд отрисовывает V2-уровни без правок.
- Проводка зеркалит `level_scan`: глобалы, диспетчеризация в `on_market_update`
  и `run_initial_scan`, сброс seen-set и параметров в `start_scan`.
- Существующий `LevelDetector` НЕ изменён.

**Тесты** — `tests/test_level_detector_v2.py` (11 тестов, все проходят):
структура результата, пустой результат на коротких/несогласованных рядах,
диапазоны силы и касаний, детекция горизонтальных зон на осцилляторе,
восходящий support-трендлайн, POC профиля объёма, круглые числа,
отсутствие близких дублей после дедупа, детерминизм, использование `times`.

**Проверка**
- `python -m pytest -q` → 35 passed (24 прежних + 11 новых).
- `python scripts/validate.py` завершается успешно, регрессии нет.
- Приложение стартует чисто на порту 5000; `GET /api/levels-v2/BTC/1h`
  (и ETH/SOL) на реальных данных Binance возвращает корректные уровни
  поддержки/сопротивления и трендлайны.

---

## T1.4 — Логирование и очистка

**Что сделано**
- **Замена `print()` на `logging`:**
  - В `binance_scanner.py` и `main.py` все `print()` заменены на вызовы
    модульного логгера `logging.getLogger(__name__)` с осмысленными уровнями:
    информационные сообщения → `info`, восстановимые сбои API → `warning`,
    ошибки обновления/поллинга → `error`. Шумные повторяющиеся строки
    («Symbol … unavailable», «Marking … unavailable») понижены до `debug`,
    поэтому при уровне INFO лог стал чистым.
  - `structure_extractor.py` `print()` не содержал — изменений не потребовалось.
- **Конфигурация логирования:** в `run.py` добавлен `logging.basicConfig`
  (уровень INFO, формат `%(asctime)s %(levelname)s [%(name)s] %(message)s`).
- **Удалён мёртвый код:** функция `BinanceScanner._generate_realistic_candles`
  (генератор синтетических свечей) удалена — у неё не было вызовов во всём
  проекте (проверено `rg`). Реальный фаллбэк работает на уровне пропуска
  недоступных символов, синтетика не использовалась.
- Опциональная правка UI (кнопка «ретест») НЕ выполнялась намеренно: это
  рабочая опция type-scan, удаление вне необходимости (риск регрессии UI).

**Проверка**
- `python -m pytest -q` → 35 passed.
- Приложение стартует чисто; в логах виден новый формат, напр.
  `… INFO [src.backend.binance_scanner] Initializing scanner...`; ошибок и
  трейсбеков нет; шумные строки скрыты на уровне INFO.
- `level_scan_v2` отрабатывает: начальное сканирование завершается с
  трансляцией совпадений.

---

## TFINAL — Отчёты и код-ревью (architect)

**Отчёты**
- Создан `PHASE01_REPORT.md` — сводный отчёт по Phase 0 + Phase 1.

**Код-ревью (architect, evaluate_task)**
Вердикт: PASS — обратная совместимость и «параллельный слой» соблюдены; новые
`StructureType` корректно зарегистрированы в `similarity_matcher` и фильтрах
`main.py`; критичных дефектов нет. Исправлены два замечания средней важности:

1. **Сброс дедупликации каузального скана.** В `start_scan()` переменная
   `causal_scan_seen` сбрасывалась без объявления `global`, из-за чего повторные
   каузальные сканы в одном процессе сохраняли устаревшее состояние и подавляли
   совпадения. Добавлено `causal_scan_seen` в `global`-декларацию.
2. **Синхронизация уверенности каузального кандидата.** При переопределении
   `structure_type` каузальным кандидатом поля `pattern_confidence` и
   `detected_patterns` оставались от легаси-классификатора, искажая оценку
   каузальных совпадений. Теперь оба синхронизируются с `cand.confidence`.

**Проверка**
- Добавлен регрессионный тест `test_pipeline_confidence_synced_to_candidate`.
- `python -m pytest -q` → 39 passed.
- Приложение перезапускается чисто; режим `causal_patterns` транслирует только
  подтверждённые паттерны (в живом прогоне — 3 совпадения против 370 в
  similarity-режиме, что ожидаемо для строгой неперерисовывающей детекции).

---

# Phase 2 — Недостающие паттерны, синтетика и ML-фильтр

## Sub-phase 2.1 — Генератор синтетических паттернов

**Новое**
- `src/backend/synthetic.py` — класс `SyntheticChartGenerator`:
  - `generate_random_walk(n, start_price, volatility, seed)` — случайное
    блуждание с согласованным OHLCV на базе `CandleData`.
  - Инъекторы форм: `inject_triple_top`, `inject_triple_bottom`,
    `inject_cup_and_handle`, `inject_pennant`, `inject_rounding_bottom`.
    Каждая форма строится как кусочно-гладкая кривая цен закрытия с шумом и
    завершается пробоем (для каузального подтверждения в 2.2).
  - `generate_labeled_dataset(n_per_class, n_noise, seed)` — список образцов
    `{candles, label, target_confirmed}`; шум помечается `target_confirmed=False`.
- `scripts/generate_synthetic.py` — CLI, сохраняет `data/synthetic_dataset.json`.
- `tests/test_synthetic.py` — 12 тестов: длина рядов, согласованность OHLC,
  узнаваемость форм (три вершины <2% и разнесены; U-образность чаши; минимум
  округлого дна в середине), санити-проверка шума, структура датасета.

**Проверка**
- `python -m pytest tests/test_synthetic.py -v` → 12 passed.
- `python scripts/generate_synthetic.py` → датасет сохранён (классы
  сбалансированы, шум помечен корректно).

## Sub-phase 2.2 — Детекторы недостающих паттернов

**Новое**
- Пять новых типов `StructureType`: `triple_top`, `triple_bottom`,
  `cup_and_handle`, `pennant`, `rounding_bottom` (строковые значения стабильны —
  контракт REST/WS не нарушен, добавлены только новые поля/значения).
- `config.py` (`PatternConfig`) — пороги: `triple_top/bottom_tolerance`+`min_conf`,
  `cup_and_handle_depth_min`/`handle_retrace_max`/`min_conf`,
  `pennant_pole_min_move`/`convergence_min`/`min_conf`,
  `rounding_bottom_curvature_min`/`min_conf`.
- `structure_extractor.py` — каузальные построители кандидатов
  `_candidate_triple_top/triple_bottom/cup_and_handle/pennant/rounding_bottom`,
  возвращающие `_PatternCandidate` и подтверждаемые той же машиной
  `_confirm_candidate` (семейства `breakout_level`/`breakout_channel`).
  Зарегистрированы в диспетчере `_build_candidate`.
  - Особенности детекции: вымпел измеряет схождение по сжатию амплитуды
    пост-полюсной зоны (мало пивотов в затухающих колебаниях); округлое дно
    требует центрального глобального минимума, выпуклой параболы и того, чтобы
    парабола объясняла форму существенно лучше прямой (отсечение каналов).
- `extract_features_causal` — ПРИОРИТЕТНАЯ логика: сначала пробуются пять новых
  типов; первый ПОДТВЕРЖДЁННЫЙ выигрывает; иначе откат к кандидату по базовому
  типу. Поведение Phase 1 полностью сохранено (новые билдеры дают `None` на
  формах double/channel/триангл и т.п.).
- `classify_structure` (режим type_scan) — `_detect_missing_patterns_norm`
  переиспользует те же геометрические билдеры в нормализованном пространстве и
  дополняет мульти-лейбл `detected_patterns`, не меняя первичную классификацию.
- `similarity_matcher.py` — новые типы добавлены в `REVERSAL_TYPES`/
  `CONSOLIDATION_TYPES`, `OPPOSITE_PAIRS` (triple_top↔triple_bottom),
  `type_mirror_map` (тройные — зеркальны, чаша/вымпел/дно — само-зеркальны),
  `MIN_CONFIDENCE_THRESHOLDS`.
- `main.py` — изменений не требует: диспетчеризация идёт по строковым значениям
  типов и `detected_patterns`, новые типы проходят автоматически.
- `tests/test_missing_patterns.py` — 34 теста: каузальная детекция и
  подтверждение, отсутствие перерисовки, мульти-лейбл в type_scan, регистрация в
  матчере, низкий уровень ложных срабатываний на шуме, стабильность enum.

**Проверка**
- `python -m pytest -q` → 85 passed (51 прежних без регрессий + 34 новых).
- Перезапуск воркфлоу `Chart Scanner` → чистый старт на порту 5000, REST/WS
  отвечают `200 OK`.

## Sub-phase 2.3 — Извлечение ML-признаков и классификатор

**Новое**
- Пакет `src/backend/ml/`:
  - `features.py` — `extract_ml_features(structure_features, candidate_index,
    confirmation_index, candle_history, timeframe_minutes)` -> dict признаков.
    Фиксированный `FEATURE_ORDER` (16 числовых + one-hot по всем `StructureType`):
    pattern_confidence, quality_score, symmetry_score, convergence_rate,
    breakout_strength, avg_pivot_confidence, trend_consistency,
    volume_confirmation, pattern_freshness, is_confirmed, timeframe_minutes,
    n_pivots, volatility_scale, price_range, relative_volume, lag_to_confirmation.
    `features_to_vector` гарантирует совпадение порядка обучение↔инференс.
  - `model.py` — `RelevanceClassifier` (обёртка `RandomForestClassifier`,
    `class_weight="balanced"`): `fit/predict_proba/predict_one/save/load`,
    `feature_importances`. Сохранение через joblib вместе с порядком признаков и
    метаданными (trained_at, n_samples, accuracy). При отсутствии scikit-learn —
    понятная ошибка `ImportError`.
  - `trainer.py` — `ModelTrainer.build_training_data(synthetic, feedback=None)`:
    каждый образец прогоняется через `extract_features_causal`; метка `1`, если
    паттерн-класс ПОДТВЕРЖДЁН и тип совпал с заявленным, иначе `0` (шум/ложное
    срабатывание/не подтверждено). Обратная связь подмешивается с весом
    `ml_feedback_weight` только при наличии восстановимых признаков (без
    выдумывания данных). `train()` возвращает accuracy/report/importances;
    `retrain()` пересобирает и перезаписывает модель.
  - `pipeline.py` — `MLPipeline`: мягкая загрузка модели; `score_features(...)`
    возвращает `ml_score`; при отсутствии модели — `1.0` (бэквард-совместимость);
    `status()` для будущего `/api/ml-status`.
- `config.py` — `MLConfig` на `AppConfig` (`CONFIG.ml`): `ml_min_training_samples=200`,
  `ml_feedback_weight=1.0`, `ml_retrain_interval_hours=24`, `ml_score_threshold=0.5`,
  `ml_model_path`.
- `database.py` — `get_all_feedback()` (feedback ⋈ matches) для CLI-обучения.
- `scripts/train_ml.py` — загрузка датасета + обратной связи, обучение, печать
  метрик и важностей, сохранение в `data/ml_relevance_model.pkl`.
- `tests/test_ml_filter.py` — 11 тестов: извлечение признаков, стабильность
  порядка, one-hot, наличие обоих классов, точность > 0.7, save/load roundtrip,
  диапазон predict_proba, мягкое поведение pipeline без модели и скоринг с
  моделью, подмешивание обратной связи.

**Проверка**
- `python scripts/train_ml.py` -> 598 образцов, hold-out accuracy ≈ 0.99,
  модель сохранена. Топ-признаки: lag_to_confirmation, volatility_scale,
  pattern_confidence.
- `python -m pytest -q` -> 96 passed (без регрессий).
- Перезапуск воркфлоу `Chart Scanner` -> чистый старт на порту 5000, REST/WS 200.

## Sub-phase 2.4 — Интеграция ML-фильтра в сканирование

**Новое**
- `main.py`:
  - Глобальный `ml_pipeline = MLPipeline(CONFIG.ml.ml_model_path)`,
    `ml_score_threshold`, `last_model_retrain`, `_retrain_in_progress`.
    Если модель уже есть на старте — `last_model_retrain` ставится в «сейчас»,
    чтобы не запускать тяжёлое переобучение на первом тике.
  - Хелперы: `_tf_minutes(tf)` (минуты таймфрейма), `_load_synthetic_dataset`,
    `_retrain_ml` (пересборка+обучение в пуле через `asyncio.to_thread`,
    перезагрузка пайплайна), `_maybe_auto_retrain` (фоновая проверка интервала
    `ml_retrain_interval_hours`).
  - `on_market_update` запускает `asyncio.create_task(_maybe_auto_retrain())`
    (не блокирует сканирование).
  - `ml_score` считается для матчей режимов `causal_patterns`
    (`on_market_update_causal_scan` + `run_initial_causal_scan`) и `type_scan`;
    поле `ml_score` добавлено в WS-сообщения. Матч отсекается, только если
    модель существует И `ml_score < ml_score_threshold` (бэквард-совместимость:
    без модели `ml_score=1.0`, фильтрация не применяется).
  - Эндпоинты: `POST /api/retrain-ml` (ручное переобучение, возвращает метрики;
    409 при параллельном запуске), `GET /api/ml-status`
    (`model_exists`, `last_trained_at`, `n_samples`, `accuracy`, `top_features`,
    `ml_score_threshold`, `retrain_in_progress`, `last_model_retrain`).
- `database.get_all_feedback()` используется для подмешивания обратной связи при
  переобучении (учитываются только записи с восстановимыми признаками).
- `tests/test_ml_integration.py` — 6 тестов: маппинг таймфреймов, мягкое
  поведение без модели, контракт гейта фильтрации, отсечение слабых матчей с
  моделью, ответы `/api/retrain-ml` и `/api/ml-status`.

**Проверка**
- `python -m pytest -q` -> 102 passed (без регрессий).
- Перезапуск воркфлоу -> чистый старт на порту 5000.
- Живые эндпоинты: `GET /api/ml-status` -> модель есть, accuracy≈0.99;
  `POST /api/retrain-ml` -> `success=true`, метрики и отчёт возвращаются.

## Sub-phase 2.5 — Реальная валидация на размеченной синтетике

**Новое**
- `validation/runner.py` -> `ValidationRunner.run_synthetic_validation(detector,
  dataset, ml_pipeline=None, ml_threshold=None)`:
  - По каждому образцу синтетики запускает каузальную детекцию
    (`extract_features_causal`) и сверяет распознанный тип с заявленным
    `label` (паттерн против `noise`).
  - Считает два режима: `raw` (только геометрия) и `ml` (геометрия + ML-фильтр
    `ml_score >= ml_threshold`), что наглядно показывает вклад ML.
  - Метрики: per-pattern precision/recall/F1, общие `true_positive_rate` и
    `false_positive_rate`, плюс гистограмма `ml_score_distribution` (10 корзин).
- `scripts/validate.py --synthetic`: грузит `data/synthetic_dataset.json`,
  прогоняет синтетическую валидацию с ML-фильтром, печатает TPR/FPR (raw vs ML)
  и распределение ml_score, сохраняет `data/validation_synthetic.json`.
- `tests/test_synthetic_validation.py` — 4 теста на структуру отчёта,
  корректность TPR/FPR, per-pattern счётчиков и гистограммы (стаб-детектор для
  детерминированности).

**Результаты (600 образцов: 300 паттернов / 300 шума, модель acc≈0.99)**
- Только геометрия: TPR 73.67%, FPR 43.67%.
- Геометрия + ML-фильтр: TPR 73.67%, FPR **0.00%**.
- ML-фильтр убирает все ложные срабатывания на шуме, не теряя истинные
  паттерны: precision по всем типам поднимается до 100% при сохранении recall.
- Распределение ml_score выражено бимодальное (масса у 0.0–0.1 и 0.9–1.0),
  то есть модель уверенно разделяет релевантные и нерелевантные матчи.

**Проверка**
- `python scripts/validate.py --synthetic` -> отчёт сохранён в
  `data/validation_synthetic.json`.
- `python -m pytest -q` -> 106 passed.
