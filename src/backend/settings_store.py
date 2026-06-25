"""Хранилище пользовательских настроек UI (Phase 3.2).

Назначение
----------
Позволяет менять часть параметров ``CONFIG`` из интерфейса в рантайме и
сохранять их между перезапусками в ``data/ui_settings.json``.

Безопасность контракта: наружу выставляется только явный «белый список»
(``SETTINGS_SCHEMA``). Любые другие поля игнорируются, значения валидируются
по типу и диапазону. Это исключает порчу конфигурации произвольными ключами.

Каждый ключ имеет вид ``"<группа>.<атрибут>"`` и указывает на поле одного из
dataclass-ов внутри ``CONFIG`` (``data``, ``structure``, ``pattern`` и т.д.).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import CONFIG

logger = logging.getLogger(__name__)

SETTINGS_PATH = "data/ui_settings.json"


@dataclass
class SettingSpec:
    """Описание одного настраиваемого параметра."""

    key: str                # "data.use_websocket"
    group: str              # "data"
    attr: str               # "use_websocket"
    type: str               # "bool" | "int" | "float"
    label: str              # человекочитаемая подпись (RU)
    category: str           # группа в UI
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    help: str = ""


# Белый список настраиваемых параметров. Порядок = порядок отображения в UI.
SETTINGS_SCHEMA: List[SettingSpec] = [
    # --- Поток данных и производительность ---
    SettingSpec(
        "data.use_websocket", "data", "use_websocket", "bool",
        "Реальный поток (WebSocket)", "Данные и производительность",
        help="Вкл — live-данные Binance по WebSocket, выкл — REST-поллинг.",
    ),
    SettingSpec(
        "data.poll_interval_sec", "data", "poll_interval_sec", "float",
        "Интервал REST-поллинга (сек)", "Данные и производительность",
        minimum=10.0, maximum=300.0, step=5.0,
        help="Период опроса при выключенном WebSocket.",
    ),
    SettingSpec(
        "data.max_concurrent_symbol_tasks", "data", "max_concurrent_symbol_tasks", "int",
        "Параллельных задач по символам", "Данные и производительность",
        minimum=1, maximum=50, step=1,
        help="Сколько символов обрабатывается одновременно при сканировании.",
    ),
    SettingSpec(
        "data.cache_structures", "data", "cache_structures", "bool",
        "Кэшировать структуры", "Данные и производительность",
        help="Не пересчитывать структуру, если свечи не изменились.",
    ),
    # --- Детекция структуры ---
    SettingSpec(
        "structure.min_quality", "structure", "min_quality", "float",
        "Мин. качество структуры", "Детекция",
        minimum=0.0, maximum=1.0, step=0.05,
        help="Структуры ниже порога качества отбрасываются.",
    ),
    # --- Пороги паттернов ---
    SettingSpec(
        "pattern.double_top_tolerance", "pattern", "double_top_tolerance", "float",
        "Допуск двойной вершины", "Паттерны",
        minimum=0.005, maximum=0.1, step=0.005,
    ),
    SettingSpec(
        "pattern.triple_top_tolerance", "pattern", "triple_top_tolerance", "float",
        "Допуск тройной вершины", "Паттерны",
        minimum=0.005, maximum=0.1, step=0.005,
    ),
    SettingSpec(
        "pattern.cup_and_handle_depth_min", "pattern", "cup_and_handle_depth_min", "float",
        "Мин. глубина чаши", "Паттерны",
        minimum=0.02, maximum=0.3, step=0.01,
    ),
    SettingSpec(
        "pattern.cup_and_handle_handle_retrace_max", "pattern",
        "cup_and_handle_handle_retrace_max", "float",
        "Макс. откат ручки", "Паттерны",
        minimum=0.1, maximum=0.9, step=0.05,
    ),
    # --- Свечные паттерны ---
    SettingSpec(
        "candle.max_age", "candle", "max_age", "int",
        "Макс. возраст свечного паттерна", "Свечные паттерны",
        minimum=1, maximum=50, step=1,
        help="Сколько свечей назад считается «свежим» паттерн.",
    ),
    # --- ML-фильтр ---
    SettingSpec(
        "ml.ml_score_threshold", "ml", "ml_score_threshold", "float",
        "Порог ML-релевантности", "ML-фильтр",
        minimum=0.0, maximum=1.0, step=0.05,
        help="Матчи с ml_score ниже порога скрываются.",
    ),
]

_SPEC_BY_KEY: Dict[str, SettingSpec] = {s.key: s for s in SETTINGS_SCHEMA}


def _coerce(spec: SettingSpec, value: Any) -> Any:
    """Приводит значение к типу спеки и валидирует диапазон. Бросает ValueError."""
    if spec.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        raise ValueError(f"{spec.key}: ожидался bool")

    if spec.type == "int":
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{spec.key}: ожидалось целое число")
    elif spec.type == "float":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{spec.key}: ожидалось число")
    else:
        raise ValueError(f"{spec.key}: неизвестный тип {spec.type}")

    if spec.minimum is not None and v < spec.minimum:
        raise ValueError(f"{spec.key}: значение {v} меньше минимума {spec.minimum}")
    if spec.maximum is not None and v > spec.maximum:
        raise ValueError(f"{spec.key}: значение {v} больше максимума {spec.maximum}")
    return v


def get_current_settings() -> Dict[str, Any]:
    """Текущие значения всех настраиваемых параметров (читает из CONFIG)."""
    result: Dict[str, Any] = {}
    for spec in SETTINGS_SCHEMA:
        group = getattr(CONFIG, spec.group)
        result[spec.key] = getattr(group, spec.attr)
    return result


def get_schema() -> List[Dict[str, Any]]:
    """Схема настроек для построения UI (метаданные + текущие значения)."""
    current = get_current_settings()
    out: List[Dict[str, Any]] = []
    for spec in SETTINGS_SCHEMA:
        out.append({
            "key": spec.key,
            "label": spec.label,
            "category": spec.category,
            "type": spec.type,
            "min": spec.minimum,
            "max": spec.maximum,
            "step": spec.step,
            "help": spec.help,
            "value": current[spec.key],
        })
    return out


def apply_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Валидирует и применяет настройки к CONFIG. Возвращает применённые значения.

    Неизвестные ключи игнорируются. При ошибке валидации любого поля —
    ValueError, при этом CONFIG не изменяется (атомарно: сперва валидируем всё).
    """
    validated: Dict[str, Any] = {}
    for key, value in updates.items():
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue  # неизвестный ключ — молча игнорируем
        validated[key] = _coerce(spec, value)

    for key, value in validated.items():
        spec = _SPEC_BY_KEY[key]
        group = getattr(CONFIG, spec.group)
        setattr(group, spec.attr, value)

    return validated


def save_settings() -> None:
    """Сохраняет текущие значения настроек в ``data/ui_settings.json``."""
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(get_current_settings(), f, ensure_ascii=False, indent=2)


def load_settings() -> Dict[str, Any]:
    """Загружает настройки из файла и применяет их к CONFIG (при старте)."""
    if not os.path.isfile(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Не удалось прочитать %s: %s", SETTINGS_PATH, e)
        return {}
    try:
        return apply_settings(data)
    except ValueError as e:
        logger.warning("Некорректные сохранённые настройки, игнорирую: %s", e)
        return {}
