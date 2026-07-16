import asyncio
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

from .image_processor import ImageProcessor
from .structure_extractor import StructureExtractor, StructureFeatures
from .similarity_matcher import SimilarityMatcher, MatchResult
from .binance_scanner import BinanceScanner
from .database import Database
from .level_detector import LevelDetector
from .level_detector_v2 import LevelDetectorV2
from .config import CONFIG
from . import settings_store
from .ml.pipeline import MLPipeline
from .ml.trainer import ModelTrainer
from .ml.features import extract_ml_features
from .synthetic import SyntheticChartGenerator


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    await database.initialize()
    # Загружаем сохранённые пользовательские настройки (Phase 3.2) и
    # синхронизируем зависящий от них глобальный порог ML.
    global ml_score_threshold
    settings_store.load_settings()
    ml_score_threshold = CONFIG.ml.ml_score_threshold
    yield
    # --- shutdown ---
    if is_scanning:
        await stop_scan()


app = FastAPI(title="Chart Structure Scanner", lifespan=lifespan)

# CORS: wildcard-происхождение ("*") несовместимо с allow_credentials=True в
# браузерах и небезопасно. По умолчанию — открытый доступ без credentials.
# Для продакшена задайте переменную окружения ALLOWED_ORIGINS (список доменов
# через запятую) — тогда включается режим с credentials.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _allowed_origins_env:
    _cors_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    # Wildcard "*" несовместим с credentials в браузерах — если оператор задал
    # его явно, отключаем credentials, чтобы не воссоздать исходный дефект.
    _cors_allow_credentials = "*" not in _cors_origins
    if not _cors_allow_credentials:
        logger.warning(
            "ALLOWED_ORIGINS содержит '*': credentials отключены (несовместимо с браузерами)."
        )
else:
    _cors_origins = ["*"]
    _cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_processor = ImageProcessor()
structure_extractor = StructureExtractor()
similarity_matcher = SimilarityMatcher()
scanner = BinanceScanner(num_symbols=50)
database = Database()
level_detector = LevelDetector()
level_detector_v2 = LevelDetectorV2()

active_websockets: List[WebSocket] = []
current_reference: Optional[StructureFeatures] = None
current_structure_id: Optional[int] = None
scan_threshold: float = 50.0
is_scanning: bool = False
scan_task: Optional[asyncio.Task] = None
_init_task: Optional[asyncio.Task] = None
_pending_initial_scan: Optional[asyncio.Task] = None
search_mode: str = "preset"  # "preset", "uploaded", "manual", "type_scan", "level_scan", "level_scan_v2", or "causal_patterns"
search_pattern_type: Optional[str] = None
search_type_filter: Optional[str] = None
search_timeframe_filter: Optional[str] = None
type_scan_seen: set = set()
causal_scan_seen: set = set()
level_scan_seen: set = set()
search_level_min_touches: int = 3
level_scan_v2_seen: set = set()
search_level_v2_min_strength: float = 0.5

# --- ML-фильтр релевантности (фаза 2.4) ---
ml_pipeline: MLPipeline = MLPipeline(model_path=CONFIG.ml.ml_model_path)
ml_score_threshold: float = CONFIG.ml.ml_score_threshold
# Кольцевой буфер недавних ml_score для гистограммы распределения (Phase 3.2).
from collections import deque as _deque
_recent_ml_scores: "_deque[float]" = _deque(maxlen=1000)


def _score_and_record(features, candle_history, timeframe_minutes: float = 0.0) -> float:
    """Считает ml_score и пишет его в кольцевой буфер для распределения."""
    s = ml_pipeline.score_features(
        features, candle_history, timeframe_minutes=timeframe_minutes
    )
    if ml_pipeline.model_exists:
        _recent_ml_scores.append(float(s))
    return s


def _ml_hard_filter_active() -> bool:
    """Включён ли ЖЁСТКИЙ ML-фильтр (отсечение матчей ниже порога).

    Модель «из коробки» обучена только на синтетике и плохо переносится на
    реальный рынок (ставит низкий ml_score почти всем реальным структурам),
    поэтому жёсткое отсечение по умолчанию ВЫКЛЮЧЕНО — ml_score используется
    лишь как бейдж и для сортировки. Фильтр включается автоматически, когда
    модель дообучена минимум на ``ml_hard_filter_min_feedback`` записях
    реальной обратной связи пользователя.
    """
    if not ml_pipeline.model_exists:
        return False
    meta = getattr(ml_pipeline.model, "metadata", {}) or {}
    try:
        n_feedback = int(meta.get("n_feedback", 0) or 0)
    except (TypeError, ValueError):
        # Битая/устаревшая метадата модели — считаем, что реального
        # feedback нет, фильтр остаётся выключенным.
        logger.warning("Некорректное значение n_feedback в метаданных модели: %r", meta.get("n_feedback"))
        n_feedback = 0
    return n_feedback >= int(CONFIG.ml.ml_hard_filter_min_feedback)


# Если модель уже есть на старте — считаем её «свежей», чтобы не запускать
# тяжёлое переобучение на первом же тике рынка.
last_model_retrain: Optional[datetime] = datetime.now() if ml_pipeline.model_exists else None
_retrain_in_progress: bool = False
# Троттлинг проверки авто-переобучения: не чаще раза в N секунд (а не каждый тик).
_last_retrain_check: Optional[datetime] = None
_RETRAIN_CHECK_INTERVAL_SEC: float = 300.0
SYNTHETIC_DATASET_PATH: str = "data/synthetic_dataset.json"


def _tf_minutes(timeframe: str) -> float:
    """Минуты таймфрейма по строке (1m/5m/.../1d). 0 если неизвестно."""
    base = timeframe.split("_w")[0] if "_w" in timeframe else timeframe
    return float(CONFIG.data.timframes.get(base, 0))


def _load_synthetic_dataset() -> list:
    """Загружает синтетический датасет для (пере)обучения; [] при ошибке."""
    try:
        if not os.path.exists(SYNTHETIC_DATASET_PATH):
            return []
        with open(SYNTHETIC_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось загрузить синтетический датасет: %s", exc)
        return []


async def _retrain_ml() -> dict:
    """Пересобирает и переобучает ML-модель, перезагружает пайплайн.

    Возвращает словарь метрик (как у ``ModelTrainer.train``) либо
    ``{"trained": False, "reason": ...}``.
    """
    global last_model_retrain, ml_pipeline

    dataset = _load_synthetic_dataset()
    if not dataset:
        return {
            "trained": False,
            "reason": (
                "Синтетический датасет не найден. Выполните "
                "scripts/generate_synthetic.py."
            ),
        }

    try:
        feedback = await database.get_all_feedback()
    except Exception:  # noqa: BLE001
        feedback = []

    trainer = ModelTrainer(model_path=CONFIG.ml.ml_model_path)
    # обучение синхронное (CPU-bound) — выполняем в пуле, чтобы не блокировать loop
    result = await asyncio.to_thread(trainer.retrain, dataset, feedback)

    last_model_retrain = datetime.now()
    if result.get("trained"):
        ml_pipeline.load()
    return result


async def _maybe_auto_retrain() -> None:
    """Фоновое авто-переобучение по истечении ml_retrain_interval_hours."""
    global _retrain_in_progress

    if _retrain_in_progress:
        return
    interval_h = float(CONFIG.ml.ml_retrain_interval_hours)
    if interval_h <= 0:
        return
    if last_model_retrain is not None:
        elapsed = (datetime.now() - last_model_retrain).total_seconds()
        if elapsed < interval_h * 3600:
            return

    _retrain_in_progress = True
    try:
        result = await _retrain_ml()
        if result.get("trained"):
            logger.info(
                "Авто-переобучение ML завершено: accuracy=%.4f",
                result.get("accuracy", 0.0),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Авто-переобучение ML не удалось: %s", exc)
    finally:
        _retrain_in_progress = False


class ThresholdUpdate(BaseModel):
    threshold: float


class FeedbackRequest(BaseModel):
    is_relevant: bool
    match_id: Optional[int] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    structure_type: Optional[str] = None


class PresetRequest(BaseModel):
    preset_type: str
    min_touches: int = 3


class StartScanRequest(BaseModel):
    mode: str = "preset"
    pattern_type: Optional[str] = None
    structure_id: Optional[int] = None
    type_filter: Optional[str] = None
    timeframe_filter: Optional[str] = None
    level_min_touches: int = 3
    level_v2_min_strength: float = 0.5


class ManualPivot(BaseModel):
    x: float
    y: float
    isHigh: bool


class ManualStructureRequest(BaseModel):
    pivots: List[ManualPivot]


async def broadcast_message(message: dict):
    """Send message to all connected WebSocket clients."""
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    
    for ws in disconnected:
        active_websockets.remove(ws)


async def on_market_update(symbol: str, timeframe: str):
    """Callback when market data updates."""
    global current_reference, scan_threshold, search_mode, search_type_filter
    global _last_retrain_check

    # Фоновая проверка авто-переобучения (не блокирует сканирование).
    # Троттлим саму проверку, чтобы не плодить задачи на каждый тик рынка.
    if not _retrain_in_progress:
        now = datetime.now()
        if (
            _last_retrain_check is None
            or (now - _last_retrain_check).total_seconds() >= _RETRAIN_CHECK_INTERVAL_SEC
        ):
            _last_retrain_check = now
            asyncio.create_task(_maybe_auto_retrain())

    if search_mode == "type_scan":
        await on_market_update_type_scan(symbol, timeframe)
        return
    
    if search_mode == "level_scan":
        await on_market_update_level_scan(symbol, timeframe)
        return
    
    if search_mode == "level_scan_v2":
        await on_market_update_level_scan_v2(symbol, timeframe)
        return
    
    if search_mode == "causal_patterns":
        await on_market_update_causal_scan(symbol, timeframe)
        return
    
    if current_reference is None:
        return
    
    structures = scanner.get_all_structures()
    
    relevant = [(s, tf, f, ts, ct) for s, tf, f, ts, ct in structures 
                if s == symbol and (tf == timeframe or tf.startswith(f"{timeframe}_w"))]
    
    if not relevant:
        return
    
    matches = similarity_matcher.find_matches(
        current_reference, relevant, scan_threshold
    )
    
    for match in matches:
        match_id = None
        base_tf = match.timeframe.split("_w")[0] if "_w" in match.timeframe else match.timeframe
        if current_structure_id:
            match_id = await database.save_match(current_structure_id, match)
        
        await broadcast_message({
            "type": "match",
            "data": {
                "match_id": match_id,
                "symbol": match.symbol,
                "timeframe": base_tf,
                "similarity_score": match.similarity_score,
                "structure_type": match.structure_type.value,
                "timestamp": match.timestamp,
                "is_mirrored": match.is_mirrored,
                "normalized_line": match.normalized_line,
                "price_change_24h": scanner.price_change_24h.get(match.symbol.replace("USDT", ""), 0),
                "pattern_time": match.pattern_time
            }
        })


async def on_market_update_type_scan(symbol: str, timeframe: str):
    """Callback for type-based scanning - checks all detected patterns."""
    global search_type_filter, type_scan_seen, search_timeframe_filter
    
    if not search_type_filter:
        return
    
    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return
    
    structures = scanner.get_all_structures()
    relevant = [(s, tf, f, ts, ct) for s, tf, f, ts, ct in structures 
                if s == symbol and (tf == timeframe or tf.startswith(f"{timeframe}_w"))]
    
    for sym, tf, features, timestamp, candle_time in relevant:
        if features is None:
            continue

        # Пропускаем устаревшие/неактивные структуры (например, тренд, против
        # которого цена уже развернулась) — как и в similarity-режиме.
        if hasattr(features, 'is_pattern_active') and not features.is_pattern_active:
            continue

        detected = getattr(features, 'detected_patterns', {}) or {}
        primary_match = features.structure_type.value == search_type_filter
        secondary_match = search_type_filter in detected
        
        if primary_match or secondary_match:
            base_tf = tf.split("_w")[0] if "_w" in tf else tf
            key = f"{sym}_{base_tf}"
            if key in type_scan_seen:
                continue
            type_scan_seen.add(key)
            
            conf = detected.get(search_type_filter, features.pattern_confidence) if secondary_match else features.pattern_confidence
            vol_conf = getattr(features, 'volume_confirmation', 0.5)
            if vol_conf > 0.7:
                conf = min(1.0, conf * (1.0 + (vol_conf - 0.7) * 0.3))
            elif vol_conf < 0.3:
                conf *= 0.85
            score = round(conf * 100, 1) if not primary_match else 100.0
            
            all_sym_tfs = set()
            for s_sym, s_tf, s_feat, s_ts, s_ct in structures:
                if s_sym == sym and s_feat is not None:
                    s_det = getattr(s_feat, 'detected_patterns', {}) or {}
                    if s_feat.structure_type.value == search_type_filter or search_type_filter in s_det:
                        s_base = s_tf.split("_w")[0] if "_w" in s_tf else s_tf
                        all_sym_tfs.add(s_base)
            mtf_count = len(all_sym_tfs)
            if mtf_count >= 3:
                score = min(100.0, score * 1.08)
            elif mtf_count >= 2:
                score = min(100.0, score * 1.04)
            score = round(score, 1)

            # ML-фильтр релевантности: считаем ml_score и отсекаем слабые матчи.
            ml_score = _score_and_record(
                features,
                scanner.get_candles(sym, base_tf) or [],
                timeframe_minutes=_tf_minutes(base_tf),
            )
            if _ml_hard_filter_active() and ml_score < ml_score_threshold:
                continue

            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": sym,
                    "timeframe": base_tf,
                    "similarity_score": score,
                    "structure_type": search_type_filter,
                    "timestamp": timestamp,
                    "is_mirrored": False,
                    "normalized_line": features.normalized_line.tolist(),
                    "price_change_24h": scanner.price_change_24h.get(sym, 0),
                    "pattern_time": candle_time,
                    "detected_patterns": detected,
                    "volume_confirmation": round(vol_conf, 2),
                    "mtf_confirmed": mtf_count >= 2,
                    "ml_score": round(ml_score, 4),
                }
            })


async def on_market_update_causal_scan(symbol: str, timeframe: str):
    """Callback for causal (non-repainting) pattern scanning.

    Запускает причинно-следственную детекцию на сырых свечах и
    публикует только ПОДТВЕРЖДЁННЫЕ паттерны (is_confirmed=True),
    исключая перерисовку (repaint).
    """
    global search_type_filter, causal_scan_seen, search_timeframe_filter

    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return

    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 20:
        return

    features = structure_extractor.extract_features_causal(candles)
    if features is None or not getattr(features, 'is_confirmed', False):
        return

    type_value = features.structure_type.value
    if search_type_filter and search_type_filter != "all" and type_value != search_type_filter:
        return

    key = f"{symbol}_{timeframe}_{type_value}_{getattr(features, 'confirmation_time', '')}"
    if key in causal_scan_seen:
        return
    causal_scan_seen.add(key)

    score = round(features.pattern_confidence * 100, 1)
    last_close_time = candles[-1].close_time if candles else None

    # ML-фильтр релевантности.
    ml_score = _score_and_record(
        features, candles, timeframe_minutes=_tf_minutes(timeframe)
    )
    if _ml_hard_filter_active() and ml_score < ml_score_threshold:
        return

    await broadcast_message({
        "type": "match",
        "data": {
            "match_id": None,
            "symbol": symbol,
            "timeframe": timeframe,
            "similarity_score": score,
            "structure_type": type_value,
            "timestamp": datetime.now().isoformat(),
            "is_mirrored": False,
            "normalized_line": features.normalized_line.tolist(),
            "price_change_24h": scanner.price_change_24h.get(symbol, 0),
            "pattern_time": getattr(features, 'confirmation_time', None) or last_close_time,
            "detected_patterns": getattr(features, 'detected_patterns', {}) or {},
            "volume_confirmation": round(getattr(features, 'volume_confirmation', 0.5), 2),
            "is_confirmed": True,
            "candidate_time": getattr(features, 'candidate_time', None),
            "confirmation_time": getattr(features, 'confirmation_time', None),
            "ml_score": round(ml_score, 4),
        }
    })


async def run_initial_causal_scan():
    """Initial causal scan across all symbols/timeframes - emits only confirmed patterns."""
    global search_type_filter, causal_scan_seen, search_timeframe_filter

    match_count = 0

    for symbol in list(scanner.symbol_data.keys()):
        data = scanner.symbol_data[symbol]
        for timeframe, candles in data.candles.items():
            if "_w" in timeframe:
                continue
            if search_timeframe_filter and timeframe != search_timeframe_filter:
                continue
            if not candles or len(candles) < 20:
                continue

            features = structure_extractor.extract_features_causal(candles)
            if features is None or not getattr(features, 'is_confirmed', False):
                continue

            type_value = features.structure_type.value
            if search_type_filter and search_type_filter != "all" and type_value != search_type_filter:
                continue

            key = f"{symbol}_{timeframe}_{type_value}_{getattr(features, 'confirmation_time', '')}"
            if key in causal_scan_seen:
                continue
            causal_scan_seen.add(key)

            score = round(features.pattern_confidence * 100, 1)
            last_close_time = candles[-1].close_time if candles else None

            # ML-фильтр релевантности.
            ml_score = _score_and_record(
                features, candles, timeframe_minutes=_tf_minutes(timeframe)
            )
            if _ml_hard_filter_active() and ml_score < ml_score_threshold:
                continue
            match_count += 1

            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "similarity_score": score,
                    "structure_type": type_value,
                    "timestamp": datetime.now().isoformat(),
                    "is_mirrored": False,
                    "normalized_line": features.normalized_line.tolist(),
                    "price_change_24h": scanner.price_change_24h.get(symbol, 0),
                    "pattern_time": getattr(features, 'confirmation_time', None) or last_close_time,
                    "detected_patterns": getattr(features, 'detected_patterns', {}) or {},
                    "volume_confirmation": round(getattr(features, 'volume_confirmation', 0.5), 2),
                    "is_confirmed": True,
                    "candidate_time": getattr(features, 'candidate_time', None),
                    "confirmation_time": getattr(features, 'confirmation_time', None),
                    "ml_score": round(ml_score, 4),
                }
            })

    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": match_count}
    })


async def on_market_update_level_scan(symbol: str, timeframe: str):
    global level_scan_seen, search_level_min_touches

    key = f"{symbol}_{timeframe}"
    if key in level_scan_seen:
        return

    sym_data = scanner.symbol_data.get(symbol)
    if not sym_data:
        return
    candles = sym_data.candles.get(timeframe, [])
    if not candles or len(candles) < 30:
        return

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    levels = level_detector.detect_levels(
        symbol, timeframe, highs, lows, closes,
        min_touches=search_level_min_touches,
        max_levels=5,
    )

    if levels:
        level_scan_seen.add(key)
        for lv in levels:
            match_data = _level_to_match_data(lv, symbol, timeframe, candles)
            await broadcast_message({"type": "match", "data": match_data})


async def run_initial_level_scan():
    global level_scan_seen, search_level_min_touches

    match_count = 0

    for symbol, sym_data in scanner.symbol_data.items():
        for timeframe, candles_list in sym_data.candles.items():
            if search_timeframe_filter and timeframe != search_timeframe_filter:
                continue
            if not candles_list or len(candles_list) < 30:
                continue

            key = f"{symbol}_{timeframe}"
            if key in level_scan_seen:
                continue

            highs = [c.high for c in candles_list]
            lows = [c.low for c in candles_list]
            closes = [c.close for c in candles_list]

            levels = level_detector.detect_levels(
                symbol, timeframe, highs, lows, closes,
                min_touches=search_level_min_touches,
                max_levels=5,
            )

            if levels:
                level_scan_seen.add(key)
                for lv in levels:
                    match_count += 1
                    match_data = _level_to_match_data(lv, symbol, timeframe, candles_list)
                    await broadcast_message({"type": "match", "data": match_data})

    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": match_count}
    })


def _level_to_match_data(lv, symbol, timeframe, candles):
    n = len(candles)
    closes = [c.close for c in candles]
    min_c, max_c = min(closes), max(closes)
    rng = max_c - min_c if max_c > min_c else 1.0
    normalized_line = [(c - min_c) / rng for c in closes]

    touch_points = []
    for t in lv.touches:
        touch_points.append({
            "candle_index": t.candle_index,
            "price": t.price,
            "deviation": round(t.deviation, 8),
            "is_high": t.is_high,
        })

    candle_time = None
    if candles:
        candle_time = candles[-1].open_time

    line_points = []
    start_idx = lv.anchor_start[0]
    end_idx = min(lv.anchor_end[0], n - 1)
    for idx in [start_idx, end_idx]:
        price_at = lv.slope * idx + lv.intercept
        line_points.append({"index": idx, "price": round(price_at, 8)})

    slope_label = "горизонтальный"
    if abs(lv.slope) > 0.0001:
        slope_label = "восходящий" if lv.slope > 0 else "нисходящий"

    return {
        "match_id": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "similarity_score": lv.quality_score,
        "structure_type": f"level_{lv.line_type}",
        "timestamp": datetime.now().isoformat(),
        "is_mirrored": False,
        "normalized_line": normalized_line,
        "price_change_24h": scanner.price_change_24h.get(symbol, 0),
        "pattern_time": candle_time,
        "is_level": True,
        "level_data": {
            "line_type": lv.line_type,
            "slope": round(lv.slope, 10),
            "intercept": round(lv.intercept, 8),
            "touch_count": lv.touch_count,
            "avg_deviation_pct": lv.avg_deviation_pct,
            "quality_score": lv.quality_score,
            "coverage": lv.coverage,
            "price_at_last": lv.price_at_last,
            "touches": touch_points,
            "line_points": line_points,
            "slope_label": slope_label,
        },
    }


def _level_v2_to_match_data(lv, symbol, timeframe, candles, is_trendline=False):
    """Преобразует уровень/трендлайн LevelDetectorV2 в формат WS-сообщения match.

    Переиспользует существующую форму ``level_data`` (line_type, slope, intercept,
    line_points, touches и т.д.), чтобы фронтенд отрисовывал V2-уровни без правок.
    """
    n = len(candles)
    closes = [c.close for c in candles]
    min_c, max_c = min(closes), max(closes)
    rng = max_c - min_c if max_c > min_c else 1.0
    normalized_line = [(c - min_c) / rng for c in closes]
    candle_time = candles[-1].open_time if candles else None

    if is_trendline:
        slope = lv["slope"]
        intercept = lv["intercept"]
        start_idx = max(0, min(lv["start_idx"], n - 1))
        end_idx = max(0, min(lv["end_idx"], n - 1))
        line_type = "support" if lv["type"] == "support_trendline" else "resistance"
        touches = [{"candle_index": int(i), "price": round(slope * i + intercept, 8),
                    "is_high": line_type == "resistance"} for i in lv["touches"]]
        touch_count = len(lv["touches"])
        first_t, last_t = (lv["touches"][0], lv["touches"][-1]) if lv["touches"] else (start_idx, end_idx)
    else:
        slope = 0.0
        intercept = lv["price"]
        start_idx, end_idx = 0, n - 1
        line_type = lv["type"]
        touches = []
        touch_count = lv["num_touches"]
        first_t, last_t = lv["first_touch_time"], lv["last_touch_time"]

    line_points = []
    for idx in [start_idx, end_idx]:
        price_at = slope * idx + intercept
        line_points.append({"index": int(idx), "price": round(price_at, 8)})

    slope_label = "горизонтальный"
    if abs(slope) > 1e-9:
        slope_label = "восходящий" if slope > 0 else "нисходящий"

    quality_score = round(lv["strength"] * 100, 1)

    return {
        "match_id": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "similarity_score": quality_score,
        "structure_type": f"level_{line_type}",
        "timestamp": datetime.now().isoformat(),
        "is_mirrored": False,
        "normalized_line": normalized_line,
        "price_change_24h": scanner.price_change_24h.get(symbol, 0),
        "pattern_time": candle_time,
        "is_level": True,
        "is_level_v2": True,
        "level_data": {
            "line_type": line_type,
            "slope": round(slope, 10),
            "intercept": round(intercept, 8),
            "touch_count": touch_count,
            "quality_score": quality_score,
            "strength": lv["strength"],
            "price_at_last": round(slope * (n - 1) + intercept, 8),
            "touches": touches,
            "line_points": line_points,
            "slope_label": slope_label,
            "is_trendline": is_trendline,
            "source": lv.get("source", "trendline"),
            "first_touch_time": first_t,
            "last_touch_time": last_t,
        },
    }


def _detect_levels_v2_for(symbol, timeframe, candles):
    """Запускает LevelDetectorV2 на свечах и возвращает (support, resistance, trendlines)."""
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    times = [c.open_time for c in candles]
    res = level_detector_v2.detect_levels(highs, lows, closes, volumes, times=times)
    return res


async def on_market_update_level_scan_v2(symbol: str, timeframe: str):
    """Callback для сканирования уровней V2 (DBSCAN/RANSAC/volume profile)."""
    global level_scan_v2_seen, search_level_v2_min_strength, search_timeframe_filter

    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return

    key = f"{symbol}_{timeframe}"
    if key in level_scan_v2_seen:
        return

    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 30:
        return

    res = _detect_levels_v2_for(symbol, timeframe, candles)
    min_strength = search_level_v2_min_strength

    emitted = False
    for lv in res["support_levels"] + res["resistance_levels"]:
        if lv["strength"] >= min_strength:
            emitted = True
            await broadcast_message({
                "type": "match",
                "data": _level_v2_to_match_data(lv, symbol, timeframe, candles, is_trendline=False),
            })
    for tl in res["trendlines"]:
        if tl["strength"] >= min_strength:
            emitted = True
            await broadcast_message({
                "type": "match",
                "data": _level_v2_to_match_data(tl, symbol, timeframe, candles, is_trendline=True),
            })

    if emitted:
        level_scan_v2_seen.add(key)


async def run_initial_level_scan_v2():
    """Начальное сканирование уровней V2 по всем символам/таймфреймам."""
    global level_scan_v2_seen, search_level_v2_min_strength, search_timeframe_filter

    match_count = 0
    min_strength = search_level_v2_min_strength

    for symbol, sym_data in scanner.symbol_data.items():
        for timeframe, candles_list in sym_data.candles.items():
            if "_w" in timeframe:
                continue
            if search_timeframe_filter and timeframe != search_timeframe_filter:
                continue
            if not candles_list or len(candles_list) < 30:
                continue

            key = f"{symbol}_{timeframe}"
            if key in level_scan_v2_seen:
                continue

            res = _detect_levels_v2_for(symbol, timeframe, candles_list)
            emitted = False
            for lv in res["support_levels"] + res["resistance_levels"]:
                if lv["strength"] >= min_strength:
                    emitted = True
                    match_count += 1
                    await broadcast_message({
                        "type": "match",
                        "data": _level_v2_to_match_data(lv, symbol, timeframe, candles_list, is_trendline=False),
                    })
            for tl in res["trendlines"]:
                if tl["strength"] >= min_strength:
                    emitted = True
                    match_count += 1
                    await broadcast_message({
                        "type": "match",
                        "data": _level_v2_to_match_data(tl, symbol, timeframe, candles_list, is_trendline=True),
                    })
            if emitted:
                level_scan_v2_seen.add(key)

    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": match_count}
    })


async def ensure_scanner_initialized():
    """Initialize scanner once, reuse for all subsequent scans."""
    if scanner.initialized:
        return
    if scanner.is_running:
        for _ in range(120):
            if scanner.initialized:
                return
            await asyncio.sleep(0.5)
        return
    await scanner.start(on_update=on_market_update, progress_callback=_init_progress_callback)


async def run_continuous_scan():
    """Run continuous market scanning - blocks until cancelled."""
    global is_scanning
    
    try:
        scanner.on_update_callback = on_market_update
        # При WebSocket-режиме поток данных идёт через ws_feed; REST-поллинг
        # запускаем только если WS выключен (или не поднялся → фаллбэк).
        if scanner.ws_feed is not None:
            scanner.ws_feed.on_update = on_market_update
        elif not scanner._poll_task or scanner._poll_task.done():
            if scanner.initialized:
                scanner._poll_task = asyncio.create_task(scanner._poll_loop())
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        is_scanning = False


async def run_initial_scan():
    """Run initial scan against all current structures."""
    global current_reference, scan_threshold, current_structure_id, search_mode, search_type_filter

    if search_mode == "type_scan":
        await run_initial_type_scan()
        return

    if search_mode == "level_scan":
        await run_initial_level_scan()
        return
    
    if search_mode == "level_scan_v2":
        await run_initial_level_scan_v2()
        return
    
    if search_mode == "causal_patterns":
        await run_initial_causal_scan()
        return
    
    if current_reference is None:
        return
    
    structures = scanner.get_all_structures()
    matches = similarity_matcher.find_matches(current_reference, structures, scan_threshold)
    
    for match in matches:
        match_id = None
        if current_structure_id:
            match_id = await database.save_match(current_structure_id, match)
        
        base_tf = match.timeframe.split("_w")[0] if "_w" in match.timeframe else match.timeframe
        await broadcast_message({
            "type": "match",
            "data": {
                "match_id": match_id,
                "symbol": match.symbol,
                "timeframe": base_tf,
                "similarity_score": match.similarity_score,
                "structure_type": match.structure_type.value,
                "timestamp": match.timestamp,
                "is_mirrored": match.is_mirrored,
                "normalized_line": match.normalized_line,
                "price_change_24h": scanner.price_change_24h.get(match.symbol.replace("USDT", ""), 0),
                "pattern_time": match.pattern_time
            }
        })
    
    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": len(matches)}
    })


async def run_initial_type_scan():
    """Run initial scan by structure type - checks all detected patterns, not just primary."""
    global search_type_filter, search_timeframe_filter
    
    if not search_type_filter:
        return
    
    structures = scanner.get_all_structures()
    match_count = 0
    seen_keys = set()
    
    for sym, tf, features, timestamp, candle_time in structures:
        if features is None:
            continue
        base_tf = tf.split("_w")[0] if "_w" in tf else tf
        if search_timeframe_filter and base_tf != search_timeframe_filter:
            continue
        
        detected = getattr(features, 'detected_patterns', {}) or {}
        primary_match = features.structure_type.value == search_type_filter
        secondary_match = search_type_filter in detected
        
        if primary_match or secondary_match:
            dedup_key = f"{sym}_{base_tf}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            
            conf = detected.get(search_type_filter, features.pattern_confidence) if secondary_match else features.pattern_confidence
            score = round(conf * 100, 1) if not primary_match else 100.0

            # ML-фильтр релевантности (как в live-пути type_scan): считаем
            # ml_score и отсекаем слабые матчи при наличии обученной модели.
            ml_score = _score_and_record(
                features,
                scanner.get_candles(sym, base_tf) or [],
                timeframe_minutes=_tf_minutes(base_tf),
            )
            if _ml_hard_filter_active() and ml_score < ml_score_threshold:
                continue

            match_count += 1
            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": sym,
                    "timeframe": base_tf,
                    "similarity_score": score,
                    "structure_type": search_type_filter,
                    "timestamp": timestamp,
                    "is_mirrored": False,
                    "normalized_line": features.normalized_line.tolist(),
                    "price_change_24h": scanner.price_change_24h.get(sym, 0),
                    "pattern_time": candle_time,
                    "detected_patterns": detected,
                    "ml_score": round(ml_score, 4),
                }
            })
    
    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": match_count}
    })


@app.post("/api/upload")
async def upload_chart(file: UploadFile = File(...)):
    """Upload a chart screenshot and extract its structure."""
    global current_reference, current_structure_id
    
    try:
        contents = await file.read()
        
        os.makedirs("uploads", exist_ok=True)
        
        import hashlib
        file_hash = hashlib.md5(contents).hexdigest()[:8]
        filename = f"{file_hash}_{file.filename or 'chart.png'}"
        filepath = f"uploads/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        price_line = image_processor.extract_price_line(contents)
        
        if len(price_line) < 20:
            raise HTTPException(status_code=400, detail="Could not extract price line from image")
        
        features = structure_extractor.extract_features(price_line)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract structure features")
        
        structure_id = await database.save_structure(
            filename,
            features
        )
        
        current_reference = features
        current_structure_id = structure_id
        
        pivot_points = []
        for p in features.pivot_points:
            pivot_points.append({
                "index": int(p.index),
                "value": float(p.value),
                "is_high": p.is_high
            })
        
        return {
            "success": True,
            "structure_id": structure_id,
            "filename": filename,
            "structure_type": features.structure_type.value,
            "trend_direction": float(features.trend_direction),
            "volatility": float(features.volatility),
            "compression_ratio": float(features.compression_ratio),
            "num_pivots": len(features.pivot_points),
            "pivot_points": pivot_points,
            "normalized_line": features.normalized_line.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_preset_pattern(preset_type: str, min_touches: int = 3) -> np.ndarray:
    """Generate a synthetic pattern based on preset type."""
    points = 100
    x = np.linspace(0, 1, points)
    
    if preset_type == 'triangle_up':
        upper = 1.0 - x * 0.4
        lower = 0.0 + x * 0.4
        mid = (upper + lower) / 2
        wave = np.sin(x * 8 * np.pi) * (0.5 - x) * 0.5
        return mid + wave
    
    elif preset_type == 'triangle_down':
        upper = 0.5 + x * 0.4
        lower = 0.5 - x * 0.4
        mid = (upper + lower) / 2
        wave = np.sin(x * 8 * np.pi) * x * 0.5
        return mid + wave
    
    elif preset_type == 'trend_up':
        trend = x * 0.8 + 0.1
        wave = np.sin(x * 6 * np.pi) * 0.08
        return trend + wave
    
    elif preset_type == 'trend_down':
        trend = (1 - x) * 0.8 + 0.1
        wave = np.sin(x * 6 * np.pi) * 0.08
        return trend + wave
    
    elif preset_type == 'range':
        wave = np.sin(x * 8 * np.pi) * 0.15 + 0.5
        return wave
    
    elif preset_type == 'compression':
        decay = np.exp(-x * 2)
        wave = np.sin(x * 10 * np.pi) * 0.3 * decay + 0.5
        return wave
    
    elif preset_type == 'level':
        base = np.random.rand(points) * 0.3 + 0.35
        level_pos = 0.7
        for i in range(min_touches):
            touch_idx = int((i + 0.5) / min_touches * points * 0.9)
            if touch_idx < points:
                base[max(0, touch_idx-3):min(points, touch_idx+3)] = level_pos + np.random.randn() * 0.02
        return base
    
    else:
        return np.sin(x * 4 * np.pi) * 0.3 + 0.5


@app.post("/api/preset")
async def select_preset(data: PresetRequest):
    """Select a preset pattern for scanning."""
    global current_reference, current_structure_id
    
    try:
        pattern = generate_preset_pattern(data.preset_type, data.min_touches)
        
        features = structure_extractor.extract_features(pattern)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract structure features")
        
        current_reference = features
        current_structure_id = None
        
        preset_names = {
            'triangle_up': 'Сходящийся треугольник',
            'triangle_down': 'Расширяющийся треугольник',
            'trend_up': 'Восходящий тренд',
            'trend_down': 'Нисходящий тренд',
            'range': 'Боковой диапазон',
            'compression': 'Сжатие/Консолидация',
            'level': f'Уровень ({data.min_touches} касаний)'
        }
        
        pivot_points = []
        for p in features.pivot_points:
            pivot_points.append({
                "index": int(p.index),
                "value": float(p.value),
                "is_high": p.is_high
            })
        
        return {
            "success": True,
            "preset_name": preset_names.get(data.preset_type, data.preset_type),
            "structure_type": features.structure_type.value,
            "trend_direction": float(features.trend_direction),
            "volatility": float(features.volatility),
            "compression_ratio": float(features.compression_ratio),
            "num_pivots": len(features.pivot_points),
            "pivot_points": pivot_points,
            "normalized_line": features.normalized_line.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manual-structure")
async def create_manual_structure(data: ManualStructureRequest):
    """Create a structure from manually placed pivot points."""
    global current_reference, current_structure_id
    
    try:
        if len(data.pivots) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 pivot points")
        
        sorted_pivots = sorted(data.pivots, key=lambda p: p.x)
        
        points = 100
        x_positions = np.linspace(0, 1, points)
        price_line = np.zeros(points)
        
        pivot_indices = []
        for p in sorted_pivots:
            idx = int(p.x * (points - 1))
            idx = max(0, min(points - 1, idx))
            pivot_indices.append((idx, p.y, p.isHigh))
        
        for i in range(len(pivot_indices) - 1):
            start_idx, start_y, _ = pivot_indices[i]
            end_idx, end_y, _ = pivot_indices[i + 1]
            
            if end_idx > start_idx:
                for j in range(start_idx, end_idx + 1):
                    t = (j - start_idx) / (end_idx - start_idx)
                    price_line[j] = start_y + t * (end_y - start_y)
        
        if pivot_indices[0][0] > 0:
            for j in range(0, pivot_indices[0][0]):
                price_line[j] = pivot_indices[0][1]
        
        if pivot_indices[-1][0] < points - 1:
            for j in range(pivot_indices[-1][0], points):
                price_line[j] = pivot_indices[-1][1]
        
        features = structure_extractor.extract_features(price_line)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract structure features")
        
        current_reference = features
        current_structure_id = None
        
        pivot_points = []
        for p in features.pivot_points:
            pivot_points.append({
                "index": int(p.index),
                "value": float(p.value),
                "is_high": p.is_high
            })
        
        return {
            "success": True,
            "structure_id": None,
            "preset_name": "Ручная структура",
            "structure_type": features.structure_type.value,
            "trend_direction": float(features.trend_direction),
            "volatility": float(features.volatility),
            "compression_ratio": float(features.compression_ratio),
            "num_pivots": len(features.pivot_points),
            "pivot_points": pivot_points,
            "normalized_line": features.normalized_line.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-scan")
async def start_scan(data: Optional[StartScanRequest] = None):
    """Start continuous market scanning."""
    global is_scanning, scan_task, _init_task, _pending_initial_scan, current_reference, search_mode, search_pattern_type, current_structure_id, search_type_filter, type_scan_seen, causal_scan_seen, search_timeframe_filter, level_scan_seen, search_level_min_touches, level_scan_v2_seen, search_level_v2_min_strength
    
    if data is None:
        data = StartScanRequest()
    
    search_mode = data.mode
    search_pattern_type = data.pattern_type
    search_type_filter = data.type_filter
    search_timeframe_filter = data.timeframe_filter if data.timeframe_filter != "all" else None
    search_level_min_touches = data.level_min_touches
    search_level_v2_min_strength = data.level_v2_min_strength
    type_scan_seen = set()
    causal_scan_seen = set()
    level_scan_seen = set()
    level_scan_v2_seen = set()

    if data.mode == "type_scan":
        if data.type_filter is None:
            raise HTTPException(status_code=400, detail="type_filter required for type_scan mode")
        current_reference = None
        current_structure_id = None
        
    elif data.mode == "preset":
        if data.pattern_type is None:
            raise HTTPException(status_code=400, detail="pattern_type required for preset mode")
        
        pattern = generate_preset_pattern(data.pattern_type)
        reference = structure_extractor.extract_features(pattern)
        
        if reference is None:
            raise HTTPException(status_code=400, detail=f"Unknown preset pattern: {data.pattern_type}")
        
        current_reference = reference
        current_structure_id = None
        
    elif data.mode == "uploaded":
        if data.structure_id is None:
            if current_reference is None:
                raise HTTPException(status_code=400, detail="No structure selected. Upload a chart or provide structure_id")
        else:
            structure = await database.get_structure(data.structure_id)
            if structure is None:
                raise HTTPException(status_code=404, detail="Structure not found")
            
            features_dict = json.loads(structure["features"]) if isinstance(structure["features"], str) else structure["features"]
            current_reference = structure_extractor.features_from_dict(features_dict)
            current_structure_id = data.structure_id
    elif data.mode == "level_scan":
        current_reference = None
        current_structure_id = None

    elif data.mode == "level_scan_v2":
        current_reference = None
        current_structure_id = None

    elif data.mode == "causal_patterns":
        current_reference = None
        current_structure_id = None

    elif data.mode == "manual":
        if current_reference is None:
            raise HTTPException(status_code=400, detail="No manual structure created. Create one first.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")
    
    if is_scanning:
        if scan_task:
            scan_task.cancel()
            try:
                await scan_task
            except asyncio.CancelledError:
                pass
    
    is_scanning = True
    
    if _pending_initial_scan and not _pending_initial_scan.done():
        _pending_initial_scan.cancel()
        try:
            await _pending_initial_scan
        except asyncio.CancelledError:
            pass
    
    if scanner.initialized and len(scanner.symbols) > 0:
        scan_task = asyncio.create_task(run_continuous_scan())
        _pending_initial_scan = asyncio.create_task(_run_initial_scan_now())
    else:
        scan_task = asyncio.create_task(run_continuous_scan())
        if not _init_task or _init_task.done():
            _init_task = asyncio.create_task(_wait_and_run_initial_scan())
        else:
            _pending_initial_scan = asyncio.create_task(_wait_init_then_scan())
    
    return {
        "status": "started",
        "mode": search_mode,
        "pattern_type": search_pattern_type,
        "num_symbols": len(scanner.symbols)
    }


async def _run_initial_scan_now():
    """Run initial scan immediately when scanner is already initialized."""
    stats = scanner.get_structure_stats()
    await broadcast_message({
        "type": "scan_status",
        "data": {"status": "scanning", "message": f"Загружено {stats['total_structures']} структур, сканирование..."}
    })
    await run_initial_scan()


async def _wait_and_run_initial_scan():
    """Wait for scanner initialization to complete, then run initial scan."""
    await broadcast_message({
        "type": "scan_status",
        "data": {"status": "initializing", "message": "Загрузка рыночных данных..."}
    })
    await ensure_scanner_initialized()
    if not is_scanning:
        return
    stats = scanner.get_structure_stats()
    await broadcast_message({
        "type": "scan_status",
        "data": {"status": "scanning", "message": f"Загружено {stats['total_structures']} структур, сканирование..."}
    })
    await run_initial_scan()


async def _wait_init_then_scan():
    """Wait for existing initialization, then run initial scan with current params."""
    await ensure_scanner_initialized()
    if not is_scanning:
        return
    await run_initial_scan()


async def _init_progress_callback(done, total):
    """Send progress updates during scanner initialization."""
    pct = int(done / total * 100) if total > 0 else 0
    await broadcast_message({
        "type": "scan_status",
        "data": {"status": "initializing", "message": f"Загрузка данных: {pct}% ({done}/{total})"}
    })


@app.post("/api/stop-scan")
async def stop_scan():
    """Stop market scanning."""
    global is_scanning, scan_task, _init_task
    
    if not is_scanning:
        return {"status": "not_running"}
    
    is_scanning = False
    if _init_task and not _init_task.done():
        _init_task.cancel()
        try:
            await _init_task
        except asyncio.CancelledError:
            pass
        _init_task = None
    if scan_task:
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    
    return {"status": "stopped"}


@app.post("/api/threshold")
async def update_threshold(data: ThresholdUpdate):
    """Update similarity threshold."""
    global scan_threshold
    scan_threshold = max(0, min(100, data.threshold))
    
    await broadcast_message({
        "type": "threshold_updated",
        "data": {"threshold": scan_threshold}
    })
    
    return {"threshold": scan_threshold}


@app.get("/api/threshold")
async def get_threshold():
    """Get current similarity threshold."""
    return {"threshold": scan_threshold}


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]
    persist: bool = True


@app.get("/api/settings")
async def get_settings():
    """Схема настраиваемых параметров с текущими значениями (для UI-панели)."""
    return {"schema": settings_store.get_schema()}


@app.post("/api/settings")
async def update_settings(data: SettingsUpdate):
    """Валидирует и применяет настройки к CONFIG в рантайме, опционально
    сохраняя их в ``data/ui_settings.json``."""
    global ml_score_threshold
    try:
        applied = settings_store.apply_settings(data.settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Синхронизируем глобальный порог ML с CONFIG.
    ml_score_threshold = CONFIG.ml.ml_score_threshold

    if data.persist:
        settings_store.save_settings()

    await broadcast_message({
        "type": "settings_updated",
        "data": {"applied": list(applied.keys())},
    })
    return {"applied": applied, "schema": settings_store.get_schema()}


@app.post("/api/restart-scan")
async def restart_scan():
    """Перезапускает текущее сканирование, чтобы применить новые настройки
    к свежим результатам (например после изменения порогов/паттернов)."""
    global scan_task, is_scanning

    was_scanning = is_scanning
    prev_mode = search_mode

    if scan_task and not scan_task.done():
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    is_scanning = False

    if not was_scanning:
        return {"status": "idle", "restarted": False}

    # Перезапускаем начальный скан и непрерывное сканирование.
    await ensure_scanner_initialized()
    await run_initial_scan()
    is_scanning = True
    scan_task = asyncio.create_task(run_continuous_scan())

    await broadcast_message({
        "type": "scan_restarted",
        "data": {"mode": prev_mode},
    })
    return {"status": "restarted", "restarted": True, "mode": prev_mode}


_market_movers_cache: Dict[str, Any] = {"data": [], "timestamp": 0}

@app.get("/api/market-movers")
async def get_market_movers():
    """Get all coins with 24h price changes for market overview."""
    import time as _time
    cache_ttl = 300
    now = _time.time()
    
    if now - _market_movers_cache["timestamp"] > cache_ttl or not _market_movers_cache["data"]:
        try:
            import aiohttp
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        movers = []
                        for item in data:
                            sym = item['symbol'].upper()
                            change = item.get('price_change_percentage_24h', 0)
                            change = round(change, 2) if change else 0
                            scanner.price_change_24h[sym] = change
                            movers.append({
                                "symbol": f"{sym}USDT",
                                "price_change_24h": change
                            })
                        movers.sort(key=lambda x: x["price_change_24h"], reverse=True)
                        _market_movers_cache["data"] = movers
                        _market_movers_cache["timestamp"] = now
        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
    
    return {"movers": _market_movers_cache["data"]}


@app.get("/api/status")
async def get_status():
    """Get current scanner status."""
    stats = scanner.get_structure_stats()
    return {
        "is_scanning": is_scanning,
        "has_reference": current_reference is not None,
        "threshold": scan_threshold,
        "num_symbols": len(scanner.symbols) if scanner.symbols else 0,
        "structure_id": current_structure_id,
        "data_available": stats.get("data_available", False),
        "total_structures": stats.get("total_structures", 0),
        "working_endpoint": stats.get("working_endpoint"),
        "last_error": stats.get("last_error")
    }


@app.get("/api/structures")
async def get_structures():
    """Get all saved structures."""
    structures = await database.get_all_structures()
    return {"structures": structures}


@app.get("/api/levels-v2/{symbol}/{timeframe}")
async def get_levels_v2(symbol: str, timeframe: str):
    """Возвращает уровни и трендлайны LevelDetectorV2 для символа/таймфрейма."""
    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 30:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "support": [],
            "resistance": [],
            "trendlines": [],
            "message": "Недостаточно данных (нужно ≥30 свечей).",
        }
    res = _detect_levels_v2_for(symbol, timeframe, candles)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "support": res["support_levels"],
        "resistance": res["resistance_levels"],
        "trendlines": res["trendlines"],
    }


@app.get("/api/structures/{structure_id}")
async def get_structure(structure_id: int):
    """Get a specific structure."""
    structure = await database.get_structure(structure_id)
    if not structure:
        raise HTTPException(status_code=404, detail="Structure not found")
    return structure


@app.delete("/api/structures/{structure_id}")
async def delete_structure(structure_id: int):
    """Delete a structure."""
    global current_reference, current_structure_id
    
    await database.delete_structure(structure_id)
    
    if current_structure_id == structure_id:
        current_reference = None
        current_structure_id = None
    
    return {"success": True}


@app.get("/api/matches/{structure_id}")
async def get_matches(structure_id: int, limit: int = 100):
    """Get matches for a structure."""
    matches = await database.get_matches(structure_id, limit)
    return {"matches": matches}


def _features_for_match(symbol: Optional[str], timeframe: Optional[str]):
    """Находит текущую структуру для symbol+timeframe в сканере и считает её
    ML-признаки. Нужно, чтобы оценка пользователя попала в дообучение модели.

    Возвращает ``dict`` признаков либо ``None``, если структура не найдена.
    """
    if not symbol or not timeframe:
        return None
    base_tf = timeframe.split("_w")[0] if "_w" in timeframe else timeframe
    for sym, tf, features, _ts, _ct in scanner.get_all_structures():
        cand_base = tf.split("_w")[0] if "_w" in tf else tf
        if sym == symbol and cand_base == base_tf and features is not None:
            try:
                return extract_ml_features(
                    features,
                    candidate_index=features.candidate_index,
                    confirmation_index=features.confirmation_index,
                    candle_history=scanner.get_candles(sym, base_tf) or [],
                    timeframe_minutes=_tf_minutes(base_tf),
                )
            except Exception:  # noqa: BLE001
                logger.warning("Не удалось извлечь ML-признаки для %s %s", sym, base_tf)
                return None
    return None


@app.post("/api/feedback")
async def submit_feedback(data: FeedbackRequest):
    """Сохраняет оценку пользователя («Релевантно»/«Нерелевантно»).

    Вместе с оценкой сохраняются ML-признаки структуры (если её удалось найти
    в сканере), чтобы накопленная обратная связь реально дообучала модель и
    включала жёсткий ML-фильтр после ``ml_hard_filter_min_feedback`` оценок.
    """
    features = _features_for_match(data.symbol, data.timeframe)
    await database.save_feedback(
        is_relevant=data.is_relevant,
        match_id=data.match_id,
        symbol=data.symbol,
        timeframe=data.timeframe,
        structure_type=data.structure_type,
        features=features,
    )
    return {"success": True, "features_captured": features is not None}


@app.post("/api/retrain-ml")
async def retrain_ml():
    """Ручной запуск переобучения ML-фильтра. Возвращает новые метрики."""
    global _retrain_in_progress

    if _retrain_in_progress:
        raise HTTPException(
            status_code=409, detail="Переобучение уже выполняется."
        )

    _retrain_in_progress = True
    try:
        result = await _retrain_ml()
    finally:
        _retrain_in_progress = False

    if not result.get("trained"):
        raise HTTPException(
            status_code=400,
            detail=result.get("reason", "Не удалось обучить ML-модель."),
        )
    return {"success": True, **result}


@app.get("/api/ml-status")
async def ml_status():
    """Состояние ML-модели: наличие, дата обучения, объём, точность, признаки."""
    status = ml_pipeline.status()
    status["ml_score_threshold"] = ml_score_threshold
    status["ml_hard_filter_active"] = _ml_hard_filter_active()
    status["ml_hard_filter_min_feedback"] = int(CONFIG.ml.ml_hard_filter_min_feedback)
    status["retrain_in_progress"] = _retrain_in_progress
    status["last_model_retrain"] = (
        last_model_retrain.isoformat() if last_model_retrain else None
    )
    # Гистограмма распределения недавних ml_score (10 бинов 0.0–1.0).
    bins = [0] * 10
    scores = list(_recent_ml_scores)
    for s in scores:
        idx = min(9, max(0, int(s * 10)))
        bins[idx] += 1
    status["ml_score_distribution"] = {
        "bins": bins,
        "total": len(scores),
        "bin_edges": [round(i / 10, 1) for i in range(11)],
    }
    return status


@app.get("/api/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded image."""
    filepath = f"uploads/{filename}"
    if os.path.isfile(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/chart/{symbol}/{timeframe}")
async def get_chart_data(symbol: str, timeframe: str):
    """Get chart data for a symbol/timeframe."""
    data = scanner.get_symbol_chart_data(symbol, timeframe)
    return {"data": data}


@app.get("/api/candles/{symbol}/{timeframe}")
async def get_candles_data(symbol: str, timeframe: str):
    """Get OHLC candle data for lightweight-charts rendering."""
    candles = scanner.get_candles(symbol, timeframe)
    if not candles:
        return {"candles": []}
    
    ohlc = []
    for c in candles:
        ohlc.append({
            "time": c.open_time // 1000,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        })
    return {"candles": ohlc}



@app.get("/api/debug-structure/{symbol}/{timeframe}")
async def get_debug_structure(symbol: str, timeframe: str):
    """Get candles + pivot points + structure info for debug visualization."""
    candles = scanner.get_candles(symbol, timeframe)

    if not candles:
        candles = await scanner._fetch_binance_klines(symbol, timeframe, limit=100)

    if not candles:
        return {"candles": [], "pivots": [], "structure_type": "unknown", "pattern_confidence": 0}

    ohlc = []
    for c in candles:
        ohlc.append({
            "time": c.open_time // 1000,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
        })

    pivots = []
    structure_type = "unknown"
    pattern_confidence = 0.0

    sym_data = scanner.symbol_data.get(symbol)
    if sym_data:
        features = sym_data.structures.get(timeframe)
        if features is not None:
            structure_type = features.structure_type.value
            pattern_confidence = round(features.pattern_confidence * 100, 1)
            for p in features.pivot_points:
                if 0 <= p.index < len(ohlc):
                    pivots.append({
                        "index": p.index,
                        "time": ohlc[p.index]["time"],
                        "value": p.value,
                        "is_high": p.is_high,
                        "confidence": round(p.confidence, 3),
                    })

    if not pivots and len(ohlc) > 0:
        import numpy as np
        closes = np.array([c.close for c in candles])
        normalized = structure_extractor.normalize_line(closes)
        raw_pivots = structure_extractor.detect_pivots(normalized)
        for p in raw_pivots:
            if 0 <= p.index < len(ohlc):
                price_val = candles[p.index].high if p.is_high else candles[p.index].low
                pivots.append({
                    "index": p.index,
                    "time": ohlc[p.index]["time"],
                    "value": price_val,
                    "is_high": p.is_high,
                    "confidence": round(p.confidence, 3),
                })
        if len(raw_pivots) > 0:
            trend = structure_extractor.calculate_trend(normalized)
            compression = structure_extractor.calculate_compression(normalized, raw_pivots)
            st, pc, _, is_active, freshness = structure_extractor.classify_structure(trend, compression, raw_pivots, normalized)
            structure_type = st.value
            pattern_confidence = round(pc * 100, 1)

    return {
        "candles": ohlc,
        "pivots": pivots,
        "structure_type": structure_type,
        "pattern_confidence": pattern_confidence,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "data": {
                "is_scanning": is_scanning,
                "has_reference": current_reference is not None,
                "threshold": scan_threshold
            }
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the main page with anti-cache measures."""
    import time
    return FileResponse("static/index.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": str(int(time.time())),
        "Last-Modified": "Thu, 01 Jan 1970 00:00:00 GMT"
    })


@app.get("/{path:path}")
async def catch_all(path: str):
    """Catch-all route for SPA."""
    file_path = f"static/{path}"
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse("static/index.html")
