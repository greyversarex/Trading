import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

from .image_processor import ImageProcessor
from .structure_extractor import StructureExtractor, StructureFeatures
from .similarity_matcher import SimilarityMatcher, MatchResult
from .binance_scanner import BinanceScanner
from .database import Database
from .candle_patterns import CandlePatternDetector, CandlePatternType
from .fibonacci_analyzer import FibonacciAnalyzer
from .level_detector import LevelDetector


app = FastAPI(title="Chart Structure Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_processor = ImageProcessor()
structure_extractor = StructureExtractor()
similarity_matcher = SimilarityMatcher()
scanner = BinanceScanner(num_symbols=50)
database = Database()
candle_detector = CandlePatternDetector()
fibo_analyzer = FibonacciAnalyzer()
level_detector = LevelDetector()

active_websockets: List[WebSocket] = []
current_reference: Optional[StructureFeatures] = None
current_structure_id: Optional[int] = None
scan_threshold: float = 50.0
is_scanning: bool = False
scan_task: Optional[asyncio.Task] = None
_init_task: Optional[asyncio.Task] = None
_pending_initial_scan: Optional[asyncio.Task] = None
search_mode: str = "preset"  # "preset", "uploaded", "manual", "type_scan", "candle_scan", or "fibo_scan"
search_pattern_type: Optional[str] = None
search_type_filter: Optional[str] = None
search_candle_filter: Optional[str] = None
search_timeframe_filter: Optional[str] = None
search_fibo_min_touches: int = 3
search_fibo_min_quality: float = 30.0
type_scan_seen: set = set()
candle_scan_seen: set = set()
fibo_scan_seen: set = set()
level_scan_seen: set = set()
search_level_min_touches: int = 3


class ThresholdUpdate(BaseModel):
    threshold: float


class FeedbackRequest(BaseModel):
    match_id: int
    is_relevant: bool


class PresetRequest(BaseModel):
    preset_type: str
    min_touches: int = 3


class StartScanRequest(BaseModel):
    mode: str = "preset"
    pattern_type: Optional[str] = None
    structure_id: Optional[int] = None
    type_filter: Optional[str] = None
    candle_filter: Optional[str] = None
    timeframe_filter: Optional[str] = None
    fibo_min_touches: int = 3
    fibo_min_quality: float = 30.0
    level_min_touches: int = 3


class ManualPivot(BaseModel):
    x: float
    y: float
    isHigh: bool


class ManualStructureRequest(BaseModel):
    pivots: List[ManualPivot]


@app.on_event("startup")
async def startup():
    await database.initialize()


@app.on_event("shutdown")
async def shutdown():
    if is_scanning:
        await stop_scan()


async def broadcast_message(message: dict):
    """Send message to all connected WebSocket clients."""
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        active_websockets.remove(ws)


async def on_market_update(symbol: str, timeframe: str):
    """Callback when market data updates."""
    global current_reference, scan_threshold, search_mode, search_type_filter, search_candle_filter
    
    if search_mode == "type_scan":
        await on_market_update_type_scan(symbol, timeframe)
        return
    
    if search_mode == "candle_scan":
        await on_market_update_candle_scan(symbol, timeframe)
        return
    
    if search_mode == "fibo_scan":
        await on_market_update_fibo_scan(symbol, timeframe)
        return
    
    if search_mode == "level_scan":
        await on_market_update_level_scan(symbol, timeframe)
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
                    "mtf_confirmed": mtf_count >= 2
                }
            })


async def on_market_update_candle_scan(symbol: str, timeframe: str):
    """Callback for candle-pattern scanning."""
    global search_candle_filter, candle_scan_seen, search_timeframe_filter
    
    if not search_candle_filter:
        return
    
    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return
    
    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 5:
        return
    
    positions = candle_detector.find_all_positions(candles, pattern_filter=search_candle_filter)
    if positions:
        key = f"{symbol}_{timeframe}"
        if key in candle_scan_seen:
            return
        candle_scan_seen.add(key)
        
        closes = [c.close for c in candles[-50:]] if len(candles) >= 50 else [c.close for c in candles]
        mn, mx = min(closes), max(closes)
        rng = mx - mn if mx > mn else 1
        normalized = [(v - mn) / rng for v in closes]
        
        last_pos = positions[-1]
        pattern_time = candles[last_pos["index"]].close_time if last_pos["index"] < len(candles) else candles[-1].close_time
        
        await broadcast_message({
            "type": "match",
            "data": {
                "match_id": None,
                "symbol": symbol,
                "timeframe": timeframe,
                "similarity_score": 100.0,
                "structure_type": search_candle_filter,
                "timestamp": datetime.now().isoformat(),
                "is_mirrored": False,
                "normalized_line": normalized,
                "price_change_24h": scanner.price_change_24h.get(symbol, 0),
                "pattern_time": pattern_time,
                "is_candle_pattern": True,
                "pattern_positions": [{"index": p["index"], "time": p["time"]} for p in positions],
                "pattern_count": len(positions)
            }
        })


async def run_initial_candle_scan():
    """Run initial scan for candle patterns across all symbols/timeframes."""
    global search_candle_filter, candle_scan_seen
    
    if not search_candle_filter:
        return
    
    match_count = 0
    
    for symbol, sym_data in scanner.symbol_data.items():
        for timeframe, candles in sym_data.candles.items():
            if search_timeframe_filter and timeframe != search_timeframe_filter:
                continue
            if not candles or len(candles) < 5:
                continue
            
            positions = candle_detector.find_all_positions(candles, pattern_filter=search_candle_filter)
            if positions:
                key = f"{symbol}_{timeframe}"
                if key in candle_scan_seen:
                    continue
                candle_scan_seen.add(key)
                match_count += 1
                
                closes = [c.close for c in candles[-50:]] if len(candles) >= 50 else [c.close for c in candles]
                mn, mx = min(closes), max(closes)
                rng = mx - mn if mx > mn else 1
                normalized = [(v - mn) / rng for v in closes]
                
                last_pos = positions[-1]
                pattern_time = candles[last_pos["index"]].close_time if last_pos["index"] < len(candles) else candles[-1].close_time
                
                await broadcast_message({
                    "type": "match",
                    "data": {
                        "match_id": None,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "similarity_score": 100.0,
                        "structure_type": search_candle_filter,
                        "timestamp": datetime.now().isoformat(),
                        "is_mirrored": False,
                        "normalized_line": normalized,
                        "price_change_24h": scanner.price_change_24h.get(symbol, 0),
                        "pattern_time": pattern_time,
                        "is_candle_pattern": True,
                        "pattern_positions": [{"index": p["index"], "time": p["time"]} for p in positions],
                        "pattern_count": len(positions)
                    }
                })
    
    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": match_count}
    })


def _fibo_result_to_match_data(result, symbol, timeframe):
    """Convert FiboResult to match data dict for broadcasting."""
    from .fibonacci_analyzer import FIBO_NAMES, FiboLevel
    touch_summary = {}
    for t in result.touches:
        if isinstance(t.level, FiboLevel):
            name = FIBO_NAMES.get(t.level, str(t.level))
        else:
            name = str(t.level)
        if name not in touch_summary:
            touch_summary[name] = {"count": 0, "bounces": 0}
        touch_summary[name]["count"] += 1
        if t.is_bounce:
            touch_summary[name]["bounces"] += 1

    return {
        "match_id": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "similarity_score": result.quality_score,
        "structure_type": "fibonacci",
        "timestamp": datetime.now().isoformat(),
        "is_mirrored": False,
        "normalized_line": result.normalized_line,
        "price_change_24h": scanner.price_change_24h.get(symbol, 0),
        "pattern_time": None,
        "fibo_data": {
            "swing_high": result.swing_high,
            "swing_low": result.swing_low,
            "is_uptrend": result.is_uptrend,
            "levels": result.levels,
            "total_touches": result.total_touches,
            "unique_levels": result.unique_levels_touched,
            "best_level": result.best_level,
            "best_level_touches": result.best_level_touches,
            "touch_summary": touch_summary
        }
    }


async def on_market_update_fibo_scan(symbol: str, timeframe: str):
    """Callback for Fibonacci level scanning."""
    global fibo_scan_seen, search_timeframe_filter, search_fibo_min_touches, search_fibo_min_quality

    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return

    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 30:
        return

    key = f"{symbol}_{timeframe}"
    if key in fibo_scan_seen:
        return

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    results = fibo_analyzer.analyze(
        symbol, timeframe, closes, highs, lows,
        min_touches=search_fibo_min_touches,
        min_quality=search_fibo_min_quality
    )

    if results:
        fibo_scan_seen.add(key)
        for result in results:
            match_data = _fibo_result_to_match_data(result, symbol, timeframe)
            await broadcast_message({"type": "match", "data": match_data})


async def run_initial_fibo_scan():
    """Run initial Fibonacci scan across all symbols/timeframes."""
    global fibo_scan_seen, search_fibo_min_touches, search_fibo_min_quality

    match_count = 0

    for symbol, sym_data in scanner.symbol_data.items():
        for timeframe, candles in sym_data.candles.items():
            if search_timeframe_filter and timeframe != search_timeframe_filter:
                continue
            if not candles or len(candles) < 30:
                continue

            key = f"{symbol}_{timeframe}"
            if key in fibo_scan_seen:
                continue

            closes = [c.close for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]

            results = fibo_analyzer.analyze(
                symbol, timeframe, closes, highs, lows,
                min_touches=search_fibo_min_touches,
                min_quality=search_fibo_min_quality
            )

            if results:
                fibo_scan_seen.add(key)
                for result in results:
                    match_count += 1
                    match_data = _fibo_result_to_match_data(result, symbol, timeframe)
                    await broadcast_message({"type": "match", "data": match_data})

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
        if not scanner._poll_task or scanner._poll_task.done():
            if scanner.initialized:
                scanner._poll_task = asyncio.create_task(scanner._poll_loop())
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        is_scanning = False


async def run_initial_scan():
    """Run initial scan against all current structures."""
    global current_reference, scan_threshold, current_structure_id, search_mode, search_type_filter, search_candle_filter
    
    if search_mode == "type_scan":
        await run_initial_type_scan()
        return
    
    if search_mode == "candle_scan":
        await run_initial_candle_scan()
        return
    
    if search_mode == "fibo_scan":
        await run_initial_fibo_scan()
        return
    
    if search_mode == "level_scan":
        await run_initial_level_scan()
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
                    "detected_patterns": detected
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
    global is_scanning, scan_task, _init_task, _pending_initial_scan, current_reference, search_mode, search_pattern_type, current_structure_id, search_type_filter, type_scan_seen, search_candle_filter, candle_scan_seen, search_timeframe_filter, fibo_scan_seen, search_fibo_min_touches, search_fibo_min_quality, level_scan_seen, search_level_min_touches
    
    if data is None:
        data = StartScanRequest()
    
    search_mode = data.mode
    search_pattern_type = data.pattern_type
    search_type_filter = data.type_filter
    search_candle_filter = data.candle_filter
    search_timeframe_filter = data.timeframe_filter if data.timeframe_filter != "all" else None
    search_fibo_min_touches = data.fibo_min_touches
    search_fibo_min_quality = data.fibo_min_quality
    search_level_min_touches = data.level_min_touches
    type_scan_seen = set()
    candle_scan_seen = set()
    fibo_scan_seen = set()
    level_scan_seen = set()
    
    if data.mode == "candle_scan":
        if data.candle_filter is None:
            raise HTTPException(status_code=400, detail="candle_filter required for candle_scan mode")
        current_reference = None
        current_structure_id = None
    
    elif data.mode == "type_scan":
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
    elif data.mode == "fibo_scan":
        current_reference = None
        current_structure_id = None

    elif data.mode == "level_scan":
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
            print(f"Error fetching market movers: {e}")
    
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


@app.post("/api/feedback")
async def submit_feedback(data: FeedbackRequest):
    """Submit feedback for a match."""
    await database.save_feedback(data.match_id, data.is_relevant)
    return {"success": True}


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


@app.get("/api/candle-patterns/{symbol}/{timeframe}")
async def get_candle_pattern_positions(symbol: str, timeframe: str, pattern: str = None):
    """Get all candle pattern positions for a symbol/timeframe."""
    candles = scanner.get_candles(symbol, timeframe)
    if not candles or len(candles) < 3:
        return {"patterns": []}
    
    positions = candle_detector.find_all_positions(candles, pattern_filter=pattern)
    return {"patterns": positions}


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
        print(f"WebSocket error: {e}")
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
