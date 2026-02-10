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

active_websockets: List[WebSocket] = []
current_reference: Optional[StructureFeatures] = None
current_structure_id: Optional[int] = None
scan_threshold: float = 50.0
is_scanning: bool = False
scan_task: Optional[asyncio.Task] = None
_init_task: Optional[asyncio.Task] = None
_pending_initial_scan: Optional[asyncio.Task] = None
search_mode: str = "preset"  # "preset", "uploaded", "manual", "type_scan", or "candle_scan"
search_pattern_type: Optional[str] = None
search_type_filter: Optional[str] = None
search_candle_filter: Optional[str] = None
search_timeframe_filter: Optional[str] = None
type_scan_seen: set = set()
candle_scan_seen: set = set()


class ThresholdUpdate(BaseModel):
    threshold: float


class FeedbackRequest(BaseModel):
    match_id: int
    is_relevant: bool


class PresetRequest(BaseModel):
    preset_type: str
    min_touches: int = 3


class StartScanRequest(BaseModel):
    mode: str = "preset"  # "preset", "uploaded", "manual", "type_scan", or "candle_scan"
    pattern_type: Optional[str] = None
    structure_id: Optional[int] = None
    type_filter: Optional[str] = None
    candle_filter: Optional[str] = None
    timeframe_filter: Optional[str] = None


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
    
    if current_reference is None:
        return
    
    structures = scanner.get_all_structures()
    
    relevant = [(s, tf, f, ts) for s, tf, f, ts in structures 
                if s == symbol and tf == timeframe]
    
    if not relevant:
        return
    
    matches = similarity_matcher.find_matches(
        current_reference, relevant, scan_threshold
    )
    
    for match in matches:
        match_id = None
        if current_structure_id:
            match_id = await database.save_match(current_structure_id, match)
        
        await broadcast_message({
            "type": "match",
            "data": {
                "match_id": match_id,
                "symbol": match.symbol,
                "timeframe": match.timeframe,
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
    """Callback for type-based scanning - no reference needed, just classify and filter."""
    global search_type_filter, type_scan_seen, search_timeframe_filter
    
    if not search_type_filter:
        return
    
    if search_timeframe_filter and timeframe != search_timeframe_filter:
        return
    
    structures = scanner.get_all_structures()
    relevant = [(s, tf, f, ts, ct) for s, tf, f, ts, ct in structures 
                if s == symbol and tf == timeframe]
    
    for sym, tf, features, timestamp, candle_time in relevant:
        if features is None:
            continue
        if features.structure_type.value == search_type_filter:
            key = f"{sym}_{tf}"
            if key in type_scan_seen:
                continue
            type_scan_seen.add(key)
            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": sym,
                    "timeframe": tf,
                    "similarity_score": 100.0,
                    "structure_type": features.structure_type.value,
                    "timestamp": timestamp,
                    "is_mirrored": False,
                    "normalized_line": features.normalized_line.tolist(),
                    "price_change_24h": scanner.price_change_24h.get(sym, 0),
                    "pattern_time": candle_time
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
    
    patterns = candle_detector.detect_all(candles)
    for pat in patterns:
        if pat.value == search_candle_filter:
            key = f"{symbol}_{timeframe}"
            if key in candle_scan_seen:
                continue
            candle_scan_seen.add(key)
            
            closes = [c.close for c in candles[-50:]] if len(candles) >= 50 else [c.close for c in candles]
            mn, mx = min(closes), max(closes)
            rng = mx - mn if mx > mn else 1
            normalized = [(v - mn) / rng for v in closes]
            
            pattern_time = candles[-1].close_time
            
            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "similarity_score": 100.0,
                    "structure_type": pat.value,
                    "timestamp": datetime.now().isoformat(),
                    "is_mirrored": False,
                    "normalized_line": normalized,
                    "price_change_24h": scanner.price_change_24h.get(symbol, 0),
                    "pattern_time": pattern_time
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
            
            patterns = candle_detector.detect_all(candles)
            for pat in patterns:
                if pat.value == search_candle_filter:
                    key = f"{symbol}_{timeframe}"
                    if key in candle_scan_seen:
                        continue
                    candle_scan_seen.add(key)
                    match_count += 1
                    
                    closes = [c.close for c in candles[-50:]] if len(candles) >= 50 else [c.close for c in candles]
                    mn, mx = min(closes), max(closes)
                    rng = mx - mn if mx > mn else 1
                    normalized = [(v - mn) / rng for v in closes]
                    
                    pattern_time = candles[-1].close_time
                    
                    await broadcast_message({
                        "type": "match",
                        "data": {
                            "match_id": None,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "similarity_score": 100.0,
                            "structure_type": pat.value,
                            "timestamp": datetime.now().isoformat(),
                            "is_mirrored": False,
                            "normalized_line": normalized,
                            "price_change_24h": scanner.price_change_24h.get(symbol, 0),
                            "pattern_time": pattern_time
                        }
                    })
    
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
    
    if current_reference is None:
        return
    
    structures = scanner.get_all_structures()
    matches = similarity_matcher.find_matches(current_reference, structures, scan_threshold)
    
    for match in matches:
        match_id = None
        if current_structure_id:
            match_id = await database.save_match(current_structure_id, match)
        
        await broadcast_message({
            "type": "match",
            "data": {
                "match_id": match_id,
                "symbol": match.symbol,
                "timeframe": match.timeframe,
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
    """Run initial scan by structure type - no reference needed."""
    global search_type_filter, search_timeframe_filter
    
    if not search_type_filter:
        return
    
    structures = scanner.get_all_structures()
    match_count = 0
    
    for sym, tf, features, timestamp, candle_time in structures:
        if features is None:
            continue
        if search_timeframe_filter and tf != search_timeframe_filter:
            continue
        if features.structure_type.value == search_type_filter:
            match_count += 1
            await broadcast_message({
                "type": "match",
                "data": {
                    "match_id": None,
                    "symbol": sym,
                    "timeframe": tf,
                    "similarity_score": 100.0,
                    "structure_type": features.structure_type.value,
                    "timestamp": timestamp,
                    "is_mirrored": False,
                    "normalized_line": features.normalized_line.tolist(),
                    "price_change_24h": scanner.price_change_24h.get(sym, 0),
                    "pattern_time": candle_time
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
    global is_scanning, scan_task, _init_task, _pending_initial_scan, current_reference, search_mode, search_pattern_type, current_structure_id, search_type_filter, type_scan_seen, search_candle_filter, candle_scan_seen, search_timeframe_filter
    
    if data is None:
        data = StartScanRequest()
    
    search_mode = data.mode
    search_pattern_type = data.pattern_type
    search_type_filter = data.type_filter
    search_candle_filter = data.candle_filter
    search_timeframe_filter = data.timeframe_filter if data.timeframe_filter != "all" else None
    type_scan_seen = set()
    candle_scan_seen = set()
    
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
    elif data.mode == "manual":
        if current_reference is None:
            raise HTTPException(status_code=400, detail="No manual structure created. Create one first.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'preset', 'uploaded', 'manual', 'type_scan', or 'candle_scan'")
    
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
