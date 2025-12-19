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

active_websockets: List[WebSocket] = []
current_reference: Optional[StructureFeatures] = None
current_structure_id: Optional[int] = None
scan_threshold: float = 50.0
is_scanning: bool = False
scan_task: Optional[asyncio.Task] = None


class ThresholdUpdate(BaseModel):
    threshold: float


class FeedbackRequest(BaseModel):
    match_id: int
    is_relevant: bool


class PresetRequest(BaseModel):
    preset_type: str
    min_touches: int = 3


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
    global current_reference, scan_threshold
    
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
        if current_structure_id:
            await database.save_match(current_structure_id, match)
        
        await broadcast_message({
            "type": "match",
            "data": {
                "symbol": match.symbol,
                "timeframe": match.timeframe,
                "similarity_score": match.similarity_score,
                "structure_type": match.structure_type.value,
                "timestamp": match.timestamp,
                "is_mirrored": match.is_mirrored,
                "normalized_line": match.normalized_line
            }
        })


async def run_continuous_scan():
    """Run continuous market scanning."""
    global is_scanning
    
    try:
        await scanner.start(on_update=on_market_update)
    except asyncio.CancelledError:
        await scanner.stop()
    finally:
        is_scanning = False


async def run_initial_scan():
    """Run initial scan against all current structures."""
    global current_reference, scan_threshold, current_structure_id
    
    if current_reference is None:
        return
    
    structures = scanner.get_all_structures()
    matches = similarity_matcher.find_matches(current_reference, structures, scan_threshold)
    
    for match in matches:
        if current_structure_id:
            await database.save_match(current_structure_id, match)
        
        await broadcast_message({
            "type": "match",
            "data": {
                "symbol": match.symbol,
                "timeframe": match.timeframe,
                "similarity_score": match.similarity_score,
                "structure_type": match.structure_type.value,
                "timestamp": match.timestamp,
                "is_mirrored": match.is_mirrored,
                "normalized_line": match.normalized_line
            }
        })
    
    await broadcast_message({
        "type": "initial_scan_complete",
        "data": {"total_matches": len(matches)}
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


@app.post("/api/start-scan")
async def start_scan():
    """Start continuous market scanning."""
    global is_scanning, scan_task, current_reference
    
    if current_reference is None:
        raise HTTPException(status_code=400, detail="Please upload a chart first")
    
    if is_scanning:
        return {"status": "already_running"}
    
    is_scanning = True
    scan_task = asyncio.create_task(run_continuous_scan())
    
    await asyncio.sleep(2)
    asyncio.create_task(run_initial_scan())
    
    return {"status": "started", "num_symbols": len(scanner.symbols)}


@app.post("/api/stop-scan")
async def stop_scan():
    """Stop market scanning."""
    global is_scanning, scan_task
    
    if not is_scanning:
        return {"status": "not_running"}
    
    is_scanning = False
    if scan_task:
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    
    await scanner.stop()
    
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
    """Serve the main page."""
    return FileResponse("static/index.html")


@app.get("/{path:path}")
async def catch_all(path: str):
    """Catch-all route for SPA."""
    file_path = f"static/{path}"
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse("static/index.html")
