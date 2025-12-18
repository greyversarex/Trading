# Chart Structure Scanner

A Python-based real-time system for detecting visually similar price structures on cryptocurrency charts.

## Overview

This tool continuously scans Binance cryptocurrency markets and finds charts whose price structure is visually similar to a structure provided by the user via a screenshot. It focuses on **structural visual similarity**, not numerical indicators.

## Project Architecture

```
├── src/backend/
│   ├── main.py              # FastAPI application with WebSocket support
│   ├── image_processor.py   # OpenCV-based image processing for screenshots
│   ├── structure_extractor.py # Pivot detection and feature extraction
│   ├── similarity_matcher.py  # Geometric similarity matching
│   ├── binance_scanner.py     # Real-time Binance WebSocket integration
│   └── database.py            # SQLite storage for structures/matches
├── static/
│   └── index.html           # Frontend SPA with Tailwind CSS
├── data/                    # SQLite database storage
├── uploads/                 # Uploaded screenshots
└── run.py                   # Application entry point
```

## Core Features

### 1. Screenshot Upload
- Accepts chart screenshots from any platform
- Automatically detects and isolates the price line using OpenCV
- Removes noise (UI, grids, axes, labels)

### 2. Structure Extraction
- Extracts pivot points (local highs/lows)
- Calculates relative geometry between points
- Measures trend tendency, compression/consolidation behavior
- Classification: Compression, Accumulation, Triangle, Range, Retest, Trend

### 3. Live Market Scanning
- Connects to Binance WebSocket API
- Monitors top 50 trading pairs by volume
- Multi-timeframe analysis: 1m, 5m, 15m, 1h

### 4. Similarity Matching
- Structure-to-structure comparison (not image-to-image)
- Geometric and relational similarity
- Mirror-invariant (bullish/bearish symmetry)
- Normalized score (0-100%)
- User-configurable threshold

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, NumPy, SciPy, scikit-learn
- **Frontend**: HTML, Tailwind CSS, Chart.js, WebSocket
- **Database**: SQLite (via aiosqlite)
- **Market Data**: Binance WebSocket API

## Running the Application

```bash
python run.py
```

The server runs on port 5000.

## API Endpoints

- `POST /api/upload` - Upload chart screenshot
- `POST /api/start-scan` - Start continuous scanning
- `POST /api/stop-scan` - Stop scanning
- `POST /api/threshold` - Update similarity threshold
- `GET /api/status` - Get scanner status
- `GET /api/structures` - List saved structures
- `GET /api/matches/{id}` - Get matches for a structure
- `WS /ws` - WebSocket for real-time updates

## Recent Changes

- 2024-12: Initial implementation with all core MVP features
  - Image processing pipeline with OpenCV
  - Structure extraction with pivot detection
  - Binance WebSocket integration
  - Real-time similarity matching
  - Web interface with live updates
