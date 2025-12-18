import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import websockets
import aiohttp

from .structure_extractor import StructureExtractor, StructureFeatures


@dataclass
class CandleData:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class SymbolData:
    symbol: str
    candles: Dict[str, List[CandleData]] = field(default_factory=dict)
    structures: Dict[str, Optional[StructureFeatures]] = field(default_factory=dict)
    last_update: float = 0


class BinanceScanner:
    """Scans Binance market for price structures using WebSocket."""
    
    TIMEFRAMES = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
    }
    
    CANDLE_COUNTS = {
        "1m": 100,
        "3m": 100,
        "5m": 100,
        "15m": 100,
        "30m": 100,
        "1h": 100,
    }
    
    def __init__(self, num_symbols: int = 50):
        self.num_symbols = num_symbols
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, SymbolData] = {}
        self.structure_extractor = StructureExtractor()
        self.is_running = False
        self.ws_connections: List[Any] = []
        self.on_update_callback: Optional[Callable] = None
        self.last_structures: Dict[str, Dict[str, StructureFeatures]] = {}
    
    async def fetch_top_symbols(self) -> List[str]:
        """Fetch top trading pairs by volume from Binance."""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
                
                data = await response.json()
                
                usdt_pairs = [
                    item for item in data 
                    if item['symbol'].endswith('USDT') and 
                    float(item.get('quoteVolume', 0)) > 0
                ]
                
                usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                
                return [item['symbol'] for item in usdt_pairs[:self.num_symbols]]
    
    async def fetch_historical_candles(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch historical candle data from Binance REST API."""
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                candles = []
                for item in data:
                    candles.append(CandleData(
                        open_time=item[0],
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        close_time=item[6]
                    ))
                
                return candles
    
    async def initialize_symbols(self):
        """Initialize symbol list and fetch initial candle data."""
        self.symbols = await self.fetch_top_symbols()
        
        for symbol in self.symbols:
            self.symbol_data[symbol] = SymbolData(symbol=symbol)
        
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES.keys():
                tasks.append(self._init_symbol_timeframe(symbol, timeframe))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._update_all_structures()
    
    async def _init_symbol_timeframe(self, symbol: str, timeframe: str):
        """Initialize candles for a symbol/timeframe pair."""
        try:
            candles = await self.fetch_historical_candles(
                symbol, timeframe, self.CANDLE_COUNTS[timeframe]
            )
            self.symbol_data[symbol].candles[timeframe] = candles
        except Exception as e:
            print(f"Error initializing {symbol} {timeframe}: {e}")
            self.symbol_data[symbol].candles[timeframe] = []
    
    def _update_structure(self, symbol: str, timeframe: str):
        """Update structure for a symbol/timeframe."""
        if symbol not in self.symbol_data:
            return
        
        candles = self.symbol_data[symbol].candles.get(timeframe, [])
        if len(candles) < 20:
            return
        
        closes = [c.close for c in candles]
        features = self.structure_extractor.extract_from_candles(closes)
        
        self.symbol_data[symbol].structures[timeframe] = features
        
        if symbol not in self.last_structures:
            self.last_structures[symbol] = {}
        self.last_structures[symbol][timeframe] = features
    
    def _update_all_structures(self):
        """Update structures for all symbols and timeframes."""
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES.keys():
                self._update_structure(symbol, timeframe)
    
    async def _process_kline(self, data: dict):
        """Process incoming kline/candle data from WebSocket."""
        try:
            kline = data.get('k', {})
            symbol = kline.get('s', '')
            interval = kline.get('i', '')
            is_closed = kline.get('x', False)
            
            if symbol not in self.symbol_data:
                return
            
            if interval not in self.TIMEFRAMES:
                return
            
            candle = CandleData(
                open_time=kline.get('t', 0),
                open=float(kline.get('o', 0)),
                high=float(kline.get('h', 0)),
                low=float(kline.get('l', 0)),
                close=float(kline.get('c', 0)),
                volume=float(kline.get('v', 0)),
                close_time=kline.get('T', 0)
            )
            
            candles = self.symbol_data[symbol].candles.get(interval, [])
            
            if candles and candles[-1].open_time == candle.open_time:
                candles[-1] = candle
            else:
                candles.append(candle)
                if len(candles) > self.CANDLE_COUNTS[interval]:
                    candles.pop(0)
            
            self.symbol_data[symbol].candles[interval] = candles
            self.symbol_data[symbol].last_update = time.time()
            
            if is_closed:
                self._update_structure(symbol, interval)
                
                if self.on_update_callback:
                    await self.on_update_callback(symbol, interval)
        
        except Exception as e:
            print(f"Error processing kline: {e}")
    
    def _create_stream_name(self, symbols: List[str], timeframe: str) -> str:
        """Create WebSocket stream name for multiple symbols."""
        streams = [f"{s.lower()}@kline_{timeframe}" for s in symbols]
        return "/".join(streams)
    
    async def _run_websocket(self, symbols: List[str], timeframe: str):
        """Run WebSocket connection for a set of symbols."""
        stream_names = [f"{s.lower()}@kline_{timeframe}" for s in symbols]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_names)}"
        
        while self.is_running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self.ws_connections.append(ws)
                    
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            if 'data' in data:
                                await self._process_kline(data['data'])
                        
                        except asyncio.TimeoutError:
                            await ws.ping()
                        except Exception as e:
                            print(f"WebSocket receive error: {e}")
                            break
            
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def start(self, on_update: Optional[Callable] = None):
        """Start the scanner."""
        if self.is_running:
            return
        
        self.is_running = True
        self.on_update_callback = on_update
        
        await self.initialize_symbols()
        
        batch_size = 10
        tasks = []
        
        for timeframe in self.TIMEFRAMES.keys():
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                tasks.append(asyncio.create_task(self._run_websocket(batch, timeframe)))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the scanner."""
        self.is_running = False
        
        for ws in self.ws_connections:
            try:
                await ws.close()
            except:
                pass
        
        self.ws_connections.clear()
    
    def get_all_structures(self) -> List[tuple]:
        """Get all current structures for matching."""
        results = []
        timestamp = datetime.now().isoformat()
        
        for symbol, data in self.symbol_data.items():
            for timeframe, features in data.structures.items():
                if features is not None:
                    results.append((symbol, timeframe, features, timestamp))
        
        return results
    
    def get_symbol_chart_data(self, symbol: str, timeframe: str) -> List[dict]:
        """Get candle data for chart display."""
        if symbol not in self.symbol_data:
            return []
        
        candles = self.symbol_data[symbol].candles.get(timeframe, [])
        return [
            {
                "time": c.open_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close
            }
            for c in candles
        ]
