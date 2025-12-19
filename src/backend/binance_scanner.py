import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
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
    """Scans Binance market for price structures using REST API polling."""
    
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
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, num_symbols: int = 50):
        self.num_symbols = num_symbols
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, SymbolData] = {}
        self.structure_extractor = StructureExtractor()
        self.is_running = False
        self.on_update_callback: Optional[Callable] = None
        self.last_structures: Dict[str, Dict[str, StructureFeatures]] = {}
        self._poll_task: Optional[asyncio.Task] = None
    
    async def fetch_top_symbols(self) -> List[str]:
        """Fetch top trading pairs by volume from Binance."""
        url = f"{self.BASE_URL}/api/v3/ticker/24hr"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"Error fetching symbols: {response.status}")
                        return self._get_default_symbols()
                    
                    data = await response.json()
                    
                    usdt_pairs = [
                        item for item in data 
                        if item['symbol'].endswith('USDT') and 
                        float(item.get('quoteVolume', 0)) > 0
                    ]
                    
                    usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                    
                    return [item['symbol'] for item in usdt_pairs[:self.num_symbols]]
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return self._get_default_symbols()
    
    def _get_default_symbols(self) -> List[str]:
        """Return default symbols if API fails."""
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "XLMUSDT",
            "TRXUSDT", "NEARUSDT", "APTUSDT", "FILUSDT", "ARBUSDT"
        ]
    
    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candle data from Binance REST API."""
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
        except Exception as e:
            print(f"Error fetching candles for {symbol} {interval}: {e}")
            return []
    
    async def initialize_symbols(self):
        """Initialize symbol list and fetch initial candle data."""
        print("Fetching top symbols...")
        self.symbols = await self.fetch_top_symbols()
        print(f"Got {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            self.symbol_data[symbol] = SymbolData(symbol=symbol)
        
        await self._update_all_candles()
        self._update_all_structures()
        print(f"Initialized {len(self.symbols)} symbols with structures")
    
    async def _update_all_candles(self):
        """Fetch candles for all symbols and timeframes."""
        batch_size = 5
        
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            tasks = []
            
            for symbol in batch:
                for timeframe in self.TIMEFRAMES.keys():
                    tasks.append(self._update_symbol_candles(symbol, timeframe))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.5)
    
    async def _update_symbol_candles(self, symbol: str, timeframe: str):
        """Update candles for a specific symbol/timeframe."""
        try:
            candles = await self.fetch_candles(symbol, timeframe, self.CANDLE_COUNTS[timeframe])
            if candles:
                self.symbol_data[symbol].candles[timeframe] = candles
                self.symbol_data[symbol].last_update = time.time()
        except Exception as e:
            print(f"Error updating {symbol} {timeframe}: {e}")
    
    def _update_structure(self, symbol: str, timeframe: str) -> bool:
        """Update structure for a symbol/timeframe. Returns True if structure changed."""
        if symbol not in self.symbol_data:
            return False
        
        candles = self.symbol_data[symbol].candles.get(timeframe, [])
        if len(candles) < 20:
            return False
        
        closes = [c.close for c in candles]
        features = self.structure_extractor.extract_from_candles(closes)
        
        old_features = self.symbol_data[symbol].structures.get(timeframe)
        self.symbol_data[symbol].structures[timeframe] = features
        
        if symbol not in self.last_structures:
            self.last_structures[symbol] = {}
        self.last_structures[symbol][timeframe] = features
        
        return old_features is None or (
            old_features.structure_type != features.structure_type or
            abs(old_features.trend_direction - features.trend_direction) > 0.1
        )
    
    def _update_all_structures(self):
        """Update structures for all symbols and timeframes."""
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES.keys():
                self._update_structure(symbol, timeframe)
    
    async def _poll_loop(self):
        """Main polling loop for updating candle data."""
        while self.is_running:
            try:
                await self._update_all_candles()
                
                for symbol in self.symbols:
                    for timeframe in self.TIMEFRAMES.keys():
                        changed = self._update_structure(symbol, timeframe)
                        if changed and self.on_update_callback:
                            await self.on_update_callback(symbol, timeframe)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Error in poll loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self, on_update: Optional[Callable] = None):
        """Start the scanner."""
        if self.is_running:
            return
        
        self.is_running = True
        self.on_update_callback = on_update
        
        await self.initialize_symbols()
        
        self._poll_task = asyncio.create_task(self._poll_loop())
    
    async def stop(self):
        """Stop the scanner."""
        self.is_running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
    
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
