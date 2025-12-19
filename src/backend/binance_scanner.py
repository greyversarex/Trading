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
    """Scans crypto market for price structures using CryptoCompare API."""
    
    TIMEFRAMES = {
        "1m": "histominute",
        "5m": "histominute",
        "15m": "histominute",
        "1h": "histohour",
        "4h": "histohour",
        "1d": "histoday",
    }
    
    TIMEFRAME_LIMITS = {
        "1m": 100,
        "5m": 100,
        "15m": 100,
        "1h": 100,
        "4h": 100,
        "1d": 100,
    }
    
    TIMEFRAME_AGGREGATE = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 1,
        "4h": 4,
        "1d": 1,
    }
    
    BASE_URL = "https://min-api.cryptocompare.com/data/v2"
    working_endpoint: str = "https://min-api.cryptocompare.com"
    data_available: bool = False
    last_error: str = None
    
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
        """Fetch top crypto symbols by volume from CryptoCompare."""
        url = "https://min-api.cryptocompare.com/data/top/totalvolfull"
        params = {
            "limit": self.num_symbols,
            "tsym": "USD"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"Error fetching symbols: {response.status}")
                        return self._get_default_symbols()
                    
                    data = await response.json()
                    
                    if data.get("Response") == "Error":
                        print(f"CryptoCompare error: {data.get('Message')}")
                        return self._get_default_symbols()
                    
                    self.data_available = True
                    self.working_endpoint = "https://min-api.cryptocompare.com"
                    
                    symbols = []
                    for item in data.get("Data", []):
                        coin_info = item.get("CoinInfo", {})
                        symbol = coin_info.get("Name")
                        if symbol:
                            symbols.append(symbol)
                    
                    print(f"Got {len(symbols)} symbols from CryptoCompare")
                    return symbols if symbols else self._get_default_symbols()
                    
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            self.last_error = str(e)
            return self._get_default_symbols()
    
    def _get_default_symbols(self) -> List[str]:
        """Return default symbols if API fails."""
        return [
            "BTC", "ETH", "BNB", "SOL", "XRP",
            "ADA", "DOGE", "AVAX", "DOT", "MATIC",
            "LINK", "LTC", "ATOM", "UNI", "XLM",
            "TRX", "NEAR", "APT", "FIL", "ARB"
        ]
    
    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candle data from CryptoCompare API."""
        endpoint = self.TIMEFRAMES.get(interval, "histominute")
        aggregate = self.TIMEFRAME_AGGREGATE.get(interval, 1)
        
        url = f"{self.BASE_URL}/{endpoint}"
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": limit,
            "aggregate": aggregate
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    if data.get("Response") == "Error":
                        return []
                    
                    candles = []
                    for item in data.get("Data", {}).get("Data", []):
                        candles.append(CandleData(
                            open_time=item["time"] * 1000,
                            open=float(item["open"]),
                            high=float(item["high"]),
                            low=float(item["low"]),
                            close=float(item["close"]),
                            volume=float(item.get("volumefrom", 0)),
                            close_time=item["time"] * 1000
                        ))
                    
                    if candles:
                        self.data_available = True
                    return candles
        except Exception as e:
            print(f"Error fetching candles for {symbol} {interval}: {e}")
            return []
    
    def get_structure_stats(self) -> dict:
        """Get statistics about loaded structures."""
        total_structures = 0
        symbols_with_data = 0
        
        for symbol, data in self.symbol_data.items():
            symbol_has_structure = False
            for tf, structure in data.structures.items():
                if structure is not None:
                    total_structures += 1
                    symbol_has_structure = True
            if symbol_has_structure:
                symbols_with_data += 1
        
        return {
            "total_symbols": len(self.symbols),
            "symbols_with_data": symbols_with_data,
            "total_structures": total_structures,
            "data_available": self.data_available,
            "working_endpoint": self.working_endpoint,
            "last_error": self.last_error
        }

    async def initialize_symbols(self):
        """Initialize symbol list and fetch initial candle data."""
        print("Fetching top symbols...")
        self.symbols = await self.fetch_top_symbols()
        print(f"Got {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            self.symbol_data[symbol] = SymbolData(symbol=symbol)
        
        await self._update_all_candles()
        self._update_all_structures()
        
        stats = self.get_structure_stats()
        print(f"Initialized: {stats['symbols_with_data']}/{stats['total_symbols']} symbols with {stats['total_structures']} structures")
        if stats['last_error']:
            print(f"Warning: {stats['last_error']}")
    
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
            limit = self.TIMEFRAME_LIMITS.get(timeframe, 100)
            candles = await self.fetch_candles(symbol, timeframe, limit)
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
