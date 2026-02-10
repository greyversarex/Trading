import asyncio
import time
import random
import math
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
    """Scans crypto market for price structures using multiple API sources."""
    
    TIMEFRAMES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    
    TIMEFRAME_LIMITS = {
        "1m": 100,
        "5m": 100,
        "15m": 100,
        "1h": 100,
        "4h": 100,
        "1d": 100,
    }
    
    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
    BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
    
    BINANCE_TF_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    
    working_endpoint: str = "binance"
    data_available: bool = False
    last_error: str = None
    
    def __init__(self, num_symbols: int = 50):
        self.num_symbols = min(num_symbols, 30)
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, SymbolData] = {}
        self.structure_extractor = StructureExtractor()
        self.is_running = False
        self.on_update_callback: Optional[Callable] = None
        self.last_structures: Dict[str, Dict[str, StructureFeatures]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._base_prices: Dict[str, float] = {}
        self.price_change_24h: Dict[str, float] = {}
        self._use_real_data: bool = True
        self._api_failures: int = 0
        self.initialized: bool = False
    
    def _get_default_symbols(self) -> List[str]:
        """Return default crypto symbols."""
        symbols = [
            "BTC", "ETH", "BNB", "SOL", "XRP",
            "ADA", "DOGE", "AVAX", "DOT", "MATIC",
            "LINK", "LTC", "ATOM", "UNI", "XLM",
            "TRX", "NEAR", "APT", "FIL", "ARB",
            "OP", "INJ", "SUI", "SEI", "TIA",
            "PEPE", "SHIB", "BONK", "WIF", "FLOKI"
        ]
        return symbols[:self.num_symbols]
    
    def _get_base_price(self, symbol: str) -> float:
        """Get a realistic base price for a symbol."""
        prices = {
            "BTC": 45000, "ETH": 2500, "BNB": 300, "SOL": 100, "XRP": 0.55,
            "ADA": 0.50, "DOGE": 0.08, "AVAX": 35, "DOT": 7, "MATIC": 0.85,
            "LINK": 15, "LTC": 70, "ATOM": 10, "UNI": 6, "XLM": 0.12,
            "TRX": 0.11, "NEAR": 3.5, "APT": 9, "FIL": 5, "ARB": 1.2,
            "OP": 2.5, "INJ": 25, "SUI": 1.5, "SEI": 0.5, "TIA": 12,
            "PEPE": 0.000001, "SHIB": 0.00001, "BONK": 0.00001, "WIF": 2, "FLOKI": 0.0001
        }
        return prices.get(symbol, 10.0)
    
    def _generate_realistic_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[CandleData]:
        """Generate realistic-looking price candles with various patterns."""
        if symbol not in self._base_prices:
            self._base_prices[symbol] = self._get_base_price(symbol)
        
        base_price = self._base_prices[symbol]
        candles = []
        current_time = int(time.time() * 1000)
        tf_minutes = self.TIMEFRAMES.get(timeframe, 1)
        interval_ms = tf_minutes * 60 * 1000
        
        pattern_seed = hash(f"{symbol}_{timeframe}") % 100
        
        if pattern_seed < 15:
            pattern_type = "triangle_up"
        elif pattern_seed < 30:
            pattern_type = "triangle_down"
        elif pattern_seed < 45:
            pattern_type = "trend_up"
        elif pattern_seed < 60:
            pattern_type = "trend_down"
        elif pattern_seed < 75:
            pattern_type = "range"
        elif pattern_seed < 90:
            pattern_type = "compression"
        else:
            pattern_type = "retest"
        
        price = base_price
        volatility = base_price * 0.02
        
        for i in range(limit):
            t = i / limit
            
            if pattern_type == "triangle_up":
                trend = math.sin(t * 8 * math.pi) * (0.5 - t) * 0.3
                drift = t * 0.1
            elif pattern_type == "triangle_down":
                trend = math.sin(t * 8 * math.pi) * t * 0.3
                drift = -t * 0.05
            elif pattern_type == "trend_up":
                trend = t * 0.15 + math.sin(t * 6 * math.pi) * 0.02
                drift = 0
            elif pattern_type == "trend_down":
                trend = -t * 0.15 + math.sin(t * 6 * math.pi) * 0.02
                drift = 0
            elif pattern_type == "range":
                trend = math.sin(t * 8 * math.pi) * 0.08
                drift = 0
            elif pattern_type == "compression":
                decay = math.exp(-t * 2)
                trend = math.sin(t * 10 * math.pi) * 0.15 * decay
                drift = 0
            else:
                trend = math.sin(t * 4 * math.pi) * 0.05
                drift = 0.02 if t > 0.7 else 0
            
            noise = random.gauss(0, 0.01)
            price_change = (trend + drift + noise) * base_price
            
            open_price = price
            close_price = price + price_change * 0.3
            
            high_price = max(open_price, close_price) + abs(random.gauss(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(random.gauss(0, volatility * 0.5))
            
            price = close_price
            
            candle_time = current_time - (limit - i) * interval_ms
            
            candles.append(CandleData(
                open_time=candle_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(1000, 10000) * base_price,
                close_time=candle_time + interval_ms - 1
            ))
        
        self._base_prices[symbol] = price
        
        return candles
    
    async def _fetch_binance_24h_tickers(self) -> Dict[str, float]:
        """Fetch 24h price changes from Binance."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BINANCE_TICKER_URL,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        changes = {}
                        for item in data:
                            sym = item.get('symbol', '')
                            if sym.endswith('USDT'):
                                base = sym.replace('USDT', '')
                                pct = float(item.get('priceChangePercent', 0))
                                changes[base] = round(pct, 2)
                        return changes
        except Exception as e:
            print(f"Binance 24h ticker error: {e}")
        return {}
    
    STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "PYUSD", "GUSD", "FRAX", "USDS", "USDE"}
    
    COINGECKO_TO_BINANCE = {
        "WBT": None,
        "FIGR_HELOC": None,
        "WETH": None,
        "STETH": None,
        "WSTETH": None,
        "LEO": None,
        "BTCB": None,
        "WEETH": None,
        "SUSDE": None,
        "CBBTC": None,
    }
    
    def _is_valid_binance_symbol(self, symbol: str) -> bool:
        """Check if a symbol is likely valid on Binance as SYMBOLUSDT pair."""
        if symbol in self.STABLECOINS:
            return False
        if symbol in self.COINGECKO_TO_BINANCE and self.COINGECKO_TO_BINANCE[symbol] is None:
            return False
        if "_" in symbol or len(symbol) > 10:
            return False
        return True
    
    async def fetch_top_symbols(self) -> List[str]:
        """Get list of symbols to scan."""
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(100, self.num_symbols * 3),
                "page": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        symbols = []
                        for item in data:
                            sym = item['symbol'].upper()
                            change = item.get('price_change_percentage_24h', 0)
                            self.price_change_24h[sym] = round(change, 2) if change else 0
                            if self._is_valid_binance_symbol(sym) and len(symbols) < self.num_symbols:
                                symbols.append(sym)
                        self.data_available = True
                        print(f"Got {len(symbols)} valid Binance symbols from CoinGecko")
                        return symbols
        except Exception as e:
            print(f"CoinGecko API error: {e}, trying Binance tickers")
        
        try:
            changes = await self._fetch_binance_24h_tickers()
            if changes:
                self.price_change_24h = changes
        except:
            pass
        
        self.data_available = True
        return self._get_default_symbols()
    
    async def _fetch_binance_klines(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch real kline data from Binance public API."""
        binance_symbol = f"{symbol}USDT"
        binance_interval = self.BINANCE_TF_MAP.get(interval, interval)
        
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "limit": limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BINANCE_KLINES_URL, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = []
                        for k in data:
                            candles.append(CandleData(
                                open_time=int(k[0]),
                                open=float(k[1]),
                                high=float(k[2]),
                                low=float(k[3]),
                                close=float(k[4]),
                                volume=float(k[5]),
                                close_time=int(k[6])
                            ))
                        return candles
                    else:
                        error_text = await response.text()
                        print(f"Binance API error for {binance_symbol} {interval}: {response.status} - {error_text}")
                        return []
        except Exception as e:
            print(f"Binance klines fetch error for {symbol} {interval}: {e}")
            return []
    
    _invalid_symbols: set = set()
    
    async def _fetch_candles_session(self, session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candles using a shared session for batch operations."""
        if self._use_real_data and symbol not in self._invalid_symbols:
            binance_symbol = f"{symbol}USDT"
            binance_interval = self.BINANCE_TF_MAP.get(interval, interval)
            params = {"symbol": binance_symbol, "interval": binance_interval, "limit": limit}
            try:
                async with session.get(self.BINANCE_KLINES_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = []
                        for k in data:
                            candles.append(CandleData(
                                open_time=int(k[0]), open=float(k[1]), high=float(k[2]),
                                low=float(k[3]), close=float(k[4]), volume=float(k[5]),
                                close_time=int(k[6])
                            ))
                        return candles
                    else:
                        self._invalid_symbols.add(symbol)
            except Exception:
                self._invalid_symbols.add(symbol)
        return self._generate_realistic_candles(symbol, interval, limit)

    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candle data from Binance API, fall back to simulation if unavailable."""
        if self._use_real_data and symbol not in self._invalid_symbols:
            candles = await self._fetch_binance_klines(symbol, interval, limit)
            if candles:
                return candles
            else:
                self._invalid_symbols.add(symbol)
                print(f"Marking {symbol} as unavailable on Binance, using simulation")
        
        return self._generate_realistic_candles(symbol, interval, limit)
    
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

    async def initialize_symbols(self, progress_callback=None):
        """Initialize symbol list and fetch initial candle data."""
        print("Initializing scanner...")
        self._invalid_symbols = set()
        self.symbols = await self.fetch_top_symbols()
        print(f"Loaded {len(self.symbols)} symbols: {', '.join(self.symbols[:10])}...")
        
        for symbol in self.symbols:
            self.symbol_data[symbol] = SymbolData(symbol=symbol)
        
        print(f"Fetching real market data from Binance API...")
        await self._update_all_candles(progress_callback=progress_callback)
        self._update_all_structures()
        
        real_symbols = set()
        sim_symbols = set()
        for sym in self.symbols:
            if sym in self._invalid_symbols:
                sim_symbols.add(sym)
            else:
                real_symbols.add(sym)
        
        if real_symbols and self._use_real_data:
            self.working_endpoint = "binance"
            print(f"REAL data for {len(real_symbols)} symbols, simulated for {len(sim_symbols)}")
            if sim_symbols:
                print(f"Simulated symbols: {', '.join(sorted(sim_symbols))}")
        else:
            self.working_endpoint = "simulation"
            print(f"Using simulated data")
        
        stats = self.get_structure_stats()
        print(f"Initialized: {stats['symbols_with_data']}/{stats['total_symbols']} symbols with {stats['total_structures']} structures")
    
    async def _update_all_candles(self, progress_callback=None):
        """Fetch candles for all symbols and timeframes using concurrent batches."""
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES.keys():
                tasks.append((symbol, timeframe))
        
        total = len(tasks)
        done = 0
        batch_size = 10
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for i in range(0, total, batch_size):
                batch = tasks[i:i+batch_size]
                coros = [self._update_symbol_candles_session(session, sym, tf) for sym, tf in batch]
                await asyncio.gather(*coros, return_exceptions=True)
                done += len(batch)
                if progress_callback:
                    await progress_callback(done, total)
                if self._use_real_data:
                    await asyncio.sleep(0.15)
    
    async def _update_symbol_candles_session(self, session: aiohttp.ClientSession, symbol: str, timeframe: str):
        """Update candles for a specific symbol/timeframe using shared session."""
        try:
            limit = self.TIMEFRAME_LIMITS.get(timeframe, 100)
            candles = await self._fetch_candles_session(session, symbol, timeframe, limit)
            if candles:
                self.symbol_data[symbol].candles[timeframe] = candles
                self.symbol_data[symbol].last_update = time.time()
        except Exception as e:
            pass

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
            features is not None and old_features is not None and (
                old_features.structure_type != features.structure_type or
                abs(old_features.trend_direction - features.trend_direction) > 0.1
            )
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
                await asyncio.sleep(60)
                
                await self._update_all_candles()
                
                for symbol in self.symbols:
                    for timeframe in self.TIMEFRAMES.keys():
                        changed = self._update_structure(symbol, timeframe)
                        if changed and self.on_update_callback:
                            await self.on_update_callback(symbol, timeframe)
                
            except Exception as e:
                print(f"Error in poll loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self, on_update: Optional[Callable] = None, progress_callback=None):
        """Start the scanner."""
        if self.is_running:
            return
        
        self.is_running = True
        self.initialized = False
        self.on_update_callback = on_update
        
        await self.initialize_symbols(progress_callback=progress_callback)
        self.initialized = True
        
        self._poll_task = asyncio.create_task(self._poll_loop())
    
    async def stop(self):
        """Stop the scanner."""
        self.is_running = False
        self.initialized = False
        
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
                    candles = data.candles.get(timeframe, [])
                    last_candle_time = candles[-1].close_time if candles else None
                    results.append((symbol, timeframe, features, timestamp, last_candle_time))
        
        return results
    
    def get_candles(self, symbol: str, timeframe: str):
        """Get raw candle data for a symbol/timeframe."""
        if symbol not in self.symbol_data:
            return []
        return self.symbol_data[symbol].candles.get(timeframe, [])
    
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
