import asyncio
import logging
import time
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import aiohttp

from .structure_extractor import StructureExtractor, StructureFeatures
from .config import CONFIG

logger = logging.getLogger(__name__)


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
    
    TIMEFRAMES = dict(CONFIG.data.timframes)

    TIMEFRAME_LIMITS = {tf: CONFIG.data.default_limit for tf in CONFIG.data.timframes}
    
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
    BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
    
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
    
    def __init__(self, num_symbols: int = None):
        if num_symbols is None:
            num_symbols = CONFIG.data.num_symbols
        self.num_symbols = min(num_symbols, CONFIG.data.num_symbols)
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, SymbolData] = {}
        self.structure_extractor = StructureExtractor()
        self.is_running = False
        self.on_update_callback: Optional[Callable] = None
        self.last_structures: Dict[str, Dict[str, StructureFeatures]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._base_prices: Dict[str, float] = {}
        self.price_change_24h: Dict[str, float] = {}
        # Суточный объём торгов в USD по символам (для фильтра ликвидности).
        self.quote_volume_24h: Dict[str, float] = {}
        self.min_quote_volume_24h: float = CONFIG.data.min_quote_volume_24h
        self._use_real_data: bool = True
        self._api_failures: int = 0
        self.initialized: bool = False
        self._candle_hashes: Dict[str, int] = {}
        # Real-time WebSocket feed (Phase 3.1); None при REST-режиме/фаллбэке.
        self.ws_feed: Optional[Any] = None
    
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
                                try:
                                    self.quote_volume_24h[base] = float(
                                        item.get('quoteVolume', 0)
                                    )
                                except (TypeError, ValueError):
                                    pass
                        return changes
        except Exception as e:
            logger.warning(f"Binance 24h ticker error: {e}")
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
                            volume = item.get('total_volume', 0) or 0
                            self.quote_volume_24h[sym] = float(volume)
                            # Отсекаем неликвидные монеты (объём < порога).
                            if volume < self.min_quote_volume_24h:
                                continue
                            if self._is_valid_binance_symbol(sym) and len(symbols) < self.num_symbols:
                                symbols.append(sym)
                        self.data_available = True
                        logger.info(
                            f"Got {len(symbols)} valid Binance symbols from CoinGecko "
                            f"(volume >= ${self.min_quote_volume_24h:,.0f})"
                        )
                        return symbols
        except Exception as e:
            logger.warning(f"CoinGecko API error: {e}, trying Binance tickers")
        
        try:
            changes = await self._fetch_binance_24h_tickers()
            if changes:
                self.price_change_24h = changes
                # Строим список из ликвидных пар Binance (объём >= порога),
                # чтобы фильтр действовал и при недоступности CoinGecko.
                symbols = self._liquid_symbols_from_volume()
                if symbols:
                    self.data_available = True
                    logger.info(
                        f"Got {len(symbols)} valid Binance symbols from tickers "
                        f"(volume >= ${self.min_quote_volume_24h:,.0f})"
                    )
                    return symbols
        except Exception as e:
            logger.debug(f"Binance 24h tickers fetch failed: {e}")
        
        self.data_available = True
        logger.warning(
            "Оба источника недоступны: используются дефолтные символы "
            "(degraded-режим, ликвидностный фильтр не применяется)."
        )
        return self._get_default_symbols()

    def _liquid_symbols_from_volume(self) -> List[str]:
        """Ликвидные пары из собранных объёмов ``quote_volume_24h``.

        Возвращает валидные для Binance символы с суточным объёмом не ниже
        ``min_quote_volume_24h``, отсортированные по объёму (самые торгуемые
        первыми), не более ``num_symbols`` штук.
        """
        liquid = [
            (sym, vol)
            for sym, vol in self.quote_volume_24h.items()
            if vol >= self.min_quote_volume_24h and self._is_valid_binance_symbol(sym)
        ]
        liquid.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in liquid[: self.num_symbols]]
    
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
                        logger.warning(f"Binance API error for {binance_symbol} {interval}: {response.status} - {error_text}")
                        return []
        except Exception as e:
            logger.warning(f"Binance klines fetch error for {symbol} {interval}: {e}")
            return []
    
    _invalid_symbols: set = set()
    
    async def _fetch_candles_session(self, session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candles using a shared session for batch operations."""
        if symbol in self._invalid_symbols:
            return []
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
                    logger.debug(f"Symbol {symbol} unavailable on Binance (status {response.status}), skipping")
        except Exception as e:
            self._invalid_symbols.add(symbol)
            logger.warning(f"Failed to fetch {symbol}: {e}, skipping")
        return []

    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List[CandleData]:
        """Fetch candle data from Binance API. Returns empty list if unavailable."""
        if symbol in self._invalid_symbols:
            return []
        candles = await self._fetch_binance_klines(symbol, interval, limit)
        if candles:
            return candles
        self._invalid_symbols.add(symbol)
        logger.debug(f"Marking {symbol} as unavailable on Binance, skipping")
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

    async def initialize_symbols(self, progress_callback=None):
        """Initialize symbol list and fetch initial candle data."""
        logger.info("Initializing scanner...")
        self._invalid_symbols = set()
        self.symbols = await self.fetch_top_symbols()
        logger.info(f"Loaded {len(self.symbols)} symbols: {', '.join(self.symbols[:10])}...")
        
        for symbol in self.symbols:
            self.symbol_data[symbol] = SymbolData(symbol=symbol)
        
        logger.info(f"Fetching real market data from Binance API...")
        await self._update_all_candles(progress_callback=progress_callback)
        await self._update_all_structures_async()
        
        real_symbols = [sym for sym in self.symbols if sym not in self._invalid_symbols]
        skipped_symbols = [sym for sym in self.symbols if sym in self._invalid_symbols]
        
        if real_symbols:
            self.working_endpoint = "binance"
            logger.info(f"REAL data for {len(real_symbols)} symbols, skipped {len(skipped_symbols)} unavailable")
            if skipped_symbols:
                logger.info(f"Skipped symbols (no Binance data): {', '.join(sorted(skipped_symbols))}")
        else:
            self.working_endpoint = "no_data"
            logger.warning(f"No real market data available")
        
        stats = self.get_structure_stats()
        logger.info(f"Initialized: {stats['symbols_with_data']}/{stats['total_symbols']} symbols with {stats['total_structures']} structures")
    
    async def _update_all_candles(self, progress_callback=None):
        """Fetch candles for all symbols and timeframes using concurrent batches."""
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES.keys():
                tasks.append((symbol, timeframe))
        
        total = len(tasks)
        done = 0
        batch_size = max(1, CONFIG.data.max_concurrent_symbol_tasks)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for i in range(0, total, batch_size):
                batch = tasks[i:i+batch_size]
                coros = [self._update_symbol_candles_session(session, sym, tf) for sym, tf in batch]
                await asyncio.gather(*coros, return_exceptions=True)
                done += len(batch)
                if progress_callback:
                    await progress_callback(done, total)
                if self._use_real_data:
                    await asyncio.sleep(CONFIG.data.batch_sleep_sec)
    
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
            logger.error(f"Error updating {symbol} {timeframe}: {e}")
    
    WINDOW_SIZES = list(CONFIG.data.window_sizes)

    def _candle_hash(self, closes: list) -> int:
        """Хэш ряда close-цен для детекции изменений (кэш структур, Phase 3.3).

        Хэшируем весь ряд (а не 3–4 точки), чтобы не пропустить изменения в
        середине окна: при ложном совпадении хэша структура не пересчиталась бы
        и результат устарел. Для 100 свечей это дёшево.
        """
        if len(closes) < 3:
            return 0
        return hash((len(closes),) + tuple(closes))

    def _update_structure(self, symbol: str, timeframe: str) -> bool:
        """Update structure for a symbol/timeframe with sliding windows. Returns True if structure changed."""
        if symbol not in self.symbol_data:
            return False
        
        candles = self.symbol_data[symbol].candles.get(timeframe, [])
        if len(candles) < 20:
            return False
        
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        cache_key = f"{symbol}_{timeframe}"
        new_hash = self._candle_hash(closes)
        if CONFIG.data.cache_structures:
            if cache_key in self._candle_hashes and self._candle_hashes[cache_key] == new_hash:
                return False
            self._candle_hashes[cache_key] = new_hash

        features = self.structure_extractor.extract_from_candles(closes, volumes)
        
        old_features = self.symbol_data[symbol].structures.get(timeframe)
        self.symbol_data[symbol].structures[timeframe] = features
        
        for win_size in self.WINDOW_SIZES:
            win_key = f"{timeframe}_w{win_size}"
            if len(closes) >= win_size:
                win_closes = closes[-win_size:]
                win_volumes = volumes[-win_size:]
                win_features = self.structure_extractor.extract_from_candles(win_closes, win_volumes)
                self.symbol_data[symbol].structures[win_key] = win_features
            else:
                self.symbol_data[symbol].structures[win_key] = None

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

    async def _update_all_structures_async(self):
        """Параллельный пересчёт структур по символам (Phase 3.3).

        Расчёт структуры — CPU-bound (numpy/scipy освобождают GIL), поэтому
        выносим его в пул потоков батчами по ``max_concurrent_symbol_tasks``.
        По символу (а не по символ/ТФ) — чтобы один поток последовательно
        обрабатывал все ТФ символа и не было гонок за один и тот же объект.
        """
        batch_size = max(1, CONFIG.data.max_concurrent_symbol_tasks)

        def _update_symbol_all_tfs(sym: str):
            for timeframe in self.TIMEFRAMES.keys():
                self._update_structure(sym, timeframe)

        symbols = list(self.symbols)
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            results = await asyncio.gather(
                *(asyncio.to_thread(_update_symbol_all_tfs, sym) for sym in batch),
                return_exceptions=True,
            )
            # Не «глотаем» ошибки потоков: логируем, чтобы частичная
            # инициализация была видимой (режим degraded, а не тихий сбой).
            for sym, res in zip(batch, results):
                if isinstance(res, Exception):
                    logger.error(
                        "Ошибка пересчёта структур для %s: %s", sym, res
                    )
    
    async def _poll_loop(self):
        """Main polling loop for updating candle data."""
        while self.is_running:
            try:
                await asyncio.sleep(CONFIG.data.poll_interval_sec)
                
                await self._update_all_candles()
                
                for symbol in self.symbols:
                    for timeframe in self.TIMEFRAMES.keys():
                        changed = self._update_structure(symbol, timeframe)
                        if changed and self.on_update_callback:
                            await self.on_update_callback(symbol, timeframe)
                
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
                await asyncio.sleep(CONFIG.data.error_sleep_sec)
    
    async def start(self, on_update: Optional[Callable] = None, progress_callback=None):
        """Start the scanner."""
        if self.is_running:
            return
        
        self.is_running = True
        self.initialized = False
        self.on_update_callback = on_update
        
        await self.initialize_symbols(progress_callback=progress_callback)
        self.initialized = True

        # Real-time режим: WebSocket-фид при включённом флаге, иначе REST-поллинг.
        if CONFIG.data.use_websocket:
            started = await self._start_ws_feed()
            if not started:
                logger.warning("WS-фид не запущен, фаллбэк на REST-поллинг")
                self._poll_task = asyncio.create_task(self._poll_loop())
        else:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def _start_ws_feed(self) -> bool:
        """Поднять WebSocket-фид Binance. Возвращает False при ошибке (→ фаллбэк)."""
        try:
            from .binance_ws_feed import BinanceWebSocketFeed

            self.ws_feed = BinanceWebSocketFeed(self, on_update=self.on_update_callback)
            started = await self.ws_feed.start(self.symbols, list(self.TIMEFRAMES.keys()))
            if not started:
                self.ws_feed = None
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Ошибка запуска WS-фида: {e}")
            self.ws_feed = None
            return False

    async def stop(self):
        """Stop the scanner."""
        self.is_running = False
        self.initialized = False

        if self.ws_feed is not None:
            try:
                await self.ws_feed.stop()
            except Exception:  # noqa: BLE001
                pass
            self.ws_feed = None

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
    
    def get_all_structures(self) -> List[tuple]:
        """Get all current structures for matching, including sliding window sub-segments."""
        results = []
        timestamp = datetime.now().isoformat()
        
        for symbol, data in self.symbol_data.items():
            for tf_key, features in data.structures.items():
                if features is not None:
                    base_tf = tf_key.split("_w")[0] if "_w" in tf_key else tf_key
                    candles = data.candles.get(base_tf, [])
                    last_candle_time = candles[-1].close_time if candles else None
                    results.append((symbol, tf_key, features, timestamp, last_candle_time))
        
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
