"""Реальный поток данных Binance по WebSocket (Phase 3.1).

Подключается к публичному стриму klines Binance и обновляет свечи в
``BinanceScanner.symbol_data`` в реальном времени, вызывая колбэк рыночных
обновлений только по ЗАКРЫТЫМ свечам (``k.x == True``) — чтобы не плодить
перерасчёты структуры на каждый тик.

Особенности:
  * Комбинированный стрим ``/stream?streams=...`` (имя ``{symbol}@kline_{tf}``).
  * Автопереподключение с экспоненциальной задержкой (cap = ``max_delay``).
  * Heartbeat через встроенный ping/pong библиотеки ``websockets``.
  * Не заменяет серверный WebSocket ``/ws`` (браузер ↔ сервер) — это отдельный
    исходящий клиент Binance.

REST-поллинг остаётся фаллбэком, если ``CONFIG.data.use_websocket`` = False
или если WebSocket не удаётся поднять.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List, Optional

import websockets

from .config import CONFIG

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443"
# Максимум стримов на одно соединение по документации Binance.
MAX_STREAMS_PER_CONNECTION = 1024


class BinanceWebSocketFeed:
    """Асинхронный клиент klines-стрима Binance, обновляющий сканер in-place."""

    def __init__(
        self,
        scanner,
        on_update: Optional[Callable] = None,
        max_delay: Optional[float] = None,
        heartbeat_interval: Optional[float] = None,
    ):
        self.scanner = scanner
        self.on_update = on_update
        self.max_delay = (
            max_delay
            if max_delay is not None
            else CONFIG.data.ws_reconnect_max_delay_sec
        )
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else CONFIG.data.ws_heartbeat_interval_sec
        )
        self.symbols: List[str] = []
        self.timeframes: List[str] = []
        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._ws = None
        # Флаг: удалось ли хотя бы раз успешно подключиться (для решения о фаллбэке).
        self.connected_once: bool = False
        # Событие первого успешного подключения — для детерминированного фаллбэка.
        self._connected_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------ utils
    def _stream_names(self) -> List[str]:
        """Список имён стримов вида ``btcusdt@kline_1m``."""
        names = []
        for sym in self.symbols:
            pair = f"{sym.lower()}usdt"
            for tf in self.timeframes:
                names.append(f"{pair}@kline_{tf}")
        return names

    def _build_url(self) -> str:
        """URL комбинированного стрима из текущих символов/таймфреймов."""
        streams = "/".join(self._stream_names())
        return f"{BINANCE_WS_BASE}/stream?streams={streams}"

    @staticmethod
    def _next_backoff(attempt: int, max_delay: float) -> float:
        """Экспоненциальная задержка переподключения с ограничением сверху."""
        delay = min(max_delay, (2.0 ** attempt))
        return float(delay)

    @staticmethod
    def _base_from_pair(pair_symbol: str) -> Optional[str]:
        """``BTCUSDT`` -> ``BTC`` (только пары к USDT)."""
        if pair_symbol.endswith("USDT"):
            return pair_symbol[:-4]
        return None

    # --------------------------------------------------------------- lifecycle
    async def start(self, symbols: List[str], timeframes: List[str]) -> bool:
        """Подписаться на klines и дождаться первого подключения.

        Возвращает ``True``, если соединение с Binance установлено в пределах
        ``ws_startup_timeout_sec``; иначе останавливает фоновую задачу и
        возвращает ``False`` — это сигнал вызывающему перейти на REST-поллинг
        (детерминированный фаллбэк, а не «тихий» режим без данных).
        """
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)
        if not self.symbols or not self.timeframes:
            logger.warning("WS-фид: пустой список символов/таймфреймов, пропуск")
            return False

        n_streams = len(self.symbols) * len(self.timeframes)
        if n_streams > MAX_STREAMS_PER_CONNECTION:
            logger.warning(
                "WS-фид: %d стримов превышает лимит %d на соединение; "
                "часть данных может не обновляться",
                n_streams,
                MAX_STREAMS_PER_CONNECTION,
            )

        self.is_running = True
        self._connected_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

        timeout = CONFIG.data.ws_startup_timeout_sec
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "WS-фид: подключение не установлено за %.1f с → фаллбэк на REST",
                timeout,
            )
            await self.stop()
            return False

        logger.info(
            "WS-фид запущен: %d символов x %d ТФ = %d стримов",
            len(self.symbols),
            len(self.timeframes),
            n_streams,
        )
        return True

    async def stop(self) -> None:
        """Корректно остановить фоновый цикл и закрыть соединение."""
        self.is_running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        """Цикл подключения/прослушивания с экспоненциальным backoff."""
        attempt = 0
        while self.is_running:
            try:
                url = self._build_url()
                async with websockets.connect(
                    url,
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=self.heartbeat_interval,
                    max_queue=None,
                ) as ws:
                    self._ws = ws
                    self.connected_once = True
                    if self._connected_event is not None:
                        self._connected_event.set()
                    attempt = 0  # успех — сбрасываем backoff
                    logger.info("WS-фид подключён к Binance")
                    async for raw in ws:
                        if not self.is_running:
                            break
                        await self._on_message(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if not self.is_running:
                    break
                delay = self._next_backoff(attempt, self.max_delay)
                attempt += 1
                logger.warning(
                    "WS-фид отключён (%s); переподключение через %.1f с",
                    exc,
                    delay,
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break
            finally:
                self._ws = None

    # ----------------------------------------------------------------- message
    async def _on_message(self, raw) -> None:
        """Разобрать сообщение kline и обновить свечи/структуру сканера."""
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return

        # Комбинированный стрим оборачивает данные в {"stream":..., "data":...}.
        data = msg.get("data", msg) if isinstance(msg, dict) else None
        if not isinstance(data, dict):
            return
        if data.get("e") != "kline":
            return

        k = data.get("k") or {}
        pair_symbol = data.get("s", "")
        base = self._base_from_pair(pair_symbol)
        if base is None:
            return
        timeframe = k.get("i")
        if timeframe not in self.timeframes:
            return
        if base not in self.scanner.symbol_data:
            return

        try:
            candle = self._candle_from_kline(k)
        except (KeyError, ValueError, TypeError):
            return

        is_closed = bool(k.get("x", False))
        self._apply_candle(base, timeframe, candle, is_closed)

        # Перерасчёт структуры и колбэк — только по закрытой свече.
        if is_closed:
            changed = self.scanner._update_structure(base, timeframe)
            if changed and self.on_update is not None:
                await self.on_update(base, timeframe)

    def _candle_from_kline(self, k: dict):
        """Построить ``CandleData`` из объекта ``k`` сообщения kline."""
        # Локальный импорт, чтобы избежать циклической зависимости на уровне модуля.
        from .binance_scanner import CandleData

        return CandleData(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=int(k["T"]),
        )

    def _apply_candle(self, base: str, timeframe: str, candle, is_closed: bool) -> None:
        """Вставить/обновить свечу в буфере символа, сохраняя длину окна.

        Если пришла новая свеча (``open_time`` больше последней) — добавляем и
        обрезаем буфер до ``default_limit``. Иначе обновляем последнюю
        (свеча в процессе формирования).
        """
        sym_data = self.scanner.symbol_data[base]
        candles = sym_data.candles.get(timeframe)
        if not candles:
            sym_data.candles[timeframe] = [candle]
            sym_data.last_update = _now()
            return

        last = candles[-1]
        if candle.open_time > last.open_time:
            candles.append(candle)
            limit = CONFIG.data.default_limit
            if len(candles) > limit:
                del candles[: len(candles) - limit]
        elif candle.open_time == last.open_time:
            # та же свеча — обновляем in-place
            candles[-1] = candle
        else:
            # устаревшее/внеочередное сообщение — игнорируем, не портим буфер
            return
        sym_data.last_update = _now()


def _now() -> float:
    import time

    return time.time()
