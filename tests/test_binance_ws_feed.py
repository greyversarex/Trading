"""Тесты WebSocket-фида Binance (Phase 3.1).

Покрывают: разбор kline-сообщения и обновление свечей, добавление новой свечи
vs обновление текущей, переключение колбэка только по закрытой свече,
экспоненциальный backoff переподключения, и REST-фаллбэк при выключенном WS.

Асинхронные сценарии вызываются через ``asyncio.run`` (как в остальных тестах
проекта — без зависимости от pytest-asyncio).
"""
from __future__ import annotations

import asyncio
import json

from src.backend.binance_scanner import BinanceScanner, CandleData, SymbolData
from src.backend.binance_ws_feed import BinanceWebSocketFeed
from src.backend.config import CONFIG


def _kline_msg(symbol="BTCUSDT", tf="1m", open_time=1000, close_time=1059,
               o=10.0, h=11.0, lo=9.0, c=10.5, v=100.0, closed=False):
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_{tf}",
        "data": {
            "e": "kline",
            "s": symbol,
            "k": {
                "t": open_time, "T": close_time, "i": tf,
                "o": str(o), "h": str(h), "l": str(lo), "c": str(c),
                "v": str(v), "x": closed,
            },
        },
    })


def _make_scanner():
    scanner = BinanceScanner(num_symbols=1)
    scanner.symbols = ["BTC"]
    scanner.symbol_data = {"BTC": SymbolData(symbol="BTC")}
    return scanner


def test_stream_names_and_url():
    scanner = _make_scanner()
    feed = BinanceWebSocketFeed(scanner)
    feed.symbols = ["BTC", "ETH"]
    feed.timeframes = ["1m", "5m"]
    names = feed._stream_names()
    assert "btcusdt@kline_1m" in names
    assert "ethusdt@kline_5m" in names
    assert len(names) == 4
    url = feed._build_url()
    assert url.startswith("wss://stream.binance.com:9443/stream?streams=")
    assert "btcusdt@kline_1m" in url


def test_base_from_pair():
    assert BinanceWebSocketFeed._base_from_pair("BTCUSDT") == "BTC"
    assert BinanceWebSocketFeed._base_from_pair("ETHBTC") is None


def test_backoff_is_exponential_and_capped():
    f = BinanceWebSocketFeed._next_backoff
    assert f(0, 30.0) == 1.0
    assert f(1, 30.0) == 2.0
    assert f(2, 30.0) == 4.0
    assert f(3, 30.0) == 8.0
    # cap
    assert f(10, 30.0) == 30.0


def test_on_message_appends_new_candle():
    scanner = _make_scanner()
    feed = BinanceWebSocketFeed(scanner)
    feed.timeframes = ["1m"]

    scanner.symbol_data["BTC"].candles["1m"] = [
        CandleData(open_time=900, open=9, high=10, low=8, close=9.5,
                   volume=50, close_time=959)
    ]
    asyncio.run(feed._on_message(_kline_msg(open_time=1000, c=10.5, closed=False)))
    candles = scanner.symbol_data["BTC"].candles["1m"]
    assert len(candles) == 2
    assert candles[-1].open_time == 1000
    assert candles[-1].close == 10.5


def test_on_message_updates_current_candle():
    scanner = _make_scanner()
    feed = BinanceWebSocketFeed(scanner)
    feed.timeframes = ["1m"]
    scanner.symbol_data["BTC"].candles["1m"] = [
        CandleData(open_time=1000, open=10, high=10.2, low=9.9, close=10.1,
                   volume=20, close_time=1059)
    ]
    asyncio.run(feed._on_message(_kline_msg(open_time=1000, c=10.8, closed=False)))
    candles = scanner.symbol_data["BTC"].candles["1m"]
    assert len(candles) == 1
    assert candles[-1].close == 10.8


def test_callback_only_on_closed_candle():
    scanner = _make_scanner()
    base = [
        CandleData(open_time=i, open=10, high=11, low=9, close=10 + (i % 3),
                   volume=100, close_time=i + 59)
        for i in range(0, 30)
    ]
    scanner.symbol_data["BTC"].candles["1m"] = base

    calls = []

    async def on_update(sym, tf):
        calls.append((sym, tf))

    feed = BinanceWebSocketFeed(scanner, on_update=on_update)
    feed.timeframes = ["1m"]

    async def scenario():
        # незакрытая свеча — колбэк не вызывается
        await feed._on_message(_kline_msg(open_time=10_000, closed=False))
        assert calls == []
        # закрытая свеча — колбэк вызывается (новая структура)
        await feed._on_message(_kline_msg(open_time=20_000, closed=True))

    asyncio.run(scenario())
    assert ("BTC", "1m") in calls


def test_apply_candle_ignores_stale_out_of_order():
    """Внеочередная (более старая) свеча не перезаписывает последнюю."""
    scanner = _make_scanner()
    feed = BinanceWebSocketFeed(scanner)
    feed.timeframes = ["1m"]
    scanner.symbol_data["BTC"].candles["1m"] = [
        CandleData(open_time=1000, open=10, high=10.2, low=9.9, close=10.1,
                   volume=20, close_time=1059)
    ]
    # приходит более старая свеча (open_time=900) — должна быть проигнорирована
    asyncio.run(feed._on_message(_kline_msg(open_time=900, c=99.0, closed=False)))
    candles = scanner.symbol_data["BTC"].candles["1m"]
    assert len(candles) == 1
    assert candles[-1].open_time == 1000
    assert candles[-1].close == 10.1  # без изменений


def test_on_message_ignores_unknown_symbol_or_tf():
    scanner = _make_scanner()
    feed = BinanceWebSocketFeed(scanner)
    feed.timeframes = ["1m"]
    asyncio.run(feed._on_message(_kline_msg(symbol="DOGEUSDT", open_time=1000)))
    assert "DOGE" not in scanner.symbol_data
    asyncio.run(feed._on_message(_kline_msg(tf="3m", open_time=1000)))
    assert scanner.symbol_data["BTC"].candles.get("3m") is None


def test_rest_fallback_when_websocket_disabled(monkeypatch):
    """При use_websocket=False стартует REST-поллинг, ws_feed остаётся None."""
    scanner = _make_scanner()

    async def fake_initialize(progress_callback=None):
        scanner.symbol_data = {"BTC": SymbolData(symbol="BTC")}

    monkeypatch.setattr(scanner, "initialize_symbols", fake_initialize)

    poll_started = {"v": False}

    async def fake_poll():
        poll_started["v"] = True

    monkeypatch.setattr(scanner, "_poll_loop", fake_poll)
    monkeypatch.setattr(CONFIG.data, "use_websocket", False)

    async def scenario():
        await scanner.start(on_update=None)
        await asyncio.sleep(0.01)
        assert scanner.ws_feed is None
        assert poll_started["v"] is True
        await scanner.stop()

    asyncio.run(scenario())


def test_ws_feed_started_when_enabled(monkeypatch):
    """При use_websocket=True создаётся ws_feed и REST-поллинг не запускается."""
    scanner = _make_scanner()

    async def fake_initialize(progress_callback=None):
        scanner.symbol_data = {"BTC": SymbolData(symbol="BTC")}
        scanner.symbols = ["BTC"]

    monkeypatch.setattr(scanner, "initialize_symbols", fake_initialize)
    monkeypatch.setattr(CONFIG.data, "use_websocket", True)

    started = {"symbols": None, "timeframes": None}

    async def fake_ws_start(self, symbols, timeframes):
        started["symbols"] = symbols
        started["timeframes"] = timeframes
        return True

    monkeypatch.setattr(BinanceWebSocketFeed, "start", fake_ws_start)

    async def scenario():
        await scanner.start(on_update=None)
        await asyncio.sleep(0.01)
        assert scanner.ws_feed is not None
        assert scanner._poll_task is None
        assert started["symbols"] == ["BTC"]
        await scanner.stop()

    asyncio.run(scenario())
