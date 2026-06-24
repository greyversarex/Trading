"""Загрузка исторических свечей через публичный Binance REST API.

Используется валидационным харнессом для бэктеста детекторов на реальных
исторических данных. Возвращает список ``CandleData`` (та же структура, что и в
рабочем сканере), чтобы детекторы работали без изменений.
"""
from __future__ import annotations

import asyncio
from typing import List

import aiohttp

from ..binance_scanner import CandleData

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

_KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH")


def _normalize_symbol(symbol: str) -> str:
    """Приводит символ к формату Binance (например, ``BTC`` -> ``BTCUSDT``)."""
    s = symbol.upper().replace("/", "")
    if any(s.endswith(q) for q in _KNOWN_QUOTES):
        return s
    return f"{s}USDT"


async def fetch_history_async(symbol: str, timeframe: str, limit: int = 500) -> List[CandleData]:
    """Асинхронно загружает до ``limit`` исторических свечей с Binance.

    При любой сетевой ошибке возвращает пустой список, чтобы харнесс мог
    завершиться корректно даже без доступа к данным.
    """
    params = {
        "symbol": _normalize_symbol(symbol),
        "interval": timeframe,
        "limit": limit,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status != 200:
                    return []
                data = await response.json()
    except Exception:
        return []

    candles: List[CandleData] = []
    for k in data:
        candles.append(
            CandleData(
                open_time=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=int(k[6]),
            )
        )
    return candles


def fetch_history(symbol: str, timeframe: str, limit: int = 500) -> List[CandleData]:
    """Синхронная обёртка над :func:`fetch_history_async`."""
    return asyncio.run(fetch_history_async(symbol, timeframe, limit))
