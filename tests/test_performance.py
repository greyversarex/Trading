"""Тесты оптимизаций производительности (Phase 3.3).

Покрывают: кэширование структур по хэшу свечей (повторный расчёт пропускается,
пока свечи не изменились), отключение кэша флагом конфигурации, и параллельный
пересчёт структур по символам.
"""
from __future__ import annotations

import asyncio

from src.backend.binance_scanner import BinanceScanner, CandleData, SymbolData
from src.backend.config import CONFIG


def _make_candles(n=60, base=100.0):
    out = []
    for i in range(n):
        price = base + (i % 7) - 3 + (i * 0.1)
        out.append(CandleData(
            open_time=i, open=price, high=price + 1, low=price - 1,
            close=price, volume=1000 + (i % 5) * 10, close_time=i + 59,
        ))
    return out


def _scanner_with(symbols):
    scanner = BinanceScanner(num_symbols=len(symbols))
    scanner.symbols = list(symbols)
    scanner.symbol_data = {}
    for s in symbols:
        sd = SymbolData(symbol=s)
        sd.candles["1h"] = _make_candles()
        scanner.symbol_data[s] = sd
    return scanner


def test_structure_cache_skips_recompute_when_unchanged():
    """Повторный вызов с теми же свечами не пересчитывает структуру."""
    scanner = _scanner_with(["BTC"])
    calls = {"n": 0}
    orig = scanner.structure_extractor.extract_from_candles

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    scanner.structure_extractor.extract_from_candles = counting

    assert CONFIG.data.cache_structures is True
    first = scanner._update_structure("BTC", "1h")
    n_after_first = calls["n"]
    assert n_after_first > 0

    # Второй вызов — свечи не менялись → кэш срабатывает, расчёта нет.
    second = scanner._update_structure("BTC", "1h")
    assert second is False
    assert calls["n"] == n_after_first


def test_structure_cache_recomputes_when_candles_change():
    """При изменении свечей хэш меняется и структура пересчитывается."""
    scanner = _scanner_with(["BTC"])
    calls = {"n": 0}
    orig = scanner.structure_extractor.extract_from_candles

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    scanner.structure_extractor.extract_from_candles = counting

    scanner._update_structure("BTC", "1h")
    n1 = calls["n"]

    # Добавляем новую свечу — ряд меняется.
    scanner.symbol_data["BTC"].candles["1h"].append(
        CandleData(open_time=999, open=200, high=205, low=195, close=202,
                   volume=5000, close_time=1058)
    )
    scanner._update_structure("BTC", "1h")
    assert calls["n"] > n1


def test_cache_can_be_disabled(monkeypatch):
    """При cache_structures=False расчёт выполняется каждый раз."""
    scanner = _scanner_with(["BTC"])
    calls = {"n": 0}
    orig = scanner.structure_extractor.extract_from_candles

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    scanner.structure_extractor.extract_from_candles = counting
    monkeypatch.setattr(CONFIG.data, "cache_structures", False)

    scanner._update_structure("BTC", "1h")
    n1 = calls["n"]
    scanner._update_structure("BTC", "1h")
    assert calls["n"] > n1  # повторный расчёт без кэша


def test_candle_hash_detects_mid_series_change():
    """Хэш ловит изменение в середине ряда (а не только на краях)."""
    scanner = _scanner_with(["BTC"])
    closes = [c.close for c in scanner.symbol_data["BTC"].candles["1h"]]
    h1 = scanner._candle_hash(closes)
    closes2 = list(closes)
    closes2[len(closes2) // 2] += 5.0
    h2 = scanner._candle_hash(closes2)
    assert h1 != h2


def test_parallel_structure_update_computes_all_symbols():
    """Параллельный пересчёт заполняет структуры для всех символов."""
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    scanner = _scanner_with(symbols)

    asyncio.run(scanner._update_all_structures_async())

    # Все символы обработаны: ключ структуры по ТФ установлен (значение может
    # быть None, если ряд не прошёл порог качества — это валидный исход).
    for s in symbols:
        assert "1h" in scanner.symbol_data[s].structures


def test_parallel_and_sequential_agree():
    """Параллельный и последовательный пересчёт дают одинаковый тип структуры."""
    symbols = ["BTC", "ETH", "SOL"]
    seq = _scanner_with(symbols)
    seq._update_all_structures()

    par = _scanner_with(symbols)
    asyncio.run(par._update_all_structures_async())

    for s in symbols:
        a = seq.symbol_data[s].structures["1h"]
        b = par.symbol_data[s].structures["1h"]
        assert (a is None) == (b is None)
        if a is not None:
            assert a.structure_type == b.structure_type
