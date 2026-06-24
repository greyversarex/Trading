---
name: Scanner symbol keys
description: BinanceScanner keys candles/structures on the BASE symbol (BTC), not the trading pair (BTCUSDT).
---

# Scanner symbol-key convention

`BinanceScanner.symbol_data`, `get_candles(symbol, tf)`, `price_change_24h`, and
the per-symbol candle dicts are all keyed on the **base** asset symbol — `"BTC"`,
`"ETH"`, `"SOL"` — **not** the trading pair `"BTCUSDT"`. The `USDT` suffix is only
added/stripped at the Binance HTTP layer.

**Why it matters:** any new REST endpoint or scan mode that reads candles via
`scanner.get_candles(...)` / `scanner.symbol_data[...]` must receive the base
symbol. Passing `"BTCUSDT"` silently returns an empty candle list, which then
looks like "not enough data" rather than a wrong-key bug. This cost real debugging
time when adding `/api/levels-v2/{symbol}/{timeframe}` — direct klines fetches
worked while the live endpoint returned empty until queried with `BTC` instead of
`BTCUSDT`.

**How to apply:**
- New candle-reading endpoints should accept/expect the base symbol (mirror the
  existing `/api/candles/{symbol}/{tf}` and `/api/chart/{symbol}/{tf}`).
- When a similarity `match.symbol` is the pair form, existing code strips it with
  `.replace("USDT", "")` before `price_change_24h` lookups — follow that pattern.
- Scanner loads top-volume symbols from CoinGecko mapped to Binance; the active
  set varies per run and BTC/ETH may or may not be present. Don't hardcode test
  symbols — read whatever `scanner.symbol_data` currently holds.
