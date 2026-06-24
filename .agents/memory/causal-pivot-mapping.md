---
name: Causal pivot mapping
description: Why the causal/raw-price detection layer normalizes before detecting pivots, then maps back to raw extrema.
---

# Causal detection: detect pivots on normalized line, then map to raw

When building any raw-price geometry/pattern detection in `structure_extractor.py`,
do NOT run `detect_pivots` directly on raw close prices. It is tuned (order/prominence)
for the normalized 100-point line and **underdetects** pivots on raw closes, so
candidates (double top/bottom, H&S, triangles, channels) fail to form.

**Correct pipeline** (used by `extract_features_causal`):
1. `normalize_line(closes)` → normalized line
2. `detect_pivots(normalized)` → pivot indices
3. `_map_pivots_to_raw(norm_pivots, closes)` → snap each pivot to the nearest real
   raw extremum within ±2 bars

**Why:** Took >2 attempts — synthetic double-top would not confirm until pivots were
detected on the normalized line and snapped back to raw extrema. Confirmation logic
(`_confirm_candidate`) needs raw-price levels/slopes, but pivot *location* must come
from the normalized detector.

**How to apply:** Any new causal/raw detection (e.g. ATR-adaptive geometry in later
phases) should reuse this normalize→detect→map-to-raw flow rather than calling
`detect_pivots` on raw prices.

**Testing note:** The base classifier (`extract_features`) reliably classifies
synthetic double_top/double_bottom/channel_up/channel_down, but NOT synthetic
head_shoulders/triangles (it picks double_top/bottom instead). To test H&S/triangle
confirmation, call `_build_candidate(<type>, ...)` + `_confirm_candidate(...)`
directly instead of going through `extract_features_causal`.
