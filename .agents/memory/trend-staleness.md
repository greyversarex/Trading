---
name: Trend staleness & active-only detected_patterns
description: How plain trends are invalidated on reversal and why detected_patterns must exclude stale candidates.
---

Plain trends (TREND_UP/DOWN) in `classify_structure` are invalidated when price reverses against the trend in the window tail (strong tail-move against direction, or large drawdown from the extreme after the peak). This mirrors how structural patterns (flags, H&S, triangles) already set `is_active=False`.

**Why:** trends used to be *always* `is_active=True, freshness=1.0`, so a long-since-reversed trend still showed as "Только что" and leaked into scans — the exact user complaint.

**How to apply / gotchas:**
- Two scan paths must both honor `is_pattern_active`: the similarity path skips inactive candidates; the type-scan path (`on_market_update_type_scan` in main.py) must do the same guard, or reversed trends re-enter as *primary* matches.
- Second leak vector: `detected_patterns` (multi-label) is matched by `secondary_match` in type-scan. Build `detected_patterns` from **active candidates only** — otherwise a stale pattern that is still listed as a secondary detected pattern re-enters the results even when the primary is active.
- The causal path (`extract_features_causal`, confirmed-only) is a separate detector set and does not use `classify_structure`'s trend activity.
