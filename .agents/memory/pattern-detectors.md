---
name: Pattern detector design
description: How chart-pattern detectors integrate into the causal feature pipeline and avoid regressions.
---

New structural pattern detectors (triple top/bottom, cup&handle, pennant, rounding bottom) are added as **causal candidate builders** that return a `_PatternCandidate` and are confirmed by the shared `_confirm_candidate` families (`breakout_level`/`breakout_channel`/`trend`/`range`). They do NOT re-implement confirmation.

**Integration rule:** `extract_features_causal` tries the specific new patterns FIRST (priority loop); the first one that *confirms* wins, otherwise it falls back to the base-classified candidate.
**Why:** preserves all Phase 1 behavior — new builders return `None` on double/channel/triangle shapes, so legacy classification and its tests are untouched.
**How to apply:** when adding another detector, (1) add a builder returning `_PatternCandidate`, (2) register it in `_build_candidate` dispatch and the priority list, (3) register the `StructureType` in `similarity_matcher` (REVERSAL/CONSOLIDATION sets, OPPOSITE_PAIRS, type_mirror_map, MIN_CONFIDENCE_THRESHOLDS), (4) it auto-flows through `main.py` (dispatch is by enum string value + `detected_patterns`). For type_scan multi-label, reuse the same builders in normalized space via `_detect_missing_patterns_norm` (append-only, never changes primary type).

**Discrimination gotchas learned:**
- Rounding bottom vs rising/falling channel: require the global-min low pivot near series center (0.35–0.65 n) AND the parabola fit to beat a linear fit (resid <= 0.6*lin_resid). A channel's deepest low sits at an edge and is near-linear.
- Pennant: decaying oscillations yield too few pivots for line-fitting; measure convergence from the post-pole price *range* shrinking, exclude the trailing breakout heuristically (~65% of post-pole window), confirm a breakout above the pole top.
