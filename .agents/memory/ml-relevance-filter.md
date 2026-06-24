---
name: ML relevance filter
description: How the ML false-positive filter integrates into scanning and what it is/isn't responsible for.
---

# ML relevance filter (Phase 2)

The ML classifier scores each detected pattern match (`ml_score` in [0,1]); matches
with `ml_score < ml_score_threshold` are dropped — **only when a trained model
exists**. With no model, `ml_score = 1.0` and nothing is filtered (backward compat).

**Rule:** every scan path that emits matches must apply the SAME ml_score gate, or
behavior diverges. Both the live path and the initial-burst path of each scan mode
(type_scan, causal) must compute ml_score and filter identically.
**Why:** initial-scan and live paths are separate functions in main.py; it's easy to
wire ML into one and forget the other, producing "ghost" matches at scan start that
later vanish.

**Division of labor:** the ML filter's job is to cut false positives (precision),
NOT to find more patterns. Recall is bounded by the geometry detectors. On synthetic
validation the filter took FPR 43.67%→0.00% while keeping TPR at 73.67% — recall
gains require improving detectors, not the filter.

**Auto-retrain:** throttle the *check* (not just the retrain) in the hot market-update
callback, or you spawn an asyncio task every tick. Use a last-check timestamp gate.
