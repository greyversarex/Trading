---
name: ML relevance hard filter gating
description: Why the ML relevance filter must not hard-drop matches until trained on real feedback
---

# ML relevance hard filter

The ML relevance model (`data/ml_relevance_model.pkl`) is trained **only on synthetic
data** (clean injected patterns vs random-walk noise). It does NOT transfer to real
Binance market structures: on real data it scores almost everything below the 0.5
threshold (measured: 0/168 real structures passed; max ml_score ~0.45). A hard gate of
`model_exists and ml_score < threshold` therefore silently drops 100% of matches in
ALL scan modes → "no matches found for any structure".

**Rule:** the hard ML gate (`_ml_hard_filter_active()` in main.py) only activates once
the model is retrained on enough REAL user feedback — tracked via `n_feedback` in model
metadata vs `CONFIG.ml.ml_hard_filter_min_feedback` (default 20). Until then `ml_score`
is computed only as a badge / sort value, never as a hard cutoff.

**Why:** synthetic-trained relevance models are mis-calibrated for live data; gating on
them blocks the whole product. Feedback-based retraining is the only path to a gate that
reflects real relevance.

**How to apply:** never reintroduce a bare `model_exists and ml_score < threshold` gate.
Route all ML hard-cuts through `_ml_hard_filter_active()`. Old pickles lacking the
`n_feedback` key default to 0 (gate stays off) — that's intended.
