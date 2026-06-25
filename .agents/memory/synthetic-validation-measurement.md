---
name: Synthetic validation measurement quirk
description: How to reliably measure per-pattern recall/FPR on this scanner — validate.py logging and the full-dataset runner are unreliable in this sandbox.
---

# Measuring detector recall/FPR reliably

When tuning pattern detectors (recall/FPR targets), do NOT trust `scripts/validate.py`
output or the full-dataset `ValidationRunner.run_synthetic_validation` in this sandbox.

**Observed failures:**
- `validate.py` launched via `setsid ... &` produces EMPTY stdout/stderr logs and does
  not reliably rewrite `data/validation_synthetic.json` (buffering/detach quirk).
- The runner over the full `data/synthetic_dataset.json` (1000 samples incl. 500 noise)
  gets killed mid-run (OOM-like, no traceback) and writes no result. Even a 700-sample
  subset crashes.

**What works:**
- Run measurement scripts in the FOREGROUND with `timeout 115 python -u script.py`
  (unbuffered, prints stream live). Background/setsid loses output here.
- Compute recall directly: call `StructureExtractor.extract_features_causal(candles)`
  per sample and check `is_confirmed and structure_type.value == intended_label`.
  The validator scores PRIMARY `structure_type`, not multi-label `detected_patterns`.
- For FPR, run noise-only samples (≈200) in one foreground pass and count confirmed
  primaries; subsample the noise so it fits the ~115s window.
- `candle_from_dict` is a staticmethod on `SyntheticChartGenerator`;
  `extract_from_candles`/`extract_features` take a float closes array, not CandleData.
  `pytest-asyncio` is NOT installed — use `asyncio.run`.

**Why:** background processes silently dying / empty logs cost many wasted iterations;
the foreground + direct-causal-extraction path is the only one that gives trustworthy
numbers here.
