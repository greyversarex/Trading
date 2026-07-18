---
name: Docs/UI drift vs backend
description: replit.md and frontend contain remnants of deleted features — verify against backend code before trusting any feature claim.
---

Rule: in this project, do NOT trust replit.md or the frontend as evidence that a feature exists. Verify in `src/backend/` first.

**Why:** A full audit (July 2026) found replit.md advertises candlestick-pattern detection (19 patterns) and Fibonacci analysis — neither exists in the backend. The frontend match-type filter still lists 19 candlestick options plus "retest" that no detector ever emits (always-empty filters), and a `candle.max_age` setting exists that nothing reads. These are leftovers from deleted modules; older docs (PROJECT_AUDIT.md, PROJECT_DOCS.md) describe removed modules too.

**How to apply:** When asked about features, scanning modes, or when planning changes: grep the backend for the actual detector/mode before assuming it exists. When the user asks to "fix" candlestick/Fibonacci features, clarify that they were removed and need full implementation, not a bug fix. Full verified state of the codebase is in TECHNICAL_AUDIT.md (July 18, 2026).
