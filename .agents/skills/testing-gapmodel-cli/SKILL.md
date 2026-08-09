---
name: testing-gapmodel-cli
description: How to test the gapmodel CLI (predict/score/screen) end-to-end from the shell, including Yahoo data caching, as-of dates, and independently recomputing printed metrics.
---

# Testing the gapmodel CLI

This project is a CLI only (no UI), so testing is shell-only — do not start a
screen recording for it, collect command output as text evidence instead.

## Environment
- Python deps live in the venv at `~/.venvs/gapmodel` (already on PATH). System
  `python3` has no project deps.
- Lint: `ruff check . && ruff format --check .` · Tests: `python -m pytest tests -q`
  (fast, no network needed).
- Price data comes from Yahoo Finance and caches to `~/.cache/gapmodel/<SYMBOL>.csv`
  with sidecar `<SYMBOL>.start` / `<SYMBOL>.fields` metadata. A cold run over the
  ~155-ticker US universe takes minutes; once cached, runs finish in ~2s. If the
  cache is already warm from an earlier session, budget accordingly.
- No secrets/credentials are required. Devin Secrets Needed: none.

## Gotchas
- `--refresh`, `--cache`, `--start` and `--verbose` are **global** flags that must
  come *before* the subcommand: `python -m gapmodel --refresh screen AAPL`.
  Putting them after the subcommand fails with "unrecognized arguments".
- Use `--asof <ISO date>` to pin a completed session; mid-session volume is
  partial and makes volume-based results non-reproducible. A weekend/holiday
  `--asof` silently resolves to the last session at or before it, so check the
  `asof` column in the output rather than assuming.
- `main()` in `gapmodel/cli.py` catches ValueError/RuntimeError/KeyError/OSError and
  exits with `error: ...`, so bad input should never produce a traceback. yfinance
  itself prints its own `ERROR ...` lines on stderr for delisted tickers — that is
  library noise, not a crash.

## Strong verification technique
Do not just eyeball the printed table: reimplement the metrics independently from
the cached CSVs with the plain `csv` module (importing only the ticker list from
`gapmodel.universe`, never `gapmodel.screener`), then compare every funnel count
and every survivor row. This catches off-by-one window bugs that a
self-consistent implementation would hide — e.g. for the screener, the volume
baseline must be the 30 sessions *before* the screened session; compute the
average both excluding and including the screened session and confirm the printed
value matches the excluding one.
