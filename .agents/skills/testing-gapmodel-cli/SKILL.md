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
- The global `--start` does *not* widen the screener's history: `screen` downloads
  from its own `--screen-start` (default `2024-01-01`) and ignores `--start`.
- Use `--asof <ISO date>` to pin a completed session; mid-session volume is
  partial and makes volume-based results non-reproducible. Each symbol's frame is
  truncated to `index <= asof`, so a weekend/holiday date reads each symbol's last
  bar at or before it — per symbol, which is why the header `asof` (the *newest*
  session across survivors) can differ from an individual row. Read the per-row
  `asof` column. A past `--asof` also suppresses the staleness re-download, since
  the cache cannot be behind a session that closed long ago.
- `main()` in `gapmodel/cli.py` catches ValueError/RuntimeError/KeyError/OSError and
  exits with `error: ...`, and the argument helpers raise `SystemExit` with the same
  `error: ...` shape, so bad *input* should not produce a traceback. Anything else
  escaping pandas or yfinance still would — a traceback is a finding, not noise.
  yfinance's own `ERROR ...` lines on stderr for delisted tickers are noise.

## Strong verification technique
Do not just eyeball the printed table: reimplement the metrics independently from
the cached CSVs with the plain `csv` module (importing only the ticker list from
`gapmodel.universe`, never `gapmodel.screener`), then compare every funnel count
and every survivor row. This catches off-by-one window bugs that a
self-consistent implementation would hide — e.g. for the screener, the volume
baseline must be the 30 sessions *before* the screened session; compute the
average both excluding and including the screened session and confirm the printed
value matches the excluding one.
