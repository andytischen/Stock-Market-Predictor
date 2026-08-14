---
name: testing-gapmodel-cli
description: How to runtime-test the gapmodel CLI (predict / stocks / screen / sectors) end to end - venv, cached price data, expected runtimes, and the checks that actually catch look-ahead and ranking bugs.
---

# Testing the gapmodel CLI

Terminal-only project: a Python library + `python -m gapmodel` CLI. There is no web UI or
server, so do **not** start a browser or a screen recording — collect stdout as text evidence.

## Environment

- `source ~/.venvs/gapmodel/bin/activate` (the blueprint creates it; system python3 has no deps).
- Add `-W ignore` to every `python -m gapmodel ...` invocation: sklearn emits noisy
  `FutureWarning: 'penalty' was deprecated` lines that otherwise swamp the output.
- Yahoo Finance bars are cached in `~/.cache/gapmodel/<SYMBOL>.csv`. Runs hit the network only
  for symbols missing from the cache (a garbage ticker therefore produces yfinance `ERROR`
  lines before the CLI's own `error:` message — that is expected, not a traceback).
- `python -m pytest tests -q` takes ~1 min; `ruff check . && ruff format --check .` is instant.

## Runtime budget (walk-forward backtest per name, single-threaded per symbol)

- ~1.3 min per stock for `python -W ignore -m gapmodel stocks SYM ...`.
- The full 65-name Nasdaq universe run is ~17 min. Start it in the background *first*
  (`nohup ... > /tmp/log 2>&1 &`) and do targeted runs while it works; several runs in
  parallel are fine on this box.

## Checks that actually catch bugs (do these, not just "it printed a table")

Parse the CSV (`--csv PATH`) with pandas rather than eyeballing the table:

1. **Look-ahead**: forecast `session` must be strictly later than the last `Date` in
   `~/.cache/gapmodel/<SYM>.csv`. `features.next_session_date` only skips weekends, so the
   expected value is "next weekday after the last cached bar" (holidays are ignored).
2. **Internal arithmetic**: `edge == round(p_open_up - base_rate, 4)` *exactly*, in the printed
   table and the CSV alike — the reported edge is derived from the two reported columns either
   side of it, so a reader checking it by hand cannot find a discrepancy. (It did not always
   hold: `edge` was once computed from the unrounded probability while `p_open_up` went through
   `predict._display`, which clamps to [1e-4, 1-1e-4] and rounds to 4dp, so 17 of 65 names were
   off by 1 in the last digit. Assert equality, not a 1e-4 tolerance, or the regression returns.)
   The unrounded `StockPick.edge` property still drives the ranking.
3. **Credibility filter** (`stocks.StockPick.credible`): recompute
   `auc >= 0.55 and brier_skill > 0 and n_oos >= 500` from the CSV and assert the ranked block
   is exactly that set, the unranked block is exactly its complement, the two do not overlap,
   and each `SYM: reason` line names precisely the failed test(s). Also probe the boundaries in
   a throwaway script by constructing `StockPick(Forecast(...))` with
   `backtest={"auc":..., "brier_skill":..., "n":..., "base_rate":..., "accuracy":...}` —
   `Forecast` needs `symbol, name, region, session, probability_up, backtest, contributions`.
4. **Ranking order**: ranked block must be sorted by `abs(edge) * max(auc - 0.5, 0)` desc.
5. **`--top N` must not truncate the CSV** — the CSV always holds every name, ranked or not,
   written in report order (credible first) with a `credible` boolean column. That column is
   deliberately absent from the printed tables. `--top 0`/negatives are rejected by the parser.
6. Good tickers for exercising the filter: `ARM` (too few OOS sessions), `HOOD`/`COIN`
   (negative Brier skill), `AAPL`/`NVDA` (credible).

## Known data caveat to re-check, not to re-file

The cached **index** series (`^GSPC`, `^IXIC`, European indices, `^VIX`, `^TNX`, ...) are often
several days staler than the individual stock series. `features.as_of` forward-fills, so a
stock forecast dated e.g. 2026-08-14 can be built from cross-market returns whose last real
observation is 2026-08-04. `stocks` now discloses this itself in a `stale inputs:` footer via
`stocks.stale_inputs()` (typically "42 of 63 series stop before ..."); check the footer is
present and that its count matches `df.dropna(subset=["Close"]).index[-1]` per symbol, rather
than re-filing it as a new finding. There is no look-ahead either way. `--refresh` is the
workaround if the network allows it; the `predict`/`export` paths have the same exposure with
no such footer.

## Rendering console evidence for a PR comment

No terminal emulator is installed and there is no GUI app to screenshot. `pip install pillow`
into the venv and render the captured log to a PNG (monospace on a dark background) so the PR
comment has an image; label it clearly as rendered CLI output. Pillow is for evidence only —
do not add it to the project's dependencies.

## Pre-existing CI failure, not yours

`tests/test_score.py::test_to_frame_sorts_and_rounds` fails in GitHub Actions
(`assert np.float64(5.6) == 5.56`) and fails identically on `main`. It does **not** reproduce
locally, even in a venv pinned to CI's exact stack (Python 3.12, numpy 2.5.2, pandas 3.0.5,
pytest 9.1.1, `pip install -e .`). Expect `214 passed, 1 failed` on CI against a green local
run, and do not "fix" `score.py` to chase it. The `check` workflow is not in the repo (only
`publish-snapshot.yml` is), so its setup cannot be inspected from the checkout.
