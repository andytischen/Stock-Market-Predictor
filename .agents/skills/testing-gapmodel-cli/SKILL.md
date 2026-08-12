---
name: testing-gapmodel-cli
description: How to runtime-test the gapmodel forecasting CLI — venv, warm Yahoo cache, paired branch-vs-main comparisons, and adversarial techniques for proving a data-adjustment or feature change only affects what it should.
---

# Testing the gapmodel CLI

CLI-only project (`python -m gapmodel ...`), no UI and no web server, so **do not record** — terminal
output is the evidence. Always `source ~/.venvs/gapmodel/bin/activate` first.

## Environment

- Repo: `/home/ubuntu/repos/Stock-Market-Predictor`; venv `~/.venvs/gapmodel` (the repo blueprint's
  `initialize`/`maintenance` already create it, install deps and put it on PATH — no extra setup needed).
- Price cache: `~/.cache/gapmodel`, one CSV per symbol plus sidecar `<sym>.start` and `<sym>.fields`
  files recording the requested start date and the requested columns.
- Yahoo is the data source and can be flaky. Per-symbol failures are logged as
  `WARNING skipping <sym>: ...` / `no forecast for <sym>: ...`. **Always grep collected stderr for
  `skipping ` / `no forecast` and report those separately from code defects.**
- Global options come *before* the subcommand: `python -m gapmodel --refresh dashboard --region Asia`.
- `--refresh` is slow and rewrites the cache — see the ordering rule below.
- stderr is noisy with sklearn `penalty` FutureWarnings (pre-existing, unrelated to any PR). Grep for
  the specific string you care about rather than eyeballing stderr, e.g.
  `grep -c "is not a feature of this target" run.err`.

## Hygiene commands

```bash
ruff check . && ruff format --check . && python -m pytest tests -q
```
The suite is fast (~6s). The expected test count changes every PR — get it from the lead rather than
assuming.

## Paired branch-vs-main comparison (the highest-value regression test)

When a PR claims "existing index/market outputs are unchanged", prove it by diffing real CLI output
between the branch and `main`, not by reasoning about the code:

```bash
git worktree add /tmp/gapmain main
cd /tmp/gapmain && python -m gapmodel predict --market ^GSPC --market ^FTSE --explain > /tmp/main.txt
cd <repo>        && python -m gapmodel predict --market ^GSPC --market ^FTSE --explain > /tmp/branch.txt
diff /tmp/main.txt /tmp/branch.txt      # expect no output
git worktree remove --force /tmp/gapmain   # always clean up
```

Use `--explain` so driver names and log-odds values are compared too, not just probabilities. A
worktree is essential: it shares the cache but never touches the working tree.

**Ordering rule:** run paired comparisons **before** any `--refresh`, so both sides read a
byte-identical cache. Otherwise new bars or newly-collected columns make a code difference
indistinguishable from cache drift. If you must test post-refresh behaviour, refresh once and then
re-run *both* sides again.

## Proving a change affects only the intended targets

Two techniques that turn "I didn't see a difference" into real evidence:

1. **Inject the triggering data rather than trusting the current cache.** A new column (e.g.
   `Adj Close`) is often absent from cached CSVs, so "the index output didn't change" can be vacuous.
   Build the excluded target's features twice — once with the column injected into its source symbol,
   once without — and assert `pd.testing.assert_frame_equal` on the features *and*
   `assert_series_equal` on the labels. Make the injected factor realistic (a step function, e.g. 1%
   per quarter, matches how Yahoo's `Adj Close` actually behaves) and report how many rows it *would*
   have moved had the adjustment applied, so the test is shown to be capable of failing.
   After a real `--refresh` the column may become genuinely present — prefer that stronger version.
2. **Reproduce the old logic inline for contrast.** To show a fix matters, compute both the new and
   the previous behaviour in one script and quantify the gap (e.g. an old `ffill().fillna(1.0)`
   invented a −71.8% opening gap at the data boundary where `ffill().bfill()` gives +0.186%).

## Comparing adjusted vs unadjusted price series: mind the float noise

`Adj Close / Close` is never exactly 1.0 in floating point, so a naive `atol=1e-12` comparison
reports *every* session as differing (~5,370 of 5,435 for one name). Only above a material threshold
(~`1e-6`) does the true count appear (80 of 5,434, matching the documented figure). Always state the
threshold used. Related: a same-session multiplicative factor cancels out of intraday
`log(Close/Open)`, so that should be identical to ~1e-16 — a good invariant to assert.

## Verifying feature lags / look-ahead

The strong form: reconstruct each feature column from its source series at **both** candidate lags and
require a match at exactly one — the expected one. A column matching both lags proves nothing.

```python
mu, label = build_features("MU", panel)      # returns a (DataFrame, Series) TUPLE
```
Forward-fill the source onto a calendar index before reindexing, since features are aligned to the
target's trading sessions. Column names carry suffixes (`own_close_return_lag1`, not
`own_close_return`) — list `frame.columns` rather than guessing.

For peer/dividend-sourced columns, test on sessions that **straddle an actual dividend**: the most
recent sessions usually have none, so both candidate source series agree there and the test is
inconclusive. Find sessions where the two sources differ by more than `1e-6` first.

## The intraday path may be untestable end-to-end

`--intraday` needs hourly futures bars stamped *after* the previous cash close. If the hourly feed
stops earlier (check `load_hourly_panel(cache_dir=..., refresh=True)` and print each series'
`index[-1]`), the CLI logs `no pre-open futures bars: falling back to the daily model` and uses the
daily model — correct documented behaviour, not a bug. Confirm it is not target-specific by trying an
index target too; and confirm the `pre_*` features do exist historically via
`build_features(sym, panel, hourly=hourly)` (without `forecast_row=True`), which is the difference
between "flag ignored" and "flag works, live data unavailable".

## Forcing date-driven or calendar-driven code paths

Use a throwaway script under `/tmp` that monkeypatches module attributes in memory — never edit repo
files. Note that `cli.py` imports some names directly (e.g. `SCENARIOS` from `gapmodel.scenarios`), so
both the defining module and `gapmodel.cli` may need patching for the CLI to see the change.

## Devin Secrets Needed

None. Yahoo Finance is reachable unauthenticated from the test box.
