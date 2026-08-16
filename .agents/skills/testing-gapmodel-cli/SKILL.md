---
name: testing-gapmodel-cli
description: How to runtime-test the gapmodel CLI (predict / screen / sectors / stock / shortlist) end to end - venv, cached price data, expected runtimes, and the checks that actually catch look-ahead and ranking bugs.
---

# Testing the gapmodel CLI

Terminal-only project: a Python library + `python -m gapmodel` CLI. There is no web UI or
server, so do **not** start a browser or a screen recording — collect stdout as text evidence.

Subcommands on `main`: `markets, fetch, score, screen, predict, backtest, asia, dashboard,
export, sectors, stock, shortlist`. The screener is invoked as `screen` even though the parser
variable is named `screener`.

The two single-stock commands are easy to confuse, and testing one proves nothing about the other:

- `stock SYM` — the curated registry in `gapmodel/stocks.py`: memory and storage (`MU`, `WDC`,
  `STX`), AI accelerators (`NVDA`, `AMD`, `AVGO`) and consumer hardware (`AAPL`). Each complex
  adds `peer_*` columns from the Asian names that trade the same demand overnight.
- `shortlist [SYM ...]` — the broad ranking in `gapmodel/shortlist.py` over the ~158-name US
  universe `modelled_universe()` in `gapmodel/universe.py` (`NASDAQ + LARGE_CAP + MID_CAP`, both
  venues). `nasdaq_universe()` is the 66-name venue slice and is no longer what `shortlist` runs.

They overlap on the curated names, so assert they **agree**: `shortlist SYM` must print the same
`p_open_up` and OOS metrics as `stock SYM` for every name in `stocks.STOCKS_BY_SYMBOL`, which holds
only because `cli._shortlist_equities()` downloads the peers of curated names too. A mismatch here
is the regression this check exists for (`shortlist MU` once printed 0.6331 against `stock MU`'s
0.6881, having silently dropped the peer columns). Every curated name is also in the universe, so
there is no name that one command models and the other refuses.

## Environment

- `source ~/.venvs/gapmodel/bin/activate` (the blueprint creates it; system python3 has no deps).
- Add `-W ignore` to every `python -m gapmodel ...` invocation: under **scikit-learn 1.9.0**
  the `penalty="l2"` in `model.py` emits a `FutureWarning: 'penalty' was deprecated` per fit,
  which swamps the output. Re-check whether this is still needed if sklearn is upgraded.
- Yahoo Finance bars are cached in `~/.cache/gapmodel/`, under a **sanitised** filename:
  `_cache_path` maps `^` to `idx_`, and `/` and `=` to `_`. So `^GSPC` is `idx_GSPC.csv` and
  `CL=F` is `CL_F.csv`, while plain tickers are `AAPL.csv`. Scripting a staleness check against
  a literal `<SYMBOL>.csv` raises a missing-file error for every index and future.
- Runs hit the network only for symbols missing from the cache (a garbage ticker therefore
  produces yfinance `ERROR` lines before the CLI's own `error:` message — expected, not a
  traceback).
- `python -m pytest tests -q` takes ~1 min (237 tests); `ruff check . && ruff format --check .`
  is instant.

## Runtime budget (walk-forward backtest per name, single-threaded per symbol)

- ~1.3 min per stock for `python -W ignore -m gapmodel shortlist SYM ...`.
- The full universe is ~158 names, so budget upwards of half an hour and start it in the
  background *first* (`nohup ... > /tmp/log 2>&1 &`), doing targeted runs while it works. CPU
  contention on this box matters as much as the count: 66 names took ~11 min alone and ~23 min
  alongside six other forecast jobs.
- `--gainers N` is the cheap way to exercise the whole path: every candidate's bars are loaded but
  only N walk-forwards are fitted, so a 12-name run costs about what 12 named symbols cost.
  A staleness case must be **synthesised** — the warm cache is date-uniform, so a test that relies
  on it to produce a lagging series passes vacuously.

## Checks that actually catch bugs (do these, not just "it printed a table")

Parse the CSV (`--csv PATH`) with pandas rather than eyeballing the table:

1. **Look-ahead**: forecast `session` must be strictly later than the last `Date` in that
   symbol's cache file (sanitised name — see above). `features.next_session_date` only skips
   weekends, so the expected value is "next weekday after the last cached bar" (holidays are
   ignored).
2. **Internal arithmetic**: `edge == round(p_open_up - base_rate, 4)` *exactly*, in the printed
   table and the CSV alike — the reported edge is derived from the two reported columns either
   side of it, so a reader checking it by hand cannot find a discrepancy. (It did not always
   hold: `edge` was once computed from the unrounded probability while `p_open_up` went through
   `predict._display`, which clamps to [1e-4, 1-1e-4] and rounds to 4dp, so 17 of 65 names were
   off by 1 in the last digit. Assert equality, not a 1e-4 tolerance, or the regression returns.)
   The unrounded `StockPick.edge` property still drives the ranking.
3. **Credibility filter** (`shortlist.StockPick.credible`): recompute
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
   (negative Brier skill), `AAPL`/`NVDA` (credible — but both are curated, so they also
   exercise the shortlist/`stock` parity check above; `TSLA` is a credible name with no peers).

## Faking a stale or missing series without harming the warm cache

The global `--cache DIR` flag (before the subcommand) is the way to test anything that depends on
the *shape* of the cached data — a series that stopped updating, a name with one bar, a missing
column. Copy the cache and edit the copy; never edit `~/.cache/gapmodel` in place, since re-fetching
it costs a slow network round trip for every symbol:

```bash
cp -a ~/.cache/gapmodel /tmp/probe-cache
md5sum ~/.cache/gapmodel/AMD.csv > /tmp/warm.md5   # verify untouched afterwards
# edit /tmp/probe-cache/AMD.csv, then:
python -W ignore -m gapmodel --cache /tmp/probe-cache shortlist AMD MU JPM XOM --gainers 2
md5sum -c /tmp/warm.md5
```

Pair it with a small explicit symbol list so the run costs two fits rather than 158. Truncating one
name's CSV to an earlier date *and* inflating its final close is what distinguishes "ranked on the
panel's latest session" from "ranked on each name's own last two bars": the doctored name has the
largest own-tail move, so it must still be excluded, and a run that lists it has the bug back.
Expect the same doctored series to be named twice — once by the mover-eligibility warning on stderr
and once by the `stale inputs:` footer — which are different mechanisms; do not read one as the
other.

## Pandas `na_rep` only reaches a float column

A missing numeric field rendered with `DataFrame.to_string(na_rep="")` prints blank only while the
column still has at least one real value and is therefore `float64`. If *every* row is `None` the
column is `object` dtype and pandas prints the literal `None`, `na_rep` notwithstanding. So test a
"missing value prints blank" claim in **both** shapes — one missing among valued rows, and every row
missing — or the all-missing case will slip through. `to_csv` writes an empty field in both, so the
CSV passing says nothing about the table.

## Known data caveat to re-check, not to re-file

The cached **index** series (`^GSPC`, `^IXIC`, European indices, `^VIX`, `^TNX`, ...) are often
several days staler than the individual stock series. `features.as_of` forward-fills, so a
stock forecast dated e.g. 2026-08-14 can be built from cross-market returns whose last real
observation is 2026-08-04. `shortlist` discloses this itself in a `stale inputs:` footer via
`shortlist.stale_inputs(panel, session)`, so check the footer rather than re-filing it as a
finding. There is no look-ahead either way; `--refresh` is the workaround if the network allows
it, and the `predict`/`export` paths have the same exposure with no such footer.

Verify the footer by building a lag histogram against the **forecast session** and asserting the
flagged set is exactly `{series : (session - last bar).days > STALE_DAYS}` (5). Expect a US series
that has not opened yet (lag 1) and a partial same-session Asian peer bar (lag 0) to be *absent*:
the count deliberately does not key off the freshest bar in the panel, which counted 129 of 134
series stale because Seoul was open, and swung between 42, 67 and 129 depending on which peers a
run happened to load. A count that moves with the peer set again is a regression — only the
denominator should. The live cache holds nothing lagging 3–8 days, so probe the threshold itself
with a synthetic dict of `pd.DataFrame`s at chosen lags rather than assuming it.

## Paired branch-vs-main comparison (the highest-value regression test)

When a PR claims "existing index/market outputs are unchanged", prove it by diffing real CLI output
between the branch and `main`, not by reasoning about the code:

```bash
git worktree add --detach /tmp/gapmain origin/main   # detached: `main` may be checked out already
cd /tmp/gapmain && python -c 'import gapmodel; print(gapmodel.__file__)'   # must print /tmp/gapmain/...
cd /tmp/gapmain && python -W ignore -m gapmodel predict --market ^GSPC --market ^FTSE --explain > /tmp/main.txt
cd <repo>        && python -W ignore -m gapmodel predict --market ^GSPC --market ^FTSE --explain > /tmp/branch.txt
diff /tmp/main.txt /tmp/branch.txt      # expect no output
git worktree remove --force /tmp/gapmain   # always clean up
```

Use `--explain` so driver names and log-odds values are compared too, not just probabilities. A
worktree is essential: it shares the cache but never touches the working tree. Compare against
`origin/main` rather than the local branch, which may be behind.

The `main` side only runs `main`'s code because `python -m` puts the cwd ahead of the editable
install on `sys.path` — nothing enforces it. Anything that removes the cwd (`-I`, `PYTHONSAFEPATH`,
invoking from elsewhere) silently compares the branch with itself and the empty diff means nothing,
so print `gapmodel.__file__` first and check it points into the worktree.

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
mu, label = build_features("MU", panel)  # returns a (DataFrame, Series) TUPLE
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

## Rendering console evidence for a PR comment

No terminal emulator is installed and there is no GUI app to screenshot. `pip install pillow`
into the venv and render the captured log to a PNG (monospace on a dark background) so the PR
comment has an image; label it clearly as rendered CLI output. Pillow is for evidence only —
do not add it to the project's dependencies.

## When CI fails a test that passes locally, suspect the merge, not the environment

CI is `.github/workflows/ci.yml` (job `check`: Python 3.12, `pip install -e . ruff pytest`, then
`ruff check`, `ruff format --check`, `pytest`). It triggers on `pull_request`, so **GitHub checks
out the merge of your branch with `main`, not your branch.** A test can therefore fail on CI
while passing on every local run, because the code under test only exists in that merge.

Not hypothetical: `tests/test_score.py::test_to_frame_sorts_and_rounds` failed
(`assert np.float64(5.6) == 5.56`) on a branch whose own `score.py` was correct. `main` had
momentarily carried `round(s.last, 1)` — committed deliberately in 5488bf7 as a smoke test for
CI automation, fixed again in #63 — and the PR was being tested against it. Time was lost
pinning numpy, pandas and pytest to CI's exact versions hunting a rounding difference that was
never there.

So when local is green and CI is red:

1. `git fetch origin` **first**. A stale `origin/main` hides this, and makes "the same failure
   is on `main`" look like proof that the failure is pre-existing and none of your business.
2. Reproduce against the merge, not the branch: `git merge origin/main` (or check out
   `refs/pull/<N>/merge`) and re-run the suite.
3. Only then suspect the environment.

Merging current `main` is usually the whole fix.

## Testing `journal` (the forecast journal)

`journal` both *writes* forecasts to a journal CSV and *settles* matured rows. Two rules make
it cheap and safe to test:

- **Always pass `--log /tmp/...`.** The default is the committed `docs/forecast-log.csv`; a bare
  `journal` run rewrites a tracked file. Finish with `git status --porcelain` to prove you did
  not. Copy the committed journal to `/tmp` when you want to test against the real seed.
- **`--settle-only` skips forecasting entirely**, so it returns in seconds instead of ~1 min per
  market. Almost every check (settle arithmetic, window/min-settled, decay, exit code) can be
  driven from a *hand-built* journal CSV plus `--settle-only`. Only idempotence-of-recording
  needs a real forecast.

A hand-built journal only needs `session,symbol,market,p_open_up,status` — `read_log` widens a
frame that is missing the optional columns, which is worth exercising on purpose. Set
`status=settled` and supply your own `outcome` to test the metrics directly; set
`status=pending` with real session dates to test settlement against the cache.

### Settle on `Market.gap_symbol`, never on the index

The single most important correctness check. Yahoo repeats the previous close as the *index*
open for some markets, so the model labels those on a tracker (`Market.open_source`):
`^FTSE` → `ISF.L`, `^AXJO` → `STW.AX`. Measured from the cache on this box:

| symbol | stale-open fraction |
| --- | --- |
| `^FTSE` | 96.3% |
| `ISF.L` | 1.5% |
| `^AXJO` | 48.4% |
| `STW.AX` | 6.6% |

So code that settles against `panel[symbol]` retires almost every FTSE session as `stale` and
the market silently never reaches `--min-settled` — a failure that looks like "no data yet"
rather than a bug. Test it by journalling ~30 real tracker sessions as pending, settling, and
asserting the recorded `prev_close`/`open` are at **tracker scale** (FTSE ≈ 1050, not ≈ 10800)
and that `status=settled` dominates. `journal.opening_bars` must mirror
`features.build_features` exactly (same `dropna(subset=["Open","Close"])`, same
`is_stock(target_symbol)` gate on `dividend_adjusted`) — grading has to use the price the label
was built from, so if you think the adjustment is wrong, it must be changed in *both* places or
the score stops measuring the model. Note the trackers distribute, and leaving them unadjusted
flips the gap sign on ~4% of `STW.AX` and ~0.7% of `ISF.L` sessions; that is a property of the
model's label, not of the journal.

### Forcing the statuses without touching the real cache

`cp -r ~/.cache/gapmodel /tmp/cache-mod` and pass `--cache /tmp/cache-mod`. Then, editing the
**gap symbol's** CSV:

- `stale` — usually occurs naturally; otherwise set a session's `Open` equal to the previous
  row's `Close` (tolerance is `features.STALE_GAP_TOLERANCE = 1e-9`).
- `no-session` — drop a row whose date you have journalled, keeping later rows.
- `pending` → `settled` — a trailing bar with `Close` NaN is dropped by `opening_bars`, so the
  newest session stays pending; filling that `Close` in the copy settles it. This is exactly why
  `^FTSE` can sit pending for a session the *index* already printed.
- `late` — needs the forecast session to already be in the panel, which the model normally
  prevents; `journal._already_printed(panel, symbol, session)` is the honest thing to probe
  directly, and it must flip with the tracker's bars, not the index's.

Assert unscorable rows never move the numbers: recompute the metrics over `status == "settled"`
rows only and require an exact 4dp match to the printed table.

### Decay has two legs — test them apart

`Skill.decayed` is `brier_skill <= 0 or hit_rate < drift_rate`, where
`drift_rate = max(base_rate, 1 - base_rate)`. Comparing against `base_rate` alone is wrong: in a
market that opens *down* 67% of the time, always saying "down" scores 67%, so a 60% hit rate
with a healthy Brier skill is still no read. A journal that hits both legs proves nothing about
the drift leg, so build one where `brier_skill > 0` and `base_rate < hit_rate < drift_rate` and
require it to be flagged anyway. A worked example that does it: 30 rows, 20 down / 10 up
(`base_rate` 0.333, `drift_rate` 0.667), probabilities `[0.15]*14 + [0.52]*6 + [0.55]*4 +
[0.48]*6` → hit 0.60, Brier skill +0.345, and it must still be listed under "below their own
drift" with `--fail-on-decay` exiting 1.

### Append-only is the other thing worth attacking

`record` keys on `(symbol, session)`. Run the same command twice and require the file to be
**byte-identical** (`diff`), not merely the same row count: the bug to catch is a second run
overwriting `p_open_up` or `recorded` in place, which a row count cannot see.

Beware that a seeded journal's probabilities stop reproducing once the cache is refreshed — a
seed row and a fresh forecast for the same session can differ (seen: `^GSPC` seed 0.6206 vs
fresh 0.6233) without anything being broken. `^FTSE` reproduces longer than the rest because
`ISF.L` lags the index by a bar.

## Devin Secrets Needed

None. Yahoo Finance is reachable unauthenticated from the test box.
