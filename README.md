# Stock-Market-Predictor

A probability model for the **next opening auction** of the major global equity
indices. For every market it answers one question:

> Given everything the world knew before the bell, what is the probability that
> this index opens above its previous close?

Seventeen indices are covered across Asia, Europe and the Americas, driven by the
sessions that have already closed plus a set of cross-asset indicators (VIX, the
US 5y/10y/30y yields, the priced policy rate, Russell 2000, the semiconductor
index, ASML, the eighteen STOXX Europe 600 sectors, dollar index, USD/JPY,
EUR/USD, GBP/USD, WTI and Brent crude, gold, silver, copper, S&P 500 and Nasdaq
futures). Run
`python -m gapmodel markets` for the full list with session times.

Each market is then analysed by its own bespoke model: a separate probability
model is fitted, back-tested and explained per index, so Tokyo is never scored
with Wall Street's coefficients.

The same question can be asked of one company rather than an index —
`python -m gapmodel stock MU` — with the peers that trade the same demand
overnight added to its features. That is a narrower claim, and the section below
says exactly how much narrower.

Crude is the market's fastest read on Middle East supply risk — strikes on Iran
push it up, negotiations and a deal unwind it — so both benchmarks additionally
carry a 5-day return, 20-day realised volatility and a shock feature (the daily
move divided by the volatility already known the day before).

All eighteen STOXX Europe 600 sectors are carried as well (banks, technology,
construction & materials, oil & gas, autos, basic resources, retail, travel and
the rest), as their Xetra-listed iShares trackers — the underlying `.SXxP`
indices have no usable Yahoo symbol. Each closes after every tracked market
opens, so it is always read a session late, and each carries a 5-day return as
well as the daily one because one session of a sector index says little on its
own. Sector features go to **European targets only**: outside the region they
measurably dilute the fit.

```bash
python -m gapmodel sectors --market ^GDAXI   # split one open call by sector
python -m gapmodel predict --shock 'EXV3.DE=-3%'   # sectors are shockable too
```

`sectors` ranks the sectors by how much log-odds each one contributes to that
index's next-open probability, alongside its 1-day and 5-day move, and prints
the net. It only works for European indices, since they are the only ones
carrying the features.

The shape of the crude curve is carried alongside its level. Yahoo serves only
the generic front contract, so the two ends are read from the oil funds that
track them — `USO` rolls the front month, `USL` holds a twelve-month strip — and
only their *difference* becomes a feature, daily and over 60 sessions, so the
funds' own price levels and tracking drift cancel. Negative is contango (the
front lagging, supply comfortable), positive is backwardation. Either leg can be
shocked, which tilts the curve without moving the level:

```bash
python -m gapmodel predict --shock 'USO=-2%'   # front month sells off: deeper contango
```

What the market has priced for the Fed is carried separately from what long
bonds yield, because a 10-year yield rises on inflation and on growth alike and
cannot express "is September a hike?". The 30-day fed funds future settles on the
average effective funds rate over its delivery month, so 100 minus its price is
the rate the front month is priced for; the 13-week bill carries the same
expectation a quarter out, and the difference between the two is the tightening
(or easing) the next three months have priced in. Both readings are levels in
percentage points rather than log returns: a future priced near 100 has
meaninglessly small returns, and bill yields have sat at zero, where a log return
is undefined. The *changes* in the priced rate were built too, and dropped —
last of eighty-three features by weight, as a series moving in single basis
points outside a meeting week deserves to be.

A hawkish turn is only partly visible this way. The pricing moves, so the model
sees it; the speech that moved it, it does not. Measured over the full
walk-forward, these features are worth about ±0.004 AUC depending on the market —
inside the noise, and not the reason to keep them. They earn their place in the
attribution, where the policy legs make an otherwise unexplained call legible.

### Scheduled releases the model cannot anticipate

Every feature is a price, and a price cannot anticipate a number that has not
been published. A call for a session carrying payrolls, CPI, PCE or an FOMC
decision is built entirely from a world in which that release has not happened,
which makes it a narrower answer than it looks. `predict` says so:

```
scheduled releases this model cannot see:
  S&P 500: US payrolls at 12:30 UTC, before this open: the auction prices it and
  the model cannot — treat the probability as stale
  FTSE 100: US payrolls at 12:30 UTC, after this open: the call still stands for
  the auction but says nothing about the session
```

Every date comes from the publishing agency's own calendar — BLS for payrolls and
CPI, BEA for PCE, the Board for FOMC decisions — and none is derived from a rule,
because the rules do not hold: payrolls are conventionally the first Friday of
the month, and in 2026 the BLS scheduled them for a Wednesday in February, the
second Friday in May and a Thursday before Independence Day in July. Times are
converted from New York's clock for the date in question, so a 14:00 ET statement
reads 18:00 UTC in September and 19:00 UTC in December.

The cost of not guessing is that the tables end: **past `CALENDAR_END` the
absence of a warning means nothing was checked, not that nothing is scheduled.**
Refreshing them is a yearly job, and each `Schedule` carries the page it came
from.

## Install

```bash
pip install -r requirements.txt   # or: pip install -e .
```

## Use

```bash
python -m gapmodel markets            # what is modelled, and when each session runs
python -m gapmodel fetch              # download and cache ~20 years of daily bars
python -m gapmodel predict --explain  # probability that the next open is up
python -m gapmodel predict --intraday # add pre-open futures moves (recent window)
python -m gapmodel stock MU --explain  # one company's next open, with its drivers
python -m gapmodel predict --shock '^KS11=+2%'  # what-if: re-run under a hypothetical move
python -m gapmodel predict --shock 'CL=F=-5%' --shock 'JPY=X=+2%'  # shocks compose
python -m gapmodel backtest --reliability
python -m gapmodel dashboard --at 05:00 --html asia.html   # crude vs the Asian session
python -m gapmodel screen             # US stocks: liquid, unusually active, moving
python -m gapmodel shortlist --top 10 # rank the Nasdaq universe by demonstrated edge
python -m gapmodel scorecard          # journal today's calls and score the settled ones
```

`predict` prints one row per market with the probability and the out-of-sample
quality of that market's model:

```
           market   symbol   region    session  p_open_up  oos_auc  oos_brier_skill  oos_accuracy  base_rate
       Nikkei 225    ^N225     Asia 2026-08-03     0.8123   0.8481           0.3642        0.7699     0.5382
        S&P 500     ^GSPC Americas 2026-08-03     0.5412   0.6773           0.0886        0.6332     0.5583
```

`--explain` adds the largest log-odds contributions behind each probability, so
a forecast can always be traced back to the indicators that moved it.

### Dashboard

`dashboard` is the 05:00 GMT view: what crude did overnight, and what the Asian
session is doing with it. `--at` pins the clock (session state is read at that
UTC time, so `--at 05:00` shows Tokyo, Hong Kong, Seoul, Shanghai and Mumbai
mid-session and Sydney already shut), `--region` switches to Europe or the
Americas, and `--html` writes a standalone page.

```
Crude:
  Brent crude     90.12  1d +1.22%  5d -7.13%  vol20 4.42%  shock +0.3
  WTI crude       84.67  1d +1.28%  5d -5.34%  vol20 3.90%  shock +0.3

market               session    to open  last close  last move  p(open up)  oil log-odds
Hang Seng            open             -   25,884.43     +0.10%       31.1%        -0.047
Nikkei 225           open             -   64,362.02     +3.95%        6.8%        +0.021
ASX 200              closed       18.0h    8,976.80     +0.10%       21.2%        -0.075
```

The oil columns are the same variables the models are fitted on — the daily and
5-day crude return, its 20-day realised volatility and the shock (today's move
divided by the volatility known beforehand) — and `oil log-odds` is the net
amount those features add to, or take off, each market's probability, with the
single largest oil driver listed underneath. A crude move of two standard
deviations or more is flagged as a shock.

## How it works

1. **Data** — daily OHLC bars from Yahoo Finance, cached as CSV under
   `~/.cache/gapmodel`.
2. **Features** — log returns of every other index and every indicator, plus the
   target's own recent gaps, returns and realised gap volatility.
3. **Model** — standardised L2 logistic regression, which yields probabilities
   that are close to calibrated out of the box and coefficients you can read.
4. **Evaluation** — an expanding-window walk-forward backtest that refits every
   21 sessions and only ever scores days the model has not seen, reported as
   AUC, accuracy, log loss, Brier score and Brier skill against the base rate.

### No look-ahead by construction

Each instrument declares the UTC time at which its daily bar becomes known, and
each market declares the UTC time of its opening auction. An indicator enters a
prediction only if its bar closed *strictly before* that auction — so Tokyo's
close feeds the European open, while Wall Street's close only feeds Tokyo on the
following session. Weekends and holidays fall back to the most recent earlier
observation rather than silently borrowing a future one.

### Indicative out-of-sample results (2005–2026)

| Market | AUC | Accuracy | Brier skill |
| --- | --- | --- | --- |
| KOSPI | 0.85 | 0.78 | 0.37 |
| ASX 200 | 0.85 | 0.78 | 0.37 |
| Nikkei 225 | 0.85 | 0.77 | 0.37 |
| DAX | 0.81 | 0.74 | 0.29 |
| Euro Stoxx 50 | 0.81 | 0.73 | 0.28 |
| CAC 40 | 0.81 | 0.74 | 0.28 |
| IBEX 35 | 0.81 | 0.74 | 0.28 |
| FTSE 100 | 0.81 | 0.73 | 0.28 |
| Hang Seng | 0.79 | 0.72 | 0.25 |
| Swiss Market Index | 0.78 | 0.71 | 0.24 |
| Shanghai Composite | 0.77 | 0.72 | 0.21 |
| Nifty 50 | 0.75 | 0.73 | 0.18 |
| Nasdaq Composite | 0.71 | 0.66 | 0.13 |
| S&P 500 | 0.70 | 0.66 | 0.11 |
| S&P/TSX | 0.69 | 0.64 | 0.11 |
| Dow Jones Industrial Average | 0.67 | 0.62 | 0.08 |
| Bovespa | 0.65 | 0.62 | 0.07 |

Asian and European opens are largely explained by the US session that closed
while they slept. Wall Street's own open is much harder from daily bars alone:
what moves it — overnight futures right up to the bell — is not in a daily bar.

### `--intraday`: pre-open futures

`--intraday` adds the overnight and last-hours moves of ES, NQ, crude and gold,
measured from hourly bars as of the opening bell. Yahoo only serves ~730 days of
hourly history, so this variant trains on a much shorter window (with a smaller
warm-up), but it is exactly what the US open was missing. Compared on the *same*
600 sessions:

| Market | AUC daily | AUC intraday |
| --- | --- | --- |
| S&P 500 | 0.67 | 0.90 |
| Nasdaq Composite | 0.65 | 0.91 |
| S&P/TSX | 0.65 | 0.86 |
| DAX | 0.71 | 0.81 |
| Nikkei 225 | 0.78 | 0.84 |
| FTSE 100 | 0.73 | 0.78 |

An hourly bar is stamped with the *start* of the hour it covers, so a bar counts
as known only one hour later — otherwise the bar straddling the bell would leak
the answer into the features.

`predict --intraday` only produces a forecast when futures have actually traded
since the target's previous close, which is the point of the feature: run it in
the hours before the bell. Over a weekend it refuses rather than reporting a
fabricated zero overnight move.

Two things keep that from costing a forecast when the data, rather than the
clock, is at fault. Yahoo's hourly endpoint sometimes stops updating hours
before its finer ones do, so a stale hourly tail is topped up from 30-, 15- or
5-minute bars resampled onto the same hourly grid — but only when the gap is
small enough for those feeds to bridge, since they carry a few days of history
each. And if the pre-open features still cannot be built, the run falls back to
the daily model with a warning instead of returning nothing.

### Single stocks

`stock` forecasts one company's opening auction with the same machinery, and the
memory and storage complex is what it is pointed at first:

```bash
python -m gapmodel stock                 # every modelled stock
python -m gapmodel stock MU --explain    # one name, with its drivers
python -m gapmodel stock MU --shock '000660.KS=-4%'   # peers are shockable too
python -m gapmodel backtest --market MU --reliability  # the table below, rebuilt
```

On top of the indices and cross-asset indicators every target reads, a stock
reads its **peers**: the companies pricing the same end demand. Half of them are
Asian, which is the structural point — Seoul closes at 06:30 UTC and Tokyo at
06:00, so Samsung, SK Hynix, Tokyo Electron, Advantest and TSMC have already
traded the overnight memory story hours before New York opens, and their bars are
*same-session* information for a US listing. The US legs (NVDA, AMAT, SMH and the
other storage names) close with Wall Street and are read a session late, as every
other American bar is.

| Stock | AUC | Accuracy | Brier skill | Base rate |
| --- | --- | --- | --- | --- |
| Micron (MU) | 0.69 | 0.65 | 0.10 | 0.57 |
| Western Digital (WDC) | 0.66 | 0.62 | 0.07 | 0.53 |
| Seagate (STX) | 0.65 | 0.62 | 0.06 | 0.54 |

Better than the S&P's daily model, and for a plain reason: a single stock's gap
is more autocorrelated and more exposed to a sector move than an index average
is. The peer block is worth little over the whole 2008–2026 sample (±0.006 AUC,
inside the noise) but clearly earns its place on the last two years, where memory
dispersion has been the whole story — MU 0.712 with peers against 0.685 without.

One correction a single name needs and an index does not: Yahoo's daily bars are
split- but not dividend-adjusted, so on the morning a company goes ex-dividend
the opening print falls by roughly the dividend and the session would be labelled
a down gap the market never made. For stock targets and their peers the dividend
factor in `Adj Close` is applied to both prints of the same session, which leaves
that session's own returns alone and corrects only the previous-close-to-open
step. Seagate yields around 3%, so this is four labels a year with a known sign.
Index targets are left on their published prints.

None of this makes a single-name probability comparable to an index one. Results,
guidance, analyst actions, index changes and company news move an individual open
more than the overnight tape does, and none of them are features, so every
`stock` run prints that list underneath the table. The scheduled-release caveats
apply unchanged: a US stock opens at 13:30 UTC, an hour after a CPI print.

### What-if shocks

`--shock SYMBOL=MOVE` re-runs every market with a hypothetical move added to one
instrument and prints `p_shocked` and `p_change` alongside the live probability.
The move applies to every feature derived from that instrument's latest bar, and
inherits the same timing rules as the model: for markets opening after that
instrument's close it is *today's* move, for markets opening before it, it is the
one they will only see tomorrow — which is why a Korean rally reads as a
different sign in Tokyo than in Frankfurt. The model is linear in log-odds, so a
move far outside the training range is an extrapolation, not a forecast. Only
`stock` accepts a shock on a single-name peer: no index feature is derived from
one, so `predict` refuses it rather than printing an unchanged probability.

### Named scenarios

`--scenario NAME` applies a bundle of moves that belong to one macro event, so
the recurring ones need not be spelled out leg by leg:

```bash
python -m gapmodel predict --scenario opec-supply-increase
python -m gapmodel predict --scenario opec-supply-increase --shock 'CL=F=-6%'
```

`markets` prints every scenario with its legs. A `--shock` on an instrument the
scenario also moves replaces that leg, so a scenario can be re-sized without
editing it. Sizes describe the one-session reaction to the announcement, not the
drift that follows it.

### Data caveats

Yahoo publishes a stale opening price for some indices (it repeats the previous
close). Where that happens the opening auction is read from a liquid tracker on
the same exchange instead — `ISF.L` for the FTSE 100 and `STW.AX` for the ASX
200 — and any remaining zero-gap session is left unlabelled rather than counted
as a down open. A series with more than half stale opens is refused outright.
Bovespa is the worst of the included markets (a quarter of its opening prints
repeat the previous close), which is part of why it scores lowest.

## Asia session dashboard

`asia` is the layer below the probability model: instead of scoring an index as
a whole it asks who inside it moved, on what participation, and which outside
market it was following. (`dashboard`, above, is the crude-versus-Asia board;
this one is about the indices themselves.)

```bash
python -m gapmodel asia --out asia-dashboard.html   # standalone HTML page
python -m gapmodel asia                             # the same, as text
```

It covers the headline index of each major Asian market — Nikkei 225, KOSPI,
CSI 300, Hang Seng and the Straits Times — plus Euro Stoxx 50, the DAX and the
FTSE 100 behind them, and for each one reports:

- **Index activity**: opening gap, 1/5/20-day return, 20-day realised
  volatility, and session volume against its own 60-day average.
- **Dominant companies**: the ten heavyweights, their weight, their
  contribution to the index move in basis points (weight × move), their beta to
  the index, their volume against average and their share of traded value.
- **Outside drivers**: univariate regressions (beta, t, R², implied basis
  points) of the index return on India, the Middle East, European futures and
  Wall Street, plus the joint R² of each theme.

Every driver is lagged by the same UTC session rule the gap model uses, so a
market that closes after the index opens is only read from the previous day.

[docs/asia-session-analysis.md](docs/asia-session-analysis.md) reads a run of
it: what dominates each index, whether India matters (mostly as a co-mover),
how Middle East risk enters, and what the free data cannot show — order book
depth, margin and short interest, investor-type breakdowns and licensed index
weights are all listed on the page as gaps rather than approximated.

## JSON snapshot

`export` writes a single JSON file with one entry per market — its next-open
probability, out-of-sample quality and the largest log-odds drivers — plus the
crude readings and a terse one-line summary. It is the mobile app's data source.

```bash
python -m gapmodel export                       # to stdout
python -m gapmodel export --out snapshot.json   # to a file
```

The workflow in `.github/workflows/publish-snapshot.yml` runs it daily at 06:30
UTC and publishes the file to GitHub Pages (enable Pages with the "GitHub
Actions" source in repository settings). The published `snapshot.json` is what
the app downloads and renders.

## Shortlisting the Nasdaq universe

`stock` above forecasts a handful of names in depth, each with a hand-written
list of the peers that trade the same demand overnight. `shortlist` takes the
opposite trade: every name in `universe.py`, read with the cross-market and
cross-asset panel a US index reads plus its own history, and then filtered hard
on whether the walk-forward record justifies reading it at all. Same pipeline —
same features, same backtest, same calibration — and the same total-return
treatment of dividends, so an ex-dividend morning is not labelled a down open.
What a shortlisted name does *not* get is peers; only the curated names have
them, which the metrics beside each row price in.

```bash
python -m gapmodel shortlist                        # the whole Nasdaq universe
python -m gapmodel shortlist AAPL NVDA MSFT         # just these
python -m gapmodel shortlist --top 10 --csv out.csv # strongest ten, all of them to CSV
```

The names are targets and never features. Adding sixty shares to `MARKETS` would
hand every index sixty new collinear columns and silently change the forecasts
above, so `target_market()` describes a shortlisted name on demand instead and
the default download panel is untouched. A ticker outside the universe is
refused rather than downloaded and ranked beside the rest: add it to `NASDAQ` in
`universe.py` to forecast it.

Two columns matter more than the probability, because a naive ranking flatters
itself twice over:

- **`base_rate`** — a share's opening gap is not a coin flip to begin with.
  Twenty years of a compounding growth name leave its unconditional up-rate
  well above 50% (Nvidia's is 0.57), so the names at the top of a probability
  sort are partly just the names with the strongest drift. **`edge`** is the
  probability against that base rate, which is what the model actually claims
  to add.
- **`oos_auc`, `oos_brier_skill`, `n_oos`** — a confident probability from a
  model with no demonstrated skill is noise with a decimal point. A name is
  ranked only if all three hold: AUC at least `MIN_AUC` (0.55), *positive* Brier
  skill, and at least `MIN_OOS` (500) out-of-sample sessions. All three are
  needed. AUC alone would put a recent listing first on a couple of hundred
  predictions, where 0.72 is within noise of nothing (Arm scores exactly that on
  193); and AUC is blind to calibration, so a model that orders sessions well
  while being confidently miscalibrated — a negative Brier skill — would
  otherwise read as a pick (Robinhood, again exactly that). Names that fail are
  printed in a separate unranked block with the test they failed, and the
  ranking orders on the edge *weighted by* the skill behind it, so a bold call
  from a coin-flip model cannot outrank a modest call from a good one.

The horizon is narrow and worth restating: the target is the overnight move into
the auction, not a view on the company and not a view on the session after the
bell. The universe in `universe.py` is a hand-maintained snapshot of today's
Nasdaq names, which means the backtest metrics carry survivorship bias — the
names that were delisted or acquired are absent, so a genuinely point-in-time
universe would read worse.

## Forecast journal and live skill

The backtest measures the model against history. `scorecard` measures it against
sessions that had not happened when the probability was written down, which is
the number a reader of a live forecast can actually act on.

```bash
python -m gapmodel scorecard                       # record today, settle what printed, score
python -m gapmodel scorecard --settle-only         # score the journal without forecasting
python -m gapmodel scorecard --window 120 --fail-on-decay
```

Each run appends its forecasts to `docs/forecast-log.csv`, one row per market
and session, then fills in the realised open for the rows whose auction has
since printed. A session already in the journal is never re-forecast and never
overwritten, so a probability cannot be improved after the fact and a run
repeated twice in a morning does not get two attempts at the same open.

Scoring follows the label the model is fitted on — an opening print above the
previous close — and retires the sessions that cannot carry one rather than
counting them: an open that merely repeats the previous close (`stale`, the same
sessions the model refuses to be fitted on) and a session the market never held
(`no-session`, a holiday the journal did not know about).

A third status covers the row that looks like a forecast but is not one. The
session a model forecasts is the one after the last session it has *complete*
features for, so a market still missing an indicator for yesterday is forecast
for an auction that has already printed. Nothing is leaked — every feature is
lagged either way — but it is not a call anybody could have acted on, so it is
journalled as `late` and left out of the live record.

```
forecast journal: 320 rows  settled 288  pending 16  unscorable 16

live record over the last 60 settled sessions per market:
 market symbol  settled  hit_rate  base_rate  brier  brier_skill  mean_p       from         to
S&P 500  ^GSPC       30      0.50       0.60 0.2666      -0.1109  0.5637 2026-07-06 2026-08-14

below their own base rate — the model is not adding a read here:
  S&P 500 (^GSPC): hit 50% against a 60% base rate, Brier skill -0.111 over 30 sessions
```

Skill is measured against each market's own realised up-rate over the same
sessions, not against a coin flip: predicting "up" every morning in a market
that opens up 60% of the time is drift, not a read, and a Brier score has to
clear that constant forecast before it says anything. A market below its own
base rate is called out by name, and `--fail-on-decay` turns that into a
non-zero exit so a scheduled run can raise it. Nothing is reported for a market
with fewer than 20 settled sessions: the sampling error on a hit rate over a
handful of opens is wider than any decay worth alerting on.

## Trend score

`score` ranks an arbitrary list of tickers by a single price-derived number: the
standardised position of the latest close within its own trailing window,

```
score = (close - mean(close, window)) / stdev(close, window)
```

so a stock riding the top of a long uptrend reads strongly positive and one at
the bottom of its range reads negative. It uses the same cached Yahoo bars as
the rest of the model.

```bash
python -m gapmodel score IVZ JPM DDOG CLX BWIN   # strongest first
python -m gapmodel score IVZ JPM --window 100 --asof 2026-08-04 --csv out.csv
```

This was built to approximate the sorted, heat-mapped "Score" column of a
ThinkorSwim watchlist, whose study formula is not published. Reverse-engineering
against a 27-name sample of that column, only long-horizon trend measures
correlate with it at all, and all weakly: the 200-day price z-score (the default
`--window`) tracks it at r ≈ 0.47, on par with the 200-day Bollinger %b, while
short-window RSI/ROC/MACD/%B, the TTM-squeeze momentum and a fitted blend of many
indicators have essentially no out-of-sample skill. A long-lookback RSI shows a
higher *raw* correlation, but Wilder's RSI is path-dependent on where the price
history starts, so its value drifts with the download window and it is not
reproducible; the z-score depends only on the trailing window, so it was chosen
instead. Either way this is an *approximation of the ranking*, not a reproduction
of the column: the real one is driven by inputs a daily price bar does not
contain.

## Stock screener

The model calls the *index*. `screen` is the layer below it on the US side: of
the names inside that market, which ones are worth looking at for the session
just traded. It narrows the universe in three stages, printing how many names
survive each one, so the result reads as a funnel:

```
US universe  ->  liquid  ->  unusually active  ->  actually moving
```

```bash
python -m gapmodel screen                        # the default US universe
python -m gapmodel screen --etfs --csv movers.csv
python -m gapmodel screen --asof 2026-08-07      # screen a completed session
python -m gapmodel screen AAPL NVDA PLTR         # or just these names
python -m gapmodel screen --universe my_list.txt --min-rel-volume 2
```

```
Screen funnel:
  universe     155  US names with enough history to screen
  liquid       104  price >= $5, 30d average volume >= 5M
  active        12  volume >= 2M, relative volume >= 1.25x
  moving         8  change >= +1.0%, ATR >= 2.0% of price

symbol   last  change  volume_m  avg_volume_m  rel_volume  atr_pct       asof
  DKNG  24.03    8.39     37.76         12.06        3.13     4.55 2026-08-07
     U  43.00    5.37     18.55          9.07        2.05     5.43 2026-08-07
  LYFT  17.46    7.12     23.94         11.87        2.02     4.14 2026-08-07
  PLTR 172.01   10.32     77.38         45.48        1.70     5.39 2026-08-07
```

| stage | filter | flag |
| --- | --- | --- |
| liquid | price ≥ $5 | `--min-price` |
| liquid | 30-day average volume ≥ 5M shares | `--min-avg-volume`, `--avg-window` |
| active | today's volume ≥ 2M shares | `--min-volume` |
| active | relative volume ≥ 1.25× | `--min-rel-volume` |
| moving | daily move ≥ +1% | `--min-change` |
| moving | ATR ≥ 2% of price | `--min-atr`, `--atr-window` |

Survivors are ranked by relative volume, since that is what separates a real
move from a name that happens to be up on its usual turnover.

Two details decide what the numbers mean. **Relative volume** is measured against
the 30 sessions *before* the one being screened: including today would put the
day's own volume into its own baseline, which flattens exactly the spikes the
test is looking for (a day trading 10× its average reads as only ~7.5× on a
window that contains it). **ATR** is a plain mean of the true ranges over the
window, not Wilder's smoothing, for the same reason the trend score avoids
Wilder's RSI — that recursion is path-dependent on where the price history
starts, so its value drifts with the download window, while a mean of the last
14 true ranges depends only on those sessions. True range counts overnight gaps,
not just the session's own high-low, so a stock that gaps and then sits still is
still recognised as one that moves.

The starting universe (`gapmodel/universe.py`) is a hand-maintained snapshot of
liquid US listings — the S&P 100 plus the mid-caps and high-beta names that
actually print unusual volume, with the heavily traded ETFs behind `--etfs`. It
is deliberately a superset: every filter is applied to real bars, so a name that
has gone quiet or been delisted is dropped by the funnel rather than needing to
be pruned by hand. Symbols, `--universe` and `--etfs` are alternatives; asking for
more than one is an error rather than a silent precedence rule.

Volume for a session still in progress is partial, so a screen run mid-session
understates today's volume and relative volume and returns fewer names than the
same screen after the close; `--asof` screens a completed session. Unlike the
model panel, the screen is *about* the latest session, so a cached series whose
last bar predates the session being screened is re-downloaded even without
`--refresh` — yesterday's cache would otherwise silently re-screen yesterday.

## Project tracker

`pm` is a small task board kept in a JSON file next to the code, used to track
the work on this model and to say what is still needed to finish it.

```bash
python -m pm add "Add European sector indices" --owner ana --due 2026-09-01
python -m pm need 1 "paid intraday data feed"
python -m pm status 1 doing
python -m pm list --status doing
python -m pm report --out STATUS.md
python -m pm deck                      # the same board as a reveal.js slide deck
```

The live board is `project.json` and the current update is checked in as
[STATUS.md](STATUS.md); regenerate it with `python -m pm report --out STATUS.md`
after changing the board. `pm deck` writes the same content as slides to
[docs/status-deck.html](docs/status-deck.html) — open it in a browser to
present the update rather than read it.

`report` prints a markdown status update: percentage complete, what is done, in
progress, blocked or next, overdue tasks, and a "Resources needed" section
collecting every resource requested by a task that is not finished yet. The
board lives in `project.json` by default; pass `--file` to keep several.

## Tests

```bash
python -m pytest tests -q
```

The tests pin the behaviour that matters most: the gap definition, the session
ordering rules, that features never read the future, and that the backtest is
strictly out of sample.

For runtime testing of the CLI against real price data — cache layout, how long a
walk-forward run takes, and the checks that actually catch look-ahead and ranking
bugs — see [`.agents/skills/testing-gapmodel-cli`](.agents/skills/testing-gapmodel-cli/SKILL.md).

Not investment advice.
