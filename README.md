# Stock-Market-Predictor

A probability model for the **next opening auction** of the major global equity
indices. For every market it answers one question:

> Given everything the world knew before the bell, what is the probability that
> this index opens above its previous close?

Sixteen indices are covered across Asia, Europe and the Americas, driven by the
sessions that have already closed plus a set of cross-asset indicators (VIX, the
US 5y/10y/30y yields, Russell 2000, the semiconductor index, dollar index,
USD/JPY, EUR/USD, GBP/USD, WTI and Brent crude, gold, silver, copper, S&P 500
and Nasdaq futures). Run `python -m gapmodel markets` for the full list with
session times.

Each market is then analysed by its own bespoke model: a separate probability
model is fitted, back-tested and explained per index, so Tokyo is never scored
with Wall Street's coefficients.

Crude is the market's fastest read on Middle East supply risk — strikes on Iran
push it up, negotiations and a deal unwind it — so both benchmarks additionally
carry a 5-day return, 20-day realised volatility and a shock feature (the daily
move divided by the volatility already known the day before).

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
python -m gapmodel backtest --reliability
python -m gapmodel dashboard --at 05:00 --html asia.html   # crude vs the Asian session
python -m gapmodel web --region Asia --at 05:00             # local browser interface
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

`web` serves the same dashboard behind a lightweight local browser interface.
It starts an HTTP server (default `http://127.0.0.1:8000/`) with a small form
to switch region/time and re-render live:

```bash
python -m gapmodel web --region Asia --at 05:00
python -m gapmodel web --host 0.0.0.0 --port 8080 --no-browser
```

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
| S&P 500 | 0.70 | 0.66 | 0.12 |
| S&P/TSX | 0.69 | 0.64 | 0.11 |
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

### Data caveats

Yahoo publishes a stale opening price for some indices (it repeats the previous
close). Where that happens the opening auction is read from a liquid tracker on
the same exchange instead — `ISF.L` for the FTSE 100 and `STW.AX` for the ASX
200 — and any remaining zero-gap session is left unlabelled rather than counted
as a down open. A series with more than half stale opens is refused outright.
Bovespa is the worst of the included markets (a quarter of its opening prints
repeat the previous close), which is part of why it scores lowest.

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

Not investment advice.
