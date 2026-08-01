# Stock-Market-Predictor

A probability model for the **next opening auction** of the major global equity
indices. For every market it answers one question:

> Given everything the world knew before the bell, what is the probability that
> this index opens above its previous close?

Nine indices are covered across Asia, Europe and the Americas, driven by the
sessions that have already closed plus a set of cross-asset indicators (VIX, US
10y yield, dollar index, USD/JPY, EUR/USD, crude, gold, copper, S&P 500 and
Nasdaq futures).

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
| Nikkei 225 | 0.85 | 0.77 | 0.36 |
| ASX 200 | 0.85 | 0.77 | 0.36 |
| DAX | 0.80 | 0.73 | 0.27 |
| Euro Stoxx 50 | 0.80 | 0.72 | 0.27 |
| FTSE 100 | 0.79 | 0.72 | 0.25 |
| Hang Seng | 0.78 | 0.71 | 0.24 |
| Nasdaq Composite | 0.69 | 0.65 | 0.11 |
| S&P/TSX | 0.68 | 0.64 | 0.10 |
| S&P 500 | 0.68 | 0.63 | 0.09 |

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

### Data caveats

Yahoo publishes a stale opening price for some indices (it repeats the previous
close). Where that happens the opening auction is read from a liquid tracker on
the same exchange instead — `ISF.L` for the FTSE 100 and `STW.AX` for the ASX
200 — and any remaining zero-gap session is left unlabelled rather than counted
as a down open. A series with more than half stale opens is refused outright.

## Tests

```bash
python -m pytest tests -q
```

The tests pin the behaviour that matters most: the gap definition, the session
ordering rules, that features never read the future, and that the backtest is
strictly out of sample.

Not investment advice.
