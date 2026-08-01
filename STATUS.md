# Stock-Market-Predictor — status update 2026-08-01

Progress: 5/13 tasks complete (38%).
2 task(s) blocked.

## Completed

- **Model 16 global indices with per-market logistic models** (#1)
  - 2026-08-01: 16 indices across Asia, Europe and the Americas, each with its own fitted model.
- **Walk-forward backtest with reliability tables** (#2)
  - 2026-08-01: Expanding-window refits; AUC, Brier skill and calibration reported per market.
- **Pre-open futures features from hourly bars (--intraday)** (#3)
  - 2026-08-01: Merged in #4; limited to the ~2 years of hourly history Yahoo serves.
- **Brent and oil shock features as a geopolitical risk read** (#4)
  - 2026-08-01: Merged in #6.
- **Project tracker (pm) for status updates and resource requests** (#5)
  - 2026-08-01: Merged in #7.

## In progress

- **Nightly scheduled run publishing predictions** (#8)
  - 2026-08-01: Needs the cache to survive between runs so a failed download does not void the day.
- **Benchmark gradient boosting against the logistic baseline** (#9)
  - 2026-08-01: Only worth keeping if it beats the linear model out of sample on Brier skill, not just AUC.

## Blocked

- **Replace stale Yahoo opening prints with real auction data** (#6)
  - 2026-08-01: Tracker proxies (ISF.L, STW.AX) patch the worst cases; Bovespa still drops a quarter of its sessions as unlabelled.
- **Extend the intraday window beyond ~2 years** (#7)
  - 2026-08-01: Yahoo caps hourly history at ~730 days, so the intraday model warms up on 200 rows instead of the usual minimum.

## Up next

- **Add Middle East and Indian indices (TASI, TA-35, NIFTY 50)** (#10)
- **Confidence intervals on the reported probabilities** (#11)
- **Track live forecasts against realised opens to measure real-world skill** (#12)
- **Alert when a market's out-of-sample skill decays below its base rate** (#13)

## Resources needed

- **A machine that can sit on a multi-hour hyperparameter search** — needed for #9
- **A small database or object store for the daily forecast log** — needed for #12
- **Intraday futures history vendor (Databento or similar)** — needed for #7
- **Paid exchange opening-auction data (FTSE, ASX, Bovespa)** — needed for #6
- **Somewhere to run the job on a schedule (GitHub Actions minutes are enough)** — needed for #8
