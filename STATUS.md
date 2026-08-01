# Stock-Market-Predictor — status update 2026-08-01

Progress: 6/17 tasks complete (35%).
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
- **Asia session dashboard: heavyweights, volumes and outside drivers** (#14) — devin
  - 2026-08-01: Covers Nikkei 225, KOSPI, CSI 300, Hang Seng and STI plus the European indices behind the Asian afternoon; India, Middle East and Europe enter as lagged driver regressions.

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
  - 2026-08-01: CSI 300 (000300.SS) and Tadawul (^TASI.SR) print only in patches on Yahoo; the dashboard falls back to the Shanghai Composite for China.
- **Confidence intervals on the reported probabilities** (#11)
- **Track live forecasts against realised opens to measure real-world skill** (#12)
- **Alert when a market's out-of-sample skill decays below its base rate** (#13)
- **Participation and leverage data for the Asian dashboard** (#15) — devin
- **Licensed index weights and free float for constituent attribution** (#16) — devin
- **Timestamped news and sentiment feed as a dashboard variable** (#17)

## Resources needed

- **A machine that can sit on a multi-hour hyperparameter search** — needed for #9
- **A small database or object store for the daily forecast log** — needed for #12
- **Exchange margin balance and short-interest statistics (TSE, KRX, SSE, HKEX)** — needed for #15
- **Index provider licence (Nikkei, KRX, CSI, HSI, SGX) for live weights** — needed for #16
- **Intraday futures history vendor (Databento or similar)** — needed for #7
- **Level 2 / TAQ order book feed for depth and spreads** — needed for #15
- **Newswire feed with timestamps (Reuters or Bloomberg)** — needed for #17
- **Paid exchange opening-auction data (FTSE, ASX, Bovespa)** — needed for #6
- **Somewhere to run the job on a schedule (GitHub Actions minutes are enough)** — needed for #8
