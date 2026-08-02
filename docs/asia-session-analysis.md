# What moves the Asian session

Read of the dashboard produced by `python -m gapmodel asia`, run on
2026-08-01 over the 500 sessions ending 2026-07-31. Numbers move with every
run; regenerate before quoting them.

```bash
python -m gapmodel asia --out asia-dashboard.html   # standalone HTML page
python -m gapmodel asia                             # same content as text
```

## The indices, and who is inside them

| Country | Index | Names covered | Concentration |
| --- | --- | --- | --- |
| Japan | Nikkei 225 (`^N225`) | 10 | 38% of the index; price-weighted, so Fast Retailing alone is ~11% |
| South Korea | KOSPI (`^KS11`) | 10 | 39%; Samsung Electronics and SK Hynix are ~27% between them |
| China | CSI 300 (`000300.SS`) | 10 | 19%; the flattest book of the five |
| Hong Kong | Hang Seng (`^HSI`) | 10 | 53%; Tencent + Alibaba ~17%, then the HSBC/CCB/AIA bloc |
| Singapore | Straits Times (`^STI`) | 10 | 72%; DBS, OCBC and UOB are over 43% |

Concentration is the first thing the dashboard is for: in Singapore and Hong
Kong a handful of names *are* the index, in China they are not. The
`Index bp` column turns that into attribution — weight times the day's move —
so a session can be read as "Advantest and SoftBank did +134bp of the Nikkei's
+395bp" rather than as a single index number.

The `Beta` column is the leverage that matters here: how much index move a name
carries per unit of index move, measured over 60 sessions. Japanese
semiconductor equipment (Tokyo Electron 1.8, Advantest 2.0) and Hong Kong tech
(Alibaba 2.2, Meituan 1.9) are the high-beta expressions of their indices;
Singaporean banks sit near 1.1–1.5 and Chinese state banks below 0.5.
`Vol x avg` (session volume against its 60-session average) and `Turnover %`
(share of the covered names' traded value) say whether the move happened on
real participation or on nothing.

## The outside drivers

Every driver enters lagged so it was knowable before the index opened, using
the same UTC session rule as the gap model. Explained variance (R², 500
sessions) of the index's daily return:

| Index | India | Middle East | Europe | Global (US) | All |
| --- | --- | --- | --- | --- | --- |
| Nikkei 225 | 0.083 | 0.004 | 0.184 | 0.251 | 0.284 |
| KOSPI | 0.072 | 0.001 | 0.157 | 0.201 | 0.241 |
| CSI 300 | 0.017 | 0.033 | 0.052 | 0.055 | 0.095 |
| Hang Seng | 0.067 | 0.036 | 0.116 | 0.137 | 0.199 |
| Straits Times | 0.094 | 0.028 | 0.222 | 0.270 | 0.316 |

Reading:

- **Wall Street is still the driver of the Asian day.** The Philadelphia
  semiconductor index is the single strongest input for Japan (β 0.32, t 12.4)
  and Korea (β 0.38, t 10.3) — those two indices are, in practice, a levered
  semiconductor trade. For Singapore the S&P 500 leads instead (β 0.39,
  t 13.1), which is what a bank-heavy index looks like.
- **China is the outlier.** The CSI 300's total explained variance is 0.095
  against 0.28–0.32 for Japan and Singapore: the mainland A-share book is
  driven by domestic flow and policy that none of these series carry.
- **Europe adds real information beyond the US.** The US-listed Euro Stoxx 50
  tracker (`FEZ`, the closest free stand-in for the overnight futures print —
  Yahoo serves neither FESX nor FDAX) is significant everywhere, and for Hong
  Kong it is the top driver (β 0.39, t 7.3). European cash closes add less once
  the US session is known.

### Does India affect Asia?

Yes, but not the way it looks. The India *theme* explains 7–9% of the variance
of Japan, Korea and Singapore — but essentially all of it comes from `INDA`,
the US-hours India ETF (Nikkei β 0.49, t 5.8), while the Nifty 50 and Sensex
cash closes are indistinguishable from zero (t 0.08 for Japan, −1.3 for Korea)
and Nifty is only marginal for Singapore (t 2.2). India's own session closes at
10:00 UTC, *after* Tokyo and Hong Kong have shut, so for the next Asian open it
is a lagged neighbour, not a leader; what `INDA` contributes is a US-hours
global risk signal wearing an Indian label. Treat India as a co-mover, not a
cause — with Singapore, whose banks fund regional flow, the closest to a real
link.

## The Middle East as a variable

The Middle East enters four ways: Brent crude, the Tadawul All Share
(`^TASI.SR`), Saudi Aramco and the Tel Aviv 125. Its explained variance is the
smallest of the four themes everywhere — but it separates cleanly:

- **Energy importers barely react.** Japan 0.004 and Korea 0.001: Brent is not
  significant for either (t 0.6 and 0.2). Crude matters to the gap model as a
  *shock* variable — the size of a move relative to the volatility already
  known — rather than as a daily return, which is why the oil-shock features
  live in the model and not here.
- **Greater China reacts most.** CSI 300 0.033 and Hang Seng 0.036, and it is
  the Gulf cash markets rather than crude that carry it (Tadawul t 3.6 for
  China, 4.3 for Hong Kong). Gulf equity is itself a risk-appetite read
  overlapping the Asian session, which is likely what is being picked up.
- **Singapore is the one place crude is significant on its own** (t 2.4),
  consistent with an index carrying commodity trading houses and rig builders.

Escalation risk therefore belongs in this dashboard as an oil *shock* and a
Gulf-equity read, not as a daily crude return.

## European futures and the names behind them

The dashboard also profiles Euro Stoxx 50, DAX and FTSE 100 with their
heavyweights, because the European open is the second act of the Asian day.
Two names dominate the European direction that Asia trades against: ASML and
SAP (15.5% of Euro Stoxx 50 between them; SAP alone is 15% of the DAX), which
is why the semiconductor cycle shows up on both continents. FTSE 100 is the
exception — AstraZeneca, Shell and HSBC give it an energy/defensive tilt, and
its correlation to the Asian indices is the weakest of the three.

ASML is carried twice on purpose: as the largest Euro Stoxx weight and, since
it is the one European name Asian semiconductor supply chains trade off, as a
Europe-theme driver and a cross-asset indicator in the gap model itself. It
regresses on Tokyo (β 0.20, t 6.9, R² 0.09) and Seoul (β 0.25, t 6.2, R² 0.07)
behind only the US series — but that information is nearly all already in
`^SOX`: a walk-forward run with and without ASML moves AUC by less than
±0.001 on every index (European indices marginally up, Asian marginally down).
It is in the feature set because it is the cleanest pre-US read on the
semiconductor cycle, not because it lifts the backtest.

## What this dashboard cannot see

The proxies above are the honest limit of free daily data. Not covered, and
what each would need:

| Measure | Feed needed |
| --- | --- |
| Order book depth, spreads, queue imbalance | exchange level 2 / TAQ feed |
| Margin balances and leveraged ETF flow | exchange margin statistics (TSE, KRX, SSE) |
| Short interest and stock borrow | exchange short-sale reporting |
| Retail vs institutional participation ("userbase") | broker or exchange investor-type breakdown |
| Index weights and free float | index provider licence (Nikkei, KRX, CSI, HSI, SGX) |
| News and sentiment flow | newswire feed with timestamps |

Volume, turnover share and beta are what stand in for participation and
leverage today. The constituent weights are a hand-maintained 2025 snapshot;
they rank names and attribute moves, and should not be traded on.

Data quality is worth one warning: Yahoo publishes the CSI 300 (`000300.SS`)
only in patches, so the dashboard falls back to the Shanghai Composite for the
China index level and driver regressions and says so on the page. The Tadawul
series is similarly thin.

Not investment advice.
