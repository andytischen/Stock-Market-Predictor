"""Static description of the markets and indicators used by the model.

Every instrument is described by the UTC time at which its daily bar becomes
known (``close_utc``), expressed in hours from midnight UTC of the bar's date.
Target markets additionally declare the time of their opening auction
(``open_utc``), on the same clock, so it may be negative for markets whose
session starts on the previous calendar day (Sydney).

Those two numbers are what keeps the feature builder free of look-ahead bias:
an indicator may only be used for a prediction if its bar closed strictly
before the target market opened.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    """An indicator series available as a daily bar."""

    symbol: str
    name: str
    close_utc: float


@dataclass(frozen=True)
class Market:
    """A stock index whose opening auction we try to predict."""

    symbol: str
    name: str
    region: str
    open_utc: float
    close_utc: float
    # Yahoo prints a stale opening price for some indices (it simply repeats the
    # previous close).  Where that happens we read the opening auction from a
    # liquid tracker listed on the same exchange instead.
    open_source: str | None = None

    @property
    def gap_symbol(self) -> str:
        return self.open_source or self.symbol

    def __post_init__(self) -> None:
        if not -12.0 <= self.open_utc <= 24.0:
            raise ValueError(f"{self.symbol}: open_utc out of range")


MARKETS: tuple[Market, ...] = (
    Market("^N225", "Nikkei 225", "Asia", open_utc=0.0, close_utc=6.0),
    Market("^HSI", "Hang Seng", "Asia", open_utc=1.5, close_utc=8.0),
    # Sydney opens at 23:00 UTC on the calendar day *before* the session date.
    Market("^AXJO", "ASX 200", "Asia", open_utc=-1.0, close_utc=5.0, open_source="STW.AX"),
    Market("^KS11", "KOSPI", "Asia", open_utc=0.0, close_utc=6.5),
    Market("000001.SS", "Shanghai Composite", "Asia", open_utc=1.5, close_utc=7.0),
    Market("^NSEI", "Nifty 50", "Asia", open_utc=3.75, close_utc=10.0),
    Market("^STOXX50E", "Euro Stoxx 50", "Europe", open_utc=7.0, close_utc=15.5),
    Market("^FCHI", "CAC 40", "Europe", open_utc=7.0, close_utc=15.5),
    Market("^IBEX", "IBEX 35", "Europe", open_utc=7.0, close_utc=15.5),
    Market("^SSMI", "Swiss Market Index", "Europe", open_utc=7.0, close_utc=15.5),
    Market("^GDAXI", "DAX", "Europe", open_utc=7.0, close_utc=15.5),
    Market("^FTSE", "FTSE 100", "Europe", open_utc=7.0, close_utc=15.5, open_source="ISF.L"),
    Market("^GSPC", "S&P 500", "Americas", open_utc=13.5, close_utc=20.0),
    Market("^IXIC", "Nasdaq Composite", "Americas", open_utc=13.5, close_utc=20.0),
    Market("^DJI", "Dow Jones Industrial Average", "Americas", open_utc=13.5, close_utc=20.0),
    Market("^GSPTSE", "S&P/TSX Composite", "Americas", open_utc=13.5, close_utc=20.0),
    Market("^BVSP", "Bovespa", "Americas", open_utc=13.0, close_utc=20.0),
)

# The STOXX Europe 600 sector line-up, as the Xetra-listed iShares trackers (the
# underlying .SXxP indices have no usable Yahoo symbol). Each one is a read on a
# different part of the cycle: banks and construction on rates and domestic
# activity, retail and travel on the consumer, basic resources and chemicals on
# China, technology on the same cycle as ^SOX. All close with the European cash
# session, so every tracked market reads them a session late. History starts in
# 2008, which costs nothing: the panel already begins in 2009 on ISF.L.
SECTORS: tuple[Instrument, ...] = (
    Instrument("EXV1.DE", "Europe 600 Banks", close_utc=15.5),
    Instrument("EXH2.DE", "Europe 600 Financial Services", close_utc=15.5),
    Instrument("EXH5.DE", "Europe 600 Insurance", close_utc=15.5),
    Instrument("EXV3.DE", "Europe 600 Technology", close_utc=15.5),
    Instrument("EXV8.DE", "Europe 600 Construction & Materials", close_utc=15.5),
    Instrument("EXH4.DE", "Europe 600 Industrial Goods & Services", close_utc=15.5),
    Instrument("EXH1.DE", "Europe 600 Oil & Gas", close_utc=15.5),
    Instrument("EXH9.DE", "Europe 600 Utilities", close_utc=15.5),
    Instrument("EXV6.DE", "Europe 600 Basic Resources", close_utc=15.5),
    Instrument("EXV7.DE", "Europe 600 Chemicals", close_utc=15.5),
    Instrument("EXV5.DE", "Europe 600 Automobiles & Parts", close_utc=15.5),
    Instrument("EXV4.DE", "Europe 600 Health Care", close_utc=15.5),
    Instrument("EXH3.DE", "Europe 600 Food & Beverage", close_utc=15.5),
    Instrument("EXH7.DE", "Europe 600 Personal & Household Goods", close_utc=15.5),
    Instrument("EXH8.DE", "Europe 600 Retail", close_utc=15.5),
    Instrument("EXV9.DE", "Europe 600 Travel & Leisure", close_utc=15.5),
    Instrument("EXH6.DE", "Europe 600 Media", close_utc=15.5),
    Instrument("EXV2.DE", "Europe 600 Telecommunications", close_utc=15.5),
)

INDICATORS: tuple[Instrument, ...] = (
    Instrument("^VIX", "VIX volatility index", close_utc=21.25),
    Instrument("^TNX", "US 10y Treasury yield", close_utc=20.0),
    Instrument("^FVX", "US 5y Treasury yield", close_utc=20.0),
    Instrument("^TYX", "US 30y Treasury yield", close_utc=20.0),
    Instrument("^RUT", "Russell 2000", close_utc=20.0),
    Instrument("^SOX", "Philadelphia semiconductor index", close_utc=20.0),
    # Europe's semiconductor bellwether, and the largest weight in the Euro
    # Stoxx 50: it closes before the US indicators, so Asia reads it a session
    # earlier than ^SOX.
    Instrument("ASML.AS", "ASML", close_utc=15.5),
    *SECTORS,
    Instrument("DX-Y.NYB", "US dollar index", close_utc=21.0),
    Instrument("JPY=X", "USD/JPY", close_utc=21.0),
    Instrument("EURUSD=X", "EUR/USD", close_utc=21.0),
    Instrument("GBPUSD=X", "GBP/USD", close_utc=21.0),
    # South Korean won: the Bank of Korea intervenes to defend the exchange rate
    # against rapid depreciation, particularly during risk-off episodes that hit
    # the KOSPI hard.  Like every other Yahoo spot FX bar it is stamped at the
    # New York cut, so it is read one session late, as the other pairs are.
    Instrument("KRW=X", "USD/KRW", close_utc=21.0),
    # Bitcoin is deliberately absent: its history starts in 2014 and, because a
    # row needs every feature, adding it would cost every market nine years of
    # training data for no measurable accuracy.
    Instrument("CL=F", "WTI crude", close_utc=21.0),
    Instrument("BZ=F", "Brent crude", close_utc=21.0),
    Instrument("GC=F", "Gold", close_utc=21.0),
    Instrument("SI=F", "Silver", close_utc=21.0),
    Instrument("HG=F", "Copper", close_utc=21.0),
    Instrument("ES=F", "S&P 500 futures", close_utc=21.0),
    Instrument("NQ=F", "Nasdaq 100 futures", close_utc=21.0),
)

# Crude is the fastest-moving read on Middle East supply risk: escalation
# (strikes on Iran, tanker traffic) spikes it, de-escalation (negotiations, a
# deal) unwinds it. Both directions are informative for equity opens, so oil
# carries extra features describing the size and volatility of the move.
OIL_SYMBOLS: frozenset[str] = frozenset({"CL=F", "BZ=F"})

# Central-bank governors intervene in these pairs to defend exchange-rate
# levels or smooth volatility. Intervention episodes produce moves that are
# large relative to recent realised volatility — exactly the signal the shock
# feature is designed to capture. USD/JPY is the most actively managed of the
# three G10 pairs; EUR/USD and GBP/USD follow the ECB and Bank of England
# respectively. USD/KRW captures Bank of Korea defences during risk-off
# episodes that coincide with KOSPI sell-offs.
FX_SYMBOLS: frozenset[str] = frozenset({"JPY=X", "EURUSD=X", "GBPUSD=X", "KRW=X"})

# Sector series read as a demand signal rather than a price level: a single
# session says little, so they carry the weekly return as well as the daily one.
SECTOR_SYMBOLS: frozenset[str] = frozenset(i.symbol for i in SECTORS)

# The shape of the crude futures curve, as the pair of oil funds that track its
# two ends: USO rolls the front month, USL holds a twelve-month strip. Yahoo
# serves only the generic front contract (CL=F), and a single deferred contract
# (CLZ26.NYM) both starts in 2017 and expires, so the funds are the only source
# of a curve reading with usable history. Neither price is a feature on its own
# — CL=F already carries the level — only the difference between them is: the
# front leg lagging the strip is contango (supply comfortable), leading it is
# backwardation (supply tight).
CURVE_FRONT = "USO"
CURVE_STRIP = "USL"
# Both are US-listed funds, so their bars close with Wall Street.
CURVE_CLOSE_UTC = 20.0
# Sessions used for the slow reading: long enough that a single roll or a day of
# noise cannot dominate it, short enough to still describe the current regime.
CURVE_WINDOW = 60

# What the market thinks the Fed will do, rather than what long bonds yield.
# The 30-day fed funds future settles on the average effective funds rate over
# its delivery month, so 100 minus its price is the rate the front month is
# priced for — a direct read on the current setting and, through its own moves,
# on the odds of it changing. The 13-week bill carries the same expectation
# three months out, so the difference between the two is the tightening (or
# easing) the next quarter has priced in: the "is September a hike?" question
# the yield curve features cannot express, since a 10y yield rises both on
# inflation and on growth.
FUNDS_FUTURE = "ZQ=F"
BILL_YIELD = "^IRX"
# Funds futures settle with the other CME contracts; the bill yield is a US
# cash-market print. Neither is used as a log return — bill yields have touched
# zero, and a future priced near 100 moves in fractions of a percent — so both
# enter the model as levels and differences of levels, in percentage points.
FUNDS_CLOSE_UTC = 21.0
BILL_CLOSE_UTC = 20.0
# Sessions over which the slow change in pricing is measured: about a month, so
# it spans the run-up to a meeting without being dominated by one session.
POLICY_WINDOW = 20

MARKETS_BY_SYMBOL = {m.symbol: m for m in MARKETS}

# Regions in the order their sessions run through the day.
REGIONS: tuple[str, ...] = tuple(dict.fromkeys(m.region for m in MARKETS))


def lag_days(source_close_utc: float, target_open_utc: float) -> int:
    """0 if the source bar closes before the target opens, otherwise 1."""
    return 0 if source_close_utc < target_open_utc else 1


def market(symbol: str) -> Market:
    try:
        return MARKETS_BY_SYMBOL[symbol]
    except KeyError as exc:
        known = ", ".join(MARKETS_BY_SYMBOL)
        raise KeyError(f"unknown market {symbol!r}; known markets: {known}") from exc


def all_symbols() -> list[str]:
    symbols = [m.symbol for m in MARKETS] + [i.symbol for i in INDICATORS]
    symbols += [m.open_source for m in MARKETS if m.open_source]
    symbols += [CURVE_FRONT, CURVE_STRIP, FUNDS_FUTURE, BILL_YIELD]
    return symbols
