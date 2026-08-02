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
    Market("^GSPTSE", "S&P/TSX Composite", "Americas", open_utc=13.5, close_utc=20.0),
    Market("^BVSP", "Bovespa", "Americas", open_utc=13.0, close_utc=20.0),
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
    Instrument("DX-Y.NYB", "US dollar index", close_utc=21.0),
    Instrument("JPY=X", "USD/JPY", close_utc=21.0),
    Instrument("EURUSD=X", "EUR/USD", close_utc=21.0),
    Instrument("GBPUSD=X", "GBP/USD", close_utc=21.0),
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
# three; EUR/USD and GBP/USD follow the ECB and Bank of England respectively.
FX_SYMBOLS: frozenset[str] = frozenset({"JPY=X", "EURUSD=X", "GBPUSD=X"})

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
    return symbols
