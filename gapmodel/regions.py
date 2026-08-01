"""Who actually moves the Asian indices.

The gap model scores an index as a whole. This registry is the layer below it:
for each Asian headline index, the handful of companies that dominate it, and
the outside markets — India, the Middle East, European futures, Wall Street —
whose sessions are already closed when that index opens.

Index weights are an approximate, hand-maintained snapshot (2025). They are
used to rank names and to attribute an index move across them, so a percentage
point of drift changes an ordering at the margin and nothing else; a live
weights feed would be needed before they could be traded on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constituent:
    """A dominant member of an index."""

    symbol: str
    name: str
    weight: float
    sector: str

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 100.0:
            raise ValueError(f"{self.symbol}: weight out of range")


@dataclass(frozen=True)
class IndexProfile:
    """A headline index, its session in UTC hours, and its heavyweights."""

    symbol: str
    name: str
    country: str
    currency: str
    open_utc: float
    close_utc: float
    constituents: tuple[Constituent, ...]
    note: str = ""
    # Yahoo serves some indices only in patches. Where a broader index of the
    # same market is published reliably, it stands in when the headline series
    # has gone quiet, and the dashboard says which one it read.
    fallback: str | None = None

    def __post_init__(self) -> None:
        symbols = [c.symbol for c in self.constituents]
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"{self.symbol}: duplicate constituents")
        if self.weight_covered > 100.0:
            raise ValueError(f"{self.symbol}: constituent weights exceed the index")

    @property
    def weight_covered(self) -> float:
        return round(sum(c.weight for c in self.constituents), 2)


@dataclass(frozen=True)
class Influence:
    """An outside market tested as a driver of the Asian opens."""

    symbol: str
    name: str
    theme: str
    close_utc: float


# Japan's index is price-weighted, so a high-priced share such as Fast
# Retailing dominates it regardless of the size of the company.
NIKKEI = IndexProfile(
    symbol="^N225",
    name="Nikkei 225",
    country="Japan",
    currency="JPY",
    open_utc=0.0,
    close_utc=6.0,
    note="Price-weighted: index points follow share price, not market value.",
    constituents=(
        Constituent("9983.T", "Fast Retailing", 10.8, "Retail"),
        Constituent("8035.T", "Tokyo Electron", 6.1, "Semiconductors"),
        Constituent("6857.T", "Advantest", 5.2, "Semiconductors"),
        Constituent("9984.T", "SoftBank Group", 4.3, "Technology"),
        Constituent("6758.T", "Sony Group", 2.7, "Technology"),
        Constituent("4063.T", "Shin-Etsu Chemical", 2.2, "Materials"),
        Constituent("6098.T", "Recruit Holdings", 2.0, "Services"),
        Constituent("6954.T", "Fanuc", 1.9, "Industrials"),
        Constituent("7203.T", "Toyota Motor", 1.5, "Autos"),
        Constituent("8306.T", "Mitsubishi UFJ", 1.3, "Banks"),
    ),
)

KOSPI = IndexProfile(
    symbol="^KS11",
    name="KOSPI",
    country="South Korea",
    currency="KRW",
    open_utc=0.0,
    close_utc=6.5,
    note="Two memory makers carry roughly a quarter of the index.",
    constituents=(
        Constituent("005930.KS", "Samsung Electronics", 18.5, "Semiconductors"),
        Constituent("000660.KS", "SK Hynix", 8.0, "Semiconductors"),
        Constituent("373220.KS", "LG Energy Solution", 2.5, "Batteries"),
        Constituent("207940.KS", "Samsung Biologics", 2.3, "Healthcare"),
        Constituent("005380.KS", "Hyundai Motor", 2.0, "Autos"),
        Constituent("000270.KS", "Kia", 1.4, "Autos"),
        Constituent("068270.KS", "Celltrion", 1.3, "Healthcare"),
        Constituent("105560.KS", "KB Financial", 1.2, "Banks"),
        Constituent("005490.KS", "POSCO Holdings", 1.0, "Materials"),
        Constituent("035420.KS", "NAVER", 1.0, "Technology"),
    ),
)

CSI300 = IndexProfile(
    symbol="000300.SS",
    name="CSI 300",
    country="China",
    currency="CNY",
    open_utc=1.5,
    close_utc=7.0,
    note="Mainland A shares; a retail-heavy book with a state-owned bank core.",
    fallback="000001.SS",
    constituents=(
        Constituent("600519.SS", "Kweichow Moutai", 5.0, "Consumer"),
        Constituent("300750.SZ", "CATL", 3.2, "Batteries"),
        Constituent("601318.SS", "Ping An Insurance", 2.0, "Insurance"),
        Constituent("600036.SS", "China Merchants Bank", 1.9, "Banks"),
        Constituent("601398.SS", "ICBC", 1.4, "Banks"),
        Constituent("000858.SZ", "Wuliangye", 1.2, "Consumer"),
        Constituent("002594.SZ", "BYD", 1.2, "Autos"),
        Constituent("600900.SS", "Yangtze Power", 1.1, "Utilities"),
        Constituent("000333.SZ", "Midea Group", 1.1, "Consumer"),
        Constituent("601899.SS", "Zijin Mining", 1.0, "Materials"),
    ),
)

HANG_SENG = IndexProfile(
    symbol="^HSI",
    name="Hang Seng",
    country="Hong Kong",
    currency="HKD",
    open_utc=1.5,
    close_utc=8.0,
    note="The offshore venue for China tech, plus the HSBC/AIA financial bloc.",
    constituents=(
        Constituent("0700.HK", "Tencent", 8.4, "Technology"),
        Constituent("9988.HK", "Alibaba", 8.1, "Technology"),
        Constituent("0939.HK", "China Construction Bank", 7.5, "Banks"),
        Constituent("1299.HK", "AIA Group", 6.5, "Insurance"),
        Constituent("3690.HK", "Meituan", 5.0, "Technology"),
        Constituent("0005.HK", "HSBC Holdings", 4.8, "Banks"),
        Constituent("1810.HK", "Xiaomi", 4.5, "Technology"),
        Constituent("0388.HK", "HK Exchanges", 3.2, "Exchanges"),
        Constituent("2318.HK", "Ping An Insurance", 3.0, "Insurance"),
        Constituent("9618.HK", "JD.com", 2.0, "Technology"),
    ),
)

STRAITS_TIMES = IndexProfile(
    symbol="^STI",
    name="Straits Times Index",
    country="Singapore",
    currency="SGD",
    open_utc=1.0,
    close_utc=9.0,
    note="Three banks are over 40% of the index: a rates and credit proxy.",
    constituents=(
        Constituent("D05.SI", "DBS Group", 20.0, "Banks"),
        Constituent("O39.SI", "OCBC", 12.5, "Banks"),
        Constituent("U11.SI", "UOB", 11.0, "Banks"),
        Constituent("Z74.SI", "Singtel", 9.0, "Telecoms"),
        Constituent("S63.SI", "ST Engineering", 4.5, "Industrials"),
        Constituent("C38U.SI", "CapitaLand Integrated Trust", 4.0, "Real estate"),
        Constituent("J36.SI", "Jardine Matheson", 3.5, "Conglomerate"),
        Constituent("S68.SI", "Singapore Exchange", 3.0, "Exchanges"),
        Constituent("BN4.SI", "Keppel", 2.5, "Industrials"),
        Constituent("F34.SI", "Wilmar International", 2.0, "Consumer"),
    ),
)

ASIA_INDICES: tuple[IndexProfile, ...] = (NIKKEI, KOSPI, CSI300, HANG_SENG, STRAITS_TIMES)

# European cash indices with the names that carry them. Their futures are the
# read Asia trades against in the afternoon; Yahoo does not serve FESX or FDAX,
# so the US-listed Euro Stoxx 50 tracker below stands in for the overnight
# futures print.
EURO_STOXX = IndexProfile(
    symbol="^STOXX50E",
    name="Euro Stoxx 50",
    country="Euro area",
    currency="EUR",
    open_utc=7.0,
    close_utc=15.5,
    constituents=(
        Constituent("ASML.AS", "ASML", 8.0, "Semiconductors"),
        Constituent("SAP.DE", "SAP", 7.5, "Technology"),
        Constituent("SIE.DE", "Siemens", 3.5, "Industrials"),
        Constituent("MC.PA", "LVMH", 3.0, "Consumer"),
        Constituent("TTE.PA", "TotalEnergies", 3.0, "Energy"),
        Constituent("SU.PA", "Schneider Electric", 3.0, "Industrials"),
        Constituent("AIR.PA", "Airbus", 2.8, "Aerospace"),
        Constituent("ALV.DE", "Allianz", 2.7, "Insurance"),
        Constituent("SAN.PA", "Sanofi", 2.5, "Healthcare"),
        Constituent("IBE.MC", "Iberdrola", 2.3, "Utilities"),
    ),
)

DAX = IndexProfile(
    symbol="^GDAXI",
    name="DAX",
    country="Germany",
    currency="EUR",
    open_utc=7.0,
    close_utc=15.5,
    constituents=(
        Constituent("SAP.DE", "SAP", 15.0, "Technology"),
        Constituent("SIE.DE", "Siemens", 10.0, "Industrials"),
        Constituent("ALV.DE", "Allianz", 7.0, "Insurance"),
        Constituent("DTE.DE", "Deutsche Telekom", 6.0, "Telecoms"),
        Constituent("MUV2.DE", "Munich Re", 4.0, "Insurance"),
        Constituent("RHM.DE", "Rheinmetall", 4.0, "Defence"),
        Constituent("IFX.DE", "Infineon", 3.5, "Semiconductors"),
        Constituent("BAS.DE", "BASF", 2.5, "Chemicals"),
        Constituent("DB1.DE", "Deutsche Boerse", 2.5, "Exchanges"),
        Constituent("ADS.DE", "Adidas", 2.0, "Consumer"),
    ),
)

FTSE = IndexProfile(
    symbol="^FTSE",
    name="FTSE 100",
    country="United Kingdom",
    currency="GBP",
    open_utc=7.0,
    close_utc=15.5,
    constituents=(
        Constituent("AZN.L", "AstraZeneca", 8.0, "Healthcare"),
        Constituent("SHEL.L", "Shell", 7.0, "Energy"),
        Constituent("HSBA.L", "HSBC Holdings", 7.0, "Banks"),
        Constituent("ULVR.L", "Unilever", 4.5, "Consumer"),
        Constituent("BP.L", "BP", 3.0, "Energy"),
        Constituent("RIO.L", "Rio Tinto", 3.0, "Materials"),
        Constituent("GSK.L", "GSK", 3.0, "Healthcare"),
        Constituent("REL.L", "RELX", 3.0, "Media"),
        Constituent("BATS.L", "British American Tobacco", 3.0, "Consumer"),
        Constituent("DGE.L", "Diageo", 2.0, "Consumer"),
    ),
)

EUROPE_INDICES: tuple[IndexProfile, ...] = (EURO_STOXX, DAX, FTSE)

INFLUENCES: tuple[Influence, ...] = (
    # India trades while Asia is open and closes after Tokyo and Hong Kong, so
    # for the next Asian open it is a same-session neighbour, not a leader.
    Influence("^NSEI", "Nifty 50", "India", close_utc=10.0),
    Influence("^BSESN", "BSE Sensex", "India", close_utc=10.0),
    Influence("INDA", "India ETF (US hours)", "India", close_utc=20.0),
    # Crude is the market's live read on Middle East supply risk; Tadawul and
    # Tel Aviv are the regional cash markets that trade the same headlines.
    Influence("BZ=F", "Brent crude", "Middle East", close_utc=21.0),
    Influence("^TASI.SR", "Tadawul All Share", "Middle East", close_utc=12.0),
    Influence("^TA125.TA", "Tel Aviv 125", "Middle East", close_utc=14.5),
    Influence("2222.SR", "Saudi Aramco", "Middle East", close_utc=12.0),
    # European futures are not on Yahoo; FEZ is the US-listed Euro Stoxx 50
    # tracker and prints through the US session, which is the window that
    # matters for the next Asian bell.
    Influence("FEZ", "Euro Stoxx 50 tracker (US hours)", "Europe", close_utc=20.0),
    Influence("^STOXX50E", "Euro Stoxx 50 cash", "Europe", close_utc=15.5),
    Influence("^GDAXI", "DAX cash", "Europe", close_utc=15.5),
    Influence("^FTSE", "FTSE 100 cash", "Europe", close_utc=15.5),
    Influence("^GSPC", "S&P 500", "Global", close_utc=20.0),
    Influence("^SOX", "Philadelphia semiconductor index", "Global", close_utc=20.0),
    Influence("^VIX", "VIX volatility index", "Global", close_utc=21.25),
    Influence("DX-Y.NYB", "US dollar index", "Global", close_utc=21.0),
    Influence("JPY=X", "USD/JPY", "Global", close_utc=21.0),
)

THEMES: tuple[str, ...] = ("India", "Middle East", "Europe", "Global")


def all_profiles() -> tuple[IndexProfile, ...]:
    return ASIA_INDICES + EUROPE_INDICES


def dashboard_symbols() -> list[str]:
    """Every symbol the dashboard needs, in a stable order and deduplicated."""
    symbols: list[str] = []
    for profile in all_profiles():
        symbols.append(profile.symbol)
        if profile.fallback:
            symbols.append(profile.fallback)
        symbols.extend(c.symbol for c in profile.constituents)
    symbols.extend(i.symbol for i in INFLUENCES)
    return list(dict.fromkeys(symbols))


def profile(symbol: str) -> IndexProfile:
    for candidate in all_profiles():
        if candidate.symbol == symbol:
            return candidate
    known = ", ".join(p.symbol for p in all_profiles())
    raise KeyError(f"unknown index {symbol!r}; known indices: {known}")
