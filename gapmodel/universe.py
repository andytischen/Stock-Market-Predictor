"""The US universe the screener starts from.

A screener is only as good as the list it runs over, and Yahoo publishes no
"all US listings" endpoint that can be pulled cheaply. So the starting universe
is a hand-maintained snapshot (2025) of liquid US listings: the S&P 100 names
plus the mid-cap and retail-favourite tickers that actually print unusual
volume, and the handful of high-turnover ETFs that set the tone for them.

It is deliberately a *superset* of what any screen returns — every liquidity,
activity and movement test is applied to real bars downstream, so a name that
has since gone quiet, been acquired or delisted is dropped by the funnel (or
skipped for want of data) rather than needing to be pruned from here. The list
being a snapshot therefore costs coverage, never correctness.

Pass ``--universe FILE`` (one ticker per line) or an explicit list of symbols to
screen something else.
"""

from __future__ import annotations

from pathlib import Path

# Mega- and large-cap US listings: the names that dominate the S&P 500 and
# supply most of the market's daily turnover.
# fmt: off
LARGE_CAP: tuple[str, ...] = (
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMT", "AMZN", "AVGO",
    "AXP", "BA", "BAC", "BNY", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EOG", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS",
    "MSFT", "MU", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT",
    "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ",
    "WFC", "WMT", "XOM",
)
# fmt: on

# Mid-caps, recent listings and high-beta names: smaller than the S&P 100 but
# where unusual volume and >1% days actually show up.
# fmt: off
MID_CAP: tuple[str, ...] = (
    "AAL", "AFRM", "ALB", "APP", "ARM", "BABA", "BBY", "CCL", "CLF", "COIN",
    "CRWD", "DAL", "DASH", "DDOG", "DKNG", "ENPH", "ETSY", "FSLR", "HOOD", "IVZ",
    "KEY", "LCID", "LULU", "LYFT", "MARA", "MRNA", "MSTR", "NCLH", "NET", "NIO",
    "OKTA", "ON", "PANW", "PINS", "PLTR", "PLUG", "RBLX", "RIOT", "RIVN", "ROKU",
    "SHOP", "SMCI", "SNAP", "SNOW", "SOFI", "TTD", "TWLO", "U", "UAL", "UBER",
    "WDC", "XYZ", "ZM", "ZS",
)
# fmt: on

# The most heavily traded US ETFs. They are not stocks, but they are the
# reference for "unusually active" and are what a US screen is read against.
# fmt: off
ETFS: tuple[str, ...] = (
    "DIA", "EEM", "EFA", "GLD", "HYG", "IWM", "QQQ", "SLV", "SMH", "SPY",
    "TLT", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV", "XLY",
)
# fmt: on


# Nasdaq-listed names, drawn from the two lists above and from the curated
# single-name registry, which is why STX appears here and in neither list: the
# invariant is that `stock` and `shortlist` accept the same symbols, and a test
# asserts it. Listing venue is not something Yahoo tells us, so this is a
# hand-maintained snapshot (2025) in the same spirit as the lists it is drawn
# from: a venue slice of the forecast universe, not an authoritative index
# membership.
#
# Two consequences worth stating. Selecting today's survivors and then fitting
# on twenty years of their history is survivorship bias: the set excludes the
# Nasdaq names that were delisted or acquired, so backtest metrics here read
# better than a genuinely point-in-time universe would. And a name whose
# listing moved (Honeywell and Palantir both changed venue) is classified by
# where it trades now, not where it traded for the bulk of the sample.
# fmt: off
NASDAQ: tuple[str, ...] = (
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AVGO", "BKNG", "CHTR", "CMCSA",
    "COST", "CSCO", "GILD", "GOOG", "GOOGL", "HON", "INTC", "INTU", "ISRG",
    "KHC", "MDLZ", "META", "MSFT", "MU", "NFLX", "NVDA", "PEP", "PYPL",
    "QCOM", "SBUX", "TMUS", "TSLA", "TXN",
    "AAL", "AFRM", "APP", "ARM", "COIN", "CRWD", "DDOG", "DKNG", "ENPH",
    "ETSY", "FSLR", "HOOD", "LCID", "LULU", "LYFT", "MARA", "MRNA", "MSTR",
    "OKTA", "ON", "PANW", "PLTR", "PLUG", "RIOT", "RIVN", "ROKU", "SMCI",
    "SOFI", "STX", "TTD", "UAL", "WDC", "ZM", "ZS",
)
# fmt: on


def nasdaq_universe() -> list[str]:
    """Nasdaq-listed tickers, stable order. A venue slice of the list below."""
    return list(dict.fromkeys(NASDAQ))


def modelled_universe() -> list[str]:
    """Every single listing the per-stock gap model forecasts, stable order.

    Listing venue is not a modelling boundary. A stock's opening gap is read
    from the overnight tape and its own history, and Wall Street's auction is
    the same auction for a NYSE name as for a Nasdaq one, so restricting the
    forecast set to one venue only cost coverage: the banks, the oils and the
    industrials that lead whole sessions were unreachable. The venue slice is
    kept above because a Nasdaq-only report is still a thing to ask for, not
    because the model needs it.

    The survivorship caveat on ``NASDAQ`` applies here in full, and more widely:
    this is a snapshot of today's listings fitted over their whole history, so
    the acquired and the delisted are missing and the metrics read better than a
    point-in-time universe would.
    """
    return list(dict.fromkeys(NASDAQ + LARGE_CAP + MID_CAP))


def us_universe(include_etfs: bool = False) -> list[str]:
    """The default US screening universe, deduplicated and in a stable order."""
    symbols = LARGE_CAP + MID_CAP + (ETFS if include_etfs else ())
    return list(dict.fromkeys(symbols))


def read_universe(path: Path) -> list[str]:
    """Read a universe file: one ticker per line, ``#`` comments allowed."""
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ticker = line.split("#", 1)[0].strip().upper()
        if ticker:
            symbols.append(ticker)
    if not symbols:
        raise ValueError(f"{path} contains no tickers")
    return list(dict.fromkeys(symbols))
