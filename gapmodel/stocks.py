"""Single stocks whose opening auction is modelled the way an index's is.

An index open is a weighted average of hundreds of auctions, so it inherits the
overnight tape and little else. One company's open does not: it carries whatever
was said about that company after the previous bell, which no price series
contains. The same machinery still applies — the alignment rules, the
walk-forward, the log-odds attribution — but the honest reading of a single-name
probability is narrower, and the metrics printed beside it are the only place
that shows.

What a US-listed chipmaker *does* have, and an index does not, is a set of peers
that finish trading before New York opens. Seoul closes at 06:30 UTC and Tokyo
at 06:00, so a memory name's opening auction happens nine hours after Samsung
and SK Hynix have already priced the same demand story. Those bars are
same-session information here, which is the one structural advantage this model
has over reading the tape.
"""

from __future__ import annotations

from dataclasses import dataclass

from .markets import Instrument, Market, market
from .universe import modelled_universe

# Wall Street's clock, shared with the indices listed on it. The 09:30 ET
# auction is 13:30 UTC under daylight time and 14:30 under standard time
# (roughly November to March); the value below matches every existing US entry,
# and no indicator closes between 13.5 and 14.5, so the winter offset changes no
# lag decision.
US_OPEN_UTC = 13.5
US_CLOSE_UTC = 20.0

# The memory and storage complex, and the names that lead it. The Asian legs
# close before New York opens and are therefore read on the same session; the
# US legs are read a session late, exactly as the indicators are.
MEMORY_PEERS: tuple[Instrument, ...] = (
    Instrument("005930.KS", "Samsung Electronics", close_utc=6.5),
    Instrument("000660.KS", "SK Hynix", close_utc=6.5),
    Instrument("8035.T", "Tokyo Electron", close_utc=6.0),
    Instrument("6857.T", "Advantest", close_utc=6.0),
    Instrument("2330.TW", "TSMC", close_utc=5.5),
    Instrument("MU", "Micron Technology", close_utc=US_CLOSE_UTC),
    Instrument("WDC", "Western Digital", close_utc=US_CLOSE_UTC),
    Instrument("STX", "Seagate Technology", close_utc=US_CLOSE_UTC),
    Instrument("NVDA", "Nvidia", close_utc=US_CLOSE_UTC),
    Instrument("AMAT", "Applied Materials", close_utc=US_CLOSE_UTC),
    Instrument("SMH", "US semiconductor ETF", close_utc=US_CLOSE_UTC),
)


@dataclass(frozen=True)
class Stock:
    """A single listing whose next opening auction is forecast.

    ``peers`` are the series read for this name on top of the indices and
    cross-asset indicators every target gets: the companies whose own sessions
    price the same end demand. A stock that appears in its own peer list is
    skipped when the features are built, so one tuple can serve a whole complex.
    """

    symbol: str
    name: str
    theme: str
    peers: tuple[Instrument, ...]

    @property
    def market(self) -> Market:
        """The stock as a target: it opens and closes with Wall Street."""
        return Market(
            self.symbol,
            self.name,
            "Americas",
            open_utc=US_OPEN_UTC,
            close_utc=US_CLOSE_UTC,
        )


STOCKS: tuple[Stock, ...] = (
    Stock("MU", "Micron Technology", "memory and storage", MEMORY_PEERS),
    Stock("WDC", "Western Digital", "memory and storage", MEMORY_PEERS),
    Stock("STX", "Seagate Technology", "memory and storage", MEMORY_PEERS),
)

STOCKS_BY_SYMBOL = {s.symbol: s for s in STOCKS}

# Said next to every single-name probability. None of it is a feature, and all
# of it moves an individual open more than the overnight tape does.
# The shortlist universe: single listings the repository models without a peer
# list of their own. Held as a set because every feature build asks whether its
# target is a company.
SHORTLISTED = frozenset(modelled_universe())


BLIND_SPOTS: tuple[str, ...] = (
    "results and guidance, including anything released after the previous bell",
    "analyst actions, index changes and block trades",
    "company news: customers, export controls, capacity, litigation",
)


def stock(symbol: str) -> Stock:
    try:
        return STOCKS_BY_SYMBOL[symbol]
    except KeyError as exc:
        known = ", ".join(STOCKS_BY_SYMBOL)
        raise KeyError(f"unknown stock {symbol!r}; modelled stocks: {known}") from exc


def is_stock(symbol: str) -> bool:
    """Whether ``symbol`` is one company rather than an index of them.

    True beyond the curated registry, because ``shortlist`` forecasts a whole
    universe of names that have no peer list. What follows from it is the same
    either way: a company pays dividends and an index does not, so the bars are
    put on a total-return basis before a gap is taken from them.

    Membership, not spelling. A symbol that merely looks like a US ticker is not
    one of these, so nothing unmodelled is quietly given Wall Street's clock.
    """
    return symbol in STOCKS_BY_SYMBOL or symbol in SHORTLISTED


def target_market(symbol: str) -> Market:
    """How to align features for ``symbol``, whether it is an index or a stock.

    A shortlisted name is described on demand rather than registered in
    ``MARKETS``, which is also the set of cross-market *features*: sixty stocks
    there would hand every index sixty collinear columns and silently change the
    forecasts this repository already makes. A stock is a target only. The region
    is ``Americas`` so it reads what a US index reads — in particular it skips
    the European sector trackers, which would dilute the fit.
    """
    known = STOCKS_BY_SYMBOL.get(symbol)
    if known is not None:
        return known.market
    if symbol in SHORTLISTED:
        return Market(symbol, symbol, "Americas", open_utc=US_OPEN_UTC, close_utc=US_CLOSE_UTC)
    return market(symbol)


def peers_of(symbol: str) -> tuple[Instrument, ...]:
    """Peer series for a stock, excluding itself; nothing for an index."""
    known = STOCKS_BY_SYMBOL.get(symbol)
    if known is None:
        return ()
    return tuple(peer for peer in known.peers if peer.symbol != symbol)


def stock_symbols() -> list[str]:
    """Every symbol a stock forecast needs on top of the index panel."""
    symbols = [s.symbol for s in STOCKS]
    symbols += [peer.symbol for s in STOCKS for peer in s.peers]
    return list(dict.fromkeys(symbols))
