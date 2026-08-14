"""A price-derived trend score for an arbitrary list of tickers.

This is a standalone read on where a stock sits in its own recent history: the
standardised position of the latest close within its trailing ``window``-day
distribution,

    score = (close - mean(close, window)) / stdev(close, window)

i.e. how many standard deviations the last price is above (positive) or below
(negative) its own recent average. A stock riding the top of a long uptrend
scores strongly positive; one grinding along the bottom of a range scores
negative. The value is unbounded but in practice sits in roughly ``[-4, +4]``.

Why this and not something cleverer: it was reverse-engineered against the
sorted, heat-mapped "Score" column of a ThinkorSwim watchlist (a custom study
whose formula is not published). Across a 27-name sample only long-horizon trend
measures correlate with that column at all, and all weakly: the 200-day price
z-score tracks it at r ~= 0.47, on par with the 200-day Bollinger %b, while
short-window RSI/ROC/MACD/%B, the TTM-squeeze momentum and a fitted blend of
many indicators have essentially no out-of-sample skill. A long-lookback RSI
shows a higher *raw* correlation but is not reproducible — Wilder's RSI is
path-dependent on where the price history starts, so its value drifts with the
download window, whereas this z-score depends only on the trailing window. So
this is an *approximation* of that column's ranking, not a reproduction of it:
the true column is driven by inputs a daily price bar does not contain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CACHE, load_symbol

log = logging.getLogger(__name__)

# The window that best tracked the reference column; long enough to describe a
# trend rather than a swing.
DEFAULT_WINDOW = 200


@dataclass(frozen=True)
class TrendScore:
    """One ticker's trend score as of a given session."""

    symbol: str
    score: float
    last: float
    asof: pd.Timestamp
    window: int


def trend_score(frame: pd.DataFrame, window: int = DEFAULT_WINDOW) -> float:
    """Standardised position of the last close within its trailing ``window``.

    ``frame`` is daily bars already trimmed to the as-of date (its last row is
    the session being scored). Raises ``ValueError`` when there is not a full
    window of history, so a half-warmed series is refused rather than scored on
    a partial mean.
    """
    close = frame["Close"].dropna()
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if len(close) < window:
        raise ValueError(f"need {window} closes, have {len(close)}")
    tail = close.iloc[-window:]
    spread = tail.std()
    if spread == 0 or pd.isna(spread):
        raise ValueError("no variation over the window")
    return float((close.iloc[-1] - tail.mean()) / spread)


def score_symbols(
    symbols: list[str],
    window: int = DEFAULT_WINDOW,
    asof: pd.Timestamp | None = None,
    start: str = "2005-01-01",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> list[TrendScore]:
    """Score every ticker, sorted from strongest to weakest.

    A ticker Yahoo cannot serve, or one without a full window of history as of
    ``asof``, is skipped with a warning rather than failing the whole run.
    """
    scored: list[TrendScore] = []
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    for symbol in symbols:
        try:
            frame = load_symbol(symbol, start, cache_dir, refresh)
            if asof is not None:
                frame = frame.loc[frame.index <= asof]
            value = trend_score(frame, window=window)
        except Exception as exc:  # one dead or too-young ticker must not stop the run
            log.warning("skipping %s: %s", symbol, exc)
            continue
        # Read last/asof from the same non-null closes the score was built on, so
        # a NaN newest bar cannot show a blank price against an unscored date.
        close = frame["Close"].dropna()
        scored.append(
            TrendScore(
                symbol=symbol,
                score=value,
                last=float(close.iloc[-1]),
                asof=close.index[-1],
                window=window,
            )
        )
    if not scored:
        raise RuntimeError("no symbols could be scored")
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def to_frame(scores: list[TrendScore]) -> pd.DataFrame:
    """Tabulate scores the way the watchlist reads: symbol, last, score, date."""
    return pd.DataFrame(
        {
            "symbol": [s.symbol for s in scores],
            "last": [round(s.last, 1) for s in scores],
            "score": [round(s.score, 2) for s in scores],
            "asof": [s.asof.date().isoformat() for s in scores],
        }
    )
