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

The raw score is *absolute*: it asks only where a stock sits in its own history,
so in a broadly rising market almost every name scores positive and a watchlist
bunches up well above zero. The reference column does not behave that way - it
is centred near zero and roughly symmetric - which is the signature of a
*cross-sectional* measure, one that ranks each stock against its peers on the
same day rather than against its own past. ``relative_scores`` restates the raw
score that way: it scores a comparison universe on one session and reports each
ticker's standardised position and percentile within it. That re-centring is a
change of yardstick, not of information - it makes the output comparable in
shape to the reference column without making it any better at predicting it.
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


@dataclass(frozen=True)
class RelativeScore:
    """One ticker's trend score restated against a comparison universe."""

    symbol: str
    score: float
    relative: float
    percentile: float
    last: float
    asof: pd.Timestamp
    window: int


@dataclass(frozen=True)
class Reference:
    """The comparison distribution the relative scores were measured against."""

    session: pd.Timestamp
    count: int
    mean: float
    stdev: float
    stale: tuple[str, ...]


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


def relative_scores(
    symbols: list[str],
    universe: list[str],
    window: int = DEFAULT_WINDOW,
    asof: pd.Timestamp | None = None,
    start: str = "2005-01-01",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> tuple[list[RelativeScore], Reference]:
    """Restate each ticker's raw score as its standing within ``universe``.

    The universe is what makes the number mean something: ``relative`` is the raw
    score minus the universe mean over the universe's standard deviation, and
    ``percentile`` is the share of the universe scoring no higher. So a relative
    score of 0 is an ordinary stock *today*, whatever the market has done, and
    the output is centred by construction rather than by luck of the tape.

    A cross-section is only a cross-section if every name is measured on the same
    day, so the comparison session is the newest one any universe member reached
    (or ``asof`` when given), and members whose data stops earlier are reported
    in ``Reference.stale`` — they still count, at their own last close, but the
    caller can say how many are lagging.

    ``symbols`` need not belong to ``universe``; names that do are scored once
    and appear in both. Returned strongest first.
    """
    wanted = [s.upper() for s in symbols]
    reference = [s.upper() for s in universe]
    if not reference:
        raise ValueError("comparison universe is empty")
    scored = score_symbols(
        list(dict.fromkeys(reference + wanted)),
        window=window,
        asof=asof,
        start=start,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    by_symbol = {s.symbol: s for s in scored}
    members = [by_symbol[s] for s in dict.fromkeys(reference) if s in by_symbol]
    if len(members) < 2:
        raise ValueError(f"need 2 scored universe members to compare against, have {len(members)}")
    values = pd.Series([m.score for m in members])
    spread = float(values.std())
    if spread == 0 or pd.isna(spread):
        raise ValueError("comparison universe has no spread to normalise against")
    mean = float(values.mean())
    session = asof if asof is not None else max(m.asof for m in members)
    out = [
        RelativeScore(
            symbol=s.symbol,
            score=s.score,
            relative=(s.score - mean) / spread,
            percentile=100.0 * float((values <= s.score).sum()) / len(values),
            last=s.last,
            asof=s.asof,
            window=s.window,
        )
        for s in (by_symbol[w] for w in dict.fromkeys(wanted) if w in by_symbol)
    ]
    if not out:
        raise RuntimeError("no symbols could be scored")
    out.sort(key=lambda s: s.relative, reverse=True)
    return out, Reference(
        session=session,
        count=len(members),
        mean=mean,
        stdev=spread,
        stale=tuple(m.symbol for m in members if m.asof < session),
    )


def to_frame(scores: list[TrendScore]) -> pd.DataFrame:
    """Tabulate scores the way the watchlist reads: symbol, last, score, date."""
    return pd.DataFrame(
        {
            "symbol": [s.symbol for s in scores],
            "last": [round(s.last, 2) for s in scores],
            "score": [round(s.score, 2) for s in scores],
            "asof": [s.asof.date().isoformat() for s in scores],
        }
    )


def to_relative_frame(scores: list[RelativeScore]) -> pd.DataFrame:
    """Tabulate relative scores: the universe standing first, raw score beside it."""
    return pd.DataFrame(
        {
            "symbol": [s.symbol for s in scores],
            "last": [round(s.last, 2) for s in scores],
            "relative": [round(s.relative, 2) for s in scores],
            "pct": [round(s.percentile) for s in scores],
            "score": [round(s.score, 2) for s in scores],
            "asof": [s.asof.date().isoformat() for s in scores],
        }
    )


def render_reference(reference: Reference) -> str:
    """One-line footer stating what the relative scores were measured against."""
    line = (
        f"universe: {reference.count} names as of "
        f"{reference.session.date().isoformat()}, "
        f"raw score mean {reference.mean:+.2f} sd {reference.stdev:.2f}"
    )
    if reference.stale:
        names = ", ".join(reference.stale[:5])
        if len(reference.stale) > 5:
            names += f", +{len(reference.stale) - 5} more"
        line += f"\nstale (scored at an earlier close): {names}"
    return line
