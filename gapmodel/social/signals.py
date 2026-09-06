"""Aggregate social posts into a per-ticker signal.

The :func:`scan` function fetches posts from all configured sources,
scores them with VADER, and returns a ranked list of :class:`SocialSignal`
objects.  Each signal is labelled **LONG**, **HOLD**, or **WEAK** according
to the rules below - these are *probabilistic* indicators, not investment
advice.

Signal rules
------------
- **LONG**  - ``sentiment_mean >= 0.15`` **and** ``bullish_ratio >= 0.55``
- **WEAK**  - ``sentiment_mean <= -0.05`` **or** ``bullish_ratio < 0.40``
- **HOLD**  - everything else
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests as _requests

from .sentiment import score_text
from .sources import fetch_all

log = logging.getLogger(__name__)

_VELOCITY_CAP = 9.9
_BULLISH_THRESHOLD = 0.05


@dataclass
class SocialSignal:
    """Social-media sentiment summary for one ticker over a look-back window.

    .. warning::
        All fields are derived exclusively from posts whose ``created_utc``
        timestamp falls **before** the moment :func:`scan` is called.
        No information from future sessions is included.
    """

    ticker: str
    """Equity ticker symbol."""

    mention_count: int
    """Total posts collected across all sources within *window_hours*."""

    sentiment_mean: float
    """Mean VADER compound score across all posts in [-1, 1]."""

    bullish_ratio: float
    """Fraction of posts with compound score > 0.05."""

    velocity: float
    """Mentions in the last 1 h divided by mentions in the prior 1 h.

    Capped at 9.9.  Returns 0.0 when the prior hour had no posts (avoids
    division-by-zero while still flagging a cold start honestly).
    """

    signal: str
    """``"LONG"``, ``"HOLD"``, or ``"WEAK"``."""

    top_posts: list[str] = field(default_factory=list)
    """Up to five post bodies with the highest absolute sentiment score."""


def _classify(sentiment_mean: float, bullish_ratio: float) -> str:
    if sentiment_mean >= 0.15 and bullish_ratio >= 0.55:
        return "LONG"
    if sentiment_mean <= -0.05 or bullish_ratio < 0.40:
        return "WEAK"
    return "HOLD"


def _velocity(recent: list[dict[str, Any]], prior: list[dict[str, Any]]) -> float:
    n_prior = len(prior)
    if n_prior == 0:
        return 0.0
    return min(len(recent) / n_prior, _VELOCITY_CAP)


def _build_signal(ticker: str, posts: list[dict[str, Any]], window_hours: float) -> SocialSignal:
    """Compute a :class:`SocialSignal` from a list of raw post dicts.

    Temporal correctness: only posts with ``created_utc`` <= ``now`` are used;
    ``now`` is sampled once at the start of :func:`scan`.
    """
    now = time.time()
    cutoff = now - window_hours * 3600
    recent_cutoff = now - 3600  # last 1 h
    prior_cutoff = now - 2 * 3600  # 1 h before that

    in_window = [p for p in posts if p["created_utc"] >= cutoff]
    recent_posts = [p for p in posts if p["created_utc"] >= recent_cutoff]
    prior_posts = [p for p in posts if prior_cutoff <= p["created_utc"] < recent_cutoff]

    if not in_window:
        return SocialSignal(
            ticker=ticker,
            mention_count=0,
            sentiment_mean=0.0,
            bullish_ratio=0.0,
            velocity=0.0,
            signal="WEAK",
            top_posts=[],
        )

    scored = [(p, score_text(p["text"])) for p in in_window]
    scores = [s for _, s in scored]
    sentiment_mean = sum(scores) / len(scores)
    bullish_ratio = sum(1 for s in scores if s > _BULLISH_THRESHOLD) / len(scores)
    vel = _velocity(recent_posts, prior_posts)

    top = sorted(scored, key=lambda t: abs(t[1]), reverse=True)[:5]
    top_posts = [p["text"][:200] for p, _ in top]

    return SocialSignal(
        ticker=ticker,
        mention_count=len(in_window),
        sentiment_mean=round(sentiment_mean, 4),
        bullish_ratio=round(bullish_ratio, 4),
        velocity=round(vel, 2),
        signal=_classify(sentiment_mean, bullish_ratio),
        top_posts=top_posts,
    )


def scan(
    tickers: list[str],
    *,
    window_hours: float = 24.0,
    session: _requests.Session | None = None,
) -> list[SocialSignal]:
    """Fetch and score social posts for each ticker, returning a ranked list.

    The list is ordered by ``sentiment_mean`` descending (most bullish first).
    Signals are computed only from posts published before the current UTC
    moment; no future information is used.

    Parameters
    ----------
    tickers:
        Equity ticker symbols to scan, e.g. ``["AAPL", "TSLA"]``.
    window_hours:
        How far back to look for posts (default 24 h).
    session:
        Optional :class:`requests.Session` shared across all requests.
        Pass a monkeypatched session in tests to avoid network calls.

    Returns
    -------
    list[SocialSignal]
        One entry per ticker, sorted by ``sentiment_mean`` descending.

    .. note::
        Output is a probabilistic summary of public social-media sentiment.
        It is **not** investment advice.
    """
    signals: list[SocialSignal] = []
    for ticker in tickers:
        log.info("scanning social posts for %s", ticker)
        posts = fetch_all(ticker, session=session)
        signals.append(_build_signal(ticker, posts, window_hours))

    signals.sort(key=lambda s: s.sentiment_mean, reverse=True)
    return signals
