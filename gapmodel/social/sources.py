"""Fetch social posts mentioning a ticker from public APIs.

Supported sources
-----------------
- **Reddit** - ``/search.json`` (no auth, JSON-over-HTTP).
- **StockTwits** - ``/streams/symbol/{ticker}.json`` (no auth).

Each source returns a list of dicts with exactly three keys:

``text``
    Post body or title (str).
``source``
    Short identifier string, e.g. ``"reddit"`` or ``"stocktwits"``.
``created_utc``
    Unix timestamp (float) representing when the post was published.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests as _requests

log = logging.getLogger(__name__)

_REDDIT_URL = "https://www.reddit.com/search.json"
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

_DEFAULT_TIMEOUT = 10  # seconds

# Reddit refuses requests without a descriptive User-Agent.
_HEADERS = {"User-Agent": "stock-market-predictor/0.1 (research-only; no-auth)"}


def _utc_now() -> float:
    return time.time()


def fetch_reddit(
    ticker: str,
    *,
    limit: int = 100,
    session: _requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Return up to *limit* recent Reddit posts that mention *ticker*.

    Falls back to an empty list on any HTTP or network error so the caller's
    pipeline can continue with the remaining source.

    Parameters
    ----------
    ticker:
        The equity ticker symbol to search for (e.g. ``"AAPL"``).
    limit:
        Maximum number of posts to request (Reddit caps at 100).
    session:
        Optional :class:`requests.Session`; a new one is created if omitted.
        Pass a pre-configured or monkeypatched session in tests.
    """
    requester = session or _requests.Session()
    params: dict[str, Any] = {"q": ticker, "sort": "new", "limit": min(limit, 100)}
    try:
        response = requester.get(
            _REDDIT_URL,
            params=params,
            headers=_HEADERS,
            timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log.warning("reddit fetch for %s failed: %s", ticker, exc)
        return []

    posts: list[dict[str, Any]] = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        title: str = post.get("title", "")
        selftext: str = post.get("selftext", "")
        body = f"{title} {selftext}".strip()
        created: float = float(post.get("created_utc", _utc_now()))
        if body:
            posts.append({"text": body, "source": "reddit", "created_utc": created})
    return posts


def fetch_stocktwits(
    ticker: str,
    *,
    session: _requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Return recent StockTwits messages for *ticker*.

    Falls back to an empty list on any HTTP or network error.

    Parameters
    ----------
    ticker:
        The equity ticker symbol (e.g. ``"AAPL"``).
    session:
        Optional :class:`requests.Session` for testability.
    """
    requester = session or _requests.Session()
    url = _STOCKTWITS_URL.format(ticker=ticker)
    try:
        response = requester.get(url, headers=_HEADERS, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log.warning("stocktwits fetch for %s failed: %s", ticker, exc)
        return []

    posts: list[dict[str, Any]] = []
    for msg in data.get("messages", []):
        body: str = msg.get("body", "")
        # StockTwits uses "created_at" as an ISO 8601 string.
        created_str: str = msg.get("created_at", "")
        try:
            import datetime

            created = datetime.datetime.fromisoformat(
                created_str.replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            created = _utc_now()
        if body:
            posts.append({"text": body, "source": "stocktwits", "created_utc": created})
    return posts


def fetch_all(
    ticker: str,
    *,
    session: _requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch posts from every configured source and return them combined.

    Each element has ``text``, ``source``, and ``created_utc`` keys.
    """
    posts: list[dict[str, Any]] = []
    posts.extend(fetch_reddit(ticker, session=session))
    posts.extend(fetch_stocktwits(ticker, session=session))
    return posts
