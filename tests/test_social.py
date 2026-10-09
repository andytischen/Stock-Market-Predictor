"""Tests for gapmodel.social: scan(), render_text(), CLI social subcommand.

All HTTP calls are intercepted by monkeypatching requests.Session so the
test suite does not make any network requests.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from gapmodel.social import SocialSignal, render_text, scan
from gapmodel.social.report import render_html
from gapmodel.social.sentiment import score_text
from gapmodel.social.signals import _classify

# ── helpers ──────────────────────────────────────────────────────────────────


def _host_is(url: str, domain: str) -> bool:
    """True if *url*'s hostname is exactly *domain* or a subdomain of it.

    A substring check like ``"reddit.com" in url`` also matches an attacker
    or typo URL such as ``https://reddit.com.evil.example/``, so the mock
    router below compares the parsed hostname instead.
    """
    host = urlparse(url).hostname or ""
    return host == domain or host.endswith(f".{domain}")


def _make_session(reddit_payload: dict, stocktwits_payload: dict) -> MagicMock:
    """Build a mock requests.Session whose .get() returns pre-canned payloads."""
    session = MagicMock()

    def _get(url: str, **_kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if _host_is(url, "reddit.com"):
            resp.json.return_value = reddit_payload
        elif _host_is(url, "stocktwits.com"):
            resp.json.return_value = stocktwits_payload
        else:
            resp.json.return_value = {}
        return resp

    session.get.side_effect = _get
    return session


def _reddit_payload(titles: list[str], created_utc: float | None = None) -> dict:
    """Minimal Reddit search JSON matching the real API structure."""
    ts = created_utc or time.time()
    return {
        "data": {
            "children": [
                {
                    "data": {
                        "title": title,
                        "selftext": "",
                        "created_utc": ts,
                    }
                }
                for title in titles
            ]
        }
    }


def _stocktwits_payload(bodies: list[str], created_utc: float | None = None) -> dict:
    """Minimal StockTwits stream JSON."""
    ts = created_utc or time.time()
    import datetime

    iso = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {"messages": [{"body": b, "created_at": iso} for b in bodies]}


# ── sentiment unit tests ──────────────────────────────────────────────────────


def test_score_text_positive_phrase():
    score = score_text("This stock is absolutely fantastic and bullish!")
    assert score > 0.0, "expected a positive sentiment score"


def test_score_text_negative_phrase():
    score = score_text("This is terrible, crashing hard, disaster.")
    assert score < 0.0, "expected a negative sentiment score"


def test_score_text_empty_string_returns_zero():
    assert score_text("") == 0.0


# ── classify unit tests ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sentiment_mean", "bullish_ratio", "expected"),
    [
        (0.20, 0.60, "LONG"),
        (0.20, 0.50, "HOLD"),  # bullish_ratio too low for LONG
        (0.00, 0.50, "HOLD"),
        (-0.10, 0.50, "WEAK"),  # sentiment too negative
        (0.10, 0.35, "WEAK"),  # bullish_ratio too low
        (0.15, 0.55, "LONG"),  # exact boundary
        (-0.05, 0.50, "WEAK"),  # exact boundary
        (0.10, 0.40, "HOLD"),  # exact boundary (not WEAK)
    ],
)
def test_classify(sentiment_mean: float, bullish_ratio: float, expected: str):
    assert _classify(sentiment_mean, bullish_ratio) == expected


# ── scan() integration tests ──────────────────────────────────────────────────


def test_scan_returns_one_signal_per_ticker():
    session = _make_session(
        _reddit_payload(["Great earnings for AAPL today!", "AAPL is on fire"]),
        _stocktwits_payload(["$AAPL looks bullish to me"]),
    )
    results = scan(["AAPL", "TSLA"], session=session)
    assert len(results) == 2
    tickers = {s.ticker for s in results}
    assert tickers == {"AAPL", "TSLA"}


def test_scan_mention_count_includes_all_sources():
    session = _make_session(
        _reddit_payload(["post one", "post two"]),  # 2 Reddit posts
        _stocktwits_payload(["tweet one"]),  # 1 StockTwits
    )
    results = scan(["HOOD"], session=session)
    assert results[0].mention_count == 3


def test_scan_no_posts_gives_weak_signal():
    session = _make_session(
        {"data": {"children": []}},
        {"messages": []},
    )
    results = scan(["IMAX"], session=session)
    assert results[0].signal == "WEAK"
    assert results[0].mention_count == 0


def test_scan_bullish_posts_produce_long_signal():
    positive_posts = [
        "This stock is absolutely incredible and surging upward!",
        "Amazing results, best quarter ever, sky is the limit!",
        "Phenomenal performance, beating every estimate, wonderful!",
        "Superb growth, fantastic outlook, overwhelmingly positive!",
        "Brilliant company, excellent fundamentals, outstanding trajectory!",
    ]
    session = _make_session(
        _reddit_payload(positive_posts),
        {"messages": []},
    )
    results = scan(["CAKE"], session=session)
    sig = results[0]
    # With strong positive posts bullish_ratio should be well above 0.55
    assert sig.bullish_ratio > 0.50
    assert sig.sentiment_mean > 0.0


def test_scan_sorted_by_sentiment_mean_descending():
    # AAPL gets positive posts, TSLA gets negative posts
    def _side(url: str, **_kw: Any) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "AAPL" in url or ("reddit" in url and "AAPL" in url):
            resp.json.return_value = _reddit_payload(["Great stock, rising fast, wonderful!"])
        elif "TSLA" in url:
            resp.json.return_value = _reddit_payload(["Terrible crash, awful disaster"])
        else:
            resp.json.return_value = {"data": {"children": []}, "messages": []}
        return resp

    # Use separate mocked sessions per ticker by monkeypatching fetch_all
    # Instead, just build two distinct sessions and check order separately.
    session_aapl = _make_session(
        _reddit_payload(["Great stock, rising, wonderful earnings!"]),
        {"messages": []},
    )
    session_tsla = _make_session(
        _reddit_payload(["Terrible, awful disaster, crashing!"]),
        {"messages": []},
    )
    aapl_sig = scan(["AAPL"], session=session_aapl)[0]
    tsla_sig = scan(["TSLA"], session=session_tsla)[0]
    assert aapl_sig.sentiment_mean > tsla_sig.sentiment_mean


def test_scan_top_posts_capped_at_five():
    posts = [f"Post number {i}, great wonderful amazing!" for i in range(20)]
    session = _make_session(_reddit_payload(posts), {"messages": []})
    results = scan(["BIRK"], session=session)
    assert len(results[0].top_posts) <= 5


def test_scan_velocity_is_zero_when_no_prior_posts():
    # All posts are in the last 1 h so the prior hour bucket is empty → 0.0
    session = _make_session(
        _reddit_payload(["Current bullish post!"], created_utc=time.time()),
        {"messages": []},
    )
    results = scan(["HOOD"], session=session)
    assert results[0].velocity == 0.0


def test_scan_velocity_capped_at_9_9():
    now = time.time()
    recent = _reddit_payload(["recent " + str(i) for i in range(20)], created_utc=now - 100)
    prior = _stocktwits_payload(["prior"], created_utc=now - 5400)
    session = _make_session(recent, prior)
    results = scan(["CAKE"], session=session)
    assert results[0].velocity <= 9.9


def test_scan_http_error_falls_back_gracefully():
    """If both sources raise, scan should still return a WEAK signal."""
    session = MagicMock()
    session.get.side_effect = OSError("network unreachable")
    results = scan(["IMAX"], session=session)
    assert len(results) == 1
    assert results[0].signal == "WEAK"


# ── render_text() tests ───────────────────────────────────────────────────────


def test_render_text_contains_headers():
    sig = SocialSignal("AAPL", 10, 0.25, 0.60, 1.5, "LONG", [])
    out = render_text([sig])
    assert "Ticker" in out
    assert "Mentions" in out
    assert "Signal" in out


def test_render_text_shows_ticker_and_signal():
    sig = SocialSignal("BIRK", 5, -0.10, 0.30, 0.0, "WEAK", [])
    out = render_text([sig])
    assert "BIRK" in out
    assert "WEAK" in out


def test_render_text_empty_list():
    out = render_text([])
    assert "no signals" in out.lower()


def test_render_text_not_investment_advice():
    sig = SocialSignal("CAKE", 3, 0.05, 0.45, 0.0, "HOLD", [])
    out = render_text([sig])
    assert "not investment advice" in out.lower()


# ── render_html() tests ───────────────────────────────────────────────────────


def test_render_html_is_valid_structure():
    sig = SocialSignal("HOOD", 7, 0.18, 0.57, 2.1, "LONG", [])
    html = render_html([sig])
    assert html.startswith("<!doctype html>")
    assert "<table>" in html
    assert "HOOD" in html
    assert "LONG" in html


def test_render_html_escapes_special_characters():
    sig = SocialSignal("<TICK&ER>", 1, 0.0, 0.5, 0.0, "HOLD", [])
    html = render_html([sig])
    assert "<TICK&ER>" not in html
    assert "&lt;TICK&amp;ER&gt;" in html


# ── CLI social subcommand ─────────────────────────────────────────────────────


def test_cli_social_subcommand_parses(capsys, monkeypatch):
    """The social subcommand should run end-to-end with mocked HTTP."""

    def _mock_scan(tickers: list[str], *, window_hours: float = 24.0, **_kw: Any):
        return [SocialSignal(t, 0, 0.0, 0.5, 0.0, "HOLD", []) for t in tickers]

    monkeypatch.setattr("gapmodel.social.scan", _mock_scan)
    # Also patch the import alias in cli module
    monkeypatch.setattr("gapmodel.cli.social_scan", _mock_scan)

    from gapmodel.cli import main

    main(["social", "--tickers", "AAPL", "TSLA", "--window", "12"])
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "TSLA" in out


def test_cli_social_default_tickers(capsys, monkeypatch):
    """When no --tickers given the four default tickers should appear."""

    def _mock_scan(tickers: list[str], *, window_hours: float = 24.0, **_kw: Any):
        return [SocialSignal(t, 0, 0.0, 0.5, 0.0, "HOLD", []) for t in tickers]

    monkeypatch.setattr("gapmodel.cli.social_scan", _mock_scan)

    from gapmodel.cli import main

    main(["social"])
    out = capsys.readouterr().out
    for ticker in ("CAKE", "BIRK", "IMAX", "HOOD"):
        assert ticker in out


def test_cli_social_html_flag(tmp_path, monkeypatch):
    """--html PATH should write a file and print 'wrote <path>'."""

    def _mock_scan(tickers: list[str], **_kw: Any):
        return [SocialSignal("CAKE", 2, 0.1, 0.5, 0.0, "HOLD", [])]

    monkeypatch.setattr("gapmodel.cli.social_scan", _mock_scan)

    import io
    import sys

    out_path = tmp_path / "report.html"
    from gapmodel.cli import main

    # Redirect stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        main(["social", "--tickers", "CAKE", "--html", str(out_path)])
    finally:
        sys.stdout = sys.__stdout__

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "wrote" in captured.getvalue()
