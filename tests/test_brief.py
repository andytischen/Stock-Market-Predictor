import numpy as np
import pandas as pd
import pytest

from gapmodel.brief import (
    NOT_FORECAST,
    UNSEEN,
    Commentary,
    build_brief,
    quotes,
    read_commentary,
    render_html,
    render_text,
)
from gapmodel.events import upcoming
from gapmodel.markets import MARKETS_BY_SYMBOL
from gapmodel.predict import Forecast
from gapmodel.staleness import STALE_DAYS

NOON = pd.Timestamp("2026-08-21 12:00")


def bars(n=120, start=100.0, step=0.5, end="2026-08-21"):
    index = pd.bdate_range(end=end, periods=n)
    close = pd.Series(start + step * np.arange(n), index=index, dtype=float)
    return pd.DataFrame({"Open": close.shift(1).fillna(start), "Close": close})


@pytest.fixture
def panel():
    return {
        "BZ=F": bars(start=90.0, step=0.05),
        "CL=F": bars(start=84.0, step=-0.02),
        "GC=F": bars(start=4000.0, step=5.0),
        "^VIX": bars(start=16.0, step=-0.01),
        "^FTSE": bars(start=10000.0, step=6.0),
    }


def forecast(symbol="^FTSE", probability=0.62, auc=0.798, session="2026-08-24", caveats=()):
    meta = MARKETS_BY_SYMBOL[symbol]
    return Forecast(
        symbol=symbol,
        name=meta.name,
        region=meta.region,
        session=pd.Timestamp(session),
        probability_up=probability,
        backtest={"auc": auc},
        contributions=pd.Series({"ind_bz_f_return": 0.3}),
        caveats=caveats,
    )


def test_quotes_carry_the_session_they_came_from(panel):
    found = quotes(panel, NOON)
    assert [q.symbol for q in found] == ["BZ=F", "CL=F", "GC=F", "^VIX"]  # grouped order
    brent = found[0]
    assert brent.session == panel["BZ=F"].index[-1]
    assert brent.close == pytest.approx(panel["BZ=F"]["Close"].iloc[-1])
    assert brent.returns["1d"] > 0 and brent.lag_days == 0 and not brent.is_stale


def test_a_series_that_stopped_printing_is_flagged_rather_than_dropped(panel):
    panel["GC=F"] = bars(start=4000.0, step=5.0, end="2026-07-01")
    gold = next(q for q in quotes(panel, NOON) if q.symbol == "GC=F")
    assert gold.lag_days > STALE_DAYS and gold.is_stale


def test_a_yield_moves_in_basis_points_not_percent(panel):
    """'+0.1%' on a 4.7% ten-year reads as ten basis points and means five."""
    panel["^TNX"] = bars(start=4.0, step=0.01)
    tnx = next(q for q in quotes(panel, NOON) if q.symbol == "^TNX")
    assert tnx.is_rate and tnx.returns["1d"] == pytest.approx(1.0)
    assert tnx.returns["1w"] == pytest.approx(5.0)
    assert "+1.0bp" in render_text(build_brief(panel, [forecast()], as_of=NOON))


def test_a_window_longer_than_the_history_says_nothing(panel):
    panel["CL=F"] = bars(n=4, start=84.0)
    wti = next(q for q in quotes(panel, NOON) if q.symbol == "CL=F")
    assert np.isnan(wti.returns["1m"]) and not np.isnan(wti.returns["1d"])


def test_an_instrument_that_did_not_download_is_skipped(panel):
    panel["GC=F"] = pd.DataFrame({"Open": [], "Close": []})
    assert "GC=F" not in [q.symbol for q in quotes(panel, NOON)]


def test_calls_are_ordered_by_probability(panel):
    brief = build_brief(
        panel,
        [forecast("^FTSE", 0.3), forecast("^GDAXI", 0.8), forecast("^GSPC", 0.55)],
        as_of=NOON,
    )
    assert [c.symbol for c in brief.calls] == ["^GDAXI", "^GSPC", "^FTSE"]


def test_the_release_window_starts_at_the_first_session_called(panel):
    # PCE prints on 26 Aug: a brief for the 24th covers it, one for the 31st does not.
    covered = build_brief(panel, [forecast(session="2026-08-24")], as_of=NOON)
    assert "US PCE inflation" in [e.name for e in covered.releases]
    later = build_brief(panel, [forecast(session="2026-08-31")], as_of=NOON)
    assert "US PCE inflation" not in [e.name for e in later.releases]


def test_upcoming_covers_the_whole_window_in_order():
    events = upcoming(pd.Timestamp("2026-09-01"), 30)
    assert [e.name for e in events] == [
        "US payrolls",
        "US CPI",
        "FOMC decision",
        "US PCE inflation",
    ]
    assert [e.date for e in events] == sorted(e.date for e in events)


def test_upcoming_refuses_an_empty_window():
    with pytest.raises(ValueError, match="covers nothing"):
        upcoming(pd.Timestamp("2026-09-01"), 0)


def test_a_calendar_that_does_not_reach_the_window_is_named(panel):
    brief = build_brief(panel, [forecast(session="2027-06-01")], as_of=NOON)
    assert "US CPI" in brief.unmaintained and "FOMC decision" not in brief.unmaintained


def test_text_render_separates_the_call_from_the_tape_and_the_caveats(panel):
    caveat = "US PCE inflation at 12:30 UTC, before this open: the auction prices it"
    brief = build_brief(panel, [forecast(caveats=(caveat,))], as_of=NOON)
    out = render_text(brief)
    assert "Week-ahead brief — 2026-08-21 12:00 UTC" in out
    assert "FTSE 100" in out and "62.0%" in out and "0.798" in out
    assert "Brent crude" in out and "session 2026-08-21" in out
    assert caveat in out
    assert NOT_FORECAST[0].split(".")[0] in out
    assert UNSEEN[0][0] in out


def test_html_render_labels_provenance_and_escapes_the_notes(panel):
    written = Commentary(
        "Oil <week> ahead",
        paragraphs=("Brent is a binary on <Hormuz>.",),
        bullets=("Resistance $95.50-96.00",),
    )
    brief = build_brief(panel, [forecast()], as_of=NOON, commentary=(written,))
    html = render_html(brief)
    assert html.startswith("<!doctype html>")
    for tag in ("model output", "market data", "written view", "limitations"):
        assert f">{tag}</span>" in html
    assert "<td>FTSE 100</td>" in html and "62.0%" in html
    assert "Oil &lt;week&gt; ahead" in html and "&lt;Hormuz&gt;" in html
    assert "<li>Resistance $95.50-96.00</li>" in html
    assert "not investment advice" in html


def test_html_marks_a_stale_card(panel):
    panel["GC=F"] = bars(start=4000.0, step=5.0, end="2026-07-01")
    html = render_html(build_brief(panel, [forecast()], as_of=NOON))
    assert "card stale" in html and "stopped printing" in html


def test_read_commentary_reads_headings_paragraphs_and_bullets(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(
        "## Oil\nBrent held $94.\n- Hormuz is the binary\n- $90 support\n\n"
        "## Gold\nA debasement trade.\n",
        encoding="utf-8",
    )
    blocks = read_commentary(path)
    assert [b.title for b in blocks] == ["Oil", "Gold"]
    assert blocks[0].paragraphs == ("Brent held $94.",)
    assert blocks[0].bullets == ("Hormuz is the binary", "$90 support")
    assert blocks[1].bullets == ()


def test_read_commentary_refuses_a_file_with_no_heading(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Brent held $94.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no commentary found"):
        read_commentary(path)
