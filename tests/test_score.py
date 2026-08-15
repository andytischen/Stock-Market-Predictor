import numpy as np
import pandas as pd
import pytest

from gapmodel import score as score_module
from gapmodel.score import (
    DEFAULT_WINDOW,
    Reference,
    RelativeScore,
    TrendScore,
    relative_scores,
    render_reference,
    to_frame,
    to_relative_frame,
    trend_score,
)


def _frame(closes: list[float], end: str = "2026-08-14") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=index)


def test_trend_score_is_zscore_of_last_close():
    closes = [10.0, 12.0, 14.0, 16.0, 18.0]
    tail = np.array(closes)
    expected = (closes[-1] - tail.mean()) / tail.std(ddof=1)
    assert trend_score(_frame(closes), window=len(closes)) == pytest.approx(expected)


def test_positive_when_last_above_average_negative_when_below():
    up = _frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 20.0])
    down = _frame([20, 9, 8, 7, 6, 5, 4, 3, 2, 1.0])
    assert trend_score(up, window=10) > 0
    assert trend_score(down, window=10) < 0


def test_only_the_trailing_window_matters():
    early = [1.0] * 300
    recent = [1.0, 1.1, 1.2, 1.3, 1.4]
    frame = _frame(early + recent)
    windowed = trend_score(frame, window=len(recent))
    assert windowed == pytest.approx(trend_score(_frame(recent), window=len(recent)))


def test_refuses_short_history():
    with pytest.raises(ValueError):
        trend_score(_frame([1.0, 2.0, 3.0]), window=DEFAULT_WINDOW)


def test_refuses_flat_series():
    with pytest.raises(ValueError):
        trend_score(_frame([5.0] * 10), window=10)


def test_to_frame_sorts_and_rounds():
    scores = [
        TrendScore("AAA", 1.234, 10.0, pd.Timestamp("2026-08-04"), 200),
        TrendScore("BBB", -2.0, 5.561, pd.Timestamp("2026-08-04"), 200),
    ]
    frame = to_frame(scores)
    assert list(frame["symbol"]) == ["AAA", "BBB"]
    assert frame.loc[0, "score"] == pytest.approx(1.23)
    assert frame.loc[1, "last"] == pytest.approx(5.56)
    assert frame.loc[1, "score"] == pytest.approx(-2.0)
    assert list(frame["asof"]) == ["2026-08-04", "2026-08-04"]


WINDOW = 10

# Series chosen to spread the raw scores out: AAA 1.84 > BBB 1.64 > CCC 0.92 >
# DDD -0.87 > EEE -1.49, with ZZZ (1.83) landing between BBB and AAA. Only the
# *shape* of each series matters, so a flat body under a single jump will not do
# - that always standardises to the same value whatever the jump.
CLOSES = {
    "AAA": [10, 10, 10, 10, 10, 10.5, 11, 11.5, 12, 12.5],
    "BBB": [10, 10.2, 9.8, 10.1, 10.3, 10.2, 10.6, 10.9, 11.0, 11.2],
    "CCC": [10, 10.5, 9.5, 10.2, 9.8, 10.4, 9.9, 10.3, 10.1, 10.4],
    "DDD": [10, 10.4, 10.2, 10.6, 10.3, 10.1, 9.9, 10.0, 9.8, 9.9],
    "EEE": [12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5],
    "ZZZ": [10, 10, 10.5, 11, 12, 13, 14, 15, 16, 18],
}
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def bars(monkeypatch):
    """Serve synthetic daily bars so the cross-section is exactly known."""
    ends: dict[str, str] = {}

    def fake_load(symbol, start, cache_dir, refresh):
        if symbol not in CLOSES:
            raise ValueError(f"no data for {symbol}")
        return _frame(CLOSES[symbol], end=ends.get(symbol, "2026-08-14"))

    monkeypatch.setattr(score_module, "load_symbol", fake_load)
    return ends


def _raw(symbol: str) -> float:
    return trend_score(_frame(CLOSES[symbol]), window=WINDOW)


def test_relative_is_the_zscore_of_raw_scores_across_the_universe(bars):
    scores, reference = relative_scores(["AAA", "EEE"], UNIVERSE, window=WINDOW)

    raws = pd.Series([_raw(s) for s in UNIVERSE])
    assert reference.count == 5
    assert reference.mean == pytest.approx(raws.mean())
    assert reference.stdev == pytest.approx(raws.std())
    by_symbol = {s.symbol: s for s in scores}
    for symbol in ("AAA", "EEE"):
        expected = (_raw(symbol) - raws.mean()) / raws.std()
        assert by_symbol[symbol].relative == pytest.approx(expected)
        assert by_symbol[symbol].score == pytest.approx(_raw(symbol))


def test_relative_centres_the_universe_on_zero(bars):
    scores, _ = relative_scores(UNIVERSE, UNIVERSE, window=WINDOW)
    assert np.mean([s.relative for s in scores]) == pytest.approx(0.0, abs=1e-12)


def test_percentile_is_the_share_of_the_universe_scoring_no_higher(bars):
    scores, _ = relative_scores(UNIVERSE, UNIVERSE, window=WINDOW)
    by_symbol = {s.symbol: s for s in scores}
    assert by_symbol["AAA"].percentile == pytest.approx(100.0)
    assert by_symbol["EEE"].percentile == pytest.approx(20.0)
    assert by_symbol["CCC"].percentile == pytest.approx(60.0)


def test_the_test_series_have_the_spread_the_other_cases_assume(bars):
    ranked = sorted(UNIVERSE, key=_raw, reverse=True)
    assert ranked == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert _raw("BBB") < _raw("ZZZ") < _raw("AAA")


def test_ranks_by_relative_strongest_first(bars):
    scores, _ = relative_scores(["EEE", "AAA", "CCC"], UNIVERSE, window=WINDOW)
    assert [s.symbol for s in scores] == ["AAA", "CCC", "EEE"]


def test_scores_a_symbol_outside_the_comparison_universe(bars):
    scores, reference = relative_scores(["ZZZ"], UNIVERSE, window=WINDOW)
    assert reference.count == 5  # ZZZ does not join the distribution it is measured against
    assert scores[0].symbol == "ZZZ"
    assert scores[0].relative > 0
    # ZZZ outscores all but AAA, so four of the five members score no higher.
    assert scores[0].percentile == pytest.approx(80.0)


def test_skipped_universe_members_do_not_break_the_comparison(bars):
    scores, reference = relative_scores(["AAA"], [*UNIVERSE, "NOPE"], window=WINDOW)
    assert reference.count == 5
    assert scores[0].symbol == "AAA"


def test_refuses_a_universe_too_small_to_compare_against(bars):
    with pytest.raises(ValueError, match="need 2 scored universe members"):
        relative_scores(["AAA"], ["AAA"], window=WINDOW)


def test_refuses_an_empty_universe(bars):
    with pytest.raises(ValueError, match="universe is empty"):
        relative_scores(["AAA"], [], window=WINDOW)


def test_refuses_a_universe_with_no_spread(bars, monkeypatch):
    monkeypatch.setattr(score_module, "trend_score", lambda frame, window: 1.5)
    with pytest.raises(ValueError, match="no spread"):
        relative_scores(["AAA"], UNIVERSE, window=WINDOW)


def test_session_is_the_newest_close_and_laggards_are_flagged(bars):
    bars["EEE"] = "2026-08-11"
    _, reference = relative_scores(["AAA"], UNIVERSE, window=WINDOW)
    assert reference.session == pd.Timestamp("2026-08-14")
    assert reference.stale == ("EEE",)


def test_a_non_trading_asof_falls_back_to_the_session_everyone_was_scored_on(bars):
    """Asking as of a Saturday must not report the whole universe as lagging."""
    _, reference = relative_scores(
        ["AAA"], UNIVERSE, window=WINDOW, asof=pd.Timestamp("2026-08-15")
    )
    assert reference.session == pd.Timestamp("2026-08-14")
    assert reference.stale == ()


def test_a_genuine_laggard_is_still_flagged_under_a_non_trading_asof(bars):
    """The fallback must not empty the stale list, only stop it swallowing everyone."""
    bars["EEE"] = "2026-08-11"
    _, reference = relative_scores(
        ["AAA"], UNIVERSE, window=WINDOW, asof=pd.Timestamp("2026-08-15")
    )
    assert reference.session == pd.Timestamp("2026-08-14")
    assert reference.stale == ("EEE",)


def test_to_relative_frame_rounds_and_orders_columns():
    scores = [
        RelativeScore("AAA", 1.234, 0.876, 83.3333, 10.0, pd.Timestamp("2026-08-14"), 200),
    ]
    frame = to_relative_frame(scores)
    assert list(frame.columns) == ["symbol", "last", "relative", "pct", "score", "asof"]
    assert frame.loc[0, "relative"] == pytest.approx(0.88)
    assert frame.loc[0, "pct"] == 83
    assert frame.loc[0, "score"] == pytest.approx(1.23)
    assert frame.loc[0, "asof"] == "2026-08-14"


def test_render_reference_states_the_comparison():
    footer = render_reference(Reference(pd.Timestamp("2026-08-14"), 150, -0.25, 1.4, stale=()))
    assert "150 names as of 2026-08-14" in footer
    assert "mean -0.25 sd 1.40" in footer
    assert "stale" not in footer


def test_render_reference_summarises_a_long_stale_list():
    stale = tuple(f"S{i}" for i in range(8))
    footer = render_reference(Reference(pd.Timestamp("2026-08-14"), 150, 0.0, 1.0, stale))
    assert "S0, S1, S2, S3, S4, +3 more" in footer
    assert "S5" not in footer
