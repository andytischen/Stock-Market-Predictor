import numpy as np
import pandas as pd
import pytest

from gapmodel.journal import (
    LATE,
    NO_SESSION,
    PENDING,
    SETTLED,
    STALE,
    decayed,
    empty_log,
    read_log,
    record,
    render_text,
    settle,
    skills,
    write_log,
)
from gapmodel.predict import Forecast


def _forecast(symbol: str, session: str, probability: float) -> Forecast:
    return Forecast(
        symbol=symbol,
        name=f"name of {symbol}",
        region="Americas",
        session=pd.Timestamp(session),
        probability_up=probability,
        backtest={"base_rate": 0.54, "brier_skill": 0.02},
        contributions=pd.Series(dtype=float),
    )


def _bars(opens: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Daily bars from ``{date: (open, close)}``."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in opens])
    return pd.DataFrame(
        {"Open": [o for o, _ in opens.values()], "Close": [c for _, c in opens.values()]},
        index=index,
    )


def _journal(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = empty_log()
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)[list(frame.columns)]


def _settled(symbol: str, session: str, probability: float, outcome: float) -> dict[str, object]:
    return {
        "recorded": "2026-01-01T06:00:00Z",
        "session": session,
        "symbol": symbol,
        "market": f"name of {symbol}",
        "region": "Americas",
        "p_open_up": probability,
        "outcome": outcome,
        "status": SETTLED,
    }


def test_record_appends_one_row_per_market_and_session():
    journal, added = record(empty_log(), [_forecast("^GSPC", "2026-08-17", 0.61)])
    assert added == ["^GSPC"]
    assert list(journal["status"]) == [PENDING]
    assert journal.at[0, "session"] == "2026-08-17"
    assert journal.at[0, "p_open_up"] == pytest.approx(0.61)


def test_a_recorded_session_is_never_overwritten():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-17", 0.61)])
    again, added = record(journal, [_forecast("^GSPC", "2026-08-17", 0.39)])
    assert added == []
    assert len(again) == 1
    assert again.at[0, "p_open_up"] == pytest.approx(0.61)


def test_the_next_session_is_recorded_alongside_the_previous_one():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-17", 0.61)])
    journal, added = record(journal, [_forecast("^GSPC", "2026-08-18", 0.44)])
    assert added == ["^GSPC"]
    assert sorted(journal["session"]) == ["2026-08-17", "2026-08-18"]


def test_a_forecast_for_a_session_that_already_printed_is_journalled_late():
    panel = {"^GSPC": _bars({"2026-08-13": (99.0, 100.0), "2026-08-14": (101.0, 102.0)})}
    journal, added = record(empty_log(), [_forecast("^GSPC", "2026-08-14", 0.81)], panel)
    assert added == ["^GSPC"]
    assert journal.at[0, "status"] == LATE


def test_a_late_row_is_never_settled_or_scored():
    panel = {"^GSPC": _bars({"2026-08-13": (99.0, 100.0), "2026-08-14": (101.0, 102.0)})}
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-14", 0.81)], panel)
    journal, filled, retired = settle(journal, panel)
    assert (filled, retired) == (0, 0)
    assert pd.isna(journal.at[0, "outcome"])
    assert skills(journal, min_settled=1) == []


def test_a_forecast_for_a_session_still_to_come_is_pending():
    panel = {"^GSPC": _bars({"2026-08-13": (99.0, 100.0), "2026-08-14": (101.0, 102.0)})}
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-17", 0.61)], panel)
    assert journal.at[0, "status"] == PENDING


def test_settle_labels_an_up_gap_against_the_previous_close():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (101.0, 102.0)})}
    journal, filled, _ = settle(journal, panel)
    assert filled == 1
    assert journal.at[0, "status"] == SETTLED
    assert journal.at[0, "prev_close"] == pytest.approx(100.0)
    assert journal.at[0, "outcome"] == pytest.approx(1.0)
    assert journal.at[0, "gap"] == pytest.approx(np.log(101.0 / 100.0), abs=1e-6)


def test_settle_labels_a_down_gap():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (98.0, 99.0)})}
    journal, _, _ = settle(journal, panel)
    assert journal.at[0, "outcome"] == pytest.approx(0.0)


def test_a_session_that_has_not_printed_stays_pending():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0)})}
    journal, filled, retired = settle(journal, panel)
    assert (filled, retired) == (0, 0)
    assert journal.at[0, "status"] == PENDING


def test_an_open_repeating_the_previous_close_is_unscorable():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (100.0, 101.0)})}
    journal, filled, retired = settle(journal, panel)
    assert (filled, retired) == (0, 1)
    assert journal.at[0, "status"] == STALE
    assert pd.isna(journal.at[0, "outcome"])


def test_a_market_with_a_stale_index_open_is_settled_on_its_tracker():
    # Yahoo repeats the previous close as the FTSE's own open, which is why the
    # model labels it on ISF.L: settling on the index would retire every session.
    journal, _ = record(empty_log(), [_forecast("^FTSE", "2026-08-18", 0.61)])
    panel = {
        "^FTSE": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (100.0, 101.0)}),
        "ISF.L": _bars({"2026-08-17": (9.9, 10.0), "2026-08-18": (10.2, 10.3)}),
    }
    journal, filled, _ = settle(journal, panel)
    assert filled == 1
    assert journal.at[0, "status"] == SETTLED
    assert journal.at[0, "open"] == pytest.approx(10.2)
    assert journal.at[0, "outcome"] == pytest.approx(1.0)


def test_the_late_check_reads_the_tracker_the_session_came_from():
    panel = {
        "^FTSE": _bars({"2026-08-17": (99.0, 100.0)}),
        "ISF.L": _bars({"2026-08-17": (9.9, 10.0), "2026-08-18": (10.2, 10.3)}),
    }
    journal, _ = record(empty_log(), [_forecast("^FTSE", "2026-08-18", 0.61)], panel)
    assert journal.at[0, "status"] == LATE


def test_the_late_check_sees_the_session_whose_close_has_not_printed():
    # The row that makes a forecast late is the one settlement discards: today's
    # auction has printed, its close has not, so the model forecasts a session
    # that is already in the past.
    bars = _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (101.0, 102.0)})
    bars.loc[pd.Timestamp("2026-08-18"), "Close"] = float("nan")
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)], {"^GSPC": bars})
    assert journal.at[0, "status"] == LATE


def test_a_session_whose_auction_has_not_run_yet_is_a_forecast_not_a_late_row():
    # The source publishes tomorrow's row before the auction: an empty ``Open``
    # is a morning still to come, so the forecast stays scoreable.
    bars = _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (101.0, 102.0)})
    bars.loc[pd.Timestamp("2026-08-18"), ["Open", "Close"]] = float("nan")
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)], {"^GSPC": bars})
    assert journal.at[0, "status"] == PENDING


def test_an_open_repeating_the_previous_close_does_not_retire_the_forecast():
    # The source publishes a placeholder open for a session it has no auction
    # for. Reading that as an opening print would file the forecast late, and
    # late is terminal, so the morning could never be scored once it arrives.
    bars = _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (100.0, 101.0)})
    bars.loc[pd.Timestamp("2026-08-18"), "Close"] = float("nan")
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)], {"^GSPC": bars})
    assert journal.at[0, "status"] == PENDING


def test_a_placeholder_open_still_settles_as_stale_once_the_session_closes():
    # The same row the late check declines to read: settlement is what retires
    # it, and it does so as stale rather than never scoring it at all.
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (100.0, 101.0)})}
    journal, filled, retired = settle(journal, panel)
    assert (filled, retired) == (0, 1)
    assert journal.at[0, "status"] == STALE


def test_the_late_check_reads_a_company_on_the_same_basis_settlement_does():
    # Going ex-dividend moves the raw open by the dividend, so a print that
    # merely looks like a repeated close on raw bars is a real auction once both
    # prints are on a total-return basis -- the basis settlement grades on.
    bars = _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (100.0, 101.0)})
    bars["Adj Close"] = [99.0, 101.0]
    journal, _ = record(empty_log(), [_forecast("MU", "2026-08-18", 0.61)], {"MU": bars})
    assert journal.at[0, "status"] == LATE


def test_a_print_with_no_earlier_session_to_measure_it_against_stays_pending():
    # Nothing to compute a gap from, so the print is unverified; late is
    # terminal, and retiring a forecast on it is the expensive way to be wrong.
    bars = _bars({"2026-08-18": (100.0, 101.0)})
    bars.loc[pd.Timestamp("2026-08-18"), "Close"] = float("nan")
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)], {"^GSPC": bars})
    assert journal.at[0, "status"] == PENDING


def test_a_bar_with_no_opening_print_is_not_mistaken_for_a_holiday():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    bars = _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (101.0, 102.0)})
    bars.loc[pd.Timestamp("2026-08-18"), "Open"] = float("nan")
    journal, filled, retired = settle(journal, {"^GSPC": bars})
    assert (filled, retired) == (0, 0)
    assert journal.at[0, "status"] == PENDING


def test_a_session_the_market_never_held_is_retired_not_awaited():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-19": (101.0, 102.0)})}
    journal, filled, retired = settle(journal, panel)
    assert (filled, retired) == (0, 1)
    assert journal.at[0, "status"] == NO_SESSION
    assert pd.isna(journal.at[0, "outcome"])


def test_settled_rows_are_not_rescored():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    panel = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (101.0, 102.0)})}
    journal, _, _ = settle(journal, panel)
    revised = {"^GSPC": _bars({"2026-08-17": (99.0, 100.0), "2026-08-18": (90.0, 91.0)})}
    journal, filled, _ = settle(journal, revised)
    assert filled == 0
    assert journal.at[0, "outcome"] == pytest.approx(1.0)


def test_skill_needs_a_minimum_of_settled_sessions():
    rows = [_settled("^GSPC", f"2026-01-{day:02d}", 0.6, 1.0) for day in range(1, 6)]
    assert skills(_journal(rows), min_settled=20) == []
    assert len(skills(_journal(rows), min_settled=5)) == 1


def test_skill_only_reads_the_most_recent_window():
    stale = [_settled("^GSPC", f"2026-01-{day:02d}", 0.9, 0.0) for day in range(1, 11)]
    fresh = [_settled("^GSPC", f"2026-02-{day:02d}", 0.9, 1.0) for day in range(1, 11)]
    measured = skills(_journal(stale + fresh), window=10, min_settled=10)[0]
    assert measured.settled == 10
    assert measured.hit_rate == pytest.approx(1.0)
    assert measured.first == "2026-02-01"


def test_skill_is_measured_against_the_markets_own_base_rate():
    # Opens up on 8 of 10 sessions; a forecast that always prints the base rate
    # adds nothing, so its skill is zero rather than its 80% hit rate.
    outcomes = [1.0] * 8 + [0.0] * 2
    rows = [
        _settled("^GSPC", f"2026-01-{day:02d}", 0.8, outcome)
        for day, outcome in enumerate(outcomes, start=1)
    ]
    measured = skills(_journal(rows), min_settled=10)[0]
    assert measured.base_rate == pytest.approx(0.8)
    assert measured.hit_rate == pytest.approx(0.8)
    assert measured.brier_skill == pytest.approx(0.0)
    assert measured.decayed


def test_a_forecast_that_beats_the_base_rate_is_not_decayed():
    outcomes = [1.0] * 5 + [0.0] * 5
    rows = [
        _settled("^GSPC", f"2026-01-{day:02d}", 0.9 if outcome else 0.1, outcome)
        for day, outcome in enumerate(outcomes, start=1)
    ]
    measured = skills(_journal(rows), min_settled=10)[0]
    assert measured.hit_rate == pytest.approx(1.0)
    assert measured.brier_skill > 0.9
    assert not measured.decayed
    assert decayed([measured]) == []


def test_the_bar_is_the_drift_not_the_up_rate_in_a_falling_market():
    # Opens up on 3 of 10 sessions, so calling every open down is right 70% of
    # the time. A 60% hit rate beats the up-rate and still adds no direction.
    calls = [(0.3, 0.0)] * 6 + [(0.6, 0.0), (0.45, 1.0), (0.45, 1.0), (0.45, 1.0)]
    rows = [
        _settled("^BVSP", f"2026-01-{day:02d}", probability, outcome)
        for day, (probability, outcome) in enumerate(calls, start=1)
    ]
    measured = skills(_journal(rows), min_settled=10)[0]
    assert measured.base_rate == pytest.approx(0.3)
    assert measured.drift_rate == pytest.approx(0.7)
    assert measured.hit_rate == pytest.approx(0.6)
    assert measured.hit_rate > measured.base_rate
    # Decay is called on the hit rate alone: the probabilities are still
    # calibrated enough to beat a constant forecast on Brier score.
    assert measured.brier_skill > 0.0
    assert measured.decayed


def test_a_market_with_no_variance_reports_no_skill():
    rows = [_settled("^GSPC", f"2026-01-{day:02d}", 0.6, 1.0) for day in range(1, 11)]
    measured = skills(_journal(rows), min_settled=10)[0]
    assert np.isnan(measured.brier_skill)


def test_a_market_with_no_variance_is_not_decayed_for_missing_the_perfect_drift():
    # Its drift is a perfect 100% by construction, so no forecast can clear it
    # and there is no measurable skill to print alongside the alert.
    rows = [_settled("^GSPC", f"2026-01-{day:02d}", 0.6, 1.0) for day in range(1, 11)]
    journal = _journal(rows)
    measured = skills(journal, min_settled=10)
    assert measured[0].drift_rate == pytest.approx(1.0)
    assert measured[0].hit_rate == pytest.approx(1.0)
    assert not measured[0].decayed
    assert decayed(measured) == []
    assert "below their own drift" not in render_text(journal, measured, window=60, min_settled=10)


def test_a_market_with_no_variance_is_still_decayed_when_it_calls_the_wrong_side():
    # Every session opened up and the model said down every morning: the Brier
    # leg cannot be computed, but direction alone says it has stopped reading.
    rows = [_settled("^GSPC", f"2026-01-{day:02d}", 0.2, 1.0) for day in range(1, 11)]
    journal = _journal(rows)
    measured = skills(journal, min_settled=10)
    assert measured[0].hit_rate == pytest.approx(0.0)
    assert measured[0].decayed
    text = render_text(journal, measured, window=60, min_settled=10)
    assert "below their own drift" in text
    # The unmeasurable Brier skill is left out of the alert rather than printed.
    assert "Brier skill" not in text.rsplit("read here:", 1)[-1]


def test_a_minimum_the_window_cannot_hold_is_refused():
    rows = [_settled("^GSPC", f"2026-01-{day:02d}", 0.6, float(day % 2)) for day in range(1, 31)]
    with pytest.raises(ValueError, match="cannot exceed window"):
        skills(_journal(rows), window=10, min_settled=20)


def test_a_short_window_narrows_the_default_minimum_instead_of_failing():
    # A caller who asks for a shorter read never mentioned the 20-session
    # default, so it is capped at the window rather than thrown back at them.
    rows = [
        _settled("^GSPC", f"2026-01-{day:02d}", 0.9 if day % 2 else 0.1, float(day % 2))
        for day in range(1, 11)
    ]
    measured = skills(_journal(rows), window=10)
    assert [s.settled for s in measured] == [10]
    assert "no market has 10 settled sessions" in render_text(_journal(rows), [], window=10)


def test_skills_are_sorted_with_the_worst_last():
    good = [
        _settled("^GSPC", f"2026-01-{day:02d}", 0.9 if day % 2 else 0.1, float(day % 2))
        for day in range(1, 11)
    ]
    bad = [
        _settled("^N225", f"2026-01-{day:02d}", 0.1 if day % 2 else 0.9, float(day % 2))
        for day in range(1, 11)
    ]
    measured = skills(_journal(good + bad), min_settled=10)
    assert [s.symbol for s in measured] == ["^GSPC", "^N225"]
    assert [s.symbol for s in decayed(measured)] == ["^N225"]


def test_round_trip_through_a_csv_preserves_the_journal(tmp_path):
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    path = tmp_path / "docs" / "forecast-log.csv"
    write_log(journal, path)
    reloaded = read_log(path)
    assert list(reloaded.columns) == list(journal.columns)
    assert reloaded.at[0, "session"] == "2026-08-18"
    assert reloaded.at[0, "status"] == PENDING


def test_read_log_widens_a_journal_written_without_every_column(tmp_path):
    path = tmp_path / "forecast-log.csv"
    path.write_text("session,symbol,p_open_up,status\n2026-08-18,^GSPC,0.61,pending\n")
    reloaded = read_log(path)
    assert list(reloaded.columns) == list(empty_log().columns)
    assert pd.isna(reloaded.at[0, "outcome"])


def test_missing_journal_reads_as_empty(tmp_path):
    assert read_log(tmp_path / "nothing.csv").empty


def test_render_says_so_when_there_is_not_enough_history():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    text = render_text(journal, [], window=60)
    assert "pending 1" in text
    assert "no market has 20 settled sessions" in text


def test_render_quotes_the_minimum_it_was_asked_for():
    journal, _ = record(empty_log(), [_forecast("^GSPC", "2026-08-18", 0.61)])
    text = render_text(journal, [], window=60, min_settled=5)
    assert "no market has 5 settled sessions" in text


def test_render_calls_out_a_decayed_market():
    rows = [
        _settled("^N225", f"2026-01-{day:02d}", 0.2, 1.0 if day % 2 else 0.0)
        for day in range(1, 11)
    ]
    journal = _journal(rows)
    text = render_text(journal, skills(journal, min_settled=10), window=60)
    assert "below their own drift" in text
    assert "^N225" in text
