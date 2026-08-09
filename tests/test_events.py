import pandas as pd

from gapmodel.events import (
    CALENDAR_END,
    FOMC_DECISIONS,
    caveats,
    events_on,
)
from gapmodel.markets import market

FIRST_FRIDAY = pd.Timestamp("2026-09-04")
SECOND_FRIDAY = pd.Timestamp("2026-09-11")
FOMC_DAY = pd.Timestamp(FOMC_DECISIONS[5])


def test_payrolls_land_on_the_first_friday_only():
    assert [e.name for e in events_on(FIRST_FRIDAY)] == ["US payrolls"]
    assert events_on(SECOND_FRIDAY) == ()
    # The first of a month that is not a Friday is not a payrolls day either.
    assert events_on(pd.Timestamp("2026-09-01")) == ()


def test_an_fomc_decision_is_recognised():
    assert [e.name for e in events_on(FOMC_DAY)] == ["FOMC decision"]


def test_releases_come_back_in_time_order():
    """Both on one day would be payrolls first: 13:30 UTC before 19:00 UTC."""
    both = events_on(pd.Timestamp("2026-12-04")), events_on(FOMC_DAY)
    for found in both:
        assert list(found) == sorted(found, key=lambda event: event.time_utc)


def test_nothing_is_claimed_past_the_end_of_the_table():
    """An unpublished FOMC calendar must not read as an empty one."""
    beyond = CALENDAR_END + pd.Timedelta(days=40)
    assert not any(e.name == "FOMC decision" for e in events_on(beyond))


def test_a_release_before_the_open_makes_the_call_stale():
    # Payrolls at 13:30 UTC, exactly when Wall Street opens.
    notes = caveats(market("^GSPC"), FIRST_FRIDAY)
    assert len(notes) == 1
    assert "US payrolls" in notes[0]
    assert "13:30 UTC" in notes[0]
    assert "stale" in notes[0]


def test_a_release_after_the_open_only_limits_the_call():
    # The same 13:30 UTC release is hours after the European auction.
    notes = caveats(market("^FTSE"), FIRST_FRIDAY)
    assert len(notes) == 1
    assert "after this open" in notes[0]
    assert "says nothing about the session" in notes[0]


def test_an_fomc_decision_is_after_every_open():
    for symbol in ("^N225", "^FTSE", "^GSPC"):
        notes = caveats(market(symbol), FOMC_DAY)
        assert len(notes) == 1
        assert "19:00 UTC" in notes[0]
        assert "after this open" in notes[0]


def test_an_ordinary_session_carries_no_caveat():
    assert caveats(market("^GSPC"), SECOND_FRIDAY) == ()


def test_sydney_reads_a_release_from_the_previous_calendar_day():
    """The ASX session dated Thursday opens at 23:00 UTC on the Wednesday.

    An FOMC decision at 19:00 UTC on the Wednesday is therefore four hours
    before that auction, not a day after it.
    """
    thursday = FOMC_DAY + pd.Timedelta(days=1)
    notes = caveats(market("^AXJO"), thursday)
    assert len(notes) == 1
    assert "the previous day" in notes[0]
    assert "before this open" in notes[0]
    # Every other market's Thursday session has nothing scheduled at all.
    assert caveats(market("^GSPC"), thursday) == ()


def test_the_fomc_table_is_sorted_and_free_of_duplicates():
    dates = [pd.Timestamp(d) for d in FOMC_DECISIONS]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)
    assert max(dates) <= CALENDAR_END


def test_every_scheduled_meeting_is_a_weekday():
    for date in FOMC_DECISIONS:
        assert pd.Timestamp(date).weekday() < 5, date


def test_the_time_of_a_release_is_a_utc_hour():
    for event in events_on(FIRST_FRIDAY) + events_on(FOMC_DAY):
        assert 0.0 <= event.time_utc <= 24.0
        assert event.date == event.date.normalize()
