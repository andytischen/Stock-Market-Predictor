import pandas as pd

from gapmodel.events import (
    CALENDAR_END,
    CPI,
    FOMC,
    PAYROLLS,
    PCE,
    SCHEDULES,
    caveats,
    events_on,
)
from gapmodel.markets import last_observed_utc, market

PAYROLLS_DAY = pd.Timestamp("2026-09-04")
CPI_DAY = pd.Timestamp("2026-09-11")
FOMC_DAY = pd.Timestamp("2026-09-16")
QUIET_DAY = pd.Timestamp("2026-09-10")


def test_each_release_is_recognised_on_its_published_day():
    assert [e.name for e in events_on(PAYROLLS_DAY)] == ["US payrolls"]
    assert [e.name for e in events_on(CPI_DAY)] == ["US CPI"]
    assert [e.name for e in events_on(FOMC_DAY)] == ["FOMC decision"]
    assert [e.name for e in events_on(pd.Timestamp("2026-10-29"))] == ["US PCE inflation"]
    assert events_on(QUIET_DAY) == ()


def test_the_first_friday_rule_is_not_used():
    """The BLS scheduled three of 2026's twelve payroll reports off that rule."""
    assert "2026-02-06" not in PAYROLLS.dates  # published on Wednesday the 11th
    assert "2026-05-01" not in PAYROLLS.dates  # published on the second Friday
    assert "2026-07-03" not in PAYROLLS.dates  # published on Thursday the 2nd
    assert {"2026-02-11", "2026-05-08", "2026-07-02"} <= set(PAYROLLS.dates)
    assert events_on(pd.Timestamp("2026-05-01")) == ()


def test_a_release_before_the_open_makes_the_call_stale():
    # 08:30 in New York is 12:30 UTC in September, an hour before the bell.
    notes = caveats(market("^GSPC"), PAYROLLS_DAY)
    assert len(notes) == 1
    assert "US payrolls" in notes[0]
    assert "12:30 UTC" in notes[0]
    assert "before this open" in notes[0]
    assert "stale" in notes[0]


def test_a_release_after_the_open_only_limits_the_call():
    # The same release is hours after the European auction.
    notes = caveats(market("^FTSE"), CPI_DAY)
    assert len(notes) == 1
    assert "US CPI at 12:30 UTC, after this open" in notes[0]
    assert "says nothing about the session" in notes[0]


def test_an_fomc_decision_is_after_every_open():
    for symbol in ("^N225", "^FTSE", "^GSPC"):
        notes = caveats(market(symbol), FOMC_DAY)
        assert len(notes) == 1
        assert "18:00 UTC" in notes[0]
        assert "after this open" in notes[0]


def test_release_times_follow_new_york_across_the_clock_change():
    """14:00 ET is 18:00 UTC on daylight time and 19:00 UTC on standard time."""
    summer = caveats(market("^GSPC"), pd.Timestamp("2026-09-16"))[0]
    winter = caveats(market("^GSPC"), pd.Timestamp("2026-12-09"))[0]
    assert "18:00 UTC" in summer
    assert "19:00 UTC" in winter
    # Either way the statement lands after the opening auction.
    assert "after this open" in summer and "after this open" in winter


def test_an_ordinary_session_carries_no_caveat():
    assert caveats(market("^GSPC"), QUIET_DAY) == ()


def test_sydney_is_not_warned_about_a_decision_its_prices_already_reflect():
    """The ASX session dated Thursday opens at 23:00 UTC on the Wednesday.

    That is five hours after the Wednesday FOMC statement, but the features are
    not blind to it: the same session reads Wall Street's 20:00 close and the
    21:25 VIX, both struck after the statement. Warning about it would train the
    reader to ignore the warnings that matter.
    """
    thursday = FOMC_DAY + pd.Timedelta(days=1)
    assert caveats(market("^AXJO"), thursday) == ()
    assert caveats(market("^GSPC"), thursday) == ()


def test_sydney_is_warned_about_the_decision_it_opens_into():
    """The session dated Wednesday opens 23:00 UTC Tuesday, a day early."""
    notes = caveats(market("^AXJO"), FOMC_DAY)
    assert len(notes) == 1
    assert "18:00 UTC, after this open" in notes[0]


def test_a_release_no_price_bar_follows_is_the_only_stale_case():
    """Payrolls at 12:30 UTC fall after every bar Wall Street may read.

    Asia's closes (10:00 UTC at the latest) precede the print and Europe's come
    too late to be used, so nothing in the S&P's feature row saw it.
    """
    assert last_observed_utc(market("^GSPC").open_utc) == 10.0
    assert "before this open" in caveats(market("^GSPC"), PAYROLLS_DAY)[0]


def test_nothing_is_claimed_past_the_end_of_the_tables():
    """An unrefreshed calendar must not read as an empty one."""
    assert events_on(CALENDAR_END + pd.Timedelta(days=40)) == ()
    for schedule in SCHEDULES:
        assert max(pd.Timestamp(d) for d in schedule.dates) <= CALENDAR_END


def test_every_table_is_sorted_free_of_duplicates_and_attributed():
    for schedule in SCHEDULES:
        dates = [pd.Timestamp(d) for d in schedule.dates]
        assert dates == sorted(dates), schedule.name
        assert len(set(dates)) == len(dates), schedule.name
        assert schedule.source, schedule.name


def test_the_monthly_series_are_complete_for_the_year_they_cover():
    for schedule in (PAYROLLS, CPI):
        months = {pd.Timestamp(d).month for d in schedule.dates}
        assert months == set(range(1, 13)), schedule.name
    # The BEA calendar only shows the months still ahead, hence the short table.
    assert len(PCE.dates) == 5


def test_the_statistical_releases_share_the_half_past_eight_slot():
    assert PAYROLLS.time_et == CPI.time_et == PCE.time_et == (8, 30)
    assert FOMC.time_et == (14, 0)
