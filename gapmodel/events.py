"""Scheduled US releases the model is structurally unable to anticipate.

Every feature in the model is a price. A price already reflects what the market
knows, which makes it a good summary of the past and no guide at all to a number
that has not been published yet: the forecast for a session carrying the
payrolls report, a CPI print or an FOMC decision is built entirely from a world
in which that release has not happened. The probability is not wrong so much as
answering a narrower question than it appears to.

This module says when that is the case, so a call can be reported with the
caveat rather than at face value. Every date is taken from the publishing
agency's own calendar rather than derived from a rule, because the rules do not
hold: payrolls are conventionally the first Friday of the month, and in 2026 the
BLS scheduled them for a Wednesday in February, the second Friday in May and a
Thursday before Independence Day in July. A caveat that fires on the wrong day
teaches you to ignore it, so nothing here is guessed.

The consequence is that the tables end, and they end at different dates: the
Fed publishes its decisions two years ahead, while the BLS and BEA pages carry
only the coming twelve months. Each schedule therefore states how far it is
maintained, and past that point no flag is raised for it — the absence of a
warning means nothing was checked rather than nothing being scheduled.
``unmaintained_on`` names the series in that state so silence can be reported as
silence. Refreshing them is a yearly job: each ``Schedule`` names its page.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from .markets import Market, last_observed_utc

# Release times are published on New York's clock, and the gap between a release
# and an opening auction is what matters, so they are converted for the date in
# question rather than pinned to one offset.
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Schedule:
    """One release series, as published by the agency that puts it out.

    ``time_et`` is the announced release time on New York's clock, ``dates`` the
    published days, and ``source`` the page to refresh them from.
    ``covers_until`` is how far that page reached when it was last read, which is
    not the same as the last date in ``dates``: a table can be maintained through
    December and simply have no release in the final week.
    """

    name: str
    time_et: tuple[int, int]
    source: str
    covers_until: str
    dates: tuple[str, ...]

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.covers_until)


PAYROLLS = Schedule(
    name="US payrolls",
    time_et=(8, 30),
    source="bls.gov/schedule/news_release/empsit.htm",
    covers_until="2026-12-31",
    dates=(
        "2026-01-09",
        "2026-02-11",
        "2026-03-06",
        "2026-04-03",
        "2026-05-08",
        "2026-06-05",
        "2026-07-02",
        "2026-08-07",
        "2026-09-04",
        "2026-10-02",
        "2026-11-06",
        "2026-12-04",
    ),
)

CPI = Schedule(
    name="US CPI",
    time_et=(8, 30),
    source="bls.gov/schedule/news_release/cpi.htm",
    covers_until="2026-12-31",
    dates=(
        "2026-01-13",
        "2026-02-13",
        "2026-03-11",
        "2026-04-10",
        "2026-05-12",
        "2026-06-10",
        "2026-07-14",
        "2026-08-12",
        "2026-09-11",
        "2026-10-14",
        "2026-11-10",
        "2026-12-10",
    ),
)

# The Fed's target measure of inflation, published with personal income. The BEA
# calendar only shows the months still to come, so this table starts later in the
# year than the others; a session before its first entry is not claimed either.
PCE = Schedule(
    name="US PCE inflation",
    time_et=(8, 30),
    source="bea.gov/news/schedule",
    covers_until="2026-12-31",
    dates=(
        "2026-08-26",
        "2026-09-30",
        "2026-10-29",
        "2026-11-25",
        "2026-12-23",
    ),
)

# Decision days — the second day of each two-day meeting — with the statement at
# 14:00 ET, an hour and a half before the cash close. The Committee announces the
# following year each September, so this table runs a year past the others; every
# date after the current year is tentative until the preceding meeting confirms
# it, which is close enough for a caveat.
FOMC = Schedule(
    name="FOMC decision",
    time_et=(14, 0),
    source="federalreserve.gov/monetarypolicy/fomccalendars.htm",
    covers_until="2027-12-31",
    dates=(
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
        "2027-01-27",
        "2027-03-17",
        "2027-04-28",
        "2027-06-09",
        "2027-07-28",
        "2027-09-15",
        "2027-10-27",
        "2027-12-08",
    ),
)

SCHEDULES: tuple[Schedule, ...] = (PAYROLLS, CPI, PCE, FOMC)

# Past this date nothing at all is checked, because no table reaches it.
CALENDAR_END = max(schedule.end for schedule in SCHEDULES)


@dataclass(frozen=True)
class Event:
    """A scheduled release, on the same UTC clock as the market sessions."""

    name: str
    date: pd.Timestamp
    time_utc: float


def _utc_hours(day: pd.Timestamp, time_et: tuple[int, int]) -> float:
    """A New York time of day on ``day``, as hours from midnight UTC."""
    hour, minute = time_et
    moment = pd.Timestamp(
        year=day.year, month=day.month, day=day.day, hour=hour, minute=minute, tz=EASTERN
    ).tz_convert("UTC")
    return moment.hour + moment.minute / 60


def events_on(date: pd.Timestamp) -> tuple[Event, ...]:
    """Releases scheduled for one calendar day, earliest first."""
    day = pd.Timestamp(date).normalize()
    stamp = day.strftime("%Y-%m-%d")
    found = [
        Event(schedule.name, day, _utc_hours(day, schedule.time_et))
        for schedule in SCHEDULES
        if day <= schedule.end and stamp in schedule.dates
    ]
    return tuple(sorted(found, key=lambda event: event.time_utc))


def upcoming(date: pd.Timestamp, days: int) -> tuple[Event, ...]:
    """Releases scheduled over ``days`` calendar days from ``date`` inclusive.

    Day by day rather than by filtering the tables directly, so a schedule that
    stops before the end of the window contributes nothing past its own end
    date instead of being read as having no release there — the same
    distinction ``unmaintained_on`` exists to report.
    """
    first = pd.Timestamp(date).normalize()
    if days < 1:
        raise ValueError(f"a window of {days} days covers nothing")
    found = [
        event for offset in range(days) for event in events_on(first + pd.Timedelta(days=offset))
    ]
    return tuple(sorted(found, key=lambda event: (event.date, event.time_utc)))


def unmaintained_on(date: pd.Timestamp) -> tuple[str, ...]:
    """Series whose published calendar does not reach ``date``.

    Reported so that a day with no warning can be distinguished from a day
    nobody has checked. Every name here needs its table refreshed from the
    agency's page before a caveat on that date means anything.
    """
    day = pd.Timestamp(date).normalize()
    return tuple(s.name for s in SCHEDULES if day > s.end)


def caveats(target: Market, session: pd.Timestamp) -> tuple[str, ...]:
    """Plain-language warnings about one market's call for one session.

    A release in the window between the freshest bar the features may use and
    the opening auction is the damaging case: the auction prices it and the
    model cannot, so the probability describes a world that no longer exists by
    the time the bell rings. A release after the open leaves the call itself
    intact and ruins everything that follows it, which is worth saying out loud
    given how readily an opening-gap probability is read as a view on the day.

    A release *before* that window earns no warning: some price the model reads
    was struck after it, so the reaction is in the features. The distinction
    matters for Sydney, which opens at 23:00 UTC the previous evening and
    therefore reads Wall Street's close from after that afternoon's FOMC
    statement, not before it.
    """
    notes: list[str] = []
    day = pd.Timestamp(session).normalize()
    observed = last_observed_utc(target.open_utc)
    scheduled = [(e, e.time_utc) for e in events_on(day)]
    if target.open_utc < 0:
        # Sydney's session opens on the previous calendar day, so that day's
        # releases fall around its auction rather than long after it.
        scheduled += [(e, e.time_utc - 24.0) for e in events_on(day - pd.Timedelta(days=1))]
    for event, relative in sorted(scheduled, key=lambda pair: pair[1]):
        if relative <= observed:
            continue
        hours, minutes = divmod(round(event.time_utc * 60), 60)
        clock = f"{hours:02d}:{minutes:02d} UTC"
        if relative != event.time_utc:
            clock += " the previous day"
        if relative <= target.open_utc:
            notes.append(
                f"{event.name} at {clock}, before this open: the auction prices it "
                "and the model cannot — treat the probability as stale"
            )
        else:
            notes.append(
                f"{event.name} at {clock}, after this open: the call still stands "
                "for the auction but says nothing about the session"
            )
    return tuple(notes)
