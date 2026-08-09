"""Scheduled US releases the model is structurally unable to anticipate.

Every feature in the model is a price. A price already reflects what the market
knows, which makes it a good summary of the past and no guide at all to a number
that has not been published yet: the forecast for a session carrying the
payrolls report or an FOMC decision is built entirely from a world in which that
release has not happened. The probability is not wrong so much as answering a
narrower question than it appears to.

This module says when that is the case, so a call can be reported with the
caveat rather than at face value. Two kinds of release are covered:

* Payrolls, which follow a rule — the first Friday of the month, 13:30 UTC.
  The rule holds all but a handful of times in the BLS record (a first Friday
  falling on New Year's Day moves the release), which is accurate enough for a
  caveat and requires no maintenance.
* FOMC decisions, which follow no rule and are taken from the published
  calendar. The table therefore ends, and past its end no flag is raised: the
  absence of a warning after ``CALENDAR_END`` means nothing was checked, not
  that nothing is scheduled.

Deliberately not covered: CPI and the other BLS and BEA releases, whose dates
are announced rather than derivable, and which would need the same table
refreshed every year. Guessing them would be worse than leaving them out —
a caveat that fires on the wrong day teaches you to ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .markets import Market

# 13:30 UTC in winter, when the New York morning is 14:30 UTC and the release is
# 08:30 ET; the summer offset is the same because both dates shift together.
PAYROLLS_TIME_UTC = 13.5
# The statement lands at 14:00 ET, an hour and a half before the cash close.
FOMC_TIME_UTC = 19.0

# Decision days — the second day of each two-day meeting — from the Federal
# Reserve's published calendar. Extend this as the Board publishes further out.
FOMC_DECISIONS: tuple[str, ...] = (
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
)

# Past this date the FOMC table says nothing, so neither does this module.
CALENDAR_END = pd.Timestamp("2026-12-31")


@dataclass(frozen=True)
class Event:
    """A scheduled release, on the same UTC clock as the market sessions."""

    name: str
    date: pd.Timestamp
    time_utc: float


def _is_payrolls_day(date: pd.Timestamp) -> bool:
    """First Friday of the month."""
    return date.weekday() == 4 and date.day <= 7


def events_on(date: pd.Timestamp) -> tuple[Event, ...]:
    """Releases scheduled for one calendar day, earliest first."""
    day = pd.Timestamp(date).normalize()
    found: list[Event] = []
    if _is_payrolls_day(day):
        found.append(Event("US payrolls", day, PAYROLLS_TIME_UTC))
    if day <= CALENDAR_END and day.strftime("%Y-%m-%d") in FOMC_DECISIONS:
        found.append(Event("FOMC decision", day, FOMC_TIME_UTC))
    return tuple(sorted(found, key=lambda event: event.time_utc))


def caveats(target: Market, session: pd.Timestamp) -> tuple[str, ...]:
    """Plain-language warnings about one market's call for one session.

    A release before the opening auction is the damaging case: the auction
    prices it and the model cannot, so the probability describes a world that no
    longer exists by the time the bell rings. A release after the open leaves
    the call itself intact and ruins everything that follows it, which is worth
    saying out loud given how readily an opening-gap probability is read as a
    view on the day.
    """
    notes: list[str] = []
    day = pd.Timestamp(session).normalize()
    scheduled = [(e, e.time_utc) for e in events_on(day)]
    if target.open_utc < 0:
        # Sydney's session starts on the previous calendar day, so a release
        # that day lands before the auction rather than long after it.
        scheduled += [(e, e.time_utc - 24.0) for e in events_on(day - pd.Timedelta(days=1))]
    for event, relative in sorted(scheduled, key=lambda pair: pair[1]):
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
