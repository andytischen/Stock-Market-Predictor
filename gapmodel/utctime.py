"""UTC time-of-day helpers shared by the CLI and the browser interface."""

from __future__ import annotations

import re

import pandas as pd

# A single-digit hour is accepted because that is what a hand-typed `--at 5:00`
# looks like, and seconds because that is what a copied timestamp carries. The
# shape is still a time of day and nothing else: a date parses as no hour at all
# rather than as midnight, which is the reading that made a stale board look
# like a fresh one.
_TIME = re.compile(r"^(\d{1,2}):([0-5]\d)(?::([0-5]\d))?$")


def parse_utc_time(value: str) -> float:
    """Return ``H:MM``, ``HH:MM`` or ``HH:MM:SS`` (UTC) as hours from midnight."""
    match = _TIME.match(value.strip())
    if match is None:
        raise ValueError(f"{value!r} is not a time of day (use HH:MM)")
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if hours > 23:
        raise ValueError(f"{value!r} is not a time of day (use HH:MM)")
    return hours + minutes / 60 + seconds / 3600


def format_utc_time(hours: float) -> str:
    """``hours`` from midnight as ``HH:MM``, rounded to the nearest minute.

    Rounded on the total and not per field: rounding the minutes alone turns
    ``13:59:40`` into ``13:60``, which no time input will accept.
    """
    total = min(round(hours * 60), 24 * 60 - 1)
    return f"{total // 60:02d}:{total % 60:02d}"


def as_of(hours: float | None) -> pd.Timestamp | None:
    """Today's date at the given UTC hour, or now when no hour is given."""
    if hours is None:
        return None
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return now.normalize() + pd.Timedelta(hours=hours)
