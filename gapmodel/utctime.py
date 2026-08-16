"""UTC time-of-day helpers shared by the CLI and the browser interface."""

from __future__ import annotations

import re

import pandas as pd

_HH_MM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def parse_utc_time(value: str) -> float:
    """Return ``HH:MM`` (UTC) as hours from midnight."""
    match = _HH_MM.match(value.strip())
    if match is None:
        raise ValueError(f"{value!r} is not a time of day (use HH:MM)")
    return int(match.group(1)) + int(match.group(2)) / 60


def as_of(hours: float | None) -> pd.Timestamp | None:
    """Today's date at the given UTC hour, or now when no hour is given."""
    if hours is None:
        return None
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return now.normalize() + pd.Timedelta(hours=hours)
