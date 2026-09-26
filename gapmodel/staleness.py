"""How far behind the panel is, and when that should stop a run.

Every forecast here is built by ``features.as_of``, which forward-fills: a
series that stopped updating is read as one that did not move. That is the right
behaviour for a holiday and the wrong behaviour for a broken feed, and the two
are indistinguishable from inside the model. The difference is only visible in
how long the silence has lasted, which is what this module measures.

The reference is the session being asked about — the one being forecast, or
today for a guard running before anything has been fitted — and deliberately not
the freshest bar in the panel. Panels are mixed: a US series that has not opened yet today is
current while ending yesterday, and an Asian series downloaded mid-session
carries a partial bar for today. Anchoring on the maximum called all of Wall
Street stale because Seoul was open, and made the count swing on which symbols a
given run happened to load rather than on anything about the data.

Two anchors follow from that, and they are not the same date. A guard asks
whether data is too old to act on, which is a question about now, so it measures
against ``today()``. A report's footer annotates a forecast already produced for
a stated session, so it measures against that session — a footer anchored on
today would call a series stale relative to a date the report never mentions.
The two coincide whenever the run is current, which is what the guard enforces,
and diverge only under ``--allow-stale`` or a widened tolerance.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd

log = logging.getLogger(__name__)

# The largest lag, in calendar days, a series may carry and still be read as
# current: the comparison is ``lag > STALE_DAYS``, so five days is tolerated and
# six is not. A series that traded on the previous session is one day behind, a
# long weekend stretches that to three, and a holiday on either side of one to
# five; six is the first lag the calendar cannot explain.
STALE_DAYS = 5


class StaleInputs(RuntimeError):
    """Raised when the panel is too far behind to answer the question asked."""


# Said in full wherever a run is refused: a reader who hits the wall should not
# have to look up which flag widens it.
REMEDIES = (
    "Their last value would be forward-filled, so the forecast would read older "
    "cross-market data as though nothing had moved. Re-run with --refresh to update the "
    "cache, --max-stale-days to widen the tolerance, or --allow-stale to forecast anyway."
)


def today() -> pd.Timestamp:
    """The reference a guard measures against, before any session is known.

    Not the next session after the panel's last bar, which is what a forecast is
    dated: a feed that died last week would date its own forecast to the day
    after it died and so look perfectly current to itself. Whether data is too
    old to act on is a question about now.
    """
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def lags(panel: dict[str, pd.DataFrame], session: pd.Timestamp) -> dict[str, int]:
    """Calendar days each non-empty series in ``panel`` sits behind ``session``."""
    reference = session.normalize()
    return {
        symbol: int((reference - bars.index.max().normalize()).days)
        for symbol, bars in panel.items()
        if not bars.empty
    }


def behind(measured: dict[str, int], max_days: int = STALE_DAYS) -> list[str]:
    """The measured series lagging by more than ``max_days``, worst first.

    Worst first, so the eight names a report has room for are the eight that
    matter rather than whichever eight sort first alphabetically.
    """
    stale = [(lag, symbol) for symbol, lag in measured.items() if lag > max_days]
    return [symbol for _, symbol in sorted(stale, key=lambda e: (-e[0], e[1]))]


def stale_inputs(
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    max_days: int = STALE_DAYS,
) -> tuple[int, list[str]]:
    """The series counted, and those lagging ``session`` by more than ``max_days``."""
    measured = lags(panel, session)
    return len(measured), behind(measured, max_days)


def _at_most_eight(described: Sequence[str]) -> str:
    """The first eight of ``described``, with the rest counted rather than listed."""
    return ", ".join(described[:8]) + (
        f" and {len(described) - 8} more" if len(described) > 8 else ""
    )


def describe(measured: dict[str, int], stale: Sequence[str]) -> str:
    """The stale series named with their lags, worst first, for an error or a log."""
    return _at_most_eight([f"{symbol} ({measured[symbol]}d)" for symbol in stale])


def missing(panel: dict[str, pd.DataFrame]) -> list[str]:
    """The series that arrived with no bars at all, and so have no lag to measure.

    They are counted nowhere else: ``lags`` skips them, because a series with no
    last bar cannot be a number of days behind one, and "0 days stale" would be a
    worse answer than no answer. Naming them separately is the alternative — a
    guard that says "2 of 3 input series" reads as reassurance about the third,
    when the third may never have arrived.

    Sorted, because there is nothing to rank them by: the stale list is truncated
    worst-first, while panel order would decide which eight of these get named on
    the strength of download order alone, differently from one run to the next.
    """
    return sorted(symbol for symbol, bars in panel.items() if bars.empty)


def _say_what_never_arrived(panel: dict[str, pd.DataFrame]) -> None:
    """Name the series that arrived empty, which no staleness count covers.

    Not a refusal: a download that returned nothing is a different failure with
    a different remedy, and it is reported where it happens. Said here only so
    that a count of the series that did arrive is not read as covering it — and
    said without a denominator of its own, since a second "N of M" beside a
    smaller total reads as the two lines disagreeing.
    """
    absent = missing(panel)
    if absent:
        log.warning(
            "%d input series arrived with no bars at all, so they are neither counted "
            "nor judged for staleness: %s",
            len(absent),
            _at_most_eight(absent),
        )


def guard(
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    max_days: int = STALE_DAYS,
    allow: bool = False,
) -> None:
    """Refuse to forecast ``session`` from inputs older than ``max_days``.

    Refusing rather than dropping the stale columns: the model is fitted over a
    history in which those columns were live, so removing them at inference time
    would answer a different question from the one the backtest metrics describe,
    and would do it silently. A run that cannot be trusted should not print a
    number that looks exactly like one that can.

    ``allow`` keeps the old behaviour available for the case where reading last
    week's macro is the deliberate intent, and says so on the log rather than
    passing quietly. It goes to the log and not to stdout because ``export``
    writes its snapshot there: a warning printed alongside it would be read by
    the next program in the pipe as the first line of the JSON.
    """
    _say_what_never_arrived(panel)
    measured = lags(panel, session)
    stale = behind(measured, max_days)
    if not stale:
        return
    detail = (
        f"{len(stale)} of {len(measured)} input series have no bar within {max_days} days of "
        f"{session.date().isoformat()}: {describe(measured, stale)}"
    )
    if allow:
        log.warning("%s (--allow-stale)", detail)
        return
    raise StaleInputs(f"{detail}. {REMEDIES}")


def fresh_forecasts(
    inputs: Mapping[str, dict[str, pd.DataFrame]],
    session: pd.Timestamp,
    max_days: int = STALE_DAYS,
    allow: bool = False,
) -> list[str]:
    """The requested forecasts none of whose own inputs has gone quiet.

    ``inputs`` maps each name asked for to the series its model reads, its own
    history included. A stale series therefore costs exactly the forecasts that
    read it: a halted listing loses its own row, and takes with it the models
    that hold it as a peer — in a default ``stock`` run MU is a column in WDC's
    model, so both go, while AAPL, which never opens it, is still forecast. One
    quiet feed cancelling every unrelated name is the same failure refusing to
    forecast from it exists to prevent, in the other direction.

    What is *not* done is dropping the stale column and fitting the model
    without it: that model's AUC and Brier skill were earned over a history in
    which the column was live, so the number it printed would answer a different
    question from the one its metrics describe.
    """
    union: dict[str, pd.DataFrame] = {}
    for read in inputs.values():
        union.update(read)
    blocked = {
        target: stale
        for target, read in inputs.items()
        if (stale := behind(lags(read, session), max_days))
    }
    kept = [target for target in inputs if target not in blocked]
    if not blocked or allow:
        # Nothing lost, or the loss deliberately accepted: either way the run
        # stands as one, which is what ``guard`` says — quietly, or warning
        # under ``--allow-stale``.
        guard(union, session, max_days, allow=allow)
        return list(inputs)
    _say_what_never_arrived(union)
    measured = lags(union, session)
    stale = describe(measured, behind(measured, max_days))
    if not kept:
        # Refused rather than printed as an empty table: a command that skips
        # every name it was asked for has not answered the question, and saying
        # so as a skip would leave the reader to infer it from a blank report.
        raise StaleInputs(
            f"every requested forecast reads a series with no bar within {max_days} days "
            f"of {session.date().isoformat()}: {stale}. {REMEDIES}"
        )
    log.warning(
        "skipping %d of %d requested forecasts (%s): they read a series with no bar "
        "within %d days of %s \u2014 %s",
        len(blocked),
        len(inputs),
        _at_most_eight(sorted(blocked)),
        max_days,
        session.date().isoformat(),
        stale,
    )
    return kept
