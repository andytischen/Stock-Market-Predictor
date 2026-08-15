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
"""

from __future__ import annotations

import pandas as pd

# Calendar days a series may sit behind the session before it is called stale.
# A series that traded on the previous session is one day behind, and a long
# weekend or a holiday stretches that to four; five is the first lag the
# calendar cannot explain.
STALE_DAYS = 5


class StaleInputs(RuntimeError):
    """Raised when the panel is too far behind to answer the question asked."""


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


def stale_inputs(
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    max_days: int = STALE_DAYS,
) -> tuple[int, list[str]]:
    """The series counted, and those lagging ``session`` by more than ``max_days``.

    Worst first, so the eight names a report has room for are the eight that
    matter rather than whichever eight sort first alphabetically.
    """
    behind = [(lag, symbol) for symbol, lag in lags(panel, session).items() if lag > max_days]
    counted = len(lags(panel, session))
    return counted, [symbol for _, symbol in sorted(behind, key=lambda e: (-e[0], e[1]))]


def describe(panel: dict[str, pd.DataFrame], session: pd.Timestamp, max_days: int) -> str:
    """The stale series named with their lags, worst first, for an error or a log."""
    measured = lags(panel, session)
    _, behind = stale_inputs(panel, session, max_days)
    named = ", ".join(f"{symbol} ({measured[symbol]}d)" for symbol in behind[:8])
    return named + (f" and {len(behind) - 8} more" if len(behind) > 8 else "")


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
    week's macro is the deliberate intent, and says so in the log rather than
    passing quietly.
    """
    counted, behind = stale_inputs(panel, session, max_days)
    if not behind:
        return
    detail = (
        f"{len(behind)} of {counted} input series have no bar within {max_days} days of "
        f"{session.date().isoformat()}: {describe(panel, session, max_days)}"
    )
    if allow:
        print(f"warning: {detail} (--allow-stale)")
        return
    raise StaleInputs(
        f"{detail}. Their last value would be forward-filled, so the forecast would "
        "read older cross-market data as though nothing had moved. Re-run with "
        "--refresh to update the cache, --max-stale-days to widen the tolerance, or "
        "--allow-stale to forecast anyway."
    )
