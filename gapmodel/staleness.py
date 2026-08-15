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

import logging
from collections.abc import Sequence

import pandas as pd

log = logging.getLogger(__name__)

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


def describe(measured: dict[str, int], stale: Sequence[str]) -> str:
    """The stale series named with their lags, worst first, for an error or a log."""
    named = ", ".join(f"{symbol} ({measured[symbol]}d)" for symbol in stale[:8])
    return named + (f" and {len(stale) - 8} more" if len(stale) > 8 else "")


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
    raise StaleInputs(
        f"{detail}. Their last value would be forward-filled, so the forecast would "
        "read older cross-market data as though nothing had moved. Re-run with "
        "--refresh to update the cache, --max-stale-days to widen the tolerance, or "
        "--allow-stale to forecast anyway."
    )


def fresh_targets(
    panel: dict[str, pd.DataFrame],
    symbols: Sequence[str],
    session: pd.Timestamp,
    max_days: int = STALE_DAYS,
    allow: bool = False,
) -> list[str]:
    """The requested names whose own history is current enough to forecast.

    A stale feature is everyone's problem, because every model in the run reads
    it; a stale target is only its own. One name that stopped trading — halted,
    acquired, delisted since the universe file was written — should not decide
    whether the other sixty-five get forecast, so it is dropped by name and the
    run continues. Losing the whole ranking to it would be the same failure the
    guard exists to prevent, in the other direction.
    """
    measured = lags({s: panel[s] for s in symbols if s in panel}, session)
    stale = behind(measured, max_days)
    if not stale:
        return list(symbols)
    if allow:
        # Said even here, and for the same reason the guard says it: ``stock``
        # prints no staleness footer, so this is the only place a reader learns
        # that the name in front of them stopped trading weeks ago.
        log.warning(
            "forecasting %d of %d requested names whose own history stops more than %d "
            "days before %s (--allow-stale): %s",
            len(stale),
            len(symbols),
            max_days,
            session.date().isoformat(),
            describe(measured, stale),
        )
        return list(symbols)
    kept = [symbol for symbol in symbols if symbol not in set(stale)]
    if kept:
        # Only when something is left to skip *to*: announcing a skip and then
        # aborting the run would describe two different outcomes in two lines.
        log.warning(
            "skipping %d of %d requested names whose own history stops more than %d days "
            "before %s: %s",
            len(stale),
            len(symbols),
            max_days,
            session.date().isoformat(),
            describe(measured, stale),
        )
    else:
        raise StaleInputs(
            f"every requested name has no bar within {max_days} days of "
            f"{session.date().isoformat()}: {describe(measured, stale)}. Re-run with "
            "--refresh to update the cache, --max-stale-days to widen the tolerance, or "
            "--allow-stale to forecast anyway."
        )
    return kept
