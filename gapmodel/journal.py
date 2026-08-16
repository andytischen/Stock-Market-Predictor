"""A forecast journal: what was predicted, what happened, and what that is worth.

The walk-forward backtest measures the model against history it has already
seen the shape of. This module measures it against sessions that had not
happened when the probability was written down, which is the only number a
reader of a live forecast can act on.

The mechanics are deliberately dull. Every run appends the forecasts it made to
a CSV, one row per market and session, and then settles the rows whose opening
auction has since printed. A row is written once: a session already in the
journal is never re-forecast and never overwritten, so a probability cannot be
quietly improved after the fact, and a run that is repeated twice in a morning
does not get two attempts at the same open.

Scoring follows the label the model is fitted on -- an opening print above the
previous close, with the gaps that merely repeat the previous close left
unlabelled rather than counted as flat. A session the market never held (a
holiday the journal did not know about) is retired the same way: unscorable, and
visible as such, because silently dropping either would flatter the hit rate.

One class of row is refused a score before it is even settled. The session a
model forecasts is the one after the last session it has *complete* features
for, so a market missing an indicator for yesterday is forecast for an auction
that has already printed. Nothing is leaked -- the features are lagged either
way -- but it is not a forecast anybody could have acted on, so it is journalled
as ``late`` and left out of the live record instead of scored beside the rest.

Skill is reported against each market's own realised drift over the same
sessions, not against a coin flip. Predicting "up" every morning in a market
that opens up 54% of the time is not skill, and a Brier score has to clear that
constant forecast before it says anything. A market whose live record fails to
is called out by ``decayed`` -- which is the point of keeping the journal at
all, and needs enough settled sessions behind it to mean anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import STALE_GAP_TOLERANCE, dividend_adjusted
from .predict import Forecast
from .stocks import is_stock, target_market

log = logging.getLogger(__name__)

DEFAULT_LOG = Path("docs") / "forecast-log.csv"
# Settled sessions per market to read the live record over. A quarter of
# trading is short enough to notice a regime change and long enough that a
# hit rate is not one bad week.
DEFAULT_WINDOW = 60
# Below this many settled sessions no live metric is reported: the sampling
# error on a hit rate over a handful of opens is wider than any decay worth
# alerting on.
MIN_SETTLED = 20

PENDING = "pending"
SETTLED = "settled"
STALE = "stale"
NO_SESSION = "no-session"
LATE = "late"
UNSCORABLE = (STALE, NO_SESSION, LATE)

COLUMNS = (
    "recorded",
    "session",
    "symbol",
    "market",
    "region",
    "p_open_up",
    "oos_base_rate",
    "oos_brier_skill",
    "prev_close",
    "open",
    "gap",
    "outcome",
    "status",
)


def empty_log() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def read_log(path: Path) -> pd.DataFrame:
    """The journal at ``path``, or an empty one when it does not exist yet.

    A journal missing columns added after it was first written is widened
    rather than refused, so an old file keeps its history.
    """
    if not path.exists():
        return empty_log()
    frame = pd.read_csv(path, dtype={"session": str, "recorded": str, "symbol": str})
    for column in COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[list(COLUMNS)]


def write_log(log_frame: pd.DataFrame, path: Path) -> None:
    """Write the journal sorted by session then market, so diffs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = log_frame.sort_values(["session", "symbol"], kind="stable")
    # Enough digits that the stored prices still reproduce the stored gap: a
    # five-digit index level keeps its cents, which %.6g rounds away.
    ordered.to_csv(path, index=False, float_format="%.10g")


def _gap_symbol(symbol: str) -> str:
    """The symbol a market's opening gap is labelled on, or the market itself."""
    try:
        return target_market(symbol).gap_symbol
    except (KeyError, ValueError):
        return symbol


def opening_bars(panel: dict[str, pd.DataFrame] | None, symbol: str) -> pd.DataFrame | None:
    """The bars a market's opening auction is read from, as the model reads them.

    Yahoo repeats the previous close as the open for a few indices, so the model
    labels those markets on a liquid tracker listed on the same exchange instead
    (``Market.gap_symbol``). Settling against the index itself would grade the
    forecast on the very price the project rejected -- every session would come
    back ``stale`` -- so the journal follows the same symbol, and drops the
    unusable bars the way ``features`` does.
    """
    if not panel:
        return None
    bars = panel.get(_gap_symbol(symbol))
    if bars is None:
        return None
    bars = bars.dropna(subset=["Open", "Close"])
    return dividend_adjusted(bars) if is_stock(symbol) else bars


def _already_printed(panel: dict[str, pd.DataFrame] | None, symbol: str, session: str) -> bool:
    """Whether the source already holds a bar for the session being forecast.

    Read from the raw bars rather than the ones ``opening_bars`` keeps. The
    session a model forecasts is the one after the last bar it has *complete*
    features for, so asking the filtered bars whether that session exists can
    only ever answer no. The row that matters is exactly the one the filter
    drops: a session whose auction has printed but whose close has not, which
    pushes the forecast onto a morning that has already happened.

    An opening print is what makes a session past, not a row in the file. This
    source also publishes a bar for a session before its auction, so a row whose
    ``Open`` is still missing is a morning yet to come and stays a real forecast.
    An ``Open`` that merely repeats the previous close is the same non-event:
    ``_settle_row`` refuses to score it as an auction, and this refuses to read
    it as one, so the journal cannot retire a forecast on a price the rest of the
    project rejects. Being wrong here is expensive in one direction only --
    ``late`` is terminal, since ``settle`` revisits pending rows alone -- so an
    unconvincing print leaves the forecast pending, to be settled or retired once
    the session really is in the file.
    """
    if not panel:
        return False
    bars = panel.get(_gap_symbol(symbol))
    if bars is None or bars.empty:
        return False
    stamp = pd.Timestamp(session)
    printed = bars.dropna(subset=["Open"])
    if stamp not in printed.index:
        return False
    opening = float(printed.loc[[stamp]]["Open"].iloc[-1])
    earlier = bars.loc[bars.index < stamp, "Close"].dropna()
    if earlier.empty:
        # Nothing to compare the print against, so take it at face value.
        return True
    previous = float(earlier.iloc[-1])
    if not (opening > 0 and previous > 0):
        return False
    return abs(float(np.log(opening / previous))) > STALE_GAP_TOLERANCE


def record(
    log_frame: pd.DataFrame,
    forecasts: list[Forecast],
    panel: dict[str, pd.DataFrame] | None = None,
    recorded: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Append the forecasts for sessions the journal has not seen before.

    Returns the journal and the symbols that were added; a market whose session
    is already recorded is left exactly as it was. A forecast for a session that
    has already printed in ``panel`` is kept as ``late`` and never scored.
    """
    stamp = (recorded or pd.Timestamp.now("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    known = set(zip(log_frame["symbol"], log_frame["session"], strict=True))
    rows: list[dict[str, object]] = []
    for forecast in forecasts:
        session = forecast.session.date().isoformat()
        if (forecast.symbol, session) in known:
            log.info(
                "%s already forecast for %s: keeping the first entry", forecast.symbol, session
            )
            continue
        rows.append(
            {
                "recorded": stamp,
                "session": session,
                "symbol": forecast.symbol,
                "market": forecast.name,
                "region": forecast.region,
                "p_open_up": round(forecast.probability_up, 4),
                "oos_base_rate": round(forecast.backtest.get("base_rate", float("nan")), 4),
                "oos_brier_skill": round(forecast.backtest.get("brier_skill", float("nan")), 4),
                "prev_close": np.nan,
                "open": np.nan,
                "gap": np.nan,
                "outcome": np.nan,
                "status": LATE if _already_printed(panel, forecast.symbol, session) else PENDING,
            }
        )
    if not rows:
        return log_frame, []
    added = pd.DataFrame(rows, columns=list(COLUMNS))
    return pd.concat([log_frame, added], ignore_index=True), [str(r["symbol"]) for r in rows]


def _settle_row(bars: pd.DataFrame, session: pd.Timestamp) -> dict[str, object] | None:
    """Outcome of one recorded session, or None while it is still in the future.

    The previous close is the last bar before the session, which is the same
    step the model's label is built from. A session absent from a panel that has
    already moved past it was never held, and is retired unscorable rather than
    waited on forever.
    """
    if bars.empty or bars.index.max() < session:
        return None
    if session not in bars.index:
        return {"status": NO_SESSION}
    earlier = bars.loc[bars.index < session]
    if earlier.empty:
        return {"status": NO_SESSION}
    opening = float(bars.loc[session, "Open"])
    previous = float(earlier["Close"].iloc[-1])
    if not (opening > 0 and previous > 0):
        return {"status": NO_SESSION}
    gap = float(np.log(opening / previous))
    if abs(gap) <= STALE_GAP_TOLERANCE:
        # The source repeated the previous close instead of publishing an
        # auction print: the same sessions the model refuses to be fitted on.
        return {"status": STALE, "prev_close": previous, "open": opening, "gap": gap}
    return {
        "status": SETTLED,
        "prev_close": previous,
        "open": opening,
        "gap": round(gap, 6),
        "outcome": float(gap > 0),
    }


def settle(
    log_frame: pd.DataFrame, panel: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, int, int]:
    """Fill in the realised open for every pending row whose session has printed.

    Returns the journal, the rows that were scored, and the rows whose session
    turned out not to be scorable -- counted apart, because a morning that
    retires four holidays has not scored four sessions.
    """
    updated = log_frame.copy()
    filled = 0
    retired = 0
    for index in updated.index[updated["status"] == PENDING]:
        symbol = str(updated.at[index, "symbol"])
        bars = opening_bars(panel, symbol)
        if bars is None:
            log.warning("%s is not in the panel: leaving its rows pending", symbol)
            continue
        outcome = _settle_row(bars, pd.Timestamp(updated.at[index, "session"]))
        if outcome is None:
            continue
        for column, value in outcome.items():
            updated.at[index, column] = value
        if outcome["status"] == SETTLED:
            filled += 1
        else:
            retired += 1
    return updated, filled, retired


@dataclass(frozen=True)
class Skill:
    """One market's live record over the settled sessions in the journal."""

    symbol: str
    market: str
    settled: int
    hit_rate: float
    base_rate: float
    brier: float
    brier_skill: float
    mean_probability: float
    first: str
    last: str

    @property
    def drift_rate(self) -> float:
        """Accuracy of always calling the side the market opened most often.

        Whichever way the drift leans: a market that opened up on 30% of the
        settled sessions is called correctly 70% of the time by saying "down"
        every morning, so the up-rate alone would be a bar the model clears
        without knowing anything.
        """
        return max(self.base_rate, 1.0 - self.base_rate)

    @property
    def decayed(self) -> bool:
        """Live record no better than always predicting the market's own drift.

        A market that opened the same way on every settled session is judged on
        direction alone: its drift is a perfect 100% by construction, so no
        forecast can clear it and there is no variance for a Brier score to
        explain, but a model calling the wrong side of a one-way market is still
        a model that has stopped reading it. The bar there is a coin flip, and
        deliberately the stricter of the two comparisons: a market read no
        better than a coin is not being read at all, while a record that merely
        equals a lopsided (not total) drift is left alone.
        """
        if not np.isfinite(self.brier_skill):
            return self.hit_rate <= 0.5
        return self.brier_skill <= 0.0 or self.hit_rate < self.drift_rate


def _skill(symbol: str, rows: pd.DataFrame) -> Skill:
    probabilities = rows["p_open_up"].astype(float).to_numpy()
    outcomes = rows["outcome"].astype(float).to_numpy()
    base_rate = float(outcomes.mean())
    brier = float(np.mean((probabilities - outcomes) ** 2))
    reference = float(np.mean((base_rate - outcomes) ** 2))
    return Skill(
        symbol=symbol,
        market=str(rows["market"].iloc[-1]),
        settled=len(rows),
        hit_rate=float(np.mean((probabilities > 0.5) == (outcomes > 0.5))),
        base_rate=base_rate,
        brier=brier,
        # A market that opened up (or down) on every settled session has no
        # variance for a forecast to explain, so no skill can be measured.
        brier_skill=float("nan") if reference == 0 else 1.0 - brier / reference,
        mean_probability=float(probabilities.mean()),
        first=str(rows["session"].iloc[0]),
        last=str(rows["session"].iloc[-1]),
    )


def resolved_minimum(window: int, min_settled: int | None = None) -> int:
    """The minimum a window that short can actually hold.

    A caller who narrows the window without naming a minimum wants a shorter
    read, not an error about a default it never passed, so the default is capped
    at the window. A minimum that *was* named and cannot fit is a contradiction
    and is refused by ``skills``.
    """
    return min(MIN_SETTLED, window) if min_settled is None else min_settled


def skills(
    log_frame: pd.DataFrame, window: int = DEFAULT_WINDOW, min_settled: int | None = None
) -> list[Skill]:
    """Live record per market over its most recent ``window`` settled sessions.

    Markets with fewer than ``min_settled`` settled sessions are left out
    entirely rather than reported with a number nobody should read; left unsaid,
    the minimum is whatever the window can hold (see ``resolved_minimum``).

    An explicit ``min_settled`` above ``window`` is refused rather than honoured:
    it asks for more sessions than the window can hold, so every market would be
    dropped however long its history, and the report would blame missing history
    for a threshold that cannot be met.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    min_settled = resolved_minimum(window, min_settled)
    if min_settled > window:
        raise ValueError(
            f"min_settled ({min_settled}) cannot exceed window ({window}): "
            "no market could ever be reported"
        )
    settled_rows = log_frame.loc[log_frame["status"] == SETTLED]
    if settled_rows.empty:
        return []
    measured: list[Skill] = []
    for symbol, rows in settled_rows.groupby("symbol", sort=False):
        recent = rows.sort_values("session", kind="stable").tail(window)
        if len(recent) < min_settled:
            continue
        measured.append(_skill(str(symbol), recent))
    measured.sort(key=lambda s: (np.isnan(s.brier_skill), -s.brier_skill))
    return measured


def decayed(measured: list[Skill]) -> list[Skill]:
    """The markets whose live record has fallen to or below their own drift."""
    return [skill for skill in measured if skill.decayed]


def to_frame(measured: list[Skill]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": [s.market for s in measured],
            "symbol": [s.symbol for s in measured],
            "settled": [s.settled for s in measured],
            "hit_rate": [round(s.hit_rate, 4) for s in measured],
            "base_rate": [round(s.base_rate, 4) for s in measured],
            "brier": [round(s.brier, 4) for s in measured],
            "brier_skill": [round(s.brier_skill, 4) for s in measured],
            "mean_p": [round(s.mean_probability, 4) for s in measured],
            "from": [s.first for s in measured],
            "to": [s.last for s in measured],
        }
    )


def render_text(
    log_frame: pd.DataFrame,
    measured: list[Skill],
    window: int,
    min_settled: int | None = None,
) -> str:
    """The journal's state and the live record, as the CLI prints it."""
    min_settled = resolved_minimum(window, min_settled)
    counts = log_frame["status"].value_counts()
    unscorable = sum(int(counts.get(status, 0)) for status in UNSCORABLE)
    lines = [
        f"forecast journal: {len(log_frame)} rows  "
        f"settled {int(counts.get(SETTLED, 0))}  pending {int(counts.get(PENDING, 0))}  "
        f"unscorable {unscorable}",
        "",
    ]
    if not measured:
        lines.append(
            f"no market has {min_settled} settled sessions yet: live skill is not reported."
        )
        return "\n".join(lines)
    lines.append(f"live record over the last {window} settled sessions per market:")
    lines.append(to_frame(measured).to_string(index=False))
    losing = decayed(measured)
    if losing:
        lines.append("")
        lines.append("below their own drift — the model is not adding a read here:")
        for skill in losing:
            # No Brier skill is quoted for a one-way market: there is no
            # variance to explain, so the number would be nan.
            skill_text = (
                f", Brier skill {skill.brier_skill:+.3f}" if np.isfinite(skill.brier_skill) else ""
            )
            lines.append(
                f"  {skill.market} ({skill.symbol}): hit {skill.hit_rate:.0%} against a "
                f"{skill.drift_rate:.0%} drift{skill_text} over {skill.settled} sessions"
            )
    return "\n".join(lines)
