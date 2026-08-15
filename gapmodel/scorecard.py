"""What the model called recently, and what the opening auction actually did.

``backtest`` reports the walk-forward record over twenty years, which is the
right measure of a model and the wrong measure of this week. A model can hold a
0.85 AUC across two decades while its last month is worse than reading the base
rate off the wall, and nothing in a full-sample table would say so: a hundred
recent sessions move the fourth decimal of five thousand.

This module reads the same walk-forward predictions the backtest scores and
keeps only the tail of them, so the recent record is stated on its own terms:
what was called for each of the last sessions, what the auction did, and the
accuracy and calibration of that window against the full sample. Nothing here
refits or re-predicts, and no session is scored by a model that had seen it —
the predictions are the walk-forward's, which is what makes a recent window an
honest out-of-sample record rather than a fit to the last month.

The realised *gap size* is carried beside the binary outcome because a miss is
not one thing. Calling a down open against a gap of two basis points is the
model declining to distinguish noise; the same call against a 1.8% gap up is a
call that was wrong about the session. A window's hit rate cannot tell them
apart, so the sizes are printed and logged alongside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, dividend_adjusted, opening_gap
from .markets import MARKETS
from .model import MIN_TRAIN, Backtest, walk_forward
from .stocks import is_stock, target_market

log = logging.getLogger(__name__)

# Sessions in the recent window. A trading month: long enough for a hit rate to
# carry any information at all, short enough to still be about the present.
RECENT_WINDOW = 21

# Sessions the window needs before its calibration is allowed to raise a drift
# flag. Brier skill over a handful of sessions swings on single outcomes, and a
# flag that fires on noise is one a reader learns to ignore.
DRIFT_MIN_SESSIONS = 10


@dataclass(frozen=True)
class Call:
    """One walk-forward prediction, scored against the auction it forecast."""

    symbol: str
    session: pd.Timestamp
    probability: float
    gap: float
    outcome: int

    @property
    def direction(self) -> str:
        return "up" if self.probability >= 0.5 else "down"

    @property
    def realised(self) -> str:
        return "up" if self.outcome == 1 else "down"

    @property
    def hit(self) -> bool:
        return (self.probability >= 0.5) == (self.outcome == 1)

    @property
    def gap_pct(self) -> float:
        """The realised gap as a percentage move, not a log return."""
        return float(np.expm1(self.gap) * 100.0)

    def as_row(self) -> dict[str, object]:
        return {
            "session": self.session.date().isoformat(),
            "symbol": self.symbol,
            "p_open_up": round(float(self.probability), 4),
            "called": self.direction,
            "realised": self.realised,
            "gap_pct": round(self.gap_pct, 3),
            "hit": self.hit,
        }


@dataclass(frozen=True)
class Record:
    """One market's recent calls, with that window's record and the full one."""

    symbol: str
    name: str
    calls: tuple[Call, ...]
    window: dict[str, float]
    full: dict[str, float]

    @property
    def latest(self) -> Call:
        return self.calls[-1]

    @property
    def hits(self) -> int:
        return sum(1 for call in self.calls if call.hit)

    @property
    def skill_change(self) -> float:
        """Window Brier skill against the full sample's; negative is worse."""
        return float(self.window["brier_skill"] - self.full["brier_skill"])

    @property
    def drifting(self) -> bool:
        """Whether the window is worse calibrated than its own base rate.

        Negative Brier skill is the specific failure worth a flag: it says the
        probabilities would have been improved by replacing every one of them
        with the up-rate of the window. A fall from the full sample is not
        enough on its own — a good month regresses — and neither is a low hit
        rate, which a run of near-flat opens produces on its own.
        """
        return len(self.calls) >= DRIFT_MIN_SESSIONS and self.window["brier_skill"] < 0.0

    def as_row(self) -> dict[str, object]:
        latest = self.latest
        return {
            "symbol": self.symbol,
            "last_session": latest.session.date().isoformat(),
            "p_open_up": round(float(latest.probability), 4),
            "called": latest.direction,
            "realised": latest.realised,
            "gap_pct": round(latest.gap_pct, 3),
            "hit": latest.hit,
            "n": len(self.calls),
            "window_accuracy": round(self.window["accuracy"], 4),
            "window_brier_skill": round(self.window["brier_skill"], 4),
            "full_brier_skill": round(self.full["brier_skill"], 4),
            "skill_change": round(self.skill_change, 4),
            "drifting": self.drifting,
        }


def realised_gaps(symbol: str, panel: dict[str, pd.DataFrame]) -> pd.Series:
    """Opening gaps of the series the target's label is built from.

    The same series ``features`` labels on, including the dividend correction a
    single name needs, so a gap printed here is the one that decided the label
    rather than a second, differently-adjusted reading of the same morning.
    """
    bars = panel[target_market(symbol).gap_symbol].dropna(subset=["Open", "Close"])
    if is_stock(symbol):
        bars = dividend_adjusted(bars)
    return opening_gap(bars)


def score(
    symbol: str,
    panel: dict[str, pd.DataFrame],
    window: int = RECENT_WINDOW,
    c: float = 0.1,
    min_train: int = MIN_TRAIN,
    hourly: dict[str, pd.Series] | None = None,
) -> Record:
    """Walk-forward one market and keep the last ``window`` scored sessions."""
    features, labels = build_features(symbol, panel, hourly=hourly)
    result = walk_forward(features, labels, min_train=min_train, c=c)
    return _record(symbol, result, realised_gaps(symbol, panel), window)


def _record(symbol: str, result: Backtest, gaps: pd.Series, window: int) -> Record:
    if window < 1:
        raise ValueError(f"window must be at least one session, got {window}")
    recent = result.probabilities.iloc[-window:]
    outcomes = result.outcomes.loc[recent.index]
    calls = tuple(
        Call(
            symbol=symbol,
            session=session,
            probability=float(recent.loc[session]),
            gap=float(gaps.get(session, float("nan"))),
            outcome=int(outcomes.loc[session]),
        )
        for session in recent.index
    )
    return Record(
        symbol=symbol,
        name=target_market(symbol).name,
        calls=calls,
        window=result.window_metrics(since=recent.index[0]),
        full=result.metrics,
    )


def build_scorecard(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    window: int = RECENT_WINDOW,
    c: float = 0.1,
    min_train: int = MIN_TRAIN,
    hourly: dict[str, pd.Series] | None = None,
) -> list[Record]:
    """Score every requested market, skipping the ones that cannot be modelled.

    A market whose data is short or unusable is dropped with a warning, as it is
    everywhere else: one refused download should not cost the whole scorecard.
    """
    records: list[Record] = []
    for symbol in dict.fromkeys(symbols or [m.symbol for m in MARKETS]):
        try:
            records.append(
                score(
                    symbol,
                    panel,
                    window=window,
                    c=c,
                    min_train=min_train,
                    hourly=hourly,
                )
            )
        except Exception as exc:
            log.warning("no scorecard for %s: %s", symbol, exc)
    if not records:
        raise RuntimeError("no market could be scored")
    return records


def drifting(records: list[Record]) -> list[Record]:
    """Markets whose recent window is worse calibrated than its base rate."""
    return sorted([r for r in records if r.drifting], key=lambda r: r.window["brier_skill"])


def to_frame(records: list[Record]) -> pd.DataFrame:
    """One row per market: its latest scored call and the window's record."""
    return pd.DataFrame([r.as_row() for r in records])


def calls_frame(records: list[Record]) -> pd.DataFrame:
    """One row per scored session per market, oldest first."""
    rows = [call.as_row() for record in records for call in record.calls]
    return pd.DataFrame(rows).sort_values(["session", "symbol"]).reset_index(drop=True)


def append_log(records: list[Record], path: str | Path) -> pd.DataFrame:
    """Merge the scored calls into a CSV log, keeping one row per session.

    Idempotent on purpose: the same day scored twice, or a window overlapping
    everything already logged, must leave one row per market and session rather
    than a file that double-counts whichever days a schedule happened to repeat.
    A re-scored session replaces its old row, since the later run read the more
    complete data.
    """
    fresh = calls_frame(records)
    target = Path(path)
    if target.exists():
        previous = pd.read_csv(target)
        fresh = pd.concat([previous, fresh], ignore_index=True)
    merged = (
        fresh.drop_duplicates(subset=["session", "symbol"], keep="last")
        .sort_values(["session", "symbol"])
        .reset_index(drop=True)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False)
    return merged


def render_text(records: list[Record], window: int = RECENT_WINDOW) -> str:
    """The scorecard as a report: the window, then what it does not establish."""
    frame = to_frame(records).sort_values("window_brier_skill", ascending=False)
    sessions = sorted({r.latest.session.date().isoformat() for r in records})
    lines = [
        f"Recent record over the last {window} scored sessions (latest: {', '.join(sessions)})",
        "",
        frame.drop(columns=["drifting"]).to_string(index=False),
    ]
    flagged = drifting(records)
    if flagged:
        lines.append("")
        lines.append("worse calibrated than the base rate over this window:")
        for record in flagged:
            lines.append(
                f"  {record.name} ({record.symbol}): Brier skill "
                f"{record.window['brier_skill']:+.3f} against "
                f"{record.full['brier_skill']:+.3f} over the full sample, "
                f"{record.hits} of {len(record.calls)} calls right"
            )
    lines.append("")
    lines.append(
        f"Every probability above is the walk-forward's own, made by a model "
        f"fitted only on sessions before it, so the window is out of sample. It "
        f"is also short: {window} sessions put a hit rate's standard error near "
        f"{50.0 / np.sqrt(window):.0f} points, so a single window's fall is "
        "weak evidence of decay and a run of them is the thing to read."
    )
    return "\n".join(lines) + "\n"
