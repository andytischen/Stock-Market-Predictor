"""A ranked shortlist of next-open calls across the US universe.

``stock`` forecasts a handful of names in depth: each one carries a hand-written
peer list, the companies whose own sessions price the same end demand. That does
not scale to sixty listings, and it is not what a screen is for. This module
takes the opposite trade: every name in the universe, read with the cross-market
and cross-asset columns a US index reads plus its own history, and then filtered
hard on whether the walk-forward record justifies reading it at all.

Ranking such forecasts needs more care than sorting on the probability, for two
reasons that both flatter a naive list.

A share's opening gap is not a coin flip to begin with. Twenty years of a
compounding growth name leave its unconditional up-rate meaningfully above 50%,
so the stocks that top a probability sort are partly just the stocks with the
strongest drift. ``edge`` therefore reports the probability against that
stock's own base rate, which is what the model actually claims to add.

And a confident probability from a model with no demonstrated skill is noise
with a decimal point. Every pick carries its walk-forward record, and a name is
ranked only if that record clears all three tests in ``credible``: a ranking
ability above the coin flip, calibration that beats the base rate, and enough
out-of-sample sessions for either number to mean anything. All three are needed.
AUC alone would rank a freshly listed name on a couple of hundred sessions,
where 0.72 is within noise of nothing; and AUC is blind to calibration, so a
model that orders sessions well while being confidently miscalibrated — which
is what a negative Brier skill says — would otherwise be presented as a pick.

Two things separate a shortlist from the curated names. Only the registry has
peers, so most names here are read from the tape and their own history alone,
which the metrics beside them price in. And the universe is a snapshot of
today's listings, so fitting it over history is survivorship-biased: the
delisted and the acquired are missing, and a genuinely point-in-time universe
would read worse than this one does.

None of this is a recommendation, and the horizon is worth restating: the
target is the next opening print against the last close, an overnight move,
not a view on the company or on the session that follows the bell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import model as model_mod
from .predict import Forecast, _display, forecast_market
from .staleness import STALE_DAYS, stale_inputs
from .universe import modelled_universe

log = logging.getLogger(__name__)

# Out-of-sample AUC a name must clear before its probability is ranked at all.
# 0.5 is the coin flip; this asks for a margin over it that is small enough to
# keep a real but modest edge and large enough to exclude the noise.
MIN_AUC = 0.55

# Out-of-sample sessions a name needs before its metrics are read as evidence.
# A recent listing can post an AUC of 0.72 on two hundred predictions and mean
# nothing by it, and that is exactly the name a ranking would put first. Set to
# the walk-forward warm-up: a model should be judged on at least as many
# sessions as it was first trained on.
MIN_OOS = model_mod.MIN_TRAIN


@dataclass(frozen=True)
class StockPick:
    """One stock's next-open call, with the quality of the model behind it."""

    forecast: Forecast
    # The move the name has just made, carried so a reader can see whether a
    # call follows through on a rally or leans against it. ``None`` when the
    # bars were not to hand, which prints as an empty cell rather than a zero:
    # "unchanged" and "unknown" are different claims.
    last_change: float | None = None

    @property
    def symbol(self) -> str:
        return self.forecast.symbol

    @property
    def probability_up(self) -> float:
        return self.forecast.probability_up

    @property
    def base_rate(self) -> float:
        return float(self.forecast.backtest.get("base_rate", float("nan")))

    @property
    def auc(self) -> float:
        return float(self.forecast.backtest.get("auc", float("nan")))

    @property
    def edge(self) -> float:
        """Probability against this stock's own unconditional up-rate.

        Positive is the model leaning more bullish than the share's history
        alone would, which is the only part of the number it can take credit
        for.
        """
        return self.probability_up - self.base_rate

    @property
    def brier_skill(self) -> float:
        """Calibration against the base rate; negative is worse than guessing it."""
        return float(self.forecast.backtest.get("brier_skill", 0.0))

    @property
    def n_oos(self) -> int:
        """Out-of-sample sessions the metrics were measured over."""
        return int(self.forecast.backtest.get("n", 0))

    @property
    def credible(self) -> bool:
        """Whether the walk-forward record justifies reading the probability.

        Ranking ability, calibration, and enough sessions to establish either.
        """
        return self.auc >= MIN_AUC and self.brier_skill > 0.0 and self.n_oos >= MIN_OOS

    @property
    def direction(self) -> str:
        return "up" if self.probability_up >= 0.5 else "down"

    def as_row(self) -> dict[str, object]:
        # Rounded through the same guard the index table uses: no forecast here
        # is a certainty, so none of them prints as one. The edge is then
        # rounded from the printed probability rather than the full-precision
        # one, so that the column a reader checks by hand is the difference of
        # the two columns either side of it.
        probability = _display(self.probability_up)
        base_rate = round(self.base_rate, 4)
        return {
            "symbol": self.symbol,
            "session": self.forecast.session.date().isoformat(),
            "last_change": (None if self.last_change is None else round(self.last_change * 100, 2)),
            "p_open_up": probability,
            "base_rate": base_rate,
            "edge": round(probability - base_rate, 4),
            "oos_auc": round(self.auc, 4),
            "oos_accuracy": round(self.forecast.backtest.get("accuracy", float("nan")), 4),
            "oos_brier_skill": round(self.brier_skill, 4),
            "n_oos": self.n_oos,
            # Carried in the table so the CSV says which rows are evidence and
            # which are not: a consumer sorting the file on the raw probability
            # would otherwise walk straight into both traps above.
            "credible": self.credible,
        }


def _rank_key(pick: StockPick) -> float:
    """Edge weighted by demonstrated skill.

    Ranking on the edge alone promotes a name whose backtest says nothing, and
    ranking on AUC alone promotes a skilful model that happens to have no view
    today. The product asks for both: a departure from the stock's own drift,
    from a model that has earned the right to make one.
    """
    return abs(pick.edge) * max(pick.auc - 0.5, 0.0)


def last_change(bars: pd.DataFrame) -> float:
    """Fractional move of the last session in ``bars`` against the one before.

    Read from the raw close, which is the number a quote screen shows, so a name
    that went ex-dividend reads as the market saw it. The model's own label is
    taken from total-return bars instead, deliberately: this is reporting, not a
    feature. A session still in progress carries a partial bar, so mid-session
    this is the move so far rather than a completed one.
    """
    close = bars["Close"].dropna().astype(float)
    if len(close) < 2:
        raise ValueError(f"need two closes, have {len(close)}")
    previous = float(close.iloc[-2])
    if previous <= 0:
        raise ValueError("non-positive previous close")
    return float(close.iloc[-1]) / previous - 1.0


def _last_bar(bars: pd.DataFrame) -> pd.Timestamp:
    """The session a series ends on, to the day."""
    return bars.index.max().normalize()


def _changes(panel: dict[str, pd.DataFrame], symbols: list[str]) -> dict[str, float]:
    """Each name's last session move, skipping those the panel cannot supply."""
    moves: dict[str, float] = {}
    for symbol in dict.fromkeys(symbols):
        bars = panel.get(symbol)
        if bars is None or bars.empty:
            continue
        try:
            moves[symbol] = last_change(bars)
        except (KeyError, ValueError) as exc:
            log.warning("no last move for %s: %s", symbol, exc)
    return moves


def biggest_gainers(panel: dict[str, pd.DataFrame], symbols: list[str], count: int) -> list[str]:
    """The ``count`` names that moved up most in the panel's latest session.

    Selection is the cheap half of the work: bars are downloaded once for the
    whole universe, while each walk-forward fit costs seconds, so narrowing to
    the movers before fitting is what makes a wide universe usable in a briefing.

    Ranked on the descending move and sliced, so on a session where everything
    fell these are the smallest fallers rather than risers — the report names the
    session and says the ranking is on the move, which is true either way.

    Only names whose own last bar *is* that session are eligible. Every listing
    here trades one clock, so a series ending earlier did not trade in the
    session being ranked, and its own last two closes describe some older day: a
    halted or delisted name would otherwise hold its final move for ever and
    take a slot on every run, from the names that actually moved. Cached bars
    make that the normal case, not an exotic one, since a panel is only as
    current as its last ``--refresh``.

    A mover is chosen for having already moved, which is a reason to read its
    call and not evidence about it: yesterday's largest rise is where a gap is
    most likely to be continuation or reversal, and the model's record for that
    name is the only thing that says which. Ties break on the symbol so a run is
    reproducible.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    dated = {
        symbol: bars
        for symbol in dict.fromkeys(symbols)
        if (bars := panel.get(symbol)) is not None and not bars.empty
    }
    if not dated:
        return []
    latest = max(_last_bar(bars) for bars in dated.values())
    eligible = [symbol for symbol, bars in dated.items() if _last_bar(bars) == latest]
    behind = [symbol for symbol in dated if symbol not in set(eligible)]
    if behind:
        log.warning(
            "%d of %d candidates have no bar for %s and cannot be ranked as movers: %s",
            len(behind),
            len(dated),
            latest.date().isoformat(),
            ", ".join(behind[:8]) + (f" and {len(behind) - 8} more" if len(behind) > 8 else ""),
        )
    moves = _changes(panel, eligible)
    ranked = sorted(moves.items(), key=lambda entry: (-entry[1], entry[0]))
    return [symbol for symbol, _ in ranked[:count]]


def forecast_universe(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    c: float = 0.1,
    min_train: int = model_mod.MIN_TRAIN,
    top_drivers: int = 5,
) -> list[StockPick]:
    """Forecast the next open for each stock, skipping those without history.

    A young listing cannot supply the walk-forward warm-up and is dropped with
    a warning rather than failing the run, exactly as an unmodellable index is.
    Repeats are collapsed: the same name twice is one forecast, not two rows and
    an inflated count.
    """
    picks: list[StockPick] = []
    wanted = list(dict.fromkeys(symbols or modelled_universe()))
    moves = _changes(panel, wanted)
    for symbol in wanted:
        try:
            picks.append(
                StockPick(
                    forecast_market(
                        symbol,
                        panel,
                        c=c,
                        min_train=min_train,
                        top_drivers=top_drivers,
                    ),
                    last_change=moves.get(symbol),
                )
            )
        except Exception as exc:
            log.warning("no forecast for %s: %s", symbol, exc)
    if not picks:
        raise RuntimeError("no stock could be modelled")
    return picks


def rank(picks: list[StockPick]) -> list[StockPick]:
    """Credible names first, by demonstrated skill against their own drift."""
    return sorted([p for p in picks if p.credible], key=_rank_key, reverse=True)


def discarded(picks: list[StockPick]) -> list[StockPick]:
    """Names whose backtest does not justify reading their probability."""
    return sorted([p for p in picks if not p.credible], key=lambda p: p.auc, reverse=True)


def _why_discarded(pick: StockPick) -> str:
    """Which of the three credibility tests a name failed, for the report."""
    reasons = []
    if pick.auc < MIN_AUC:
        reasons.append(f"AUC below {MIN_AUC:g}")
    if pick.brier_skill <= 0.0:
        reasons.append("worse calibrated than its base rate")
    if pick.n_oos < MIN_OOS:
        reasons.append(f"only {pick.n_oos} out-of-sample sessions")
    return ", ".join(reasons)


def to_frame(picks: list[StockPick]) -> pd.DataFrame:
    return pd.DataFrame([p.as_row() for p in picks])


def _table(picks: list[StockPick]) -> str:
    """One printed block. The verdict is a column in the CSV, not here: each
    block is uniform in it, and the heading above already says which is which.
    """
    # ``na_rep``, so a move the panel could not supply is a blank rather than the
    # ``NaN`` pandas would print: an unknown move should not read as a number.
    # The column is cast first because ``na_rep`` only reaches a missing float:
    # left as it comes, an all-unknown column is object dtype and prints ``None``.
    frame = to_frame(picks).drop(columns=["credible"])
    frame["last_change"] = frame["last_change"].astype("float64")
    return frame.to_string(index=False, na_rep="")


def render_text(
    picks: list[StockPick],
    top: int | None = None,
    panel: dict[str, pd.DataFrame] | None = None,
    max_stale_days: int = STALE_DAYS,
    selection: str | None = None,
    as_of: pd.Timestamp | None = None,
) -> str:
    """The ranking as a report, with what the numbers do and do not support.

    ``selection`` says how the names in front of the reader were chosen, when it
    was not simply "all of them": a table of ten movers read as a whole universe
    would look like a market where every name had just risen.

    ``as_of`` is the day the report is being read on, and is what catches a panel
    that is uniformly old. The stale-input footer measures each series against the
    session being forecast, which is dated from the panel's own last bar, so a
    cache that stopped a month ago has nothing lagging within itself and says
    nothing — the one run where the reader most needs telling.
    """
    ranked = rank(picks)
    # `top is not None`, not `if top`: asking for the strongest zero names is a
    # request for none of them, not a request for all of them.
    shown = ranked[:top] if top is not None else ranked
    lines: list[str] = []
    sessions = sorted({p.forecast.session.date().isoformat() for p in picks})
    lines.append(f"Next-open direction for {len(picks)} US names (session {', '.join(sessions)})")
    if selection:
        lines.append(selection)
    lines.append("")
    if shown:
        lines.append(
            f"Ranked — AUC at least {MIN_AUC:g}, positive Brier skill, "
            f"at least {MIN_OOS} out-of-sample sessions:"
        )
        lines.append(_table(shown))
    elif not ranked:
        lines.append(
            "No name cleared the credibility tests: on this sample the model has no "
            "demonstrated edge on any single stock, and none of the probabilities "
            "below should be read as a pick."
        )
    else:
        # Names did qualify; the caller asked to see none of them. Saying "no
        # demonstrated edge" here would be a claim about the model, not the request.
        lines.append(f"{len(ranked)} names cleared the credibility tests; none requested.")
    rest = discarded(picks)
    if rest:
        lines.append("")
        lines.append("No demonstrated skill — probabilities not ranked:")
        lines.append(_table(rest))
        lines.append("")
        for entry in rest:
            lines.append(f"  {entry.symbol}: {_why_discarded(entry)}")
    # Every printed row, not just the ranked ones. All these names share one
    # market clock, so a release landing before the auction applies to the
    # unranked table too — which is the whole output when nothing is credible.
    flagged = [p for p in [*shown, *rest] if p.forecast.caveats]
    if flagged:
        lines.append("")
        lines.append("scheduled releases this model cannot see:")
        for pick in flagged:
            for note in pick.forecast.caveats:
                lines.append(f"  {pick.symbol}: {note}")
    named: list[str] = []
    counted = 0
    if panel is not None and picks:
        session = max(p.forecast.session for p in picks)
        # The threshold the run was given, not the default: a footer disagreeing
        # with the guard that let the run through says the wrong thing twice.
        counted, named = stale_inputs(panel, session, max_stale_days)
        if named:
            lines.append("")
            lines.append(
                f"stale inputs: {len(named)} of {counted} series have no bar within "
                f"{max_stale_days} days of {session.date().isoformat()}, a gap the calendar "
                "does not explain. Their last value is carried forward, so these "
                "probabilities are the model's read of older cross-market and "
                f"cross-asset data: {', '.join(named[:8])}"
                + (f" and {len(named) - 8} more" if len(named) > 8 else "")
            )
    # Not conditional on the panel: this is a fact about the forecast itself, and
    # the only disclosure a uniformly old cache produces.
    if picks and as_of is not None:
        session = max(p.forecast.session for p in picks)
        lag = int((as_of.normalize() - session.normalize()).days)
        if lag > max_stale_days:
            # Three states, because the sentence should only claim what was looked
            # at: names printed above, series compared and none behind, or nothing
            # to compare — a panel of no measurable series says as little as none.
            if named:
                whose = (
                    "The series named above are behind the rest of that panel, which "
                    "is itself behind today, so these"
                )
            elif counted:
                whose = (
                    "The whole panel stops there, so no series is behind the others "
                    "and none is named above: these"
                )
            else:
                whose = "These"
            lines.append("")
            lines.append(
                "stale run: the session forecast above follows the panel's last bar and "
                f"is {lag} days before {as_of.date().isoformat()}. "
                f"{whose} probabilities are the model's read of the market as it stood "
                "then, not this morning."
            )
    lines.append("")
    lines.append(
        "last_change is the move the name has just made, for context only; it is "
        "not part of the model's view. "
        "The target is the opening print against the previous close — an overnight "
        "move, not a view on the company or on the session after the bell. "
        "Fitted on today's listings, so the metrics carry survivorship bias. "
        "Only the names in `gapmodel stock` read their overnight peers."
    )
    return "\n".join(lines) + "\n"
