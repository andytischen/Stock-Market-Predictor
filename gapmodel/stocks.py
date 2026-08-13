"""Per-stock opening-gap forecasts for the Nasdaq universe.

The index model asks whether a market's opening auction prints above the
previous close. This module asks the same question of one Nasdaq-listed share,
reusing the same feature builder, the same walk-forward backtest and the same
calibration: a stock is simply a target on the Nasdaq cash-session clock, and
the cross-market and cross-asset columns it reads are the ones a US index
reads.

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

None of this is a recommendation, and the horizon is worth restating: the
target is the next opening print against the last close, an overnight move,
not a view on the company or on the session that follows the bell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import model as model_mod
from .markets import all_symbols, stock_market
from .predict import Forecast, _display, forecast_market
from .universe import nasdaq_universe

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
        return {
            "symbol": self.symbol,
            "session": self.forecast.session.date().isoformat(),
            # Rounded through the same guard the index table uses: no forecast
            # here is a certainty, so none of them prints as one.
            "p_open_up": _display(self.probability_up),
            "base_rate": round(self.base_rate, 4),
            "edge": round(self.edge, 4),
            "oos_auc": round(self.auc, 4),
            "oos_accuracy": round(self.forecast.backtest.get("accuracy", float("nan")), 4),
            "oos_brier_skill": round(self.brier_skill, 4),
            "n_oos": self.n_oos,
        }


def _rank_key(pick: StockPick) -> float:
    """Edge weighted by demonstrated skill.

    Ranking on the edge alone promotes a name whose backtest says nothing, and
    ranking on AUC alone promotes a skilful model that happens to have no view
    today. The product asks for both: a departure from the stock's own drift,
    from a model that has earned the right to make one.
    """
    return abs(pick.edge) * max(pick.auc - 0.5, 0.0)


def forecast_stocks(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    c: float = 0.1,
    min_train: int = model_mod.MIN_TRAIN,
    top_drivers: int = 5,
) -> list[StockPick]:
    """Forecast the next open for each stock, skipping those without history.

    A young listing cannot supply the walk-forward warm-up and is dropped with
    a warning rather than failing the run, exactly as an unmodellable index is.
    """
    picks: list[StockPick] = []
    for symbol in symbols or nasdaq_universe():
        try:
            picks.append(
                StockPick(
                    forecast_market(
                        symbol,
                        panel,
                        c=c,
                        min_train=min_train,
                        top_drivers=top_drivers,
                        target=stock_market(symbol),
                    )
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


def panel_symbols(symbols: list[str] | None = None) -> list[str]:
    """Every series a stock run needs: the indicator panel plus the stocks."""
    return list(dict.fromkeys(all_symbols() + (symbols or nasdaq_universe())))


def to_frame(picks: list[StockPick]) -> pd.DataFrame:
    return pd.DataFrame([p.as_row() for p in picks])


def render_text(picks: list[StockPick], top: int | None = None) -> str:
    """The ranking as a report, with what the numbers do and do not support."""
    ranked = rank(picks)
    shown = ranked[:top] if top else ranked
    lines: list[str] = []
    sessions = sorted({p.forecast.session.date().isoformat() for p in picks})
    lines.append(
        f"Next-open direction for {len(picks)} Nasdaq names (session {', '.join(sessions)})"
    )
    lines.append("")
    if shown:
        lines.append(
            f"Ranked — AUC at least {MIN_AUC:g}, positive Brier skill, "
            f"at least {MIN_OOS} out-of-sample sessions:"
        )
        lines.append(to_frame(shown).to_string(index=False))
    else:
        lines.append(
            "No name cleared the credibility tests: on this sample the model has no "
            "demonstrated edge on any single stock, and none of the probabilities "
            "below should be read as a pick."
        )
    rest = discarded(picks)
    if rest:
        lines.append("")
        lines.append("No demonstrated skill — probabilities not ranked:")
        lines.append(to_frame(rest).to_string(index=False))
        lines.append("")
        for entry in rest:
            lines.append(f"  {entry.symbol}: {_why_discarded(entry)}")
    flagged = [p for p in shown if p.forecast.caveats]
    if flagged:
        lines.append("")
        lines.append("scheduled releases this model cannot see:")
        for pick in flagged:
            for note in pick.forecast.caveats:
                lines.append(f"  {pick.symbol}: {note}")
    lines.append("")
    lines.append(
        "The target is the opening print against the previous close — an overnight "
        "move, not a view on the company or on the session after the bell. "
        "Fitted on today's Nasdaq names, so the metrics carry survivorship bias."
    )
    return "\n".join(lines) + "\n"
