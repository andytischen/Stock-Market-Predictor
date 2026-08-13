"""Turn the fitted models into a next-open probability report."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model as model_mod
from .events import caveats
from .features import _column_name, build_features, live_feature_row
from .markets import CURVE_FRONT, CURVE_STRIP, CURVE_WINDOW, MARKETS, Market, market

log = logging.getLogger(__name__)


@dataclass
class Forecast:
    symbol: str
    name: str
    region: str
    session: pd.Timestamp
    probability_up: float
    backtest: dict[str, float]
    contributions: pd.Series
    top_drivers: int = 5
    # Probability under the hypothetical moves asked for, if any were.
    shocked_probability: float | None = None
    # Scheduled releases that this session's probability cannot account for.
    caveats: tuple[str, ...] = ()

    @property
    def drivers(self) -> pd.Series:
        """The largest log-odds contributions behind this probability."""
        return self.contributions.head(self.top_drivers)

    def as_row(self) -> dict[str, object]:
        return {
            "market": self.name,
            "symbol": self.symbol,
            "region": self.region,
            "session": self.session.date().isoformat(),
            "p_open_up": _display(self.probability_up),
            **(
                {}
                if self.shocked_probability is None
                else {
                    "p_shocked": _display(self.shocked_probability),
                    "p_change": round(self.shocked_probability - self.probability_up, 4),
                }
            ),
            "oos_auc": round(self.backtest.get("auc", float("nan")), 4),
            "oos_brier_skill": round(self.backtest.get("brier_skill", 0.0), 4),
            "oos_accuracy": round(self.backtest.get("accuracy", 0.0), 4),
            "base_rate": round(self.backtest.get("base_rate", 0.0), 4),
        }


def _display(probability: float) -> float:
    """Never print a flat 0 or 1: no forecast here is a certainty."""
    return round(min(max(probability, 1e-4), 1 - 1e-4), 4)


def shocked_row(live: pd.DataFrame, shocks: dict[str, float]) -> pd.DataFrame:
    """Copy of ``live`` with hypothetical log returns added to some instruments.

    A shock is applied to every feature derived from that symbol's latest bar:
    the one-day and five-day returns (a move today is also part of the week),
    the VIX level, and the volatility-normalised shock feature. Symbols the
    target does not use (its own symbol, above all) are silently absent from
    its feature set and simply have no effect on it.
    """
    bumped = live.copy()
    for symbol, move in shocks.items():
        name = _column_name(symbol)
        for column in (
            f"mkt_{name}_return",
            f"mkt_{name}_return_5",
            f"ind_{name}_return",
            f"ind_{name}_return_5",
        ):
            if column in bumped:
                bumped[column] += move
        if symbol == "^VIX" and "ind_vix_level" in bumped:
            bumped["ind_vix_level"] *= np.exp(move)
        # A volatility-normalised shock feature has to move with its return.
        # The volatility itself is measured up to the previous bar, so a move
        # today leaves the denominator alone.
        vol = [c for c in bumped.columns if c.startswith(f"ind_{name}_vol_")]
        if f"ind_{name}_shock" in bumped and vol:
            sigma = bumped[vol[0]]
            bumped[f"ind_{name}_shock"] += move / sigma.where(sigma > 0)
        # The curve features are differences between the two oil funds, so a
        # move in either leg tilts them, in opposite directions.
        sign = 1.0 if symbol == CURVE_FRONT else -1.0 if symbol == CURVE_STRIP else 0.0
        if sign:
            for column in ("ind_oil_curve_return", f"ind_oil_curve_slope_{CURVE_WINDOW}"):
                if column in bumped:
                    bumped[column] += sign * move
    return bumped


def parse_shock(text: str) -> tuple[str, float]:
    """``"^KS11=+2%"`` or ``"^KS11=0.02"`` into a symbol and a log return.

    Split on the last ``=`` so symbols that contain one — every FX pair and
    future, ``CL=F``, ``JPY=X`` — remain shockable.
    """
    symbol, _, size = text.rpartition("=")
    if not symbol or not size:
        raise ValueError(f"expected SYMBOL=MOVE, got {text!r}")
    percent = size.strip().endswith("%")
    try:
        number = float(size.strip().rstrip("%"))
    except ValueError as exc:
        raise ValueError(f"{size!r} is not a move; expected e.g. {symbol}=+2%") from exc
    simple = number / 100.0 if percent else number
    if simple <= -1.0:
        raise ValueError("a move of -100% or worse is not a price")
    return symbol, float(np.log1p(simple))


def forecast_market(
    symbol: str,
    panel: dict[str, pd.DataFrame],
    c: float = 0.1,
    top_drivers: int = 5,
    hourly: dict[str, pd.Series] | None = None,
    min_train: int = model_mod.MIN_TRAIN,
    shocks: dict[str, float] | None = None,
    target: Market | None = None,
) -> Forecast:
    meta = target or market(symbol)
    features, labels = build_features(symbol, panel, forecast_row=True, hourly=hourly, target=meta)
    backtest = model_mod.walk_forward(features, labels, min_train=min_train, c=c)

    pipeline = model_mod.fit(features, labels, c=c)
    live, session = live_feature_row(symbol, panel, hourly=hourly, target=meta)
    calibrate = model_mod.calibrator(backtest)
    probability = float(calibrate(pipeline.predict_proba(live.to_numpy())[:, 1])[0])

    shocked = None
    explained = live
    if shocks:
        explained = shocked_row(live, shocks)
        shocked = float(calibrate(pipeline.predict_proba(explained.to_numpy())[:, 1])[0])

    weights = model_mod.coefficients(pipeline, list(features.columns))
    scaler = pipeline.named_steps["scale"]
    # Drivers describe the row that produced the rightmost probability, so
    # under a shock they explain the hypothetical rather than today.
    standardised = pd.Series(scaler.transform(explained.to_numpy())[0], index=features.columns)
    contributions = (weights * standardised).sort_values(key=abs, ascending=False)

    return Forecast(
        caveats=caveats(meta, session),
        symbol=symbol,
        name=meta.name,
        region=meta.region,
        session=session,
        probability_up=probability,
        backtest=backtest.metrics,
        contributions=contributions,
        top_drivers=top_drivers,
        shocked_probability=shocked,
    )


def forecast_all(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    c: float = 0.1,
    hourly: dict[str, pd.Series] | None = None,
    min_train: int = model_mod.MIN_TRAIN,
    shocks: dict[str, float] | None = None,
) -> list[Forecast]:
    results: list[Forecast] = []
    for symbol in symbols or [m.symbol for m in MARKETS]:
        try:
            results.append(
                forecast_market(
                    symbol, panel, c=c, hourly=hourly, min_train=min_train, shocks=shocks
                )
            )
        except Exception as exc:
            log.warning("no forecast for %s: %s", symbol, exc)
    if not results:
        raise RuntimeError("no market could be modelled")
    return results


def to_frame(forecasts: list[Forecast]) -> pd.DataFrame:
    return pd.DataFrame([f.as_row() for f in forecasts])
