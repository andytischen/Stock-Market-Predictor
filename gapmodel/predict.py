"""Turn the fitted models into a next-open probability report."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import model as model_mod
from .features import build_features, live_feature_row
from .markets import MARKETS, market

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
            # Never print a flat 0 or 1: no forecast here is a certainty.
            "p_open_up": round(min(max(self.probability_up, 1e-4), 1 - 1e-4), 4),
            "oos_auc": round(self.backtest.get("auc", float("nan")), 4),
            "oos_brier_skill": round(self.backtest.get("brier_skill", 0.0), 4),
            "oos_accuracy": round(self.backtest.get("accuracy", 0.0), 4),
            "base_rate": round(self.backtest.get("base_rate", 0.0), 4),
        }


def forecast_market(
    symbol: str,
    panel: dict[str, pd.DataFrame],
    c: float = 0.1,
    top_drivers: int = 5,
    hourly: dict[str, pd.Series] | None = None,
    min_train: int = model_mod.MIN_TRAIN,
) -> Forecast:
    features, labels = build_features(symbol, panel, forecast_row=True, hourly=hourly)
    backtest = model_mod.walk_forward(features, labels, min_train=min_train, c=c)

    pipeline = model_mod.fit(features, labels, c=c)
    live, session = live_feature_row(symbol, panel, hourly=hourly)
    raw = pipeline.predict_proba(live.to_numpy())[:, 1]
    probability = float(model_mod.calibrator(backtest)(raw)[0])

    weights = model_mod.coefficients(pipeline, list(features.columns))
    scaler = pipeline.named_steps["scale"]
    standardised = pd.Series(scaler.transform(live.to_numpy())[0], index=features.columns)
    contributions = (weights * standardised).sort_values(key=abs, ascending=False)

    meta = market(symbol)
    return Forecast(
        symbol=symbol,
        name=meta.name,
        region=meta.region,
        session=session,
        probability_up=probability,
        backtest=backtest.metrics,
        contributions=contributions,
        top_drivers=top_drivers,
    )


def forecast_all(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    c: float = 0.1,
    hourly: dict[str, pd.Series] | None = None,
    min_train: int = model_mod.MIN_TRAIN,
) -> list[Forecast]:
    results: list[Forecast] = []
    for symbol in symbols or [m.symbol for m in MARKETS]:
        try:
            results.append(forecast_market(symbol, panel, c=c, hourly=hourly, min_train=min_train))
        except Exception as exc:
            log.warning("no forecast for %s: %s", symbol, exc)
    if not results:
        raise RuntimeError("no market could be modelled")
    return results


def to_frame(forecasts: list[Forecast]) -> pd.DataFrame:
    return pd.DataFrame([f.as_row() for f in forecasts])
