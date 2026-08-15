"""Serialise a forecast run to the JSON snapshot the mobile app consumes.

The snapshot is a single small file: one entry per market with its next-open
probability, its out-of-sample quality and the largest log-odds drivers behind
it, the crude readings the models feed on, and a terse one-line summary. It is
what a scheduled job publishes to static hosting for the app to download; it
holds no look-ahead beyond what ``predict`` already reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .dashboard import OilReading
from .events import SCHEDULES, unmaintained_on
from .predict import Forecast, _display
from .stocks import target_market

_COVERAGE = {schedule.name: schedule for schedule in SCHEDULES}


def _session_open_utc(symbol: str, session: pd.Timestamp) -> str:
    """ISO-8601 UTC timestamp of a market's opening auction on ``session``.

    ``open_utc`` is hours from midnight UTC of the session date and may be
    negative for a session that starts the previous calendar day (Sydney).
    """
    open_utc = target_market(symbol).open_utc
    moment = session.normalize() + pd.Timedelta(hours=open_utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _market_entry(f: Forecast) -> dict[str, object]:
    entry: dict[str, object] = {
        "market": f.name,
        "symbol": f.symbol,
        "region": f.region,
        "session": f.session.date().isoformat(),
        "session_open_utc": _session_open_utc(f.symbol, f.session),
        "p_open_up": _display(f.probability_up),
        "oos_auc": round(f.backtest.get("auc", float("nan")), 4),
        "oos_brier_skill": round(f.backtest.get("brier_skill", 0.0), 4),
        "oos_accuracy": round(f.backtest.get("accuracy", 0.0), 4),
        "base_rate": round(f.backtest.get("base_rate", 0.0), 4),
        "drivers": [
            {"name": str(name), "log_odds": round(float(value), 4)}
            for name, value in f.drivers.items()
        ],
    }
    if f.caveats:
        entry["caveats"] = list(f.caveats)
    unchecked = _unchecked(f.session)
    if unchecked:
        entry["unchecked_releases"] = unchecked
    if f.shocked_probability is not None:
        entry["p_shocked"] = _display(f.shocked_probability)
        entry["p_change"] = round(f.shocked_probability - f.probability_up, 4)
    return entry


def _unchecked(session: pd.Timestamp) -> list[dict[str, str]]:
    """Release series whose published calendar does not reach ``session``.

    A consumer reading an empty ``caveats`` cannot otherwise tell a session with
    nothing scheduled from one nobody has a calendar for, which is the same
    silence the terminal output refuses to keep.
    """
    return [
        {"series": name, "table_ends": _COVERAGE[name].covers_until}
        for name in unmaintained_on(session)
    ]


def _crude_entry(reading: OilReading) -> dict[str, object]:
    return {
        "symbol": reading.symbol,
        "name": reading.name,
        "as_of": reading.as_of.date().isoformat(),
        "close": round(reading.close, 4),
        "return_1d": round(reading.return_1d, 4),
        "return_5d": round(reading.return_5d, 4),
        "volatility_20d": round(reading.volatility_20d, 4),
        "shock": round(reading.shock, 4),
        "is_shock": reading.is_shock,
    }


def summarise(forecasts: list[Forecast], oil: list[OilReading]) -> str:
    """A terse, SMS-length line: the crude moves and the headline index calls.

    Prefers the S&P 500 and Nasdaq (the calls readers ask about first); when
    neither is in the run it falls back to the two highest-probability markets.
    """
    parts: list[str] = []
    for reading in oil:
        parts.append(f"{reading.name.replace(' crude', '')} {reading.return_1d:+.1%}")

    by_symbol = {f.symbol: f for f in forecasts}
    headline = [by_symbol[s] for s in ("^GSPC", "^IXIC") if s in by_symbol]
    if not headline:
        headline = sorted(forecasts, key=lambda f: f.probability_up, reverse=True)[:2]
    calls = ", ".join(f"{f.name} {f.probability_up:.0%}" for f in headline)

    crude = ", ".join(parts)
    if crude and calls:
        return f"{crude} | {calls}"
    return crude or calls


def build_snapshot(
    forecasts: list[Forecast],
    oil: list[OilReading],
    generated_at: datetime | None = None,
) -> dict[str, object]:
    moment = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summarise(forecasts, oil),
        "markets": [_market_entry(f) for f in forecasts],
        "crude": [_crude_entry(r) for r in oil],
    }


def dumps(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=False)
