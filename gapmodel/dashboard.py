"""A point-in-time dashboard: where oil is, and what Asia is doing about it.

At 05:00 UTC Tokyo, Hong Kong, Seoul, Shanghai and Sydney are mid-session while
Mumbai has just opened and Europe has not, so the interesting question is how
the overnight crude move is feeding into the Asian opens the model still has to
call. The dashboard pairs the crude readings the model actually consumes with
those markets' session state and next-open probabilities, and shows how much of
each probability comes from the oil features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd

from .features import OIL_VOL_WINDOW, _column_name, log_return
from .markets import INDICATORS, MARKETS, OIL_SYMBOLS
from .predict import Forecast

OIL_FEATURE_PREFIXES = tuple(f"ind_{_column_name(symbol)}_" for symbol in sorted(OIL_SYMBOLS))
# A market is called "pre-open" once the auction is this close.
PREOPEN_HOURS = 3.0


@dataclass
class OilReading:
    """The crude numbers the model feeds on, for one benchmark."""

    symbol: str
    name: str
    as_of: pd.Timestamp
    close: float
    return_1d: float
    return_5d: float
    volatility_20d: float
    shock: float

    @property
    def direction(self) -> str:
        return "up" if self.return_1d > 0 else "down" if self.return_1d < 0 else "flat"

    @property
    def is_shock(self) -> bool:
        """A move large enough to stand out against the known volatility."""
        return abs(self.shock) >= 2.0


@dataclass
class MarketRow:
    """One market's session state and next-open call."""

    symbol: str
    name: str
    session_state: str
    hours_to_open: float
    last_close: float
    last_return: float
    forecast: Forecast | None = None

    @property
    def probability_up(self) -> float | None:
        return None if self.forecast is None else self.forecast.probability_up

    @property
    def oil_contribution(self) -> float:
        """Net log-odds the oil features add to this market's probability."""
        if self.forecast is None:
            return 0.0
        oil = self.forecast.contributions[
            [name.startswith(OIL_FEATURE_PREFIXES) for name in self.forecast.contributions.index]
        ]
        return float(oil.sum())

    @property
    def top_oil_driver(self) -> tuple[str, float] | None:
        if self.forecast is None:
            return None
        oil = self.forecast.contributions[
            [name.startswith(OIL_FEATURE_PREFIXES) for name in self.forecast.contributions.index]
        ]
        if oil.empty:
            return None
        return str(oil.index[0]), float(oil.iloc[0])


@dataclass
class Dashboard:
    as_of: pd.Timestamp
    region: str
    oil: list[OilReading] = field(default_factory=list)
    markets: list[MarketRow] = field(default_factory=list)


def _hour_of_day(moment: pd.Timestamp) -> float:
    return moment.hour + moment.minute / 60


def session_state(open_utc: float, close_utc: float, hour: float) -> tuple[str, float]:
    """Whether a session is open at ``hour`` UTC, and how long until it opens.

    ``open_utc`` may be negative for a session that starts on the previous
    calendar day (Sydney at 23:00), so it is read modulo the clock.
    """
    opening = open_utc % 24
    closing = close_utc % 24
    hours_to_open = (opening - hour) % 24
    if opening <= closing:
        is_open = opening <= hour < closing
    else:  # the session runs through midnight
        is_open = hour >= opening or hour < closing
    if is_open:
        return "open", 0.0
    state = "pre-open" if hours_to_open <= PREOPEN_HOURS else "closed"
    return state, hours_to_open


def oil_readings(panel: dict[str, pd.DataFrame]) -> list[OilReading]:
    names = {i.symbol: i.name for i in INDICATORS}
    readings = []
    for symbol in sorted(OIL_SYMBOLS):
        frame = panel.get(symbol)
        if frame is None:
            continue
        close = frame["Close"].dropna()
        if len(close) < OIL_VOL_WINDOW + 2:
            continue
        returns = log_return(close)
        # As in the feature builder, the volatility is the one known before
        # today's move, so the shock is scaled by a regime already in the price.
        vol = float(returns.rolling(OIL_VOL_WINDOW).std().shift(1).iloc[-1])
        move = float(returns.iloc[-1])
        readings.append(
            OilReading(
                symbol=symbol,
                name=names.get(symbol, symbol),
                as_of=close.index[-1],
                close=float(close.iloc[-1]),
                return_1d=move,
                return_5d=float(log_return(close, 5).iloc[-1]),
                volatility_20d=vol,
                shock=move / vol if vol > 0 else float("nan"),
            )
        )
    return readings


def build_dashboard(
    panel: dict[str, pd.DataFrame],
    forecasts: list[Forecast],
    as_of: pd.Timestamp | None = None,
    region: str = "Asia",
) -> Dashboard:
    moment = as_of or pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    hour = _hour_of_day(moment)
    by_symbol = {f.symbol: f for f in forecasts}

    rows = []
    for meta in MARKETS:
        if meta.region != region or meta.symbol not in panel:
            continue
        close = panel[meta.symbol]["Close"].dropna()
        if close.empty:
            continue
        state, hours = session_state(meta.open_utc, meta.close_utc, hour)
        returns = log_return(close)
        rows.append(
            MarketRow(
                symbol=meta.symbol,
                name=meta.name,
                session_state=state,
                hours_to_open=hours,
                last_close=float(close.iloc[-1]),
                last_return=float(returns.iloc[-1]) if len(returns) > 1 else float("nan"),
                forecast=by_symbol.get(meta.symbol),
            )
        )
    if not rows:
        raise ValueError(f"no {region} market could be read from the panel")

    rows.sort(key=lambda r: (r.session_state != "pre-open", r.hours_to_open, r.name))
    return Dashboard(as_of=moment, region=region, oil=oil_readings(panel), markets=rows)


def _pct(value: float) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:+.2%}"


def render_text(board: Dashboard) -> str:
    lines = [
        f"{board.region} dashboard — {board.as_of:%Y-%m-%d %H:%M} UTC",
        "",
        "Crude:",
    ]
    if not board.oil:
        lines.append("  no crude history loaded")
    for reading in board.oil:
        flag = "  <- shock" if reading.is_shock else ""
        lines.append(
            f"  {reading.name:<12} {reading.close:>8.2f}  1d {_pct(reading.return_1d)}"
            f"  5d {_pct(reading.return_5d)}  vol20 {reading.volatility_20d:.2%}"
            f"  shock {reading.shock:+.1f}{flag}"
        )

    lines += [
        "",
        f"{'market':<20} {'session':<9} {'to open':>8} {'last close':>11} "
        f"{'last move':>10} {'p(open up)':>11} {'oil log-odds':>13}",
    ]
    for row in board.markets:
        to_open = "-" if row.session_state == "open" else f"{row.hours_to_open:.1f}h"
        probability = "n/a" if row.probability_up is None else f"{row.probability_up:.1%}"
        lines.append(
            f"{row.name:<20} {row.session_state:<9} {to_open:>8} {row.last_close:>11,.2f} "
            f"{_pct(row.last_return):>10} {probability:>11} {row.oil_contribution:>+13.3f}"
        )

    lines += ["", "Oil driver behind each call:"]
    for row in board.markets:
        driver = row.top_oil_driver
        if driver is None:
            continue
        lines.append(f"  {row.name:<20} {driver[0]:<24} {driver[1]:+.3f}")
    return "\n".join(lines) + "\n"


def render_html(board: Dashboard) -> str:
    def oil_row(reading: OilReading) -> str:
        cells = [
            escape(reading.name),
            f"{reading.close:,.2f}",
            _pct(reading.return_1d),
            _pct(reading.return_5d),
            f"{reading.volatility_20d:.2%}",
            f"{reading.shock:+.1f}",
        ]
        css = ' class="shock"' if reading.is_shock else ""
        return f"<tr{css}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def market_row(row: MarketRow) -> str:
        driver = row.top_oil_driver
        cells = [
            escape(row.name),
            escape(row.session_state),
            "-" if row.session_state == "open" else f"{row.hours_to_open:.1f}h",
            f"{row.last_close:,.2f}",
            _pct(row.last_return),
            "n/a" if row.probability_up is None else f"{row.probability_up:.1%}",
            f"{row.oil_contribution:+.3f}",
            escape(driver[0]) if driver else "-",
        ]
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    oil_head = "".join(
        f"<th>{h}</th> " for h in ("benchmark", "close", "1d", "5d", "vol 20d", "shock")
    )
    market_head = "".join(
        f"<th>{h}</th>"
        for h in (
            "market",
            "session",
            "to open",
            "last close",
            "last move",
            "p(open up)",
            "oil log-odds",
            "top oil driver",
        )
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{escape(board.region)} dashboard — {board.as_of:%Y-%m-%d %H:%M} UTC</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 1.3rem; }}
 table {{ border-collapse: collapse; margin-bottom: 2rem; }}
 th, td {{ padding: .35rem .8rem; border-bottom: 1px solid #ddd; text-align: right; }}
 th:first-child, td:first-child, td:last-child {{ text-align: left; }}
 tr.shock td {{ font-weight: 600; color: #a00; }}
 caption {{ text-align: left; font-weight: 600; padding-bottom: .4rem; }}
</style>
<h1>{escape(board.region)} dashboard — {board.as_of:%Y-%m-%d %H:%M} UTC</h1>
<table><caption>Crude</caption><tr>{oil_head}</tr>
{"".join(oil_row(r) for r in board.oil)}
</table>
<table><caption>{escape(board.region)} markets</caption><tr>{market_head}</tr>
{"".join(market_row(r) for r in board.markets)}
</table>
<p>Probabilities are next-open calls from each market's own model; the oil
column is the net log-odds those probabilities take from the crude features.</p>
</html>
"""
