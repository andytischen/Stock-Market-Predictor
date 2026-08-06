"""Break one index's next-open call down by European sector.

The probability model consumes each STOXX Europe 600 sector as a daily and a
weekly return, so the log-odds behind a forecast can be split by sector: which
parts of the market the call is actually leaning on, and in which direction.
Only European targets carry sector features, so only they can be broken down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .features import _column_name, log_return
from .markets import SECTORS
from .predict import Forecast


@dataclass
class SectorRow:
    """One sector's recent move and what the model makes of it."""

    symbol: str
    name: str
    close: float
    return_1d: float
    return_5d: float
    # Log-odds this sector's features add to the target's probability, and the
    # single largest of them.
    contribution: float
    top_feature: str


@dataclass
class SectorBoard:
    market: str
    session: pd.Timestamp
    probability_up: float
    as_of: pd.Timestamp
    rows: list[SectorRow] = field(default_factory=list)

    @property
    def net_contribution(self) -> float:
        return sum(row.contribution for row in self.rows)


def build_sector_board(panel: dict[str, pd.DataFrame], forecast: Forecast) -> SectorBoard:
    contributions = forecast.contributions
    as_of: pd.Timestamp | None = None
    rows: list[SectorRow] = []
    for sector in SECTORS:
        frame = panel.get(sector.symbol)
        if frame is None:
            continue
        close = frame["Close"].dropna()
        if len(close) < 6:
            continue
        as_of = close.index[-1] if as_of is None else max(as_of, close.index[-1])
        prefix = f"ind_{_column_name(sector.symbol)}_"
        mine = contributions[[name.startswith(prefix) for name in contributions.index]]
        rows.append(
            SectorRow(
                symbol=sector.symbol,
                name=sector.name,
                close=float(close.iloc[-1]),
                return_1d=float(log_return(close).iloc[-1]),
                return_5d=float(log_return(close, 5).iloc[-1]),
                contribution=float(mine.sum()),
                top_feature=str(mine.index[0]) if not mine.empty else "-",
            )
        )
    if not rows:
        raise ValueError("no sector history loaded")
    if not any(row.contribution for row in rows):
        raise ValueError(f"{forecast.symbol} carries no sector features; only European markets do")
    rows.sort(key=lambda r: abs(r.contribution), reverse=True)
    return SectorBoard(
        market=forecast.name,
        session=forecast.session,
        probability_up=forecast.probability_up,
        as_of=as_of or forecast.session,
        rows=rows,
    )


def render_text(board: SectorBoard) -> str:
    lines = [
        f"{board.market} — next open {board.session.date()}, p(up) {board.probability_up:.1%}",
        f"sector bars as of {board.as_of.date()}; log-odds is this sector's share of the call",
        "",
        f"{'sector':<40} {'close':>9} {'1d':>8} {'5d':>8} {'log-odds':>9}  driver",
    ]
    for row in board.rows:
        lines.append(
            f"{row.name:<40} {row.close:>9.2f} {row.return_1d:>+8.2%} "
            f"{row.return_5d:>+8.2%} {row.contribution:>+9.3f}  {row.top_feature}"
        )
    lines += ["", f"net sector log-odds {board.net_contribution:+.3f}"]
    return "\n".join(lines) + "\n"
