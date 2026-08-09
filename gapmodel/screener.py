"""A progressive screen over a US universe: liquid, then active, then moving.

The gap model calls the *index*. This is the layer below it on the US side: of
the names inside that market, which ones are worth looking at for the session
just traded. It narrows in three stages, each cutting the list further, so the
survivors are stocks that are liquid enough to trade, busier than they usually
are, and already moving:

    universe  ->  liquid  ->  active  ->  moving

* **liquid** — a price floor (no sub-$5 quotes, where a tick is a percent) and a
  floor on the 30-session average volume.
* **active** — today's volume in absolute terms, and relative to that 30-session
  average. Relative volume is the one that says *unusual*: 5m shares is quiet for
  one name and a stampede for another.
* **moving** — the day's return, and the average true range as a percentage of
  price. Return says it moved today, ATR says the name moves at all, which keeps
  a 1% pop in a normally becalmed stock from reading like a trend.

Two details matter for the numbers to mean what they say:

Relative volume is measured against the 30 sessions *before* the one being
screened, not including it. Including today would put the day's own volume in
its own baseline, which flattens exactly the spikes the test is looking for (a
day trading 10x its average would read as ~7.5x on a 30-day window).

The average true range is a simple mean of the true ranges over the window, not
Wilder's smoothing. Wilder's is path-dependent on where the price history
starts, so its value drifts with the download window; a plain mean of the last
``atr_window`` true ranges depends only on those sessions and is reproducible.

Volume for a session still in progress is partial, so a screen run mid-session
understates today's volume and relative volume, and will return fewer names than
the same screen run after the close. Use ``--asof`` to screen a completed
session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CACHE, load_symbol

log = logging.getLogger(__name__)

# Andrew's filters: the defaults the screen ships with.
MIN_PRICE = 5.0
MIN_VOLUME = 2_000_000
MIN_AVG_VOLUME = 5_000_000
MIN_REL_VOLUME = 1.25
MIN_CHANGE = 0.01
MIN_ATR = 0.02
AVG_WINDOW = 30
ATR_WINDOW = 14

# Enough history to warm the longest window plus a margin for holidays; the
# screen reads the recent tail only, so it does not need the model's 20 years.
DEFAULT_START = "2024-01-01"


@dataclass(frozen=True)
class Criteria:
    """The thresholds a name has to clear, stage by stage."""

    min_price: float = MIN_PRICE
    min_volume: float = MIN_VOLUME
    min_avg_volume: float = MIN_AVG_VOLUME
    min_rel_volume: float = MIN_REL_VOLUME
    min_change: float = MIN_CHANGE
    min_atr: float = MIN_ATR
    avg_window: int = AVG_WINDOW
    atr_window: int = ATR_WINDOW

    def __post_init__(self) -> None:
        if self.avg_window < 2:
            raise ValueError(f"avg_window must be at least 2, got {self.avg_window}")
        if self.atr_window < 2:
            raise ValueError(f"atr_window must be at least 2, got {self.atr_window}")


DEFAULT_CRITERIA = Criteria()


@dataclass(frozen=True)
class Reading:
    """One ticker's screening metrics as of a given session."""

    symbol: str
    last: float
    change: float
    volume: float
    avg_volume: float
    rel_volume: float
    atr: float
    asof: pd.Timestamp


@dataclass(frozen=True)
class Stage:
    """One narrowing step: what it tests, and how many names came through."""

    name: str
    description: str
    kept: int


@dataclass(frozen=True)
class Screen:
    """The result of a run: the funnel, and the names that survived it."""

    criteria: Criteria
    stages: tuple[Stage, ...]
    readings: tuple[Reading, ...]
    asof: pd.Timestamp | None


def read_metrics(
    frame: pd.DataFrame,
    criteria: Criteria = DEFAULT_CRITERIA,
    symbol: str = "",
) -> Reading:
    """Screening metrics for one ticker from its daily bars.

    ``frame`` is daily bars already trimmed to the as-of session (its last row
    is the session being screened). Raises ``ValueError`` when there is not
    enough history to fill the windows, so a freshly listed name is skipped
    rather than screened against a partial average.
    """
    bars = frame.dropna(subset=["Close"])
    for column in ("Open", "High", "Low", "Volume"):
        if column not in bars.columns:
            raise ValueError(f"no {column} data")
    needed = max(criteria.avg_window, criteria.atr_window) + 1
    if len(bars) < needed:
        raise ValueError(f"need {needed} sessions, have {len(bars)}")

    close = bars["Close"].astype(float)
    prior_close = float(close.iloc[-2])
    last = float(close.iloc[-1])
    if prior_close <= 0:
        raise ValueError("non-positive previous close")

    volume = bars["Volume"].astype(float)
    today_volume = float(volume.iloc[-1])
    # The baseline excludes the session being screened: see the module docstring.
    baseline = volume.iloc[-(criteria.avg_window + 1) : -1]
    avg_volume = float(baseline.mean())
    if avg_volume <= 0 or pd.isna(avg_volume):
        raise ValueError("no volume over the window")

    return Reading(
        symbol=symbol,
        last=last,
        change=last / prior_close - 1,
        volume=today_volume,
        avg_volume=avg_volume,
        rel_volume=today_volume / avg_volume,
        atr=average_true_range(bars, criteria.atr_window) / last,
        asof=close.index[-1],
    )


def average_true_range(bars: pd.DataFrame, window: int = ATR_WINDOW) -> float:
    """Mean true range over the last ``window`` sessions, in price terms.

    True range is the widest of the session's own range and the two gapped
    ranges against the previous close, so an overnight gap counts as movement.
    """
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)
    previous = bars["Close"].astype(float).shift(1)
    spans = pd.concat([high - low, (high - previous).abs(), (previous - low).abs()], axis=1)
    true_range = spans.max(axis=1).dropna()
    if len(true_range) < window:
        raise ValueError(f"need {window} true ranges, have {len(true_range)}")
    return float(true_range.iloc[-window:].mean())


def _funnel(readings: list[Reading], criteria: Criteria) -> tuple[tuple[Stage, ...], list[Reading]]:
    """Apply the stages in order, counting survivors at each one."""
    liquid = [
        r
        for r in readings
        if r.last >= criteria.min_price and r.avg_volume >= criteria.min_avg_volume
    ]
    active = [
        r
        for r in liquid
        if r.volume >= criteria.min_volume and r.rel_volume >= criteria.min_rel_volume
    ]
    moving = [r for r in active if r.change >= criteria.min_change and r.atr >= criteria.min_atr]
    stages = (
        Stage("universe", "US names with enough history to screen", len(readings)),
        Stage(
            "liquid",
            f"price >= ${criteria.min_price:g}, "
            f"{criteria.avg_window}d average volume >= {criteria.min_avg_volume / 1e6:g}M",
            len(liquid),
        ),
        Stage(
            "active",
            f"volume >= {criteria.min_volume / 1e6:g}M, "
            f"relative volume >= {criteria.min_rel_volume:g}x",
            len(active),
        ),
        Stage(
            "moving",
            f"change >= {criteria.min_change:+.1%}, ATR >= {criteria.min_atr:.1%} of price",
            len(moving),
        ),
    )
    return stages, moving


def screen(
    symbols: list[str],
    criteria: Criteria = DEFAULT_CRITERIA,
    asof: pd.Timestamp | None = None,
    start: str = DEFAULT_START,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> Screen:
    """Screen every ticker, returning the funnel and the survivors.

    A ticker Yahoo cannot serve, or one without enough history as of ``asof``,
    is skipped with a warning rather than failing the whole run: a screen over
    a few hundred names always meets a few dead ones.
    """
    readings: list[Reading] = []
    for symbol in symbols:
        try:
            frame = load_symbol(symbol, start, cache_dir, refresh, require=("Volume",))
            if asof is not None:
                frame = frame.loc[frame.index <= asof]
            reading = read_metrics(frame, criteria, symbol=symbol)
        except Exception as exc:  # one dead or too-young ticker must not stop the run
            log.warning("skipping %s: %s", symbol, exc)
            continue
        readings.append(reading)
    if not readings:
        raise RuntimeError("no symbols could be screened")
    stages, survivors = _funnel(readings, criteria)
    # Most unusual first: relative volume is what separates a real move from a
    # name that happens to be up on its usual turnover.
    survivors.sort(key=lambda r: r.rel_volume, reverse=True)
    return Screen(
        criteria=criteria,
        stages=stages,
        readings=tuple(survivors),
        asof=max(r.asof for r in readings),
    )


def to_frame(readings: tuple[Reading, ...]) -> pd.DataFrame:
    """Tabulate the survivors the way a screener reads."""
    return pd.DataFrame(
        {
            "symbol": [r.symbol for r in readings],
            "last": [round(r.last, 2) for r in readings],
            "change": [round(r.change * 100, 2) for r in readings],
            "volume_m": [round(r.volume / 1e6, 2) for r in readings],
            "avg_volume_m": [round(r.avg_volume / 1e6, 2) for r in readings],
            "rel_volume": [round(r.rel_volume, 2) for r in readings],
            "atr_pct": [round(r.atr * 100, 2) for r in readings],
            "asof": [r.asof.date().isoformat() for r in readings],
        }
    )


def render_text(result: Screen) -> str:
    """The funnel, then the surviving names."""
    lines = ["Screen funnel:"]
    for stage in result.stages:
        lines.append(f"  {stage.name:<10} {stage.kept:>5}  {stage.description}")
    lines.append("")
    if result.readings:
        lines.append(to_frame(result.readings).to_string(index=False))
    else:
        lines.append("nothing cleared every filter")
    return "\n".join(lines) + "\n"
