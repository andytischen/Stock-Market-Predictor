"""Pre-open futures features built from hourly bars.

A daily bar cannot see what happens overnight, which is exactly what moves the
Wall Street open.  Yahoo serves roughly two years of hourly history, so these
features are only available on a recent window and are opt-in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .data import DEFAULT_CACHE
from .markets import Market

log = logging.getLogger(__name__)

# Instruments that trade through the night and lead the cash open.
INTRADAY_SYMBOLS: tuple[str, ...] = ("ES=F", "NQ=F", "CL=F", "GC=F")
MAX_HOURLY_PERIOD = "730d"
MOMENTUM_HOURS = 3
# Yahoo timestamps an hourly bar with the *start* of the hour it covers, so a
# bar is only complete — and only usable — one hour after its timestamp.
BAR_DURATION = pd.Timedelta(hours=1)
# The bell of the session being forecast lies in the future, so the newest bar
# is allowed to stand in for it — but only for this long, after which the cache
# is treated as too stale to describe the pre-open state.
MAX_STALENESS = pd.Timedelta(hours=24)
# How long a downloaded hourly file may be reused. Keyed on when it was
# fetched, not on its last bar: when the market is shut the newest bar is old
# however recently it was downloaded.
CACHE_TTL = BAR_DURATION


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "idx_").replace("=", "_")
    return cache_dir / f"{safe}_1h.csv"


def load_hourly(symbol: str, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.Series:
    """Hourly closes indexed in UTC, cached on disk."""
    path = _cache_path(cache_dir, symbol)
    now = pd.Timestamp.now(tz="UTC")
    # Unlike daily bars, an old hourly file is useless: it cannot describe the
    # run-up to the next bell, so it is refreshed rather than reused.
    fresh = (
        path.exists() and now - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") <= CACHE_TTL
    )
    if fresh and not refresh:
        close = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        return close.tz_localize("UTC") if close.index.tz is None else close.tz_convert("UTC")

    raw = yf.download(
        symbol,
        period=MAX_HOURLY_PERIOD,
        interval="1h",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no hourly data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
    close = raw["Close"].astype(float).dropna().sort_index()
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated(keep="last")]
    cache_dir.mkdir(parents=True, exist_ok=True)
    close.to_frame("Close").to_csv(path)
    return close


def load_hourly_panel(
    symbols: tuple[str, ...] = INTRADAY_SYMBOLS,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict[str, pd.Series]:
    panel: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            panel[symbol] = load_hourly(symbol, cache_dir, refresh)
        except Exception as exc:
            log.warning("skipping hourly %s: %s", symbol, exc)
    if not panel:
        raise RuntimeError("no hourly data could be loaded")
    return panel


def _positions_before(close: pd.Series, cutoffs: pd.DatetimeIndex) -> np.ndarray:
    """Index of the last bar completed before each cutoff, or -1 if there is none.

    A cutoff past the end of the series resolves to the final bar only while it
    is within ``MAX_STALENESS``; beyond that the series simply does not cover
    the moment asked about, and pretending otherwise would report a zero move.
    """
    positions = close.index.searchsorted(cutoffs - BAR_DURATION, side="right") - 1
    covered = cutoffs <= close.index[-1] + BAR_DURATION + MAX_STALENESS
    return np.where(covered, positions, -1)


def _move(close: pd.Series, to_pos: np.ndarray, from_pos: np.ndarray) -> np.ndarray:
    """Log move between two bars, NaN unless both exist and differ."""
    values = close.to_numpy()
    usable = (to_pos >= 0) & (from_pos >= 0) & (to_pos != from_pos)
    out = np.full(len(to_pos), np.nan)
    out[usable] = np.log(values[to_pos[usable]] / values[from_pos[usable]])
    return out


def preopen_features(
    target: Market, dates: pd.DatetimeIndex, hourly: dict[str, pd.Series]
) -> pd.DataFrame:
    """Overnight and last-hours futures moves, as known at the opening bell.

    Everything is measured against the bell itself: the reference point is the
    target's previous close, so the feature spans exactly the same window as
    the gap it predicts.
    """
    bell = dates.tz_localize("UTC") + pd.Timedelta(hours=target.open_utc)
    previous_close = bell - pd.Timedelta(hours=24 + target.open_utc - target.close_utc)
    momentum_from = bell - pd.Timedelta(hours=MOMENTUM_HOURS)

    columns: dict[str, np.ndarray] = {}
    for symbol, close in hourly.items():
        name = symbol.replace("=", "_").replace("-", "_").lower()
        at_bell = _positions_before(close, bell)
        columns[f"pre_{name}_overnight"] = _move(
            close, at_bell, _positions_before(close, previous_close)
        )
        columns[f"pre_{name}_momentum"] = _move(
            close, at_bell, _positions_before(close, momentum_from)
        )
    return pd.DataFrame(columns, index=dates).replace([np.inf, -np.inf], np.nan)
