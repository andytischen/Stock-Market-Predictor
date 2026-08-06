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
# Finer intervals to fall back on, coarsest first, when the hourly endpoint
# lags behind the market. They cover a shorter history, which is why they are
# only ever used to extend the tail of the hourly series.
FALLBACK_INTERVALS: tuple[str, ...] = ("30m", "15m", "5m")
FALLBACK_PERIOD = "5d"
# The finer feeds only reach back over ``FALLBACK_PERIOD``, so an hourly series
# older than that cannot be bridged: splicing would leave a hole in the middle.
MAX_FALLBACK_GAP = pd.Timedelta(days=4)
# How far the newest hourly bar may trail the present before the finer feeds
# are consulted. Two bar durations tolerates the usual publication lag.
STALE_AFTER = 2 * BAR_DURATION


def _download(symbol: str, interval: str, period: str) -> pd.Series:
    """Closes for one symbol at one interval, indexed in UTC."""
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no {interval} data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
    close = raw["Close"].astype(float).dropna().sort_index()
    close.index = pd.to_datetime(close.index, utc=True)
    return close[~close.index.duplicated(keep="last")]


def _extend_tail(symbol: str, close: pd.Series) -> pd.Series:
    """Top the hourly series up from a finer feed when it has fallen behind.

    Yahoo's hourly endpoint sometimes stops updating hours before the finer
    ones do, which strands the pre-open features on bars too old to describe
    the run-up to the bell. Sub-hourly bars are resampled to the same hourly
    grid — Yahoo stamps a bar with the start of the span it covers, so the
    resample is left-labelled and left-closed to match — and only the part
    newer than the hourly series is appended. If nothing finer is fresher, the
    series is returned untouched.
    """
    if close.empty:
        return close
    behind = pd.Timestamp.now(tz="UTC") - close.index[-1]
    if not STALE_AFTER < behind <= MAX_FALLBACK_GAP:
        return close
    for interval in FALLBACK_INTERVALS:
        try:
            fine = _download(symbol, interval, FALLBACK_PERIOD)
        except Exception as exc:
            log.debug("no %s bars for %s: %s", interval, symbol, exc)
            continue
        hourly = fine.resample(BAR_DURATION, label="left", closed="left").last().dropna()
        newer = hourly[hourly.index > close.index[-1]]
        if newer.empty:
            continue
        log.info(
            "%s: hourly feed stale at %s, extended to %s from %s bars",
            symbol,
            close.index[-1],
            newer.index[-1],
            interval,
        )
        return pd.concat([close, newer]).sort_index()
    return close


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

    close = _extend_tail(symbol, _download(symbol, "1h", MAX_HOURLY_PERIOD))
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
