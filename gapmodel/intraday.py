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


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "idx_").replace("=", "_")
    return cache_dir / f"{safe}_1h.csv"


def load_hourly(symbol: str, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.Series:
    """Hourly closes indexed in UTC, cached on disk."""
    path = _cache_path(cache_dir, symbol)
    if path.exists() and not refresh:
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
    close = raw["Close"].astype(float).sort_index()
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


def _last_before(close: pd.Series, cutoffs: pd.DatetimeIndex) -> np.ndarray:
    """Most recent close completed before each cutoff (NaN when none exists)."""
    positions = close.index.searchsorted(cutoffs - BAR_DURATION, side="right") - 1
    values = np.full(len(cutoffs), np.nan)
    valid = positions >= 0
    values[valid] = close.to_numpy()[positions[valid]]
    return values


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
        at_bell = _last_before(close, bell)
        columns[f"pre_{name}_overnight"] = np.log(at_bell / _last_before(close, previous_close))
        columns[f"pre_{name}_momentum"] = np.log(at_bell / _last_before(close, momentum_from))
    return pd.DataFrame(columns, index=dates).replace([np.inf, -np.inf], np.nan)
