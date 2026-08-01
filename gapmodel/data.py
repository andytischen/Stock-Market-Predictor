"""Download and cache daily bars from Yahoo Finance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .markets import all_symbols

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path.home() / ".cache" / "gapmodel"
FIELDS = ("Open", "High", "Low", "Close")


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "idx_").replace("=", "_")
    return cache_dir / f"{safe}.csv"


def _cached_start(path: Path) -> pd.Timestamp | None:
    """Start date the cache was downloaded with, or None if unknown.

    The first bar in the file cannot serve as this date: it is a trading day,
    always later than the requested start, and for young instruments later by
    years — comparing against it would re-download the whole panel every run.
    """
    meta = path.with_suffix(".start")
    if not meta.exists():
        return None
    try:
        return pd.Timestamp(meta.read_text().strip())
    except ValueError:
        return None


def _download(symbol: str, start: str) -> pd.DataFrame:
    raw = yf.download(symbol, start=start, interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
    frame = raw.loc[:, [c for c in FIELDS if c in raw.columns]].astype(float)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load_symbol(
    symbol: str,
    start: str = "2005-01-01",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily bars for ``symbol``, using an on-disk CSV cache."""
    path = _cache_path(cache_dir, symbol)
    requested = pd.Timestamp(start)
    frame = None
    if path.exists() and not refresh:
        covered = _cached_start(path)
        if covered is not None and covered <= requested:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame is None:
        frame = _download(symbol, start)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        path.with_suffix(".start").write_text(requested.date().isoformat())
    return frame.loc[frame.index >= requested]


def load_panel(
    symbols: list[str] | None = None,
    start: str = "2005-01-01",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every requested symbol, skipping the ones Yahoo cannot serve."""
    # Fail before downloading anything if the cache is not usable.
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols or all_symbols():
        try:
            panel[symbol] = load_symbol(symbol, start, cache_dir, refresh)
        except Exception as exc:  # a single dead ticker must not kill a run
            log.warning("skipping %s: %s", symbol, exc)
    if not panel:
        raise RuntimeError("no symbols could be loaded")
    return panel
