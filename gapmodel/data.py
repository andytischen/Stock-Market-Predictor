"""Download and cache daily bars from Yahoo Finance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .markets import all_symbols

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path.home() / ".cache" / "gapmodel"
# ``Adj Close`` carries Yahoo's dividend factor, which single equities need and
# indices do not: see ``features.dividend_adjusted``.
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


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


def _cached_fields(path: Path) -> frozenset[str] | None:
    """Fields asked of Yahoo when the cache was written, or None if unknown.

    What matters is what was *requested*, not what came back: a ticker Yahoo
    serves no volume for would otherwise be re-downloaded on every run.
    """
    meta = path.with_suffix(".fields")
    if not meta.exists():
        return None
    return frozenset(field for field in meta.read_text().split(",") if field)


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
    require: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return daily bars for ``symbol``, using an on-disk CSV cache.

    ``require`` names columns the caller cannot do without; a cache written
    before those columns were collected at all is re-downloaded rather than
    served with them missing. A column Yahoo simply does not publish for a
    symbol stays absent, and the cache is still used.
    """
    path = _cache_path(cache_dir, symbol)
    requested = pd.Timestamp(start)
    frame = None
    if path.exists() and not refresh:
        covered = _cached_start(path)
        if covered is not None and covered <= requested:
            collected = _cached_fields(path)
            if require and (collected is None or any(c not in collected for c in require)):
                frame = None
            else:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame is None:
        frame = _download(symbol, start)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        path.with_suffix(".start").write_text(requested.date().isoformat())
        path.with_suffix(".fields").write_text(",".join(FIELDS))
    return frame.loc[frame.index >= requested]


def load_panel(
    symbols: list[str] | None = None,
    start: str = "2005-01-01",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    require: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    """Load every requested symbol, skipping the ones Yahoo cannot serve."""
    # Fail before downloading anything if the cache is not usable.
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols or all_symbols():
        try:
            panel[symbol] = load_symbol(symbol, start, cache_dir, refresh, require)
        except Exception as exc:  # a single dead ticker must not kill a run
            log.warning("skipping %s: %s", symbol, exc)
    if not panel:
        raise RuntimeError("no symbols could be loaded")
    return panel
