"""Leakage-free feature construction for opening-gap prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .intraday import preopen_features
from .markets import (
    FX_SYMBOLS,
    INDICATORS,
    MARKETS,
    OIL_SYMBOLS,
    SECTOR_SYMBOLS,
    Market,
    lag_days,
    market,
)

MIN_HISTORY = 60
OIL_VOL_WINDOW = 20
# Volatility window for the FX-intervention shock feature.  Kept identical to
# the oil window so both shock series share a comparable normalisation scale,
# but defined separately so it can be tuned independently.
FX_VOL_WINDOW = 20
# A gap of exactly zero means the source repeated the previous close instead of
# publishing a real opening print; such sessions cannot be labelled.
STALE_GAP_TOLERANCE = 1e-9
MAX_STALE_FRACTION = 0.5


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    positive = close.where(close > 0)
    return np.log(positive / positive.shift(periods))


def opening_gap(bars: pd.DataFrame) -> pd.Series:
    """Log return from the previous close to today's opening print."""
    return np.log(bars["Open"] / bars["Close"].shift(1))


def as_of(source: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """Value of ``source`` known ``lag_days`` calendar days before each date.

    Missing calendar days (weekends, holidays) fall back to the most recent
    earlier observation, so a stale-but-known value is used rather than NaN.
    """
    cut = dates - pd.Timedelta(days=lag_days)
    calendar = pd.date_range(min(source.index.min(), cut.min()), max(source.index.max(), cut.max()))
    return pd.Series(source.reindex(calendar).ffill().reindex(cut).to_numpy(), index=dates)


def _column_name(symbol: str) -> str:
    """Symbol turned into a feature-name fragment."""
    cleaned = symbol.lstrip("^")
    for character in "=-.":
        cleaned = cleaned.replace(character, "_")
    return cleaned.lower()


def _lag_days(source_close_utc: float, target: Market) -> int:
    """0 if the source bar closes before the target opens, otherwise 1."""
    return lag_days(source_close_utc, target.open_utc)


def build_features(
    target_symbol: str,
    panel: dict[str, pd.DataFrame],
    forecast_row: bool = False,
    hourly: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the design matrix and the up/down label for one market.

    With ``forecast_row`` the frame is extended by one row for the next
    session, whose label is unknown and whose features only use information
    available before that session's opening auction.

    With ``hourly`` the pre-open futures moves are added, which restricts the
    sample to the window those hourly bars cover.
    """
    target = market(target_symbol)
    if target_symbol not in panel:
        raise KeyError(f"no price history loaded for {target_symbol}")

    gap_symbol = target.gap_symbol
    if gap_symbol not in panel:
        raise KeyError(f"no opening prices loaded for {gap_symbol}")

    bars = panel[gap_symbol].dropna(subset=["Open", "Close"])
    if len(bars) < MIN_HISTORY:
        raise ValueError(f"{gap_symbol}: only {len(bars)} usable rows")
    if forecast_row:
        next_date = next_session_date(bars.index[-1])
        blank = pd.DataFrame(np.nan, index=[next_date], columns=bars.columns, dtype=float)
        bars = pd.concat([bars.astype(float), blank])

    dates = pd.DatetimeIndex(bars.index)
    gap = opening_gap(bars)
    features: dict[str, pd.Series] = {
        "own_gap_lag1": gap.shift(1),
        "own_close_return_lag1": log_return(bars["Close"]).shift(1),
        "own_intraday_return_lag1": np.log(bars["Close"] / bars["Open"]).shift(1),
        "own_gap_mean_20": gap.shift(1).rolling(20).mean(),
        "own_gap_std_20": gap.shift(1).rolling(20).std(),
        "own_close_return_5": log_return(bars["Close"], 5).shift(1),
        "calendar_gap_days": pd.Series(dates.to_series().diff().dt.days.to_numpy(), index=dates),
    }

    for other in MARKETS:
        if other.symbol == target_symbol or other.symbol not in panel:
            continue
        close = panel[other.symbol]["Close"].dropna()
        lag = _lag_days(other.close_utc, target)
        name = _column_name(other.symbol)
        features[f"mkt_{name}_return"] = as_of(log_return(close), dates, lag)
        features[f"mkt_{name}_return_5"] = as_of(log_return(close, 5), dates, lag)

    for indicator in INDICATORS:
        if indicator.symbol not in panel:
            continue
        close = panel[indicator.symbol]["Close"].dropna()
        lag = _lag_days(indicator.close_utc, target)
        name = _column_name(indicator.symbol)
        returns = log_return(close)
        features[f"ind_{name}_return"] = as_of(returns, dates, lag)
        if indicator.symbol == "^VIX":
            features["ind_vix_level"] = as_of(close, dates, lag)
        if indicator.symbol in OIL_SYMBOLS:
            # Volatility is taken as of the previous bar so the shock is scaled
            # by a regime the market already knew about.
            vol = returns.rolling(OIL_VOL_WINDOW).std().shift(1)
            features[f"ind_{name}_return_5"] = as_of(log_return(close, 5), dates, lag)
            features[f"ind_{name}_vol_{OIL_VOL_WINDOW}"] = as_of(vol, dates, lag)
            features[f"ind_{name}_shock"] = as_of(returns / vol.where(vol > 0), dates, lag)
        elif indicator.symbol in FX_SYMBOLS:
            # Central-bank intervention produces a move that is large relative
            # to recent realised volatility.  The shock feature normalises the
            # daily return by the preceding-bar volatility so the model can
            # distinguish a routine 0.5% drift from a 3-sigma BoJ defence.
            vol = returns.rolling(FX_VOL_WINDOW).std().shift(1)
            features[f"ind_{name}_return_5"] = as_of(log_return(close, 5), dates, lag)
            features[f"ind_{name}_vol_{FX_VOL_WINDOW}"] = as_of(vol, dates, lag)
            features[f"ind_{name}_shock"] = as_of(returns / vol.where(vol > 0), dates, lag)
        elif indicator.symbol in SECTOR_SYMBOLS:
            features[f"ind_{name}_return_5"] = as_of(log_return(close, 5), dates, lag)

    frame = pd.DataFrame(features, index=dates)
    if hourly:
        frame = frame.join(preopen_features(target, dates, hourly))
    frame = frame.replace([np.inf, -np.inf], np.nan)
    real_open = gap.abs() > STALE_GAP_TOLERANCE
    stale_fraction = float((~real_open & gap.notna()).sum() / max(gap.notna().sum(), 1))
    if stale_fraction > MAX_STALE_FRACTION:
        raise ValueError(
            f"{gap_symbol}: {stale_fraction:.0%} of opening prints repeat the "
            "previous close, the series cannot be modelled"
        )
    label = gap.gt(0).astype(float).where(gap.notna() & real_open)

    complete = frame.notna().all(axis=1)
    missing = frame.columns[frame.iloc[-1].isna()].tolist() if len(frame) else []
    frame, label = frame.loc[complete], label.loc[complete]
    if int(label.notna().sum()) < MIN_HISTORY:
        raise ValueError(f"{target_symbol}: only {len(frame)} complete feature rows")
    if forecast_row and (frame.empty or frame.index[-1] != dates[-1]):
        detail = ", ".join(missing[:4]) or "unknown"
        if all(name.startswith("pre_") for name in missing):
            detail += " (no futures trading since the previous close)"
        raise ValueError(f"{target_symbol}: indicators missing for the next session: {detail}")
    return frame, label


def next_session_date(last_session: pd.Timestamp) -> pd.Timestamp:
    """Next weekday after the last observed session (holidays are ignored)."""
    nxt = last_session + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def live_feature_row(
    target_symbol: str,
    panel: dict[str, pd.DataFrame],
    hourly: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Feature row for the next, not yet observed, opening auction."""
    frame, _ = build_features(target_symbol, panel, forecast_row=True, hourly=hourly)
    return frame.iloc[[-1]], frame.index[-1]
