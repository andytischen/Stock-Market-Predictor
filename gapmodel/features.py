"""Leakage-free feature construction for opening-gap prediction."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from .intraday import preopen_features
from .markets import (
    BILL_CLOSE_UTC,
    BILL_YIELD,
    CURVE_CLOSE_UTC,
    CURVE_FRONT,
    CURVE_STRIP,
    CURVE_WINDOW,
    FUNDS_CLOSE_UTC,
    FUNDS_FUTURE,
    FX_SYMBOLS,
    INDICATORS,
    MARKETS,
    OIL_SYMBOLS,
    SECTOR_SYMBOLS,
    Market,
    lag_days,
)
from .stocks import is_stock, peers_of, target_market

MIN_HISTORY = 60
OIL_VOL_WINDOW = 20
# Volatility window for the FX-intervention shock feature.  Kept identical to
# the oil window so both shock series share a comparable normalisation scale,
# but defined separately so it can be tuned independently.
FX_VOL_WINDOW = 20
# Cross-market moves are read in standard deviations of the volatility regime
# the target already knew about, not in raw percent.
MKT_VOL_WINDOW = 60
# A move beyond this many deviations is held at the edge: past it the linear
# model is extrapolating out of every sample it was ever fitted on.
MKT_SHOCK_CLIP = 4.0
# A gap of exactly zero means the source repeated the previous close instead of
# publishing a real opening print; such sessions cannot be labelled.
STALE_GAP_TOLERANCE = 1e-9
MAX_STALE_FRACTION = 0.5
# Feature blocks built from two legs or from neither: the crude curve is a
# spread and the policy rate a premium, so one leg alone measures nothing.
PAIRED_INPUTS = ((CURVE_FRONT, CURVE_STRIP), (FUNDS_FUTURE, BILL_YIELD))


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    positive = close.where(close > 0)
    return np.log(positive / positive.shift(periods))


def opening_gap(bars: pd.DataFrame) -> pd.Series:
    """Log return from the previous close to today's opening print."""
    return np.log(bars["Open"] / bars["Close"].shift(1))


def dividend_adjusted(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars on a total-return basis, so going ex-dividend is not a down gap.

    Yahoo's daily bars are split-adjusted but not dividend-adjusted, so a single
    company's opening print falls by roughly the dividend on the morning it goes
    ex and the label for that session records a gap the market never made. It is
    a handful of sessions a year per name, but the sign is always the same one.
    ``Adj Close`` carries the dividend factor; applying it to both prints of the
    same session leaves that session's own returns untouched and corrects only
    the previous-close-to-open step. An index pays nothing, so it is left alone,
    and a cache written before the column was collected simply is not corrected.

    The factor is cumulative and rises towards 1, so a gap in it is carried both
    ways: filling leading rows with 1.0 instead of the first factor published
    would put a made-up gap of the whole accumulated discount at the boundary.
    """
    if "Adj Close" not in bars:
        return bars
    factor = bars["Adj Close"] / bars["Close"].where(bars["Close"] > 0)
    factor = factor.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)
    return bars.assign(Open=bars["Open"] * factor, Close=bars["Close"] * factor)


def total_return_close(bars: pd.DataFrame) -> pd.Series:
    """Closing series to take returns from: dividend-adjusted where published."""
    column = "Adj Close" if "Adj Close" in bars else "Close"
    return bars[column].dropna()


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


def _shock(returns: pd.Series, dates: pd.DatetimeIndex, lag: int) -> pd.Series:
    """``returns`` in deviations of their own trailing volatility, clipped.

    The denominator is measured up to the previous bar, so the session being
    scaled is not part of the scale it is judged by.
    """
    vol = returns.rolling(MKT_VOL_WINDOW).std().shift(1)
    scaled = (returns / vol.where(vol > 0)).clip(-MKT_SHOCK_CLIP, MKT_SHOCK_CLIP)
    return as_of(scaled, dates, lag)


def _lag_days(source_close_utc: float, target: Market) -> int:
    """0 if the source bar closes before the target opens, otherwise 1."""
    return lag_days(source_close_utc, target.open_utc)


def curve_features(
    panel: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, target: Market
) -> dict[str, pd.Series]:
    """Shape of the crude curve: the front month against the twelve-month strip.

    Two readings, both differences of log returns so the funds' own price levels
    and their tracking drift cancel: today's move of the front leg relative to
    the strip, and the same over ``CURVE_WINDOW`` sessions. Negative is contango
    — the front lagging, supply comfortable — and positive is backwardation.
    Absent from the panel, the features are simply not built.
    """
    if CURVE_FRONT not in panel or CURVE_STRIP not in panel:
        return {}
    front = panel[CURVE_FRONT]["Close"].dropna()
    strip = panel[CURVE_STRIP]["Close"].dropna()
    lag = _lag_days(CURVE_CLOSE_UTC, target)
    daily = log_return(front) - log_return(strip)
    slow = log_return(front, CURVE_WINDOW) - log_return(strip, CURVE_WINDOW)
    return {
        "ind_oil_curve_return": as_of(daily.dropna(), dates, lag),
        f"ind_oil_curve_slope_{CURVE_WINDOW}": as_of(slow.dropna(), dates, lag),
    }


def peer_features(
    target_symbol: str, panel: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, target: Market
) -> dict[str, pd.Series]:
    """What the companies trading the same end demand did, daily and weekly.

    Only single stocks have peers; an index is the average of its own. The lag
    is the ordinary one, which is the point: Seoul and Tokyo close before New
    York opens, so their sessions are same-day information for a US chipmaker,
    while the US legs are read a session late like every other Wall Street bar.

    Peers are single companies too, so their returns are taken from the
    dividend-adjusted close: an ex-dividend date is not a fall in demand.
    """
    built: dict[str, pd.Series] = {}
    for peer in peers_of(target_symbol):
        if peer.symbol not in panel:
            continue
        close = total_return_close(panel[peer.symbol])
        lag = _lag_days(peer.close_utc, target)
        name = _column_name(peer.symbol)
        built[f"peer_{name}_return"] = as_of(log_return(close), dates, lag)
        built[f"peer_{name}_return_5"] = as_of(log_return(close, 5), dates, lag)
    return built


def policy_features(
    panel: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, target: Market
) -> dict[str, pd.Series]:
    """What the market has priced for the policy rate, now and a quarter out.

    Two readings in percentage points: the rate the front fed funds future is
    priced for, and the tightening priced into the next quarter as the 13-week
    bill's premium over it. A widening spread is the market pulling a hike
    forward, which is the part of a hawkish turn a price-only model can actually
    observe; the words spoken to cause it remain invisible.

    The daily and monthly *changes* in the priced rate were measured and
    dropped: they ranked last of eighty-three features by log-odds weight, which
    is what one should expect of a series that moves in single basis points
    outside a meeting week.

    Levels, not log returns: a future priced near 100 has meaninglessly small
    returns, and bill yields have sat at zero, where a log return is undefined.
    The two legs close an hour apart, so both are read on the later of the two
    clocks and the bill is carried forward onto the future's sessions.
    """
    if FUNDS_FUTURE not in panel or BILL_YIELD not in panel:
        return {}
    price = panel[FUNDS_FUTURE]["Close"].dropna()
    bill = panel[BILL_YIELD]["Close"].dropna()
    if price.empty or bill.empty:
        return {}
    implied = 100.0 - price
    lag = _lag_days(max(FUNDS_CLOSE_UTC, BILL_CLOSE_UTC), target)
    spread = bill.reindex(implied.index.union(bill.index)).ffill().reindex(implied.index) - implied
    return {
        "ind_policy_rate": as_of(implied, dates, lag),
        "ind_policy_tightening_3m": as_of(spread.dropna(), dates, lag),
    }


def feature_symbols(target_symbol: str, available: Collection[str] | None = None) -> set[str]:
    """The panel series a model for ``target_symbol`` reads.

    The download is one list for every target, so a loaded panel is wider than
    any single model: the STOXX 600 sector trackers are skipped outside Europe,
    and an opening-price stand-in is read only as the gap source of its own
    index. Kept beside ``build_features`` because it has to answer for the same
    branches — a column added there and not here would be read by a model
    nothing was checked against.

    ``available`` names the series the panel actually carries, which decides the
    paired blocks: a spread needs both its legs, so when one download failed the
    surviving leg is read by nothing and naming it would judge the run on a
    series no feature was built from.
    """
    target = target_market(target_symbol)
    read = {target_symbol, target.gap_symbol}
    read |= {other.symbol for other in MARKETS if other.symbol != target_symbol}
    read |= {
        indicator.symbol
        for indicator in INDICATORS
        if indicator.symbol not in SECTOR_SYMBOLS or target.region == "Europe"
    }
    read |= {peer.symbol for peer in peers_of(target_symbol)}
    for pair in PAIRED_INPUTS:
        if available is None or set(pair) <= set(available):
            read |= set(pair)
    return read


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
    target = target_market(target_symbol)
    if target_symbol not in panel:
        raise KeyError(f"no price history loaded for {target_symbol}")

    gap_symbol = target.gap_symbol
    if gap_symbol not in panel:
        raise KeyError(f"no opening prices loaded for {gap_symbol}")

    bars = panel[gap_symbol].dropna(subset=["Open", "Close"])
    if is_stock(target_symbol):
        bars = dividend_adjusted(bars)
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
        # In percent a cross-market move is not comparable across regimes: a 6%
        # Kospi session is a four-sigma event in a calm sample and an ordinary
        # one in a violent month. Fitted on the calm years, the model reads the
        # violent month as certainty and states 0.01 for opens that then come up
        # small, which costs far more than being merely wrong. Each move is
        # therefore divided by the volatility the source had already shown -
        # measured to the previous bar, as the oil and FX shocks are - and held
        # at the edge of the range any fit has seen.
        features[f"mkt_{name}_shock"] = _shock(log_return(close), dates, lag)
        features[f"mkt_{name}_shock_5"] = _shock(log_return(close, 5), dates, lag)
        features[f"mkt_{name}_vol_{MKT_VOL_WINDOW}"] = as_of(
            log_return(close).rolling(MKT_VOL_WINDOW).std().shift(1), dates, lag
        )

    for indicator in INDICATORS:
        if indicator.symbol not in panel:
            continue
        # European sector read-across is a European story: outside the region it
        # measurably dilutes the fit, so those markets keep the whole-index and
        # cross-asset indicators only.
        if indicator.symbol in SECTOR_SYMBOLS and target.region != "Europe":
            continue
        # The sector read-across is carried by iShares trackers, which are
        # funds and distribute: on an ex-distribution morning every one of the
        # eighteen prints a fall of up to 3% that the sectors never made, and
        # European targets read all of them at once. The dividend factor makes
        # those sessions total-return, as it does for a single stock's peers.
        close = total_return_close(panel[indicator.symbol])
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

    features.update(peer_features(target_symbol, panel, dates, target))
    features.update(curve_features(panel, dates, target))
    features.update(policy_features(panel, dates, target))

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
