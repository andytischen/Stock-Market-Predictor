"""An evaluation dashboard for the Asian session.

Three questions, one page:

1. *What is the index doing?*  Return, realised volatility, the size of the
   opening gap, turnover against its own recent average.
2. *Who is doing it?*  The dominant constituents, their contribution to the
   index move in basis points, their beta to the index (how much index risk a
   given name carries) and whether their volume is running hot.
3. *What is doing it to them?*  Univariate and joint regressions of the index
   return on the outside markets that were already closed when it opened —
   India, the Middle East (crude, Tadawul, Tel Aviv), European futures and
   Wall Street.

Every driver enters with the same lag rule the gap model uses: a market that
closes after the index opens can only be read from the previous day, so no
number on this page is one the market could not have seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import as_of, log_return, opening_gap
from .markets import lag_days
from .regions import (
    ASIA_INDICES,
    EUROPE_INDICES,
    INFLUENCES,
    THEMES,
    Constituent,
    IndexProfile,
    Influence,
)

log = logging.getLogger(__name__)

# Sessions used for the recent-activity baselines (beta, average volume).
ACTIVITY_WINDOW = 60
# Sessions used for the driver regressions: long enough for a t-statistic to
# mean something, short enough to describe the current regime.
REGRESSION_WINDOW = 500
MIN_REGRESSION_ROWS = 60
TRADING_DAYS = 252
# A headline series is only preferred over its fallback if it actually printed
# over this recent stretch of calendar days.
RECENCY_WINDOW = 60

# Data this dashboard cannot see, and the feed that would be needed for it.
# Printed with the dashboard so a reader never mistakes a proxy for the thing.
DATA_GAPS: tuple[tuple[str, str], ...] = (
    ("Order book depth, spreads, queue imbalance", "exchange level 2 / TAQ feed"),
    ("Margin balances and leveraged ETF flow", "exchange margin statistics (TSE, KRX, SSE)"),
    ("Short interest and stock borrow", "exchange short-sale reporting"),
    ("Retail vs institutional participation", "broker or exchange investor-type breakdown"),
    ("Index weights and free float", "index provider licence (Nikkei, KRX, CSI, HSI, SGX)"),
    ("News and sentiment flow", "newswire feed with timestamps (Reuters, Bloomberg)"),
)


@dataclass
class IndexSnapshot:
    """Everything the dashboard shows for one index."""

    profile: IndexProfile
    source: str
    session: pd.Timestamp
    metrics: dict[str, float]
    constituents: pd.DataFrame
    drivers: pd.DataFrame
    themes: pd.DataFrame


@dataclass
class Dashboard:
    generated: pd.Timestamp
    asia: list[IndexSnapshot] = field(default_factory=list)
    europe: list[IndexSnapshot] = field(default_factory=list)

    @property
    def snapshots(self) -> list[IndexSnapshot]:
        return self.asia + self.europe

    def theme_matrix(self) -> pd.DataFrame:
        """Explanatory power of each theme, one row per Asian index."""
        rows = []
        for snapshot in self.asia:
            row = {"index": snapshot.profile.name, "country": snapshot.profile.country}
            for theme in THEMES:
                match = snapshot.themes.loc[snapshot.themes["theme"] == theme, "r2"]
                row[theme] = float(match.iloc[0]) if len(match) else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)


def _pct(value: float) -> float:
    return round(100.0 * float(value), 3)


def _returns(bars: pd.DataFrame) -> pd.Series:
    return log_return(bars["Close"].dropna())


def _traded_volume(bars: pd.DataFrame) -> pd.Series:
    """Volume with the non-prints dropped: Yahoo writes 0 where it has none."""
    volume = bars.get("Volume")
    if volume is None:
        return pd.Series(dtype=float)
    return volume.where(volume > 0).dropna()


def _last_session(bars: pd.DataFrame) -> pd.Timestamp | None:
    closes = bars["Close"].dropna()
    return None if closes.empty else pd.Timestamp(closes.index[-1])


def _recent_sessions(bars: pd.DataFrame, reference: pd.Timestamp, days: int) -> int:
    """Sessions printed in the ``days`` before ``reference``.

    The reference is the freshest date in the comparison, not the series' own
    last bar: a series that stopped printing months ago would otherwise look as
    busy as one that is up to date.
    """
    closes = bars["Close"].dropna()
    return int((closes.index > reference - pd.Timedelta(days=days)).sum())


def index_source(profile: IndexProfile, panel: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """The series to read the index from: the headline one unless it has gone quiet."""
    primary = panel.get(profile.symbol)
    fallback = panel.get(profile.fallback) if profile.fallback else None
    primary_last = _last_session(primary) if primary is not None else None
    fallback_last = _last_session(fallback) if fallback is not None else None
    if primary_last is None:
        if fallback is None or fallback_last is None:
            raise KeyError(f"no price history loaded for {profile.symbol}")
        return str(profile.fallback), fallback
    if fallback is None or fallback_last is None:
        return profile.symbol, primary
    reference = max(primary_last, fallback_last)
    primary_sessions = _recent_sessions(primary, reference, RECENCY_WINDOW)
    if primary_sessions >= _recent_sessions(fallback, reference, RECENCY_WINDOW):
        return profile.symbol, primary
    log.warning(
        "%s printed %d sessions in the last %d days; reading %s instead",
        profile.symbol,
        primary_sessions,
        RECENCY_WINDOW,
        profile.fallback,
    )
    return str(profile.fallback), fallback


def index_metrics(bars: pd.DataFrame, window: int = ACTIVITY_WINDOW) -> dict[str, float]:
    """Headline activity numbers for one index."""
    closes = bars["Close"].dropna()
    returns = log_return(closes).dropna()
    if returns.empty:
        raise ValueError("no usable closes")
    recent = returns.tail(window)
    metrics = {
        "close": float(closes.iloc[-1]),
        "return_1d": _pct(returns.iloc[-1]),
        "return_5d": _pct(returns.tail(5).sum()),
        "return_20d": _pct(returns.tail(20).sum()),
        "volatility_20d": _pct(returns.tail(20).std() * np.sqrt(TRADING_DAYS)),
        "up_days_20d": _pct((returns.tail(20) > 0).mean()),
        "drawdown_from_high": _pct(np.log(closes.iloc[-1] / closes.tail(window).max())),
        "sessions": float(len(returns)),
    }
    if "Open" in bars.columns:
        gap = opening_gap(bars.dropna(subset=["Open", "Close"])).dropna()
        metrics["opening_gap"] = _pct(gap.iloc[-1]) if len(gap) else float("nan")
    volume = _traded_volume(bars)
    if not volume.empty:
        average = float(volume.tail(window).mean())
        metrics["volume"] = float(volume.iloc[-1])
        metrics["volume_vs_average"] = (
            round(float(volume.iloc[-1]) / average, 3) if average else 0.0
        )
        metrics["volume_trend_20d"] = (
            round(float(volume.tail(20).mean()) / average, 3) if average else 0.0
        )
    # `recent` describes the same window the betas below are measured over.
    metrics["volatility_window"] = _pct(recent.std() * np.sqrt(TRADING_DAYS))
    return metrics


def _beta(asset: pd.Series, benchmark: pd.Series, window: int) -> float:
    """Slope of the asset on the index: how much index move a name carries."""
    joined = pd.concat([asset, benchmark], axis=1, join="inner").dropna().tail(window)
    if len(joined) < 20:
        return float("nan")
    x = joined.iloc[:, 1].to_numpy()
    y = joined.iloc[:, 0].to_numpy()
    variance = float(np.var(x))
    if variance <= 0:
        return float("nan")
    return round(float(np.cov(y, x, bias=True)[0, 1] / variance), 3)


def constituent_table(
    profile: IndexProfile,
    panel: dict[str, pd.DataFrame],
    window: int = ACTIVITY_WINDOW,
) -> pd.DataFrame:
    """One row per dominant company, ranked by index weight."""
    index_returns = _returns(index_source(profile, panel)[1])
    rows: list[dict[str, object]] = []
    for member in profile.constituents:
        bars = panel.get(member.symbol)
        if bars is None or bars["Close"].dropna().empty:
            log.warning("no prices for %s (%s)", member.symbol, member.name)
            continue
        rows.append(_constituent_row(member, bars, index_returns, window))
    if not rows:
        raise ValueError(f"{profile.symbol}: no constituent prices available")
    frame = pd.DataFrame(rows)
    turnover = frame["turnover"].where(frame["turnover"] > 0).sum()
    frame["turnover_share"] = (
        (100.0 * frame["turnover"] / turnover).round(2) if turnover else float("nan")
    )
    return frame.sort_values("weight", ascending=False).reset_index(drop=True)


def _constituent_row(
    member: Constituent,
    bars: pd.DataFrame,
    index_returns: pd.Series,
    window: int,
) -> dict[str, object]:
    returns = _returns(bars)
    last = float(returns.iloc[-1]) if len(returns) else float("nan")
    close = float(bars["Close"].dropna().iloc[-1])
    volume = _traded_volume(bars)
    average_volume = float(volume.tail(window).mean()) if len(volume) else float("nan")
    latest_volume = float(volume.iloc[-1]) if len(volume) else float("nan")
    return {
        "symbol": member.symbol,
        "name": member.name,
        "sector": member.sector,
        "weight": member.weight,
        "return_1d": _pct(last),
        "return_5d": _pct(returns.tail(5).sum()) if len(returns) >= 5 else float("nan"),
        "return_20d": _pct(returns.tail(20).sum()) if len(returns) >= 20 else float("nan"),
        # Weight times move: what this name did to the index, in basis points.
        "contribution_bp": round(member.weight * _pct(last), 2),
        "beta_to_index": _beta(returns, index_returns, window),
        "volume_vs_average": (
            round(latest_volume / average_volume, 2) if average_volume > 0 else float("nan")
        ),
        "turnover": close * latest_volume if latest_volume == latest_volume else float("nan"),
    }


def breadth(constituents: pd.DataFrame) -> dict[str, float]:
    """How much of the covered index rose, and what it added up to."""
    weight = constituents["weight"].sum()
    up = constituents.loc[constituents["return_1d"] > 0, "weight"].sum()
    return {
        "weight_covered": round(float(weight), 2),
        "weight_advancing": round(100.0 * float(up) / float(weight), 1) if weight else 0.0,
        "contribution_bp": round(float(constituents["contribution_bp"].sum()), 2),
        "top_name": str(constituents.iloc[0]["name"]) if len(constituents) else "",
        "largest_contributor": str(
            constituents.loc[constituents["contribution_bp"].abs().idxmax(), "name"]
        )
        if len(constituents)
        else "",
    }


def _aligned_driver(
    driver: Influence,
    panel: dict[str, pd.DataFrame],
    profile: IndexProfile,
    dates: pd.DatetimeIndex,
) -> pd.Series | None:
    """Driver return as it was known before ``profile`` opened, per date."""
    bars = panel.get(driver.symbol)
    if bars is None or bars["Close"].dropna().empty:
        return None
    returns = log_return(bars["Close"].dropna())
    lag = lag_days(driver.close_utc, profile.open_utc)
    return as_of(returns, dates, lag)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with an intercept; returns coefficients, t-stats and R²."""
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = len(y) - design.shape[1]
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / total if total > 0 and dof > 0 else float("nan")
    if dof <= 0:
        return coefficients, np.full(design.shape[1], np.nan), r2
    sigma2 = float((residuals**2).sum()) / dof
    try:
        covariance = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return coefficients, np.full(design.shape[1], np.nan), r2
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(errors > 0, coefficients / errors, np.nan)
    return coefficients, t_stats, r2


def driver_table(
    profile: IndexProfile,
    panel: dict[str, pd.DataFrame],
    influences: tuple[Influence, ...] = INFLUENCES,
    window: int = REGRESSION_WINDOW,
) -> pd.DataFrame:
    """Univariate regression of the index return on each outside driver."""
    source, bars = index_source(profile, panel)
    target = _returns(bars).tail(window)
    dates = pd.DatetimeIndex(target.index)
    rows: list[dict[str, object]] = []
    for driver in influences:
        if driver.symbol in (profile.symbol, source):
            continue
        aligned = _aligned_driver(driver, panel, profile, dates)
        if aligned is None:
            continue
        pair = pd.concat([target.rename("y"), aligned.rename("x")], axis=1).dropna()
        if len(pair) < MIN_REGRESSION_ROWS:
            continue
        x = pair["x"].to_numpy()
        y = pair["y"].to_numpy()
        coefficients, t_stats, r2 = _ols(x[:, None], y)
        last_move = float(pair["x"].iloc[-1])
        rows.append(
            {
                "driver": driver.name,
                "symbol": driver.symbol,
                "theme": driver.theme,
                "lag_days": lag_days(driver.close_utc, profile.open_utc),
                "beta": round(float(coefficients[1]), 3),
                "t_stat": round(float(t_stats[1]), 2),
                "r2": round(float(r2), 4),
                "correlation": round(float(np.corrcoef(x, y)[0, 1]), 3),
                "observations": len(pair),
                "last_move": _pct(last_move),
                # What that beta implies the last driver move was worth.
                "implied_bp": round(float(coefficients[1]) * last_move * 10000.0, 1),
            }
        )
    if not rows:
        raise ValueError(f"{profile.symbol}: no driver had enough overlapping history")
    return pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False).reset_index(drop=True)


def theme_table(
    profile: IndexProfile,
    panel: dict[str, pd.DataFrame],
    influences: tuple[Influence, ...] = INFLUENCES,
    window: int = REGRESSION_WINDOW,
) -> pd.DataFrame:
    """Joint explanatory power of each theme, and of all themes together."""
    source, bars = index_source(profile, panel)
    target = _returns(bars).tail(window)
    dates = pd.DatetimeIndex(target.index)
    columns: dict[str, pd.Series] = {}
    themes: dict[str, list[str]] = {}
    for driver in influences:
        if driver.symbol in (profile.symbol, source):
            continue
        aligned = _aligned_driver(driver, panel, profile, dates)
        if aligned is None:
            continue
        columns[driver.symbol] = aligned
        themes.setdefault(driver.theme, []).append(driver.symbol)
    if not columns:
        raise ValueError(f"{profile.symbol}: no drivers available")
    frame = pd.concat([target.rename("y"), pd.DataFrame(columns, index=dates)], axis=1).dropna()
    if len(frame) < MIN_REGRESSION_ROWS:
        raise ValueError(f"{profile.symbol}: only {len(frame)} rows overlap every driver")
    y = frame["y"].to_numpy()
    rows = []
    for theme in THEMES:
        symbols = themes.get(theme)
        if not symbols:
            continue
        _, _, r2 = _ols(frame[symbols].to_numpy(), y)
        rows.append({"theme": theme, "drivers": len(symbols), "r2": round(float(r2), 4)})
    _, _, joint = _ols(frame[list(columns)].to_numpy(), y)
    rows.append({"theme": "All themes", "drivers": len(columns), "r2": round(float(joint), 4)})
    frame_out = pd.DataFrame(rows)
    frame_out["observations"] = len(frame)
    return frame_out


def snapshot(
    profile: IndexProfile,
    panel: dict[str, pd.DataFrame],
    window: int = ACTIVITY_WINDOW,
    regression_window: int = REGRESSION_WINDOW,
    influences: tuple[Influence, ...] = INFLUENCES,
) -> IndexSnapshot:
    source, bars = index_source(profile, panel)
    metrics = index_metrics(bars, window)
    constituents = constituent_table(profile, panel, window)
    metrics.update(breadth(constituents))
    return IndexSnapshot(
        profile=profile,
        source=source,
        session=pd.Timestamp(bars["Close"].dropna().index[-1]),
        metrics=metrics,
        constituents=constituents,
        drivers=driver_table(profile, panel, influences, regression_window),
        themes=theme_table(profile, panel, influences, regression_window),
    )


def build_dashboard(
    panel: dict[str, pd.DataFrame],
    window: int = ACTIVITY_WINDOW,
    regression_window: int = REGRESSION_WINDOW,
    influences: tuple[Influence, ...] = INFLUENCES,
) -> Dashboard:
    """Snapshot every Asian index, then the European indices behind them."""
    dashboard = Dashboard(generated=pd.Timestamp.now("UTC").floor("s"))
    for group, target in ((ASIA_INDICES, dashboard.asia), (EUROPE_INDICES, dashboard.europe)):
        for profile in group:
            try:
                target.append(snapshot(profile, panel, window, regression_window, influences))
            except Exception as exc:
                log.warning("no snapshot for %s: %s", profile.symbol, exc)
    if not dashboard.asia:
        raise RuntimeError("no Asian index could be evaluated")
    return dashboard
