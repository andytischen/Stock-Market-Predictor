import numpy as np
import pandas as pd
import pytest

from gapmodel.dashboard import (
    breadth,
    build_dashboard,
    constituent_table,
    driver_table,
    index_metrics,
    index_source,
    snapshot,
    theme_table,
)
from gapmodel.regions import (
    ASIA_INDICES,
    EUROPE_INDICES,
    INFLUENCES,
    Constituent,
    IndexProfile,
    Influence,
    all_profiles,
    dashboard_symbols,
    profile,
)
from gapmodel.report import render_html, render_text

TOKYO = IndexProfile(
    symbol="^TEST",
    name="Test index",
    country="Testland",
    currency="TST",
    open_utc=0.0,
    close_utc=6.0,
    constituents=(
        Constituent("BIG.T", "Big Co", 20.0, "Technology"),
        Constituent("SMALL.T", "Small Co", 5.0, "Banks"),
    ),
)

DRIVER = Influence("^WALLST", "Wall Street", "Global", close_utc=20.0)
LOCAL = Influence("^LOCAL", "Local neighbour", "India", close_utc=-1.0)


def bars(
    n: int = 400,
    seed: int = 0,
    drift: float = 0.0,
    volume: float = 1e6,
    returns: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    steps = rng.normal(drift, 0.01, n) if returns is None else returns
    close = 100 * np.exp(np.cumsum(steps))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
            "Volume": np.full(n, volume) * rng.uniform(0.5, 1.5, n),
        },
        index=dates,
    )


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 400
    driver_returns = rng.normal(0, 0.01, n)
    # The index follows the driver's *previous* session, which is what the lag
    # rule is supposed to recover.
    index_returns = 0.6 * np.roll(driver_returns, 1) + rng.normal(0, 0.004, n)
    index_returns[0] = 0.0
    return {
        "^TEST": bars(n, seed=1, returns=index_returns),
        "BIG.T": bars(n, seed=2),
        "SMALL.T": bars(n, seed=3),
        "^WALLST": bars(n, seed=4, returns=driver_returns),
        "^LOCAL": bars(n, seed=5),
    }


def test_registry_is_consistent():
    for candidate in all_profiles():
        assert candidate.constituents
        assert candidate.weight_covered <= 100.0
        assert profile(candidate.symbol) is candidate
    symbols = dashboard_symbols()
    assert len(symbols) == len(set(symbols))
    assert {p.symbol for p in ASIA_INDICES + EUROPE_INDICES} <= set(symbols)
    assert {i.symbol for i in INFLUENCES} <= set(symbols)


def test_weights_above_the_index_are_rejected():
    with pytest.raises(ValueError):
        IndexProfile(
            symbol="^BAD",
            name="Bad",
            country="Nowhere",
            currency="XXX",
            open_utc=0.0,
            close_utc=6.0,
            constituents=(Constituent("A", "A", 60.0, "X"), Constituent("B", "B", 60.0, "X")),
        )


def test_index_metrics_report_the_last_session(panel):
    metrics = index_metrics(panel["^TEST"])
    closes = panel["^TEST"]["Close"]
    expected = 100.0 * np.log(closes.iloc[-1] / closes.iloc[-2])
    assert metrics["return_1d"] == pytest.approx(expected, abs=1e-3)
    assert metrics["volatility_20d"] > 0
    assert metrics["volume_vs_average"] > 0


def test_contribution_is_weight_times_move(panel):
    table = constituent_table(TOKYO, panel)
    row = table.loc[table["symbol"] == "BIG.T"].iloc[0]
    assert row["contribution_bp"] == pytest.approx(row["weight"] * row["return_1d"], abs=1e-6)
    assert table["weight"].is_monotonic_decreasing
    summary = breadth(table)
    assert summary["weight_covered"] == pytest.approx(25.0)
    assert 0.0 <= summary["weight_advancing"] <= 100.0


def test_beta_measures_index_sensitivity(panel):
    doubled = panel["^TEST"].copy()
    returns = np.log(doubled["Close"]).diff().fillna(0.0)
    doubled["Close"] = 100 * np.exp(np.cumsum(2 * returns))
    doubled["Open"] = doubled["Close"]
    levered = dict(panel, **{"BIG.T": doubled})
    table = constituent_table(TOKYO, levered)
    assert table.loc[table["symbol"] == "BIG.T", "beta_to_index"].iloc[0] == pytest.approx(
        2.0, abs=0.05
    )


def test_driver_regression_finds_the_lagged_leader(panel):
    table = driver_table(TOKYO, panel, influences=(DRIVER, LOCAL))
    wall_street = table.loc[table["symbol"] == "^WALLST"].iloc[0]
    # Wall Street closes after Tokyo opens, so it may only be read a day late.
    assert wall_street["lag_days"] == 1
    assert wall_street["beta"] == pytest.approx(0.6, abs=0.1)
    assert wall_street["t_stat"] > 5
    unrelated = table.loc[table["symbol"] == "^LOCAL"].iloc[0]
    assert unrelated["lag_days"] == 0
    assert abs(unrelated["t_stat"]) < abs(wall_street["t_stat"])


def test_theme_table_totals_at_least_each_theme(panel):
    table = theme_table(TOKYO, panel, influences=(DRIVER, LOCAL))
    joint = table.loc[table["theme"] == "All themes", "r2"].iloc[0]
    for theme in ("Global", "India"):
        assert joint >= table.loc[table["theme"] == theme, "r2"].iloc[0] - 1e-9


def test_fallback_series_is_used_when_the_headline_goes_quiet(panel):
    quiet = IndexProfile(
        symbol="^TEST",
        name="Test index",
        country="Testland",
        currency="TST",
        open_utc=0.0,
        close_utc=6.0,
        constituents=TOKYO.constituents,
        fallback="^LOCAL",
    )
    stale = dict(panel, **{"^TEST": panel["^TEST"].iloc[:-90]})
    assert index_source(quiet, panel)[0] == "^TEST"
    assert index_source(quiet, stale)[0] == "^LOCAL"


def test_zero_volume_is_not_counted_as_a_session(panel):
    blank = panel["^TEST"].copy()
    blank["Volume"] = 0.0
    metrics = index_metrics(blank)
    assert "volume_vs_average" not in metrics


def test_snapshot_and_rendering(panel):
    from gapmodel import dashboard as dashboard_mod

    single = dashboard_mod.snapshot(TOKYO, panel, influences=(DRIVER, LOCAL))
    assert single.source == "^TEST"
    assert set(single.constituents["symbol"]) == {"BIG.T", "SMALL.T"}

    built = dashboard_mod.Dashboard(generated=pd.Timestamp("2026-01-01"), asia=[single])
    text = render_text(built)
    assert "Test index" in text and "Big Co" in text
    html = render_html(built)
    assert html.startswith("<!doctype html>")
    assert "Test index" in html and "<table>" in html
    assert built.theme_matrix().loc[0, "index"] == "Test index"


def test_build_dashboard_needs_an_asian_index():
    with pytest.raises(RuntimeError):
        build_dashboard({"BIG.T": bars(200)})


def test_snapshot_is_importable_from_the_package():
    assert callable(snapshot)
