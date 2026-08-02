import numpy as np
import pandas as pd
import pytest

from gapmodel.features import _lag_days, as_of, build_features, opening_gap
from gapmodel.markets import INDICATORS, MARKETS, all_symbols, market
from gapmodel.model import walk_forward


def synthetic_bars(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
        },
        index=dates,
    )


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    return {
        symbol: synthetic_bars(seed=seed)
        for seed, symbol in enumerate(["^GSPC", "^N225", "^FTSE", "^VIX", "ES=F", "CL=F", "JPY=X", "KRW=X"])
    }


def test_opening_gap_uses_previous_close():
    bars = pd.DataFrame(
        {"Open": [10.0, 11.0], "Close": [10.0, 12.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    gap = opening_gap(bars)
    assert np.isnan(gap.iloc[0])
    assert gap.iloc[1] == pytest.approx(np.log(11.0 / 10.0))


def test_lag_days_respects_session_order():
    # Tokyo closes at 06:00 UTC, before Europe opens at 07:00 -> same-day usable.
    assert _lag_days(6.0, market("^GDAXI")) == 0
    # Wall Street closes at 20:00 UTC, after Tokyo's 00:00 open -> previous day.
    assert _lag_days(20.0, market("^N225")) == 1


def test_asml_is_a_lagged_indicator():
    asml = next(i for i in INDICATORS if i.symbol == "ASML.AS")
    assert asml.close_utc == 15.5
    assert "ASML.AS" in all_symbols()
    # Amsterdam closes after every tracked market opens, so its close is always
    # read a session late -- never same-day.
    assert all(_lag_days(asml.close_utc, market(m.symbol)) == 1 for m in MARKETS)


def test_as_of_never_reads_the_future():
    dates = pd.bdate_range("2020-01-01", periods=5)
    source = pd.Series(range(5), index=dates, dtype=float)
    same_day = as_of(source, dates, lag_days=0)
    previous = as_of(source, dates, lag_days=1)
    assert list(same_day) == [0, 1, 2, 3, 4]
    assert previous.iloc[-1] == 3  # yesterday's value, not today's


def test_build_features_is_aligned_and_finite(panel):
    features, labels = build_features("^GSPC", panel)
    assert features.notna().all().all()
    assert set(labels.dropna().unique()) <= {0.0, 1.0}
    assert features.index.equals(labels.index)
    assert any(col.startswith("mkt_") for col in features.columns)
    assert any(col.startswith("ind_") for col in features.columns)


def test_oil_carries_shock_features(panel):
    features, _ = build_features("^GSPC", panel)
    assert {
        "ind_cl_f_return",
        "ind_cl_f_return_5",
        "ind_cl_f_vol_20",
        "ind_cl_f_shock",
    } <= set(features.columns)
    assert (features["ind_cl_f_vol_20"] > 0).all()


def test_oil_shock_is_the_move_scaled_by_known_volatility(panel):
    close = panel["CL=F"]["Close"]
    returns = np.log(close / close.shift(1))
    vol = returns.rolling(20).std().shift(1)
    features, _ = build_features("^GSPC", panel)
    # Wall Street opens at 13:30 UTC, before crude's 21:00 close -> yesterday's bar,
    # carried forward over the weekend like every other as-of lookup.
    shock = returns / vol
    calendar = pd.date_range(shock.index.min(), shock.index.max())
    expected = shock.reindex(calendar).ffill().reindex(features.index - pd.Timedelta(days=1))
    assert features["ind_cl_f_shock"].to_numpy() == pytest.approx(expected.to_numpy())


def test_fx_carries_shock_features(panel):
    features, _ = build_features("^GSPC", panel)
    # USD/JPY is in the panel; expect the same shock set as oil.
    assert {
        "ind_jpy_x_return",
        "ind_jpy_x_return_5",
        "ind_jpy_x_vol_20",
        "ind_jpy_x_shock",
    } <= set(features.columns)
    assert (features["ind_jpy_x_vol_20"] > 0).all()


def test_fx_shock_is_the_move_scaled_by_known_volatility(panel):
    close = panel["JPY=X"]["Close"]
    returns = np.log(close / close.shift(1))
    vol = returns.rolling(20).std().shift(1)
    features, _ = build_features("^GSPC", panel)
    # Wall Street opens at 13:30 UTC, before JPY=X's 21:00 close -> yesterday's bar.
    shock = returns / vol
    calendar = pd.date_range(shock.index.min(), shock.index.max())
    expected = shock.reindex(calendar).ffill().reindex(features.index - pd.Timedelta(days=1))
    assert features["ind_jpy_x_shock"].to_numpy() == pytest.approx(expected.to_numpy())


def test_krw_carries_shock_features(panel):
    features, _ = build_features("^GSPC", panel)
    # USD/KRW closes at 06:30 UTC (Seoul close), before the US open at 13:30 UTC,
    # so it is a same-day indicator for European and US markets.
    assert {
        "ind_krw_x_return",
        "ind_krw_x_return_5",
        "ind_krw_x_vol_20",
        "ind_krw_x_shock",
    } <= set(features.columns)
    assert (features["ind_krw_x_vol_20"] > 0).all()


def test_forecast_row_is_unlabelled_and_last(panel):
    features, labels = build_features("^GSPC", panel, forecast_row=True)
    assert np.isnan(labels.iloc[-1])
    assert labels.iloc[:-1].notna().all()
    assert features.index[-1] > features.index[-2]


def test_walk_forward_is_out_of_sample_and_calibratable(panel):
    features, labels = build_features("^GSPC", panel)
    result = walk_forward(features, labels, min_train=400, refit_every=50)
    assert result.probabilities.between(0, 1).all()
    assert result.probabilities.index[0] == features.index[400]
    assert set(result.metrics) >= {"auc", "brier", "brier_skill", "accuracy"}
    assert not result.reliability().empty


def test_window_metrics_restricts_to_date_range(panel):
    from gapmodel.model import Backtest

    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(42)
    probabilities = pd.Series(rng.uniform(0.3, 0.7, len(index)), index=index)
    outcomes = pd.Series(rng.integers(0, 2, len(index)).astype(float), index=index)
    bt = Backtest(probabilities=probabilities, outcomes=outcomes)

    since = pd.Timestamp("2021-01-01")
    wm = bt.window_metrics(since=since)
    # The windowed count must be less than the full series length.
    assert int(wm["n"]) < len(probabilities)
    # All sessions in the window are on or after the cutoff.
    assert probabilities.loc[probabilities.index >= since].shape[0] == int(wm["n"])


def test_window_metrics_raises_when_window_is_empty(panel):
    from gapmodel.model import Backtest

    index = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    probabilities = pd.Series(rng.uniform(0.3, 0.7, len(index)), index=index)
    outcomes = pd.Series(rng.integers(0, 2, len(index)).astype(float), index=index)
    bt = Backtest(probabilities=probabilities, outcomes=outcomes)

    with pytest.raises(ValueError, match="no out-of-sample predictions"):
        bt.window_metrics(since=pd.Timestamp("2099-01-01"))


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        market("^NOPE")


def test_calibration_pulls_overconfident_probabilities_back():
    from gapmodel.model import Backtest, calibrator

    index = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    # Out of sample the model never went past 0.8 and was right 70% of the time.
    probabilities = pd.Series(rng.uniform(0.2, 0.8, len(index)), index=index)
    outcomes = pd.Series(rng.binomial(1, probabilities), index=index)

    calibrate = calibrator(Backtest(probabilities=probabilities, outcomes=outcomes))
    assert calibrate(np.array([0.9999]))[0] < 0.9
    # Probabilities inside the earned range are left broadly alone.
    assert abs(calibrate(np.array([0.6]))[0] - 0.6) < 0.1
