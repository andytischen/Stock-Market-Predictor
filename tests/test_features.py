import numpy as np
import pandas as pd
import pytest

from gapmodel.features import (
    _column_name,
    _lag_days,
    as_of,
    build_features,
    feature_symbols,
    opening_gap,
)
from gapmodel.markets import INDICATORS, MARKETS, SECTOR_SYMBOLS, all_symbols, market
from gapmodel.model import walk_forward


def test_the_dow_is_modelled_alongside_the_other_wall_street_indices():
    dow = market("^DJI")
    assert dow.region == "Americas"
    # It opens and closes with the S&P, so the two read the same indicator lags.
    spx = market("^GSPC")
    assert (dow.open_utc, dow.close_utc) == (spx.open_utc, spx.close_utc)
    assert "^DJI" in all_symbols()


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
        for seed, symbol in enumerate(
            [
                "^GSPC",
                "^N225",
                "^FTSE",
                "^GDAXI",
                "^VIX",
                "ES=F",
                "CL=F",
                "JPY=X",
                "KRW=X",
                "EXH8.DE",
                "EXV3.DE",
            ]
        )
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


def test_sectors_carry_a_weekly_return_but_no_shock(panel):
    features, _ = build_features("^GDAXI", panel)
    assert {"ind_exh8_de_return", "ind_exh8_de_return_5"} <= set(features.columns)
    assert not any(col.startswith("ind_exh8_de_vol") for col in features.columns)
    assert "ind_exh8_de_shock" not in features.columns
    # Xetra closes after every tracked market opens, so it is always read late.
    retail = next(i for i in INDICATORS if i.symbol == "EXH8.DE")
    assert all(_lag_days(retail.close_utc, market(m.symbol)) == 1 for m in MARKETS)


def test_sector_features_reach_european_markets_only(panel):
    european, _ = build_features("^GDAXI", panel)
    overseas, _ = build_features("^GSPC", panel)
    sectors = {f"ind_{_column_name(s)}_return" for s in SECTOR_SYMBOLS}
    assert sectors & set(european.columns)
    assert not sectors & set(overseas.columns)


@pytest.mark.parametrize("target", ["^GSPC", "^GDAXI"])
def test_the_named_inputs_are_the_ones_the_model_reads(panel, target):
    """What the staleness guard is entitled to refuse a run over.

    Both directions matter: a series left out must make no difference to the
    design matrix, and a series named must make one, or the guard is judging a
    run on a feed it never reads.
    """
    named = feature_symbols(target)
    whole, _ = build_features(target, panel)
    restricted, _ = build_features(target, {s: b for s, b in panel.items() if s in named})
    pd.testing.assert_frame_equal(whole, restricted)
    for symbol in (set(panel) & named) - {target}:
        without, _ = build_features(target, {s: b for s, b in panel.items() if s != symbol})
        assert set(without.columns) < set(whole.columns), f"{symbol} is named but unread"


def test_a_sector_tracker_is_an_input_in_europe_and_not_elsewhere():
    """The asymmetry `build_features` applies, in the form a guard can check."""
    assert "EXH8.DE" in feature_symbols("^GDAXI")
    assert "EXH8.DE" not in feature_symbols("^GSPC")
    # A single name is given Wall Street's clock and reads what a US index does.
    assert "EXH8.DE" not in feature_symbols("MU")


def test_an_opening_stand_in_is_an_input_to_its_own_index_only():
    """`ISF.L` is read as the FTSE's opening auction, and by nothing else."""
    assert market("^FTSE").gap_symbol == "ISF.L"
    assert "ISF.L" in feature_symbols("^FTSE")
    assert "ISF.L" not in feature_symbols("^GSPC")


def test_a_peer_is_an_input_to_the_names_it_leads(panel):
    """A memory name reads Samsung's session; an index does not."""
    assert "005930.KS" in feature_symbols("MU")
    assert "005930.KS" not in feature_symbols("^GSPC")


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


def test_calibrating_forward_never_reads_the_session_it_maps():
    from gapmodel.model import Backtest, calibrated

    index = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(1)
    # An overconfident record: the model speaks in near-certainties and is right
    # about two thirds of the time.
    raw = pd.Series(rng.choice([0.004, 0.996], len(index)), index=index)
    outcomes = pd.Series(
        [int(rng.random() < (0.66 if p > 0.5 else 0.34)) for p in raw], index=index
    )

    record = calibrated(Backtest(probabilities=raw, outcomes=outcomes), min_history=250, step=21)

    # The first block is dropped: nothing had been earned to calibrate it with.
    assert record.probabilities.index[0] == index[250]
    assert record.probabilities.between(0.05, 0.95).all()
    assert (
        record.metrics["brier"]
        < Backtest(
            probabilities=raw.loc[record.probabilities.index],
            outcomes=outcomes.loc[record.probabilities.index],
        ).window_metrics()["brier"]
    )

    # Flipping outcomes after a session cannot change the probability shown for
    # it: its calibration was fitted on the sessions before it and nothing else.
    tampered = outcomes.copy()
    tampered.iloc[300:] = 1 - tampered.iloc[300:]
    later = calibrated(Backtest(probabilities=raw, outcomes=tampered), min_history=250, step=21)
    assert later.probabilities.iloc[:50].equals(record.probabilities.iloc[:50])


def test_a_short_record_is_returned_uncalibrated():
    from gapmodel.model import Backtest, calibrated

    index = pd.bdate_range("2020-01-01", periods=40)
    raw = pd.Series(np.linspace(0.1, 0.9, len(index)), index=index)
    outcomes = pd.Series([i % 2 for i in range(len(index))], index=index)
    record = Backtest(probabilities=raw, outcomes=outcomes)

    assert calibrated(record, min_history=250) is record


def test_sector_trackers_are_read_on_a_total_return_basis(panel):
    """An ex-distribution print is not eighteen sectors selling off."""
    tracker = panel["EXV3.DE"].copy()
    session = tracker.index[700]
    # The fund goes ex a 2% distribution: the price drops, the factor does not.
    tracker["Adj Close"] = tracker["Close"]
    tracker.loc[session:, "Close"] *= 0.98
    tracker.loc[session:, "Adj Close"] = tracker.loc[session:, "Close"] / 0.98
    europe = dict(panel, **{"EXV3.DE": tracker})

    corrected, _ = build_features("^GDAXI", europe)
    raw = dict(panel, **{"EXV3.DE": tracker.drop(columns=["Adj Close"])})
    uncorrected, _ = build_features("^GDAXI", raw)

    column = "ind_exv3_de_return"
    read_on = corrected.index[corrected.index >= session]
    ex_day = (uncorrected.loc[read_on, column] - corrected.loc[read_on, column]).idxmin()
    # Only the ex-distribution session differs, and there by the distribution.
    assert uncorrected.loc[ex_day, column] - corrected.loc[ex_day, column] == pytest.approx(
        np.log(0.98), abs=1e-9
    )
    others = corrected.loc[read_on, column].drop(ex_day)
    assert others.to_numpy() == pytest.approx(
        uncorrected.loc[others.index, column].to_numpy(), abs=1e-12
    )


def test_a_cross_market_move_is_read_in_deviations_of_its_own_regime(panel):
    """The same 6% session is a shock in a calm month and routine in a wild one."""
    from gapmodel.features import MKT_SHOCK_CLIP

    features, _ = build_features("^GDAXI", panel)
    shock = features["mkt_n225_shock"]
    assert shock.abs().max() <= MKT_SHOCK_CLIP + 1e-12

    nikkei = panel["^N225"].copy()
    calm, wild = nikkei.index[300], nikkei.index[700]
    # Ten times the volatility for the run-up to the second session, the same
    # move on the day itself.
    close = nikkei["Close"].astype(float)
    steps = np.log(close).diff()
    steps.iloc[601:700] *= 10
    steps.iloc[300] = steps.iloc[700] = np.log(1.06)
    nikkei["Close"] = np.exp(np.log(close.iloc[0]) + steps.fillna(0.0).cumsum())
    regimes, _ = build_features("^GDAXI", dict(panel, **{"^N225": nikkei}))

    read = regimes["mkt_n225_shock"]
    # The identical 6% session reads as a large shock in the calm regime and a
    # small one after a hundred sessions of ten-fold volatility.
    assert read.loc[calm] > 3.0
    assert 0.0 < read.loc[wild] < 1.0
