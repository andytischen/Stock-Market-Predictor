import numpy as np
import pandas as pd
import pytest

from gapmodel.features import _as_of, _lag_days, build_features, opening_gap
from gapmodel.markets import market
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
        for seed, symbol in enumerate(["^GSPC", "^N225", "^FTSE", "^VIX", "ES=F"])
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


def test_as_of_never_reads_the_future():
    dates = pd.bdate_range("2020-01-01", periods=5)
    source = pd.Series(range(5), index=dates, dtype=float)
    same_day = _as_of(source, dates, lag_days=0)
    previous = _as_of(source, dates, lag_days=1)
    assert list(same_day) == [0, 1, 2, 3, 4]
    assert previous.iloc[-1] == 3  # yesterday's value, not today's


def test_build_features_is_aligned_and_finite(panel):
    features, labels = build_features("^GSPC", panel)
    assert features.notna().all().all()
    assert set(labels.dropna().unique()) <= {0.0, 1.0}
    assert features.index.equals(labels.index)
    assert any(col.startswith("mkt_") for col in features.columns)
    assert any(col.startswith("ind_") for col in features.columns)


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


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        market("^NOPE")
