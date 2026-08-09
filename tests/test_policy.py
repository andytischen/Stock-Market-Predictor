import pandas as pd
import pytest

from gapmodel.features import build_features, policy_features
from gapmodel.markets import BILL_YIELD, FUNDS_FUTURE, all_symbols, market
from tests.test_features import synthetic_bars


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    built = {
        symbol: synthetic_bars(seed=seed)
        for seed, symbol in enumerate(
            ["^GSPC", "^N225", "^VIX", "ES=F", "CL=F", "JPY=X", "KRW=X", BILL_YIELD]
        )
    }
    # A funds future trades a shade under 100, unlike a synthetic equity series.
    future = synthetic_bars(seed=99)
    built[FUNDS_FUTURE] = future / future["Close"].iloc[0] * 96.5
    return built


def test_the_policy_symbols_are_downloaded():
    assert {FUNDS_FUTURE, BILL_YIELD} <= set(all_symbols())


def test_the_priced_rate_is_a_hundred_minus_the_future(panel):
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    built = policy_features(panel, dates, market("^GSPC"))
    # Wall Street opens before the funds future settles, so it reads yesterday.
    expected = 100.0 - panel[FUNDS_FUTURE]["Close"].iloc[-2]
    assert built["ind_policy_rate"].iloc[-1] == pytest.approx(expected)


def test_tokyo_reads_the_same_bar_as_wall_street_does(panel):
    """Both open before the future settles, so neither can use today's price."""
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    priced = 100.0 - panel[FUNDS_FUTURE]["Close"]
    for symbol in ("^GSPC", "^N225"):
        built = policy_features(panel, dates, market(symbol))
        assert built["ind_policy_rate"].iloc[-1] == pytest.approx(priced.iloc[-2])


def test_tightening_is_the_bill_over_the_priced_rate(panel):
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    built = policy_features(panel, dates, market("^GSPC"))
    priced = 100.0 - panel[FUNDS_FUTURE]["Close"].iloc[-2]
    bill = panel[BILL_YIELD]["Close"].iloc[-2]
    assert built["ind_policy_tightening_3m"].iloc[-1] == pytest.approx(bill - priced)


def test_a_zero_bill_yield_still_produces_a_spread(panel):
    """Bill yields have sat at zero, where a log return would be undefined."""
    panel[BILL_YIELD] = panel[BILL_YIELD].assign(Close=0.0)
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    built = policy_features(panel, dates, market("^GSPC"))
    priced = 100.0 - panel[FUNDS_FUTURE]["Close"].iloc[-2]
    assert built["ind_policy_tightening_3m"].iloc[-1] == pytest.approx(-priced)
    # Only the first session is blank, for want of a bar before the series began.
    assert built["ind_policy_tightening_3m"].iloc[1:].notna().all()


def test_only_the_two_levels_are_built(panel):
    """The change features were measured as inert and removed; stay removed."""
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    built = policy_features(panel, dates, market("^GSPC"))
    assert set(built) == {"ind_policy_rate", "ind_policy_tightening_3m"}


def test_policy_features_are_absent_without_both_legs(panel):
    del panel[BILL_YIELD]
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    assert policy_features(panel, dates, market("^GSPC")) == {}
    features, _ = build_features("^GSPC", panel)
    assert not any(column.startswith("ind_policy") for column in features.columns)


def test_the_model_uses_the_policy_features_when_they_are_there(panel):
    features, _ = build_features("^GSPC", panel)
    assert "ind_policy_rate" in features.columns
    assert "ind_policy_tightening_3m" in features.columns
    assert features["ind_policy_rate"].notna().all()
