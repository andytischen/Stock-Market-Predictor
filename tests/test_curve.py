import math

import numpy as np
import pandas as pd
import pytest

from gapmodel.cli import build_parser
from gapmodel.features import build_features, curve_features
from gapmodel.markets import CURVE_FRONT, CURVE_STRIP, CURVE_WINDOW, all_symbols, market
from gapmodel.predict import shocked_row
from tests.test_features import synthetic_bars


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    return {
        symbol: synthetic_bars(seed=seed)
        for seed, symbol in enumerate(
            ["^GSPC", "^N225", "^VIX", "ES=F", "CL=F", "JPY=X", "KRW=X", CURVE_FRONT, CURVE_STRIP]
        )
    }


def test_curve_symbols_are_downloaded():
    assert {CURVE_FRONT, CURVE_STRIP} <= set(all_symbols())


def test_curve_is_the_difference_between_the_two_legs(panel):
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    built = curve_features(panel, dates, market("^GSPC"))
    front = panel[CURVE_FRONT]["Close"]
    strip = panel[CURVE_STRIP]["Close"]
    expected = np.log(front / front.shift(1)) - np.log(strip / strip.shift(1))
    # Wall Street opens before the funds close, so it reads yesterday's bar.
    assert built["ind_oil_curve_return"].iloc[-1] == pytest.approx(expected.iloc[-2])
    assert f"ind_oil_curve_slope_{CURVE_WINDOW}" in built


def test_curve_features_are_absent_without_both_legs(panel):
    del panel[CURVE_STRIP]
    dates = pd.DatetimeIndex(panel["^GSPC"].index)
    assert curve_features(panel, dates, market("^GSPC")) == {}
    features, _ = build_features("^GSPC", panel)
    assert not any(col.startswith("ind_oil_curve") for col in features.columns)


def test_shocking_a_leg_tilts_the_curve_in_its_own_direction():
    live = pd.DataFrame(
        {"ind_oil_curve_return": [0.0], f"ind_oil_curve_slope_{CURVE_WINDOW}": [0.02]}
    )
    front = shocked_row(live, {CURVE_FRONT: 0.03})
    strip = shocked_row(live, {CURVE_STRIP: 0.03})
    assert front["ind_oil_curve_return"].iloc[0] == pytest.approx(0.03)
    assert strip["ind_oil_curve_return"].iloc[0] == pytest.approx(-0.03)
    assert front[f"ind_oil_curve_slope_{CURVE_WINDOW}"].iloc[0] == pytest.approx(0.05)


def test_curve_legs_are_shockable_from_the_command_line():
    parsed = build_parser().parse_args(["predict", "--shock", f"{CURVE_FRONT}=-2%"])
    assert parsed.shock == [(CURVE_FRONT, pytest.approx(math.log(0.98)))]
