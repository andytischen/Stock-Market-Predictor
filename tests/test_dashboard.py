import numpy as np
import pandas as pd
import pytest

from gapmodel.dashboard import (
    build_dashboard,
    oil_readings,
    render_html,
    render_text,
    session_state,
)
from gapmodel.markets import MARKETS_BY_SYMBOL
from gapmodel.predict import Forecast

FIVE_AM = pd.Timestamp("2026-08-03 05:00")


def bars(n=120, start=100.0, step=0.5):
    index = pd.bdate_range("2026-01-01", periods=n)
    close = pd.Series(start + step * np.arange(n), index=index, dtype=float)
    return pd.DataFrame({"Open": close.shift(1).fillna(start), "Close": close})


@pytest.fixture
def panel():
    return {
        "^N225": bars(),
        "^HSI": bars(start=200.0, step=-0.3),
        "^GSPC": bars(start=50.0),
        "CL=F": bars(start=80.0, step=0.1),
        "BZ=F": bars(start=84.0, step=0.1),
    }


def forecast(symbol, probability, oil_weight):
    meta = MARKETS_BY_SYMBOL[symbol]
    contributions = pd.Series(
        {
            "ind_cl_f_shock": oil_weight,
            "ind_bz_f_return": oil_weight / 2,
            "mkt_gspc_return": 0.4,
        }
    )
    return Forecast(
        symbol=symbol,
        name=meta.name,
        region=meta.region,
        session=pd.Timestamp("2026-08-04"),
        probability_up=probability,
        backtest={"auc": 0.6},
        contributions=contributions,
    )


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("^N225", "open"),  # 00:00-06:00 UTC
        ("^NSEI", "open"),  # opened 03:45 UTC
        ("^STOXX50E", "pre-open"),  # opens 07:00 UTC
        ("^GSPC", "closed"),  # opens 13:30 UTC
    ],
)
def test_session_state_at_five_am_utc(symbol, expected):
    meta = MARKETS_BY_SYMBOL[symbol]
    state, _ = session_state(meta.open_utc, meta.close_utc, 5.0)
    assert state == expected


def test_a_session_running_through_midnight_is_open():
    state, hours = session_state(-1.0, 5.0, 2.0)  # Sydney: 23:00 -> 05:00 UTC
    assert state == "open" and hours == 0.0


def test_a_session_about_to_start_is_pre_open():
    state, hours = session_state(7.0, 15.5, 5.0)  # Europe seen at 05:00
    assert state == "pre-open" and hours == 2.0


def test_hours_to_open_wrap_around_the_clock():
    state, hours = session_state(1.5, 8.0, 23.0)  # Hong Kong seen late the evening before
    assert state == "pre-open" and hours == 2.5


def test_oil_readings_use_the_previously_known_volatility(panel):
    readings = oil_readings(panel)
    assert [r.symbol for r in readings] == ["BZ=F", "CL=F"]
    wti = next(r for r in readings if r.symbol == "CL=F")
    assert wti.return_1d > 0 and wti.direction == "up"
    assert wti.volatility_20d > 0
    assert wti.close == pytest.approx(panel["CL=F"]["Close"].iloc[-1])


def test_oil_readings_skip_a_benchmark_without_enough_history(panel):
    panel["BZ=F"] = bars(n=5)
    assert [r.symbol for r in oil_readings(panel)] == ["CL=F"]


def test_dashboard_only_covers_the_requested_region(panel):
    board = build_dashboard(panel, [forecast("^N225", 0.7, 0.5)], as_of=FIVE_AM, region="Asia")
    assert sorted(r.symbol for r in board.markets) == ["^HSI", "^N225"]
    assert board.as_of == FIVE_AM


def test_dashboard_refuses_a_region_it_cannot_read(panel):
    with pytest.raises(ValueError, match="no Europe market"):
        build_dashboard(panel, [], as_of=FIVE_AM, region="Europe")


def test_oil_contribution_sums_only_the_crude_features(panel):
    board = build_dashboard(panel, [forecast("^N225", 0.7, 0.5)], as_of=FIVE_AM)
    row = next(r for r in board.markets if r.symbol == "^N225")
    assert row.oil_contribution == pytest.approx(0.75)  # 0.5 + 0.25, not the 0.4 index term
    assert row.top_oil_driver == ("ind_cl_f_shock", 0.5)


def test_a_market_without_a_forecast_reports_no_probability(panel):
    board = build_dashboard(panel, [], as_of=FIVE_AM)
    row = board.markets[0]
    assert row.probability_up is None
    assert row.oil_contribution == 0.0 and row.top_oil_driver is None
    assert "n/a" in render_text(board)


def test_text_render_shows_crude_and_the_open_calls(panel):
    board = build_dashboard(panel, [forecast("^N225", 0.7, 0.5)], as_of=FIVE_AM)
    out = render_text(board)
    assert "Asia dashboard — 2026-08-03 05:00 UTC" in out
    assert "WTI crude" in out and "Brent crude" in out
    assert "Nikkei 225" in out and "70.0%" in out
    assert "+0.750" in out
    assert "ind_cl_f_shock" in out


def test_html_render_escapes_and_flags_a_shock(panel):
    closes = panel["CL=F"]["Close"]
    closes.iloc[-1] = closes.iloc[-2] * 1.15  # a move far outside the known vol
    board = build_dashboard(panel, [forecast("^N225", 0.7, 0.5)], as_of=FIVE_AM)
    assert next(r for r in board.oil if r.symbol == "CL=F").is_shock

    html = render_html(board)
    assert html.startswith("<!doctype html>")
    assert 'class="shock"' in html
    assert "<td>Nikkei 225</td>" in html and "70.0%" in html


def test_forecast_drivers_stay_capped_at_the_top_contributions():
    f = forecast("^N225", 0.7, 0.5)
    f.top_drivers = 2
    assert list(f.drivers.index) == ["ind_cl_f_shock", "ind_bz_f_return"]
