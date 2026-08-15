import pandas as pd
import pytest

from gapmodel.markets import SECTORS
from gapmodel.predict import Forecast
from gapmodel.sectors import build_sector_board, render_text
from tests.test_features import synthetic_bars


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    return {s.symbol: synthetic_bars(seed=seed) for seed, s in enumerate(SECTORS)}


def forecast(contributions: pd.Series) -> Forecast:
    return Forecast(
        symbol="^GDAXI",
        name="DAX",
        region="Europe",
        session=pd.Timestamp("2026-08-04"),
        probability_up=0.42,
        backtest={},
        contributions=contributions,
    )


def test_board_sums_each_sector_and_ranks_by_size(panel):
    board = build_sector_board(
        panel,
        forecast(
            pd.Series(
                {
                    "ind_exh8_de_return": 0.10,
                    "ind_exh8_de_return_5": -0.02,
                    "ind_exv1_de_return": -0.50,
                    "ind_cl_f_return": 9.0,
                }
            )
        ),
    )
    banks, retail = board.rows[0], next(r for r in board.rows if r.symbol == "EXH8.DE")
    assert banks.symbol == "EXV1.DE"
    assert banks.contribution == pytest.approx(-0.50)
    # The two retail features net off; the oil feature belongs to no sector.
    assert retail.contribution == pytest.approx(0.08)
    assert retail.top_feature == "ind_exh8_de_return"
    assert board.net_contribution == pytest.approx(-0.42)
    assert len(board.rows) == len(SECTORS)


def test_board_refuses_a_market_without_sector_features(panel):
    with pytest.raises(ValueError, match="no sector features"):
        build_sector_board(panel, forecast(pd.Series({"ind_cl_f_return": 1.0})))


def test_render_names_every_sector_and_the_net(panel):
    board = build_sector_board(panel, forecast(pd.Series({"ind_exh8_de_return": 0.3})))
    out = render_text(board)
    assert "DAX" in out and "Europe 600 Retail" in out
    assert "net sector log-odds +0.300" in out
