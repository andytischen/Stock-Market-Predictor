import numpy as np
import pandas as pd
import pytest

from gapmodel.cli import build_parser
from gapmodel.features import _lag_days, as_of, build_features, log_return
from gapmodel.markets import SECTOR_SYMBOLS, market
from gapmodel.stocks import (
    STOCKS_BY_SYMBOL,
    peers_of,
    stock,
    stock_symbols,
    target_market,
)


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
    symbols = [
        "MU",
        "WDC",
        "000660.KS",
        "8035.T",
        "NVDA",
        "SMH",
        "^GSPC",
        "^N225",
        "^VIX",
        "CL=F",
        "EXH8.DE",
    ]
    return {symbol: synthetic_bars(seed=seed) for seed, symbol in enumerate(symbols)}


def test_a_stock_is_a_target_on_wall_streets_clock():
    mu = target_market("MU")
    spx = market("^GSPC")
    assert mu.region == "Americas"
    assert (mu.open_utc, mu.close_utc) == (spx.open_utc, spx.close_utc)
    # Indices still resolve through the market registry, untouched.
    assert target_market("^GSPC") is spx


def test_an_unmodelled_symbol_is_refused_rather_than_invented():
    with pytest.raises(KeyError):
        target_market("TSLA")
    with pytest.raises(KeyError, match="modelled stocks"):
        stock("TSLA")


def test_a_stock_is_not_its_own_peer_and_an_index_has_none():
    peers = {p.symbol for p in peers_of("MU")}
    assert "MU" not in peers
    assert {"000660.KS", "005930.KS", "SMH"} <= peers
    # One peer tuple serves the whole complex, so the others still appear.
    assert "WDC" in peers
    assert peers_of("^GSPC") == ()


def test_seoul_and_tokyo_are_same_session_information_for_a_us_listing():
    """The reason a single US chipmaker is worth modelling separately at all."""
    mu = target_market("MU")
    by_symbol = {p.symbol: p for p in peers_of("MU")}
    assert _lag_days(by_symbol["000660.KS"].close_utc, mu) == 0
    assert _lag_days(by_symbol["8035.T"].close_utc, mu) == 0
    # Wall Street's own bars close after the bell they would be predicting.
    assert _lag_days(by_symbol["NVDA"].close_utc, mu) == 1


def test_peer_features_are_built_for_stocks_only(panel):
    stock_features, _ = build_features("MU", panel)
    index_features, _ = build_features("^GSPC", panel)
    assert {"peer_000660_ks_return", "peer_000660_ks_return_5"} <= set(stock_features.columns)
    assert not any(c.startswith("peer_") for c in index_features.columns)
    # A US listing is not European, so it reads no STOXX sector series.
    assert not {
        c for c in stock_features.columns if any(s[:4].lower() in c for s in SECTOR_SYMBOLS)
    }


def test_a_peer_that_closed_before_the_bell_is_read_on_the_same_day(panel):
    features, _ = build_features("MU", panel)
    expected = as_of(log_return(panel["000660.KS"]["Close"]), features.index, 0)
    assert features["peer_000660_ks_return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_a_peer_that_closed_with_wall_street_is_read_a_session_late(panel):
    features, _ = build_features("MU", panel)
    expected = as_of(log_return(panel["NVDA"]["Close"]), features.index, 1)
    assert features["peer_nvda_return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_a_missing_peer_costs_the_feature_and_not_the_run(panel):
    without = {k: v for k, v in panel.items() if k != "8035.T"}
    features, _ = build_features("MU", without)
    assert not any(c.startswith("peer_8035") for c in features.columns)
    assert features.notna().all().all()


def test_the_download_list_covers_every_peer_exactly_once():
    symbols = stock_symbols()
    assert len(symbols) == len(set(symbols))
    for name, s in STOCKS_BY_SYMBOL.items():
        assert name in symbols
        assert all(p.symbol in symbols for p in s.peers)


def test_the_command_takes_lower_case_and_refuses_unmodelled_names(capsys):
    args = build_parser().parse_args(["stock", "mu", "--explain"])
    assert args.symbols == ["MU"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["stock", "TSLA"])
    assert "unknown stock" in capsys.readouterr().err


def test_a_peer_can_be_shocked_like_any_other_instrument(panel):
    from gapmodel.features import live_feature_row
    from gapmodel.predict import shocked_row

    live, _ = live_feature_row("MU", panel)
    bumped = shocked_row(live, {"000660.KS": 0.05})
    for column in ("peer_000660_ks_return", "peer_000660_ks_return_5"):
        assert bumped[column].iloc[0] == pytest.approx(live[column].iloc[0] + 0.05)
