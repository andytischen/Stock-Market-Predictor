import numpy as np
import pandas as pd
import pytest

from gapmodel.cli import build_parser
from gapmodel.features import (
    _lag_days,
    as_of,
    build_features,
    dividend_adjusted,
    log_return,
    opening_gap,
)
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
    # A well-formed US ticker that is in neither the registry nor the shortlist
    # universe: looking like a stock is not being one.
    with pytest.raises(KeyError):
        target_market("BRKB")
    # In the shortlist universe, so it has a clock, but still not one of the
    # names carrying a hand-written peer list.
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


def test_going_ex_dividend_is_not_recorded_as_an_opening_gap():
    """The one label error a single company has and an index does not."""
    bars = synthetic_bars(n=10)
    # Yahoo's factor steps down on the ex-dividend morning and holds thereafter.
    factor = pd.Series(1.0, index=bars.index)
    factor.iloc[:5] = 0.99
    bars = bars.assign(**{"Adj Close": bars["Close"] * factor})
    ex_date = bars.index[5]

    raw = opening_gap(bars)
    adjusted = opening_gap(dividend_adjusted(bars))
    # A 1% dividend paid out of the previous close is not a 1% fall.
    assert raw[ex_date] == pytest.approx(adjusted[ex_date] + np.log(0.99), abs=1e-12)
    # Every other session, and every session's own intraday return, is untouched.
    others = [d for d in bars.index[1:] if d != ex_date]
    assert raw[others].to_numpy() == pytest.approx(adjusted[others].to_numpy())
    intraday = dividend_adjusted(bars)
    assert (intraday["Close"] / intraday["Open"]).to_numpy() == pytest.approx(
        (bars["Close"] / bars["Open"]).to_numpy()
    )


def test_an_index_is_left_on_its_published_prints(panel):
    """Only the single names are corrected; the index tables must not move."""
    dividend_paying = {
        symbol: bars.assign(**{"Adj Close": bars["Close"] * 0.9}) for symbol, bars in panel.items()
    }
    before, labels_before = build_features("^GSPC", panel)
    after, labels_after = build_features("^GSPC", dividend_paying)
    assert labels_after.to_numpy() == pytest.approx(labels_before.to_numpy())
    assert after["own_gap_lag1"].to_numpy() == pytest.approx(before["own_gap_lag1"].to_numpy())


def test_a_peers_dividend_is_not_read_as_a_fall_in_demand(panel):
    factor = pd.Series(1.0, index=panel["000660.KS"].index)
    factor.iloc[:100] = 0.98
    hynix = panel["000660.KS"]
    with_dividend = dict(panel)
    with_dividend["000660.KS"] = hynix.assign(**{"Adj Close": hynix["Close"] * factor})

    features, _ = build_features("MU", with_dividend)
    expected = as_of(log_return(with_dividend["000660.KS"]["Adj Close"]), features.index, 0)
    assert features["peer_000660_ks_return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_only_a_single_name_run_accepts_a_shock_on_a_peer(capsys):
    """An index reads no peer, so a shock on one there would do nothing at all."""
    args = build_parser().parse_args(["stock", "MU", "--shock", "000660.KS=-4%"])
    assert args.shock == [("000660.KS", pytest.approx(np.log1p(-0.04)))]
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "--shock", "MU=+5%"])
    assert "unknown instrument" in capsys.readouterr().err
    # Instruments both kinds of model read stay shockable on either command.
    assert build_parser().parse_args(["predict", "--shock", "CL=F=+3%"]).shock[0][0] == "CL=F"
    assert build_parser().parse_args(["stock", "--shock", "CL=F=+3%"]).shock[0][0] == "CL=F"


def test_the_metrics_behind_a_stock_can_be_reproduced_from_the_command_line():
    """The per-stock table in the README needs no code editing to rebuild."""
    args = build_parser().parse_args(["backtest", "--market", "mu", "--reliability"])
    assert args.market == ["MU"]
    assert build_parser().parse_args(["backtest", "--market", "^GSPC"]).market == ["^GSPC"]


def test_a_dividend_paid_before_the_first_factor_is_not_a_gap_at_the_boundary():
    """A leading block of unadjusted bars must not invent one enormous gap."""
    bars = synthetic_bars(n=10)
    adj = bars["Close"] * 0.6
    adj.iloc[:4] = np.nan  # Yahoo served no factor for the earliest bars
    bars = bars.assign(**{"Adj Close": adj})

    gap = opening_gap(dividend_adjusted(bars))
    raw = opening_gap(bars)
    assert gap.to_numpy()[1:] == pytest.approx(raw.to_numpy()[1:])


def test_a_move_in_something_the_target_never_reads_is_called_unmodelled(panel, caplog):
    """Including the company's own price: no model may predict itself."""
    from gapmodel.features import live_feature_row
    from gapmodel.predict import shocked_row

    live, _ = live_feature_row("MU", panel)
    with caplog.at_level("WARNING"):
        bumped = shocked_row(live, {"MU": 0.05})
    assert bumped.equals(live)
    assert "MU is not a feature of this target" in caplog.text


def test_a_peer_can_be_shocked_like_any_other_instrument(panel, caplog):
    from gapmodel.features import live_feature_row
    from gapmodel.predict import shocked_row

    live, _ = live_feature_row("MU", panel)
    with caplog.at_level("WARNING"):
        bumped = shocked_row(live, {"000660.KS": 0.05})
    assert "not a feature" not in caplog.text
    for column in ("peer_000660_ks_return", "peer_000660_ks_return_5"):
        assert bumped[column].iloc[0] == pytest.approx(live[column].iloc[0] + 0.05)
