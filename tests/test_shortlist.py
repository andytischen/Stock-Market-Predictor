import pandas as pd
import pytest

from gapmodel.cli import _shortlist_equities, build_parser
from gapmodel.features import _lag_days, build_features
from gapmodel.markets import MARKETS, MARKETS_BY_SYMBOL, all_symbols, market
from gapmodel.predict import Forecast
from gapmodel.shortlist import (
    MIN_AUC,
    MIN_OOS,
    StockPick,
    discarded,
    forecast_universe,
    rank,
    render_text,
    stale_inputs,
    to_frame,
)
from gapmodel.stocks import (
    STOCKS_BY_SYMBOL,
    US_CLOSE_UTC,
    US_OPEN_UTC,
    is_stock,
    peers_of,
    target_market,
)
from gapmodel.universe import NASDAQ, nasdaq_universe
from tests.test_features import synthetic_bars

TICKER = "AAPL"


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    """An indicator panel plus one stock, all on the same business days."""
    return {
        symbol: synthetic_bars(seed=seed)
        for seed, symbol in enumerate(
            ["^GSPC", "^IXIC", "^N225", "^FTSE", "^VIX", "ES=F", "CL=F", "JPY=X", "KRW=X", TICKER]
        )
    }


def pick(
    symbol: str,
    probability: float,
    auc: float,
    base_rate: float = 0.5,
    brier_skill: float = 0.05,
    n: int = 1000,
) -> StockPick:
    return StockPick(
        Forecast(
            symbol=symbol,
            name=symbol,
            region="Americas",
            session=pd.Timestamp("2026-08-13"),
            probability_up=probability,
            backtest={
                "auc": auc,
                "base_rate": base_rate,
                "accuracy": 0.55,
                "brier_skill": brier_skill,
                "n": n,
            },
            contributions=pd.Series({"mkt_gspc_return": 0.4}),
        )
    )


def test_a_stock_sits_on_the_nasdaq_cash_session_clock():
    apple = target_market(TICKER)
    assert (apple.open_utc, apple.close_utc) == (US_OPEN_UTC, US_CLOSE_UTC)
    # It reads the indicator lags a US index reads, so the Nasdaq Composite is
    # the reference: same auction, same close.
    composite = market("^IXIC")
    assert (apple.open_utc, apple.close_utc) == (composite.open_utc, composite.close_utc)
    # Region drives the sector read-across, which is European only.
    assert apple.region == "Americas"
    # Its own opening print is the label, so there is no tracker substitution.
    assert apple.gap_symbol == TICKER


def test_stocks_are_targets_and_never_features():
    """Registering stocks as markets would change every index forecast."""
    assert TICKER not in MARKETS_BY_SYMBOL
    assert TICKER not in {m.symbol for m in MARKETS}
    # The default download panel is unchanged, so `fetch` does not grow.
    assert not set(NASDAQ) & set(all_symbols())


def test_the_universe_is_a_stable_deduplicated_list():
    universe = nasdaq_universe()
    assert universe == list(dict.fromkeys(universe))
    assert "AAPL" in universe and "NVDA" in universe
    # Names listed elsewhere have no business in a Nasdaq universe.
    for elsewhere in ("IBM", "ORCL", "CRM", "JPM", "XOM"):
        assert elsewhere not in universe


def test_a_repeated_ticker_is_forecast_once(panel):
    """Refitting the same name twice would double-count it in the header."""
    picks = forecast_universe(panel, symbols=[TICKER, TICKER, TICKER])
    assert [p.symbol for p in picks] == [TICKER]
    assert "for 1 Nasdaq names" in render_text(picks)


def test_a_shortlisted_name_is_a_single_company_without_a_peer_list():
    """The universe reuses the single-name machinery `stock` established.

    Being a stock is what puts the bars on a total-return basis, so a dividend
    is not labelled a down open. What the universe does not get is peers: those
    are hand-written per complex, and only the curated names carry them.
    """
    assert is_stock(TICKER)
    assert peers_of(TICKER) == ()
    assert peers_of("MU"), "a curated name keeps its overnight peers"
    # An index is not a company; a foreign listing is not on Wall Street's
    # clock; and a ticker that is merely well-formed is not modelled at all.
    assert not is_stock("^IXIC")
    assert not is_stock("005930.KS")
    assert not is_stock("BRKB")
    assert target_market("^IXIC") == market("^IXIC")


def test_a_stock_reads_wall_street_a_session_late_and_tokyo_same_day(panel):
    """The look-ahead guarantee has to hold for a stock exactly as for an index."""
    apple = target_market(TICKER)
    # Wall Street's 20:00 close lands after the 13:30 auction -> previous day.
    assert _lag_days(market("^GSPC").close_utc, apple) == 1
    # Tokyo closes at 06:00, hours before the US opens -> same day is legitimate.
    assert _lag_days(market("^N225").close_utc, apple) == 0


def test_build_features_accepts_a_stock_target(panel):
    features, labels = build_features(TICKER, panel)
    assert features.notna().all().all()
    assert features.index.equals(labels.index)
    assert set(labels.dropna().unique()) <= {0.0, 1.0}
    # Its own history, the indices as read-across, and the cross-asset panel.
    assert any(c.startswith("own_") for c in features.columns)
    assert "mkt_gspc_return" in features.columns
    assert "ind_vix_level" in features.columns
    # The stock is the target, so it is not also one of its own columns.
    assert not any(c.startswith("mkt_aapl") for c in features.columns)


def test_a_stock_target_leaves_the_index_features_untouched(panel):
    """Adding stock support must not change the forecasts already made."""
    before, _ = build_features("^IXIC", panel)
    build_features(TICKER, panel)
    after, _ = build_features("^IXIC", panel)
    pd.testing.assert_frame_equal(before, after)


def test_a_symbol_that_is_neither_index_nor_us_listing_is_still_an_error(panel):
    """A US ticker is described on demand; a foreign one must not be guessed at."""
    panel["7203.T"] = synthetic_bars(seed=99)
    with pytest.raises(KeyError, match="unknown market"):
        build_features("7203.T", panel)


def test_edge_measures_the_probability_against_the_stocks_own_drift():
    # A share that rose on 56% of opens is not a 56% call, it is a coin flip
    # dressed as one.
    drifting = pick("X", probability=0.56, auc=0.60, base_rate=0.56)
    assert drifting.edge == pytest.approx(0.0)
    leaning = pick("Y", probability=0.66, auc=0.60, base_rate=0.56)
    assert leaning.edge == pytest.approx(0.10)
    assert leaning.direction == "up"
    assert pick("Z", probability=0.40, auc=0.60).direction == "down"


def test_only_names_with_demonstrated_skill_are_ranked():
    skilled = pick("GOOD", probability=0.70, auc=MIN_AUC + 0.05)
    noise = pick("NOISE", probability=0.99, auc=0.51)
    picks = [noise, skilled]
    # The 99% call comes from a coin-flip model, so it is not a pick at all.
    assert [p.symbol for p in rank(picks)] == ["GOOD"]
    assert [p.symbol for p in discarded(picks)] == ["NOISE"]
    assert skilled.credible and not noise.credible


def test_a_strong_auc_on_too_few_sessions_is_not_evidence():
    """A recent listing can post a fine AUC on a couple of hundred predictions."""
    young = pick("ARM", probability=0.70, auc=0.72, n=MIN_OOS - 1)
    assert not young.credible
    assert rank([young]) == []
    seasoned = pick("ARM", probability=0.70, auc=0.72, n=MIN_OOS)
    assert seasoned.credible


def test_a_miscalibrated_model_is_not_a_pick():
    """AUC orders sessions; it says nothing about whether the level is right."""
    orders_well_but_wrong = pick("HOOD", probability=0.66, auc=0.63, brier_skill=-0.02)
    assert not orders_well_but_wrong.credible
    assert rank([orders_well_but_wrong]) == []


def test_the_ranking_weights_the_edge_by_demonstrated_skill():
    # A large edge from a barely-skilled model ranks below a smaller edge from
    # a model that has actually earned it.
    weak_but_bold = pick("BOLD", probability=0.80, auc=0.56)
    strong_and_modest = pick("SOLID", probability=0.62, auc=0.70)
    assert [p.symbol for p in rank([weak_but_bold, strong_and_modest])] == ["SOLID", "BOLD"]


def test_a_downward_call_ranks_on_the_size_of_its_edge():
    """Direction is not quality: a confident down-call is a pick too."""
    down = pick("DOWN", probability=0.25, auc=0.65)
    flat = pick("FLAT", probability=0.51, auc=0.65)
    assert [p.symbol for p in rank([down, flat])] == ["DOWN", "FLAT"]


def test_forecast_universe_skips_a_name_it_cannot_model(panel, caplog):
    """A young listing must not take the whole run down with it."""
    panel["YOUNG"] = synthetic_bars(n=80, seed=42)
    picks = forecast_universe(panel, symbols=[TICKER, "YOUNG"], min_train=500)
    assert [p.symbol for p in picks] == [TICKER]
    assert "YOUNG" in caplog.text


def test_forecast_universe_reports_when_nothing_can_be_modelled(panel):
    with pytest.raises(RuntimeError, match="no stock could be modelled"):
        forecast_universe(panel, symbols=["MISSING"])


def test_a_real_forecast_is_calibrated_and_explained(panel):
    picks = forecast_universe(panel, symbols=[TICKER], min_train=500)
    entry = picks[0]
    assert 0.0 < entry.probability_up < 1.0
    assert entry.forecast.session > panel[TICKER].index[-1]
    assert 0.0 <= entry.auc <= 1.0
    assert len(entry.forecast.drivers) == 5


def test_the_table_reports_the_quality_next_to_the_probability():
    frame = to_frame([pick("GOOD", probability=0.70, auc=0.62, base_rate=0.55)])
    row = frame.iloc[0]
    assert row["symbol"] == "GOOD"
    assert row["p_open_up"] == 0.70
    assert row["edge"] == pytest.approx(0.15)
    assert row["oos_auc"] == 0.62
    assert row["n_oos"] == 1000


def test_the_report_says_which_test_each_discarded_name_failed():
    picks = [
        pick("GOOD", 0.70, auc=0.62),
        pick("FLIP", 0.99, auc=0.51),
        pick("YOUNG", 0.70, auc=0.72, n=193),
        pick("SKEW", 0.66, auc=0.63, brier_skill=-0.02),
    ]
    text = render_text(picks)
    assert "FLIP: AUC below 0.55" in text
    assert "YOUNG: only 193 out-of-sample sessions" in text
    assert "SKEW: worse calibrated than its base rate" in text


def test_the_report_separates_picks_from_noise_and_states_the_horizon():
    text = render_text([pick("GOOD", 0.70, auc=0.62), pick("NOISE", 0.99, auc=0.51)])
    assert "GOOD" in text and "NOISE" in text
    assert "Ranked" in text
    assert "No demonstrated skill" in text
    # The two claims a reader most needs: what is being predicted, and that the
    # metrics are flattered by the universe.
    assert "opening print against the previous close" in text
    assert "survivorship bias" in text


def test_the_report_says_so_when_no_name_has_an_edge():
    text = render_text([pick("NOISE", 0.99, auc=0.51)])
    assert "no demonstrated edge on any single stock" in text
    assert "Ranked" not in text


def test_a_real_forecast_carries_enough_history_to_be_judged(panel):
    """The synthetic fixture must itself clear the sample-size test."""
    entry = forecast_universe(panel, symbols=[TICKER], min_train=500)[0]
    assert entry.n_oos == entry.forecast.backtest["n"]
    assert entry.brier_skill == entry.forecast.backtest["brier_skill"]


def test_the_report_can_show_only_the_strongest_names():
    picks = [pick("A", 0.75, auc=0.65), pick("B", 0.60, auc=0.65), pick("C", 0.99, auc=0.51)]
    text = render_text(picks, top=1)
    assert "A" in text
    # B is credible but beyond the cut; C is shown regardless, as discarded.
    ranked_block = text.split("No demonstrated skill")[0]
    assert " B " not in ranked_block


def test_the_report_carries_the_release_caveats():
    entry = pick("GOOD", 0.70, auc=0.62)
    entry.forecast.caveats = ("US CPI at 12:30 UTC, before this open",)
    text = render_text([entry])
    assert "scheduled releases this model cannot see" in text
    assert "US CPI" in text


def test_the_cli_exposes_the_shortlist_and_normalises_its_tickers():
    args = build_parser().parse_args(["shortlist", "aapl", "nvda", "--top", "5"])
    assert args.symbols == ["AAPL", "NVDA"]
    assert args.top == 5
    parsed = build_parser().parse_args(["shortlist"])
    assert parsed.symbols == []


def test_a_name_in_both_commands_is_read_from_the_same_features():
    """`shortlist MU` and `stock MU` must not print different probabilities.

    The peers are what make the curated forecast better; downloading them only
    for one command would make the answer depend on which one asked.
    """
    equities = _shortlist_equities(["AAPL", "MU"])
    assert equities[:2] == ["AAPL", "MU"]
    assert {"005930.KS", "000660.KS", "SMH"} <= set(equities)
    # A name with no peers adds nothing, and nothing repeats.
    assert _shortlist_equities(["AAPL"]) == ["AAPL"]
    assert equities == list(dict.fromkeys(equities))


def test_the_cli_refuses_a_ticker_the_shortlist_does_not_model():
    """Better than downloading whatever the name matched and ranking it."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["shortlist", "brkb"])


def test_asking_for_no_names_shows_none_rather_than_all_of_them():
    picks = [pick("GOOD", 0.70, auc=0.62), pick("ALSO", 0.65, auc=0.60)]
    text = render_text(picks, top=0)
    assert "GOOD" not in text and "ALSO" not in text
    # Silence about the request is not a verdict on the model.
    assert "no demonstrated edge" not in text
    assert "2 names cleared" in text
    assert "GOOD" in render_text(picks, top=1)


def test_the_cli_rejects_a_top_that_cannot_mean_anything():
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["shortlist", "--top", bad])
    assert build_parser().parse_args(["shortlist", "--top", "4"]).top == 4


def test_release_caveats_reach_the_unranked_table_too():
    """When nothing is credible the unranked table is the whole output."""
    noise = pick("NOISE", 0.99, auc=0.51)
    noise.forecast.caveats = ("US CPI at 12:30 UTC, before this open",)
    text = render_text([noise])
    assert "scheduled releases this model cannot see" in text
    assert "NOISE: US CPI" in text


def test_the_csv_carries_the_verdict_but_the_printed_table_does_not():
    picks = [pick("GOOD", 0.70, auc=0.62), pick("NOISE", 0.99, auc=0.51)]
    frame = to_frame(picks)
    assert list(frame["credible"]) == [True, False]
    # Redundant in print: each block is uniform and its heading says which.
    assert "credible" not in render_text(picks)


def test_the_printed_edge_is_the_difference_of_the_printed_columns():
    """A reader checking the column by hand must not find it off by a digit."""
    row = to_frame([pick("X", 0.52214, auc=0.6, base_rate=0.50553)]).iloc[0]
    assert row["edge"] == round(row["p_open_up"] - row["base_rate"], 4)


SESSION = pd.Timestamp("2026-08-13")


def _ending(bars: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    return bars.set_axis(pd.date_range(end=last, periods=len(bars), freq="B"))


def test_the_report_discloses_inputs_that_stopped_updating(panel):
    """A stale macro series is forward-filled, so it has to be called out."""
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    # The stock is current; the S&P and crude stopped a week ago.
    for stale in ("^GSPC", "CL=F"):
        panel[stale] = _ending(panel[stale], SESSION - pd.Timedelta(days=7))
    counted, behind = stale_inputs(panel, SESSION)
    assert counted == len(panel)
    assert "^GSPC" in behind and "CL=F" in behind and TICKER not in behind

    text = render_text([pick("GOOD", 0.70, auc=0.62)], panel=panel)
    assert "stale inputs" in text
    assert "carried forward" in text
    assert "^GSPC" in text
    # Without the panel the report cannot know, so it must not claim otherwise.
    assert "stale inputs" not in render_text([pick("GOOD", 0.70, auc=0.62)])


def test_the_worst_lag_is_named_first(panel):
    """Eight names are printed; they should be the eight furthest behind."""
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    panel["^GSPC"] = _ending(panel["^GSPC"], SESSION - pd.Timedelta(days=30))
    panel["CL=F"] = _ending(panel["CL=F"], SESSION - pd.Timedelta(days=8))
    _, behind = stale_inputs(panel, SESSION)
    assert behind[:2] == ["^GSPC", "CL=F"]


def test_a_panel_current_for_the_session_raises_no_staleness_note(panel):
    """Wall Street is not stale because Seoul was open.

    A US series that has not opened yet ends on the previous session, and a peer
    downloaded mid-session in Asia carries a partial bar for the session itself.
    Both are current, so the older reading of the two must not become the
    yardstick that condemns the rest of the panel.
    """
    panel = {
        symbol: _ending(bars, SESSION - pd.Timedelta(days=1)) for symbol, bars in panel.items()
    }
    panel["^N225"] = _ending(panel["^N225"], SESSION)
    counted, behind = stale_inputs(panel, SESSION)
    assert counted == len(panel)
    assert behind == []
    assert "stale inputs" not in render_text([pick("GOOD", 0.70, auc=0.62)], panel=panel)


def test_probabilities_are_never_a_certainty():
    """A stock forecast inherits the index model's refusal to print 0 or 1."""
    assert to_frame([pick("X", 1.0, auc=0.6)]).iloc[0]["p_open_up"] == 0.9999
    assert to_frame([pick("X", 0.0, auc=0.6)]).iloc[0]["p_open_up"] == 0.0001


def test_every_curated_stock_can_also_be_shortlisted():
    """The two commands must not disagree about what a modelled name is.

    A curated name absent here is accepted by ``stock`` and refused by
    ``shortlist``, so the parity between them cannot even be checked.
    """
    assert set(STOCKS_BY_SYMBOL) <= set(nasdaq_universe())


def test_the_universe_covers_a_useful_slice_of_the_nasdaq():
    universe = nasdaq_universe()
    assert len(universe) >= 50
    assert all(symbol == symbol.upper() for symbol in universe)
    # Indices carry a caret; these are all single listings.
    assert all(not symbol.startswith("^") for symbol in universe)
