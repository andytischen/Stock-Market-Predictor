import pandas as pd
import pytest

from gapmodel.cli import build_parser
from gapmodel.features import _lag_days, build_features
from gapmodel.markets import (
    MARKETS,
    MARKETS_BY_SYMBOL,
    NASDAQ_CLOSE_UTC,
    NASDAQ_OPEN_UTC,
    all_symbols,
    market,
    stock_market,
)
from gapmodel.predict import Forecast
from gapmodel.stocks import (
    MIN_AUC,
    MIN_OOS,
    StockPick,
    discarded,
    forecast_stocks,
    panel_symbols,
    rank,
    render_text,
    stale_inputs,
    to_frame,
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
    apple = stock_market(TICKER)
    assert (apple.open_utc, apple.close_utc) == (NASDAQ_OPEN_UTC, NASDAQ_CLOSE_UTC)
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
    picks = forecast_stocks(panel, symbols=[TICKER, TICKER, TICKER])
    assert [p.symbol for p in picks] == [TICKER]
    assert "for 1 Nasdaq names" in render_text(picks)


def test_a_stock_run_loads_the_indicators_as_well_as_the_stocks():
    symbols = panel_symbols([TICKER])
    assert TICKER in symbols
    assert "^GSPC" in symbols and "^VIX" in symbols
    assert symbols == list(dict.fromkeys(symbols))


def test_a_stock_reads_wall_street_a_session_late_and_tokyo_same_day(panel):
    """The look-ahead guarantee has to hold for a stock exactly as for an index."""
    apple = stock_market(TICKER)
    # Wall Street's 20:00 close lands after the 13:30 auction -> previous day.
    assert _lag_days(market("^GSPC").close_utc, apple) == 1
    # Tokyo closes at 06:00, hours before the US opens -> same day is legitimate.
    assert _lag_days(market("^N225").close_utc, apple) == 0


def test_build_features_accepts_a_stock_target(panel):
    features, labels = build_features(TICKER, panel, target=stock_market(TICKER))
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
    build_features(TICKER, panel, target=stock_market(TICKER))
    after, _ = build_features("^IXIC", panel)
    pd.testing.assert_frame_equal(before, after)


def test_an_unknown_symbol_without_a_target_is_still_an_error(panel):
    with pytest.raises(KeyError, match="unknown market"):
        build_features(TICKER, panel)


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


def test_forecast_stocks_skips_a_name_it_cannot_model(panel, caplog):
    """A young listing must not take the whole run down with it."""
    panel["YOUNG"] = synthetic_bars(n=80, seed=42)
    picks = forecast_stocks(panel, symbols=[TICKER, "YOUNG"], min_train=500)
    assert [p.symbol for p in picks] == [TICKER]
    assert "YOUNG" in caplog.text


def test_forecast_stocks_reports_when_nothing_can_be_modelled(panel):
    with pytest.raises(RuntimeError, match="no stock could be modelled"):
        forecast_stocks(panel, symbols=["MISSING"])


def test_a_real_forecast_is_calibrated_and_explained(panel):
    picks = forecast_stocks(panel, symbols=[TICKER], min_train=500)
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
    entry = forecast_stocks(panel, symbols=[TICKER], min_train=500)[0]
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


def test_the_cli_exposes_the_stock_forecast():
    args = build_parser().parse_args(["stocks", "aapl", "nvda", "--top", "5"])
    assert args.symbols == ["aapl", "nvda"]
    assert args.top == 5
    parsed = build_parser().parse_args(["stocks"])
    assert parsed.symbols == []


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
            build_parser().parse_args(["stocks", "--top", bad])
    assert build_parser().parse_args(["stocks", "--top", "4"]).top == 4


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


def test_the_report_discloses_inputs_that_stopped_updating(panel):
    """A stale macro series is forward-filled, so it has to be called out."""
    fresh = pd.Timestamp("2026-08-13")
    panel = {symbol: bars.copy() for symbol, bars in panel.items()}
    # The stock is current; the S&P and crude stopped a week ago.
    panel[TICKER].index = pd.date_range(end=fresh, periods=len(panel[TICKER]), freq="B")
    for stale in ("^GSPC", "CL=F"):
        panel[stale].index = pd.date_range(
            end=fresh - pd.Timedelta(days=7), periods=len(panel[stale]), freq="B"
        )
    freshest, behind = stale_inputs(panel)
    assert freshest == fresh
    assert "^GSPC" in behind and "CL=F" in behind and TICKER not in behind

    text = render_text([pick("GOOD", 0.70, auc=0.62)], panel=panel)
    assert "stale inputs" in text
    assert "carried forward" in text
    assert "^GSPC" in text
    # Without the panel the report cannot know, so it must not claim otherwise.
    assert "stale inputs" not in render_text([pick("GOOD", 0.70, auc=0.62)])


def test_a_fully_current_panel_raises_no_staleness_note(panel):
    aligned = pd.date_range(end=pd.Timestamp("2026-08-13"), periods=600, freq="B")
    current = {symbol: bars.iloc[-600:].set_axis(aligned) for symbol, bars in panel.items()}
    freshest, behind = stale_inputs(current)
    assert freshest == aligned[-1]
    assert behind == []
    assert "stale inputs" not in render_text([pick("GOOD", 0.70, auc=0.62)], panel=current)


def test_probabilities_are_never_a_certainty():
    """A stock forecast inherits the index model's refusal to print 0 or 1."""
    assert to_frame([pick("X", 1.0, auc=0.6)]).iloc[0]["p_open_up"] == 0.9999
    assert to_frame([pick("X", 0.0, auc=0.6)]).iloc[0]["p_open_up"] == 0.0001


def test_the_universe_covers_a_useful_slice_of_the_nasdaq():
    universe = nasdaq_universe()
    assert len(universe) >= 50
    assert all(symbol == symbol.upper() for symbol in universe)
    # Indices carry a caret; these are all single listings.
    assert all(not symbol.startswith("^") for symbol in universe)
