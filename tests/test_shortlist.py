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
    biggest_gainers,
    discarded,
    forecast_universe,
    last_change,
    rank,
    render_text,
    stale_inputs,
    to_frame,
)
from gapmodel.stocks import (
    SHORTLISTED,
    STOCKS_BY_SYMBOL,
    US_CLOSE_UTC,
    US_OPEN_UTC,
    is_stock,
    peers_of,
    target_market,
)
from gapmodel.universe import NASDAQ, modelled_universe, nasdaq_universe
from tests.test_features import synthetic_bars

TICKER = "AAPL"
# A shortlisted name carrying no hand-written peer complex of its own.
PEERLESS = "TSLA"


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
    assert "for 1 US names" in render_text(picks)


def test_a_shortlisted_name_is_a_single_company_without_a_peer_list():
    """The universe reuses the single-name machinery `stock` established.

    Being a stock is what puts the bars on a total-return basis, so a dividend
    is not labelled a down open. What the universe does not get is peers: those
    are hand-written per complex, and only the curated names carry them.
    """
    assert is_stock(PEERLESS)
    assert peers_of(PEERLESS) == ()
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
    assert "mkt_gspc_shock" in features.columns
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
    assert _shortlist_equities([PEERLESS]) == [PEERLESS]
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


def test_the_footer_reports_the_tolerance_the_run_was_given(panel):
    """A footer saying five days under `--max-stale-days 30` contradicts the
    guard that let the run through."""
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    panel["^GSPC"] = _ending(panel["^GSPC"], SESSION - pd.Timedelta(days=7))
    picks = [pick("GOOD", 0.70, auc=0.62)]
    # A seven-day lag is stale at the default and current at thirty.
    assert "within 5 days" in render_text(picks, panel=panel)
    assert "stale inputs" not in render_text(picks, panel=panel, max_stale_days=30)
    # And the wording follows the threshold, not just the filtering.
    panel["CL=F"] = _ending(panel["CL=F"], SESSION - pd.Timedelta(days=40))
    assert "within 30 days" in render_text(picks, panel=panel, max_stale_days=30)


def test_a_uniformly_old_panel_is_disclosed_by_the_session_it_forecast(panel):
    """The one run the per-series footer cannot describe.

    Lags are measured against the session being forecast, which is dated from
    the panel's own last bar, so a cache that stopped a month ago has nothing
    lagging within itself: every series is equally old and none is named. The
    report has to say the forecast is a month old instead.
    """
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    picks = [pick("GOOD", 0.70, auc=0.62)]
    read_on = SESSION + pd.Timedelta(days=30)
    assert "stale inputs" not in render_text(picks, panel=panel, as_of=read_on)
    text = render_text(picks, panel=panel, as_of=read_on)
    assert "stale run" in text and "30 days before" in text
    # A report read the morning it was built says nothing, and the widened
    # tolerance the run was given is honoured here too.
    assert "stale run" not in render_text(picks, panel=panel, as_of=SESSION)
    assert "stale run" not in render_text(picks, panel=panel, as_of=read_on, max_stale_days=40)


def test_the_two_stale_footers_do_not_contradict_each_other(panel):
    """A panel can be old *and* be ragged, and then both footers print."""
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    panel["^GSPC"] = _ending(panel["^GSPC"], SESSION - pd.Timedelta(days=60))
    text = render_text(
        [pick("GOOD", 0.70, auc=0.62)], panel=panel, as_of=SESSION + pd.Timedelta(days=30)
    )
    assert "^GSPC" in text
    # Having just named one, the run footer cannot claim none was named.
    assert "none is named above" not in text
    assert "behind the rest of that panel" in text


def test_an_old_panel_ragged_within_tolerance_is_not_called_uniform(panel):
    """Nothing named is not the same fact as every series stopping together.

    A panel can be a month old and still be a few days ragged inside that
    month, which names nobody: what the footer knows is that no series lags
    the forecast session by more than the tolerance, not that they all end on
    the same bar.
    """
    panel = {symbol: _ending(bars, SESSION) for symbol, bars in panel.items()}
    panel["^GSPC"] = _ending(panel["^GSPC"], SESSION - pd.Timedelta(days=3))
    text = render_text(
        [pick("GOOD", 0.70, auc=0.62)], panel=panel, as_of=SESSION + pd.Timedelta(days=30)
    )
    assert "stale inputs" not in text
    assert "No series is more than 5 days behind that session" in text


@pytest.mark.parametrize("given", [None, {}])
def test_the_run_footer_claims_nothing_about_series_it_did_not_compare(given):
    """No panel and a panel of nothing measurable are the same evidence: none.

    The age of the forecast is still known — it is read off the picks — but
    "no series is behind the others" would be a finding about inputs that were
    never compared.
    """
    text = render_text(
        [pick("GOOD", 0.70, auc=0.62)], panel=given, as_of=SESSION + pd.Timedelta(days=30)
    )
    assert "stale run" in text and "30 days before" in text
    assert "no series is behind" not in text and "named above" not in text


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
    assert set(STOCKS_BY_SYMBOL) <= set(modelled_universe())


def test_the_universe_covers_a_useful_slice_of_the_nasdaq():
    universe = nasdaq_universe()
    assert len(universe) >= 50
    assert all(symbol == symbol.upper() for symbol in universe)
    # Indices carry a caret; these are all single listings.
    assert all(not symbol.startswith("^") for symbol in universe)


def test_the_modelled_universe_reaches_past_the_nasdaq():
    """Venue is not a modelling boundary: NYSE names have the same auction."""
    universe = modelled_universe()
    assert universe == list(dict.fromkeys(universe))
    assert set(nasdaq_universe()) <= set(universe)
    assert len(universe) > len(nasdaq_universe())
    # The names a Nasdaq-only list could never call, one per sleeve.
    for elsewhere in ("JPM", "XOM", "CAT", "IBM", "CRM"):
        assert elsewhere in universe
    assert all(symbol == symbol.upper() for symbol in universe)
    assert all(not symbol.startswith("^") for symbol in universe)
    # A wider universe is only reachable if the parser accepts it.
    assert all(symbol in SHORTLISTED for symbol in universe)


def test_a_nyse_name_is_modelled_on_wall_streets_clock():
    """Widening the list is worth nothing if the new names have no target."""
    bank = target_market("JPM")
    assert (bank.open_utc, bank.close_utc) == (US_OPEN_UTC, US_CLOSE_UTC)
    assert bank.region == "Americas"
    assert is_stock("JPM"), "a bank pays dividends: its bars need total return"


def _closing(prices: list[float], end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily bars ending on the given closes, on consecutive business days."""
    index = pd.date_range(end=end if end is not None else SESSION, periods=len(prices), freq="B")
    return pd.DataFrame({"Open": prices, "Close": prices}, index=index)


def test_the_last_move_is_read_from_the_close_before_it():
    assert last_change(_closing([100.0, 110.0])) == pytest.approx(0.10)
    assert last_change(_closing([100.0, 95.0])) == pytest.approx(-0.05)


def test_a_name_without_two_closes_has_no_last_move():
    for unusable in ([100.0], [0.0, 100.0]):
        with pytest.raises(ValueError):
            last_change(_closing(unusable))


def test_the_biggest_gainers_are_the_names_that_rose_most():
    panel = {
        "UP": _closing([100.0, 112.0]),
        "MID": _closing([100.0, 104.0]),
        "FLAT": _closing([100.0, 100.0]),
        "DOWN": _closing([100.0, 90.0]),
    }
    assert biggest_gainers(panel, list(panel), 2) == ["UP", "MID"]
    # Asking for more names than moved is not an error; it is all of them.
    assert biggest_gainers(panel, list(panel), 10) == ["UP", "MID", "FLAT", "DOWN"]
    with pytest.raises(ValueError):
        biggest_gainers(panel, list(panel), 0)


def test_ties_break_on_the_symbol_so_a_run_is_reproducible():
    panel = {"BBB": _closing([100.0, 105.0]), "AAA": _closing([100.0, 105.0])}
    assert biggest_gainers(panel, ["BBB", "AAA"], 2) == ["AAA", "BBB"]


def test_a_name_the_panel_cannot_price_is_skipped_not_ranked(caplog):
    """A missing or one-bar series must not silently rank as unchanged."""
    panel = {"UP": _closing([100.0, 105.0]), "NEW": _closing([100.0])}
    assert biggest_gainers(panel, ["UP", "NEW", "ABSENT"], 3) == ["UP"]
    assert "NEW" in caplog.text


def test_a_name_that_did_not_trade_the_latest_session_is_not_ranked_as_a_mover(caplog):
    """Otherwise a halted or delisted name holds its final move for ever.

    Its own last two closes are a real move on some older day, so the rank would
    be filled by a name that did not trade at all in the session on the heading,
    displacing one that did. Cached bars make this the normal case.
    """
    panel = {
        "STALE": _closing([100.0, 140.0], end=SESSION - pd.offsets.BDay(10)),
        "UP": _closing([100.0, 105.0]),
        "MID": _closing([100.0, 102.0]),
    }
    assert biggest_gainers(panel, list(panel), 2) == ["UP", "MID"]
    assert "STALE" in caplog.text


def test_the_table_reports_the_move_the_name_has_just_made(panel):
    entry = forecast_universe(panel, symbols=[TICKER], min_train=500)[0]
    expected = last_change(panel[TICKER])
    assert entry.last_change == pytest.approx(expected)
    row = to_frame([entry]).iloc[0]
    assert row["last_change"] == pytest.approx(round(expected * 100, 2))
    # Context, not a claim: the report says so rather than leaving a reader to
    # read the column as part of the forecast.
    assert "last_change is the move the name has just made" in render_text([entry])


def test_an_unknown_last_move_prints_empty_rather_than_zero():
    """Zero would assert the name was unchanged, which is a different claim.

    And ``NaN`` in a column of percentages reads as a number that went wrong
    rather than as a value nobody has, so the printed table blanks it too.
    """
    row = to_frame([pick("X", 0.6, auc=0.6)]).iloc[0]
    assert row["last_change"] is None or pd.isna(row["last_change"])
    text = render_text(
        [
            StockPick(pick("KNOWN", 0.70, auc=0.62).forecast, last_change=0.0123),
            pick("UNKNOWN", 0.70, auc=0.62),
        ]
    )
    assert "1.23" in text
    assert "NaN" not in text and "None" not in text
    # And when no row has a move: the column is then all-missing, where pandas
    # would print `None` per row unless it is a float column to begin with.
    every = render_text([pick("A", 0.70, auc=0.62), pick("B", 0.71, auc=0.63)])
    assert "NaN" not in every and "None" not in every


def test_the_report_says_when_the_names_were_chosen_for_moving():
    """Ten movers read as a universe would look like a market of only risers."""
    text = render_text([pick("GOOD", 0.70, auc=0.62)], selection="the 1 biggest gainers")
    assert "the 1 biggest gainers" in text
    assert "biggest gainers" not in render_text([pick("GOOD", 0.70, auc=0.62)])


def test_the_cli_exposes_the_gainers_pass():
    args = build_parser().parse_args(["shortlist", "--gainers", "10"])
    assert args.gainers == 10
    assert build_parser().parse_args(["shortlist"]).gainers is None
    for bad in ("0", "-2", "ten"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["shortlist", "--gainers", bad])
