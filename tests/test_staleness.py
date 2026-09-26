from pathlib import Path

import pandas as pd
import pytest

from gapmodel import cli
from gapmodel.cli import _model_inputs, build_parser, main
from gapmodel.staleness import (
    STALE_DAYS,
    StaleInputs,
    fresh_forecasts,
    guard,
    lags,
    stale_inputs,
    today,
)

SESSION = pd.Timestamp("2026-08-14")


def bars(last: pd.Timestamp, rows: int = 10) -> pd.DataFrame:
    # Daily, not business-daily: a weekend `end` would snap back to the Friday
    # and quietly change the lag under test.
    index = pd.date_range(end=last, periods=rows, freq="D")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=index)


def panel(**lags_by_symbol: int) -> dict[str, pd.DataFrame]:
    return {
        symbol: bars(SESSION - pd.Timedelta(days=lag)) for symbol, lag in lags_by_symbol.items()
    }


def judged(loaded: dict[str, pd.DataFrame], targets: list[str], **how) -> list[str]:
    """``fresh_forecasts`` over what each target's own model reads, as the CLI does."""
    return fresh_forecasts(
        {target: _model_inputs(loaded, [target]) for target in targets}, SESSION, **how
    )


def alone(loaded: dict[str, pd.DataFrame], targets: list[str], **how) -> list[str]:
    """``fresh_forecasts`` where each name is the only series its model reads."""
    return fresh_forecasts({target: {target: loaded[target]} for target in targets}, SESSION, **how)


def test_a_run_on_current_inputs_is_left_alone():
    """Yesterday's close is what a market that has not opened yet knows."""
    guard(panel(AAPL=1, **{"^N225": 0, "^GSPC": 2}), SESSION)


def test_a_run_on_dead_inputs_fails_rather_than_forecasting_from_them():
    with pytest.raises(StaleInputs) as raised:
        guard(panel(AAPL=1, **{"^GSPC": 10, "CL=F": 9}), SESSION)
    message = str(raised.value)
    assert "2 of 3 input series have no bar within 5 days of 2026-08-14" in message
    # The lag itself, worst first, so the cause is visible without a second command.
    assert "^GSPC (10d), CL=F (9d)" in message
    assert "AAPL" not in message
    assert "--allow-stale" in message


def test_the_tolerance_is_the_first_lag_the_calendar_cannot_explain():
    assert STALE_DAYS == 5
    guard(panel(**{"^GSPC": 5}), SESSION)
    with pytest.raises(StaleInputs):
        guard(panel(**{"^GSPC": 6}), SESSION)


def test_a_wider_tolerance_is_honoured_in_both_directions():
    guard(panel(**{"^GSPC": 9}), SESSION, max_days=10)
    with pytest.raises(StaleInputs):
        guard(panel(AAPL=1), SESSION, max_days=0)


def test_forecasting_from_stale_inputs_on_purpose_still_says_so(caplog, capsys):
    with caplog.at_level("WARNING"):
        guard(panel(**{"^GSPC": 10}), SESSION, allow=True)
    assert "^GSPC (10d)" in caplog.text and "--allow-stale" in caplog.text
    # Logged, not printed: `export` writes its JSON to stdout, and a warning
    # there would be the first line of the document.
    assert capsys.readouterr().out == ""


def test_one_dead_listing_does_not_cancel_the_names_around_it(caplog):
    """A stale series costs the forecasts that read it, and no others."""
    universe = panel(AAPL=1, MSFT=1, HALTED=30)
    with caplog.at_level("WARNING"):
        kept = alone(universe, ["AAPL", "MSFT", "HALTED"])
    assert kept == ["AAPL", "MSFT"]
    assert "skipping 1 of 3 requested forecasts (HALTED)" in caplog.text
    # And the series behind it, which is what a reader can act on.
    assert "HALTED (30d)" in caplog.text


def test_a_run_whose_every_forecast_is_dead_fails_rather_than_forecasting_nothing(caplog):
    with pytest.raises(StaleInputs) as raised, caplog.at_level("WARNING"):
        alone(panel(AAPL=30, MSFT=30), ["AAPL", "MSFT"])
    message = str(raised.value)
    # Said as a refusal of the request, not as a count of series: a reader who
    # asked for two names is told both are gone, and how to widen the tolerance.
    assert "every requested forecast reads a series with no bar within 5 days" in message
    assert "--allow-stale" in message
    # Not "skipping" and then aborting: two lines describing two outcomes.
    assert "skipping" not in caplog.text


def test_the_footer_describes_the_series_the_run_read(caplog):
    """A skipped name's last value is not carried forward, so it is not an input.

    Handed the whole loaded panel the report would count the name it dropped and
    say its stale value was read anyway, which is the opposite of what happened.
    """
    loaded = panel(AAPL=1, ZM=30, **{"^GSPC": 1})
    with caplog.at_level("WARNING"):
        kept = alone(loaded, ["AAPL", "ZM"])
    counted, stale = stale_inputs({s: loaded[s] for s in kept}, SESSION)
    assert (counted, stale) == (1, [])


def test_forecasts_are_kept_when_the_stale_read_is_the_deliberate_one(caplog):
    """Kept, and still named: `stock` prints no footer, so this is the only place
    a reader learns the name in front of them stopped trading."""
    with caplog.at_level("WARNING"):
        assert alone(panel(AAPL=30), ["AAPL"], allow=True) == ["AAPL"]
    assert "AAPL (30d)" in caplog.text and "--allow-stale" in caplog.text


def test_a_forecast_is_judged_on_the_series_its_own_model_reads():
    """A run of the panel is narrower than the download, and narrower per name.

    ``stock`` and ``shortlist`` load their names in bulk, so the panel holds
    listings a given model never opens. Those cannot decide its forecast. A peer
    is a different thing: it is a column in that model, and stale it is read as
    a company that did not move.
    """
    loaded = panel(AAPL=1, MSFT=1, WDC=1, **{"^GSPC": 1, "005930.KS": 1})
    # MSFT is loaded but read by no model here, and Samsung is a memory peer
    # that an AAPL model never opens.
    assert set(_model_inputs(loaded, ["AAPL"])) == {"^GSPC", "AAPL"}
    # WDC is in MU's peer list, so a MU model reads it as a feature.
    assert "WDC" in _model_inputs(loaded, ["MU"])


def test_a_halted_curated_name_costs_the_models_that_hold_it_as_a_peer(caplog):
    """How far one silence reaches for ``stock``, whose names are mutual peers.

    MU is a column in WDC's model and in STX's, so a halted MU takes all three
    down: their metrics were earned over a history in which it was live, and
    fitting them without it would answer a different question. AAPL never opens
    MU, so it is still forecast — the run loses the names that read the dead
    feed rather than every name it was asked for.
    """
    halted = panel(MU=30, WDC=1, STX=1, AAPL=1, **{"^GSPC": 1})
    with caplog.at_level("WARNING"):
        assert judged(halted, ["MU", "WDC", "STX", "AAPL"]) == ["AAPL"]
    assert "skipping 3 of 4 requested forecasts (MU, STX, WDC)" in caplog.text
    assert "MU (30d)" in caplog.text
    # Asked for alone, the halted name is the whole run, so the run is refused.
    with pytest.raises(StaleInputs, match=r"MU \(30d\)"):
        judged(halted, ["MU"])


def test_a_series_no_model_in_the_run_reads_cannot_refuse_it():
    """The panel is one download; a run of it is narrower than the whole list.

    A European sector tracker is not a column in a US model, and an
    opening-price stand-in is read only as the gap source of its own index.
    Guarding a US run on those refuses a forecast over a feed it never opens.
    """
    loaded = panel(**{"^GSPC": 1, "^FTSE": 1, "EXH8.DE": 20, "ISF.L": 20})
    assert judged(loaded, ["^GSPC"]) == ["^GSPC"]
    # The same two series, read by the models that do read them.
    with pytest.raises(StaleInputs, match=r"EXH8\.DE"):
        judged(loaded, ["^GDAXI"])
    with pytest.raises(StaleInputs, match=r"ISF\.L"):
        judged(loaded, ["^FTSE"])


def test_half_a_paired_block_is_read_by_nothing_and_so_judges_nothing():
    """The crude curve is a spread and the policy rate a premium.

    Both are built from two legs or from neither, so when one download fails the
    survivor is a series no feature is derived from — and refusing a run over it
    is the same over-broad judgement as guarding a sector tracker nobody reads.
    """
    orphaned = panel(**{"^GSPC": 1, "USO": 20, "ZQ=F": 20})
    assert judged(orphaned, ["^GSPC"]) == ["^GSPC"]
    # Both legs present: the feature is built, so its silence stops the run.
    paired = panel(**{"^GSPC": 1, "USO": 20, "USL": 20})
    with pytest.raises(StaleInputs, match=r"USO"):
        judged(paired, ["^GSPC"])


def test_a_leg_that_arrived_empty_does_not_complete_its_pair():
    """A key with no bars is a partner in name only.

    A download that returned nothing builds no feature, so it must not be what
    turns its partner's silence into a refusal.
    """
    loaded = panel(**{"^GSPC": 1, "ZQ=F": 20})
    loaded["^IRX"] = pd.DataFrame()
    assert judged(loaded, ["^GSPC"]) == ["^GSPC"]


def test_an_empty_leg_is_still_named_as_a_download_that_returned_nothing(caplog):
    """Not judged for staleness, but not swallowed by the pair rule either.

    Dropping the pair answers the freshness question — nothing was built from the
    survivor — and that is a different question from whether the leg arrived at
    all, which has its own remedy and is reported on its own line.
    """
    loaded = panel(**{"^GSPC": 1, "ZQ=F": 1})
    loaded["^IRX"] = pd.DataFrame()
    with caplog.at_level("WARNING"):
        judged(loaded, ["^GSPC"])
    assert "^IRX" in caplog.text and "no bars at all" in caplog.text


def test_a_run_that_names_no_target_is_judged_on_everything_loaded():
    """Nothing is known about what it will read, so nothing is excused."""
    with pytest.raises(StaleInputs):
        guard(_model_inputs(panel(**{"EXH8.DE": 20}), []), SESSION)


def test_an_empty_series_is_neither_counted_nor_flagged():
    """It has no last bar to be behind; a download that returned nothing is a
    different failure, reported where the download happens."""
    counted, behind = stale_inputs({"AAPL": bars(SESSION), "GONE": pd.DataFrame()}, SESSION)
    assert (counted, behind) == (1, [])
    guard({"GONE": pd.DataFrame()}, SESSION)


def test_a_series_that_never_arrived_is_named_rather_than_hidden(caplog, capsys):
    """A count of the series that arrived reads as reassurance about the rest.

    It cannot be given a lag — it has no last bar to be behind one — so it is
    named separately instead, and on the log, where `export`'s JSON is not.
    """
    with caplog.at_level("WARNING"):
        guard({"AAPL": bars(SESSION), "GONE": pd.DataFrame()}, SESSION)
    assert "GONE" in caplog.text and "no bars at all" in caplog.text
    assert capsys.readouterr().out == ""


def test_the_series_that_never_arrived_are_named_in_a_fixed_order(caplog):
    """Only eight are named, so which eight cannot depend on the download order.

    The stale list is truncated worst-first, which the absent list has no
    equivalent of — none of them has a lag — so it is sorted instead, rather than
    naming whichever eight the panel happens to hold first.
    """
    absent = {symbol: pd.DataFrame() for symbol in "JIHGFEDCBA"}
    with caplog.at_level("WARNING"):
        guard({"AAPL": bars(SESSION), **absent}, SESSION)
    assert "A, B, C, D, E, F, G, H and 2 more" in caplog.text


def test_lag_is_measured_in_whole_days_from_any_time_of_day():
    """An intraday timestamp is a bar for that date, not a fraction of a lag."""
    intraday = {"AAPL": bars(SESSION - pd.Timedelta(days=9))}
    assert lags(intraday, SESSION + pd.Timedelta(hours=13.5)) == {"AAPL": 9}


def test_the_guard_reference_is_a_bare_date_in_utc():
    """A feed that died last week must not vouch for itself.

    The reference cannot be the next session after the panel's last bar, which
    is how a forecast is dated: a dead panel would date its own forecast one day
    past its own last bar and read as current. It is today, in UTC, to the day —
    tz-naive and normalised, because the bars it is compared against are.
    """
    reference = today()
    assert reference.tz is None
    assert reference == reference.normalize()
    assert abs(reference - pd.Timestamp.now(tz="UTC").tz_localize(None)) <= pd.Timedelta(days=1)


@pytest.mark.parametrize(
    "command", ["predict", "stock", "shortlist", "export", "dashboard", "sectors"]
)
def test_every_forecasting_command_is_guarded(command):
    args = build_parser().parse_args([command])
    assert args.max_stale_days == STALE_DAYS
    assert args.allow_stale is False


@pytest.mark.parametrize(
    "command", ["predict", "stock", "shortlist", "export", "dashboard", "sectors"]
)
def test_no_command_offers_a_probability_from_a_dead_panel(command, monkeypatch, capsys):
    """Whichever command a reader reaches for, the same cache gives the same answer.

    ``dashboard`` and ``sectors`` print a probability for the next open exactly as
    ``predict`` does, so one refusing while another prints from the same dead feed
    would tell a reader that the data is fine as long as they ask differently.
    """
    monkeypatch.setattr(cli, "load_panel", lambda **kwargs: panel(**{"^GSPC": 40, "CL=F": 40}))
    with pytest.raises(SystemExit) as raised:
        main([command])
    assert "no bar within 5 days" in str(raised.value)
    # Refused before anything was fitted, so there is no half-report on stdout.
    assert capsys.readouterr().out == ""


def test_the_tolerance_must_be_a_positive_number_of_days():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-stale-days", "-1", "predict"])


def test_the_unattended_publish_asks_for_a_tolerance_the_parser_accepts():
    """The one caller nobody is watching, parsed rather than eyeballed.

    ``--max-stale-days`` is a global flag, so writing it after the subcommand is
    a usage error rather than a wider tolerance: the daily job would fail every
    morning instead of only during a closure, and the first symptom would be a
    Pages snapshot that quietly stopped changing.
    """
    workflow = (Path(__file__).parent.parent / ".github/workflows/publish-snapshot.yml").read_text()
    published = next(
        line.split("python -m gapmodel")[1].split()
        for line in workflow.splitlines()
        if "python -m gapmodel" in line and "export" in line
    )
    args = build_parser().parse_args(published)
    # Wide enough for Golden Week, and still a boundary rather than an off switch.
    assert args.max_stale_days == 12
    assert args.allow_stale is False
