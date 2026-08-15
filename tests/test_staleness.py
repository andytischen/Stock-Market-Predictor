import pandas as pd
import pytest

from gapmodel.cli import _shared_inputs, build_parser
from gapmodel.staleness import (
    STALE_DAYS,
    StaleInputs,
    fresh_targets,
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
    """A stale feature is read by every model in the run; a stale target by one."""
    universe = panel(AAPL=1, MSFT=1, HALTED=30)
    with caplog.at_level("WARNING"):
        kept = fresh_targets(universe, ["AAPL", "MSFT", "HALTED"], SESSION)
    assert kept == ["AAPL", "MSFT"]
    assert "HALTED (30d)" in caplog.text


def test_a_run_whose_every_target_is_dead_fails_rather_than_forecasting_nothing():
    with pytest.raises(StaleInputs) as raised:
        fresh_targets(panel(AAPL=30, MSFT=30), ["AAPL", "MSFT"], SESSION)
    assert "every requested name" in str(raised.value)


def test_targets_are_kept_when_the_stale_read_is_the_deliberate_one():
    assert fresh_targets(panel(AAPL=30), ["AAPL"], SESSION, allow=True) == ["AAPL"]


def test_a_target_is_guarded_for_itself_and_a_peer_for_everyone():
    """Which series can fail a whole run, and which only lose their own row.

    ``stock`` and ``shortlist`` load their names in bulk, so the panel holds
    listings this run may never forecast. Those cannot decide the run. A peer of
    something requested is a different thing: it is a column in another name's
    model, and stale it is read as a company that did not move.
    """
    loaded = panel(AAPL=1, MSFT=1, WDC=1, **{"^GSPC": 1, "005930.KS": 1})
    # MSFT and WDC are loaded but unasked for, so neither can fail this run.
    assert set(_shared_inputs(loaded, ["AAPL"])) == {"^GSPC", "005930.KS"}
    # WDC is in MU's peer list, so a MU run reads it as a feature.
    assert "WDC" in _shared_inputs(loaded, ["MU"])


def test_an_empty_series_is_neither_counted_nor_flagged():
    """It has no last bar to be behind; a download that returned nothing is a
    different failure, reported where the download happens."""
    counted, behind = stale_inputs({"AAPL": bars(SESSION), "GONE": pd.DataFrame()}, SESSION)
    assert (counted, behind) == (1, [])
    guard({"GONE": pd.DataFrame()}, SESSION)


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


@pytest.mark.parametrize("command", ["predict", "stock", "shortlist", "export"])
def test_every_forecasting_command_is_guarded(command):
    args = build_parser().parse_args([command])
    assert args.max_stale_days == STALE_DAYS
    assert args.allow_stale is False


def test_the_tolerance_must_be_a_positive_number_of_days():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-stale-days", "-1", "predict"])
