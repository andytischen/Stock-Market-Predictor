import pandas as pd
import pytest

from gapmodel.cli import build_parser
from gapmodel.staleness import STALE_DAYS, StaleInputs, guard, lags, stale_inputs, today

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


def test_forecasting_from_stale_inputs_on_purpose_still_says_so(capsys):
    guard(panel(**{"^GSPC": 10}), SESSION, allow=True)
    printed = capsys.readouterr().out
    assert "warning" in printed and "^GSPC (10d)" in printed and "--allow-stale" in printed


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
