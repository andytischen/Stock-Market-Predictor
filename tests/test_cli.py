import math

import pandas as pd
import pytest

from gapmodel.cli import _last_monday_5am, _since_timestamp, build_parser, main


def test_unknown_market_is_rejected_at_parse_time(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["predict", "--market", "^NOPE"])
    assert exit_info.value.code == 2
    assert "unknown market" in capsys.readouterr().err


def test_non_positive_regularisation_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["--regularisation", "0", "predict"])
    assert "greater than 0" in capsys.readouterr().err


def test_unusable_cache_reports_an_error_without_downloading(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("no download should be attempted")

    monkeypatch.setattr("gapmodel.data.load_symbol", fail)
    with pytest.raises(SystemExit) as exit_info:
        main(["--cache", "/proc/nope/x", "fetch"])
    assert "error:" in str(exit_info.value)


def test_markets_command_lists_every_market(capsys):
    main(["markets"])
    out = capsys.readouterr().out
    assert "^GSPC" in out and "Nikkei 225" in out


def test_last_monday_5am_is_a_monday_at_5am():

    ts = _last_monday_5am()
    assert ts.weekday() == 0, "should be a Monday"
    assert ts.hour == 5 and ts.minute == 0


def test_since_timestamp_last_week():
    import argparse

    args = argparse.Namespace(last_week=True, since=None)
    ts = _since_timestamp(args)
    assert ts is not None
    assert ts.weekday() == 0


def test_since_timestamp_explicit_date():
    import argparse

    import pandas as pd

    args = argparse.Namespace(last_week=False, since="2026-07-28")
    ts = _since_timestamp(args)
    assert ts == pd.Timestamp("2026-07-28")


def test_since_timestamp_none_when_neither_flag():
    import argparse

    args = argparse.Namespace(last_week=False, since=None)
    assert _since_timestamp(args) is None


def test_backtest_last_week_in_parser():
    parser = build_parser()
    args = parser.parse_args(["backtest", "--last-week"])
    assert args.last_week is True
    assert args.since is None


def test_backtest_since_in_parser():
    parser = build_parser()
    args = parser.parse_args(["backtest", "--since", "2026-07-28"])
    assert args.since == "2026-07-28"
    assert not args.last_week


def test_shock_parsing_accepts_percentages_and_fractions():
    from gapmodel.predict import parse_shock

    symbol, move = parse_shock("^KS11=+2%")
    assert symbol == "^KS11"
    assert abs(move - math.log(1.02)) < 1e-12
    assert abs(parse_shock("^KS11=0.02")[1] - math.log(1.02)) < 1e-12
    assert parse_shock("^KS11=-2%")[1] < 0


def test_shock_rejects_nonsense():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--shock", "^NOPE=+2%"])
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--shock", "^KS11"])


def test_shock_moves_only_the_features_of_that_symbol():
    from gapmodel.predict import shocked_row

    live = pd.DataFrame(
        {"mkt_ks11_return": [0.01], "mkt_ks11_return_5": [0.02], "mkt_n225_return": [0.03]}
    )
    bumped = shocked_row(live, {"^KS11": 0.1})
    assert bumped["mkt_ks11_return"].iloc[0] == pytest.approx(0.11)
    assert bumped["mkt_ks11_return_5"].iloc[0] == pytest.approx(0.12)
    assert bumped["mkt_n225_return"].iloc[0] == pytest.approx(0.03)
