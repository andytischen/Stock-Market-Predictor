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


def test_web_command_starts_server_with_expected_arguments(monkeypatch):
    called = {}

    monkeypatch.setattr("gapmodel.cli._panel", lambda _args: {"dummy": None})
    monkeypatch.setattr("gapmodel.cli._hourly", lambda _args: None)

    def fake_serve_dashboard(panel, hourly, **kwargs):
        called["panel"] = panel
        called["hourly"] = hourly
        called["kwargs"] = kwargs

    monkeypatch.setattr("gapmodel.cli.serve_dashboard", fake_serve_dashboard)
    main(["web", "--region", "Europe", "--at", "05:00", "--no-browser", "--port", "8123"])

    assert called["panel"] == {"dummy": None}
    assert called["hourly"] is None
    assert called["kwargs"]["region"] == "Europe"
    assert called["kwargs"]["at"] == 5.0
    assert called["kwargs"]["port"] == 8123
    assert called["kwargs"]["launch_browser"] is False
def test_last_monday_5am_is_a_monday_at_5am():
    import pandas as pd

    ts = _last_monday_5am()
    assert ts.weekday() == 0, "should be a Monday"
    assert ts.hour == 5 and ts.minute == 0


def test_since_timestamp_last_week():
    import argparse

    import pandas as pd

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
