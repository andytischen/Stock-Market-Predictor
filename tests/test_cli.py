import pytest

from gapmodel.cli import main


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
