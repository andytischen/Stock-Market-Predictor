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
