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


def test_shock_accepts_symbols_containing_equals():
    from gapmodel.predict import parse_shock

    assert parse_shock("CL=F=-5%") == ("CL=F", pytest.approx(math.log(0.95)))
    assert parse_shock("JPY=X=+2%") == ("JPY=X", pytest.approx(math.log(1.02)))
    assert build_parser().parse_args(["predict", "--shock", "CL=F=-5%"]).shock == [
        ("CL=F", pytest.approx(math.log(0.95)))
    ]


def test_shock_moves_every_feature_derived_from_the_instrument():
    from gapmodel.predict import shocked_row

    live = pd.DataFrame(
        {
            "ind_vix_return": [0.0],
            "ind_vix_level": [20.0],
            "ind_cl_f_return": [0.0],
            "ind_cl_f_return_5": [0.01],
            "ind_cl_f_vol_20": [0.02],
            "ind_cl_f_shock": [0.5],
        }
    )
    bumped = shocked_row(live, {"^VIX": math.log(1.1), "CL=F": 0.04})
    assert bumped["ind_vix_return"].iloc[0] == pytest.approx(math.log(1.1))
    assert bumped["ind_vix_level"].iloc[0] == pytest.approx(22.0)
    assert bumped["ind_cl_f_return"].iloc[0] == pytest.approx(0.04)
    assert bumped["ind_cl_f_return_5"].iloc[0] == pytest.approx(0.05)
    # The volatility denominator is measured to the previous bar, so it stays.
    assert bumped["ind_cl_f_vol_20"].iloc[0] == pytest.approx(0.02)
    assert bumped["ind_cl_f_shock"].iloc[0] == pytest.approx(0.5 + 0.04 / 0.02)


def test_screen_flags_are_scaled_into_criteria(monkeypatch, tmp_path):
    from gapmodel import cli
    from gapmodel.screener import Screen

    captured = {}

    def fake_screen(symbols, **kwargs):
        captured["symbols"] = symbols
        captured["criteria"] = kwargs["criteria"]
        captured["asof"] = kwargs["asof"]
        return Screen(criteria=kwargs["criteria"], stages=(), readings=(), asof=None)

    monkeypatch.setattr(cli, "screen", fake_screen)
    main(
        [
            "--cache",
            str(tmp_path),
            "screen",
            "nvda",
            "--min-volume",
            "3",
            "--min-avg-volume",
            "8",
            "--min-change",
            "2.5",
            "--min-atr",
            "4",
            "--asof",
            "2026-08-07",
        ]
    )
    criteria = captured["criteria"]
    assert captured["symbols"] == ["NVDA"]
    assert criteria.min_volume == pytest.approx(3e6)
    assert criteria.min_avg_volume == pytest.approx(8e6)
    assert criteria.min_change == pytest.approx(0.025)
    assert criteria.min_atr == pytest.approx(0.04)
    assert captured["asof"] == pd.Timestamp("2026-08-07")


def test_screen_defaults_to_the_us_universe(monkeypatch, tmp_path):
    from gapmodel import cli
    from gapmodel.screener import Screen

    seen = {}

    def fake_screen(symbols, **kwargs):
        seen["symbols"] = symbols
        return Screen(criteria=kwargs["criteria"], stages=(), readings=(), asof=None)

    monkeypatch.setattr(cli, "screen", fake_screen)
    main(["--cache", str(tmp_path), "screen"])
    assert "AAPL" in seen["symbols"] and "SPY" not in seen["symbols"]


def test_screen_rejects_an_unreadable_universe_file(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["screen", "--universe", str(tmp_path / "missing.txt")])
    assert "error:" in str(exit_info.value)


def test_intraday_falls_back_to_the_daily_model_when_futures_bars_are_missing(monkeypatch):
    """A stale futures feed should cost sharpness, not the whole forecast."""
    import argparse

    from gapmodel import cli

    attempts = []

    def fake_forecast_all(panel, **kwargs):
        attempts.append(kwargs.get("hourly"))
        if kwargs.get("hourly") is not None:
            raise RuntimeError("no market could be modelled")
        return ["forecast"]

    monkeypatch.setattr(cli, "forecast_all", fake_forecast_all)
    args = argparse.Namespace(market=["^IXIC"], regularisation=0.1)

    assert cli._forecast({}, args, {"ES=F": None}) == ["forecast"]
    assert attempts == [{"ES=F": None}, None]


def test_the_daily_model_failing_is_still_an_error(monkeypatch):
    import argparse

    from gapmodel import cli

    def fake_forecast_all(panel, **kwargs):
        raise RuntimeError("no market could be modelled")

    monkeypatch.setattr(cli, "forecast_all", fake_forecast_all)
    args = argparse.Namespace(market=["^IXIC"], regularisation=0.1)

    with pytest.raises(RuntimeError):
        cli._forecast({}, args, None)


def test_a_total_hourly_outage_still_yields_a_daily_forecast(monkeypatch, tmp_path):
    """Yahoo refusing every hourly request must not cost the forecast either."""
    import argparse

    from gapmodel import cli

    def refuse(**kwargs):
        raise RuntimeError("no hourly data could be loaded")

    monkeypatch.setattr(cli, "load_hourly_panel", refuse)
    args = argparse.Namespace(intraday=True, cache=str(tmp_path), refresh=False)

    assert cli._hourly(args) is None
