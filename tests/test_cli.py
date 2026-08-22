import math

import pandas as pd
import pytest

from gapmodel.cli import _last_monday, _since_timestamp, build_parser, main
from gapmodel.markets import MARKETS
from gapmodel.staleness import StaleInputs


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
    monkeypatch.setattr("gapmodel.cli._fresh_enough", lambda _panel, _args, targets: list(targets))

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
    assert called["kwargs"]["symbols"]["Europe"] == [
        m.symbol for m in MARKETS if m.region == "Europe"
    ]


def test_web_serves_only_the_markets_that_passed_the_staleness_check(monkeypatch):
    called = {}

    monkeypatch.setattr("gapmodel.cli._panel", lambda _args: {"dummy": None})
    monkeypatch.setattr("gapmodel.cli._hourly", lambda _args: None)
    monkeypatch.setattr("gapmodel.cli._fresh_enough", lambda _panel, _args, _targets: ["^N225"])
    monkeypatch.setattr(
        "gapmodel.cli.serve_dashboard", lambda *_args, **kwargs: called.update(kwargs)
    )
    main(["web", "--no-browser"])

    assert called["symbols"]["Asia"] == ["^N225"]
    assert called["symbols"]["Europe"] == []


def test_web_refuses_to_serve_a_panel_that_is_too_stale(monkeypatch):
    monkeypatch.setattr("gapmodel.cli._panel", lambda _args: {"dummy": None})
    monkeypatch.setattr("gapmodel.cli._hourly", lambda _args: None)

    def stale(*_args, **_kwargs):
        raise StaleInputs("^N225 is 9 days behind")

    monkeypatch.setattr("gapmodel.cli._fresh_enough", stale)

    def fail(*_args, **_kwargs):
        raise AssertionError("no server should be started on stale inputs")

    monkeypatch.setattr("gapmodel.cli.serve_dashboard", fail)
    with pytest.raises(SystemExit) as exit_info:
        main(["web", "--no-browser"])
    assert "9 days behind" in str(exit_info.value)


@pytest.mark.parametrize("port", ["-1", "65536", "http"])
def test_a_port_outside_the_tcp_range_is_rejected_at_parse_time(port, capsys):
    with pytest.raises(SystemExit):
        main(["web", "--port", port])
    assert "--port" in capsys.readouterr().err


def test_port_zero_is_accepted_so_the_os_can_pick_one():
    args = build_parser().parse_args(["web", "--port", "0"])
    assert args.port == 0


def test_at_rejects_anything_that_is_not_a_time_of_day(capsys):
    with pytest.raises(SystemExit):
        main(["dashboard", "--at", "2024-01-01"])
    assert "is not a time of day" in capsys.readouterr().err


def test_at_accepts_a_single_digit_hour():
    assert build_parser().parse_args(["dashboard", "--at", "5:00"]).at == 5.0


def test_last_monday_is_a_monday_at_midnight():
    """Sessions are indexed on the normalised date, so the cutoff must be too."""
    ts = _last_monday()
    assert ts.weekday() == 0, "should be a Monday"
    assert ts == ts.normalize()


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


def test_since_timestamp_drops_the_timezone():
    """An aware cutoff cannot be compared against the tz-naive session index."""
    import argparse

    import pandas as pd

    args = argparse.Namespace(last_week=False, since="2026-07-28T12:00:00+02:00")
    assert _since_timestamp(args) == pd.Timestamp("2026-07-28 10:00:00")


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


def test_backtest_rejects_both_window_flags(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["backtest", "--since", "2026-07-28", "--last-week"])
    assert "not allowed with" in capsys.readouterr().err


def test_scorecard_accepts_a_window_a_stock_and_a_log():
    parser = build_parser()
    args = parser.parse_args(
        ["scorecard", "--market", "mu", "--window", "40", "--log", "docs/log.csv"]
    )
    assert args.market == ["MU"]
    assert args.window == 40
    assert args.log == "docs/log.csv"


def test_scorecard_rejects_an_unmodelled_symbol(capsys):
    with pytest.raises(SystemExit):
        main(["scorecard", "--market", "^NOPE"])
    assert "unknown market" in capsys.readouterr().err


def test_scorecard_rejects_an_empty_window_before_fitting_anything(capsys):
    with pytest.raises(SystemExit):
        main(["scorecard", "--window", "0"])
    assert "must be greater than 0" in capsys.readouterr().err


def test_journal_and_scorecard_are_separate_commands():
    # Both score the model's calls, but scorecard reads the walk-forward's own
    # predictions and journal reads what was written down before each open.
    parser = build_parser()
    live = parser.parse_args(
        ["journal", "--market", "^FTSE", "--window", "90", "--min-settled", "5", "--settle-only"]
    )
    assert live.func is not parser.parse_args(["scorecard"]).func
    assert live.market == ["^FTSE"]
    assert (live.window, live.min_settled) == (90, 5)
    assert live.settle_only and not live.fail_on_decay
    assert live.log.endswith("forecast-log.csv")


def test_a_shorter_journal_window_leaves_the_minimum_to_be_narrowed():
    # Unset rather than 20: a caller narrowing the window never named the
    # default, so it is capped at the window instead of refused.
    assert build_parser().parse_args(["journal", "--window", "10"]).min_settled is None


def test_journal_refuses_a_minimum_its_window_cannot_hold_before_fitting_anything(tmp_path):
    # Rejected up front: the window is trimmed before the minimum is applied, so
    # the run would fit every model and then report nothing for any market.
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "journal",
                "--log",
                str(tmp_path / "forecast-log.csv"),
                "--window",
                "10",
                "--min-settled",
                "20",
            ]
        )
    assert "exceeds --window" in str(exit_info.value)


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


def test_the_gainers_line_counts_the_names_the_report_holds(monkeypatch, capsys):
    """A count taken before the drops would claim movers the table does not hold."""
    from gapmodel import cli
    from tests.test_shortlist import pick

    bars = pd.DataFrame(
        {"Close": [100.0, 105.0]}, index=pd.to_datetime(["2026-08-13", "2026-08-14"])
    )
    monkeypatch.setattr(cli, "_panel", lambda _args: {})
    monkeypatch.setattr(
        cli, "load_panel", lambda **_kwargs: {"AAPL": bars, "MSFT": bars, "NVDA": bars}
    )
    monkeypatch.setattr(cli, "biggest_gainers", lambda *_args: ["AAPL", "MSFT", "NVDA"])
    # Both drop paths at once: MSFT is stale, and NVDA is fresh but too short to
    # train, so only the pick that forecast_universe returns reaches the table.
    monkeypatch.setattr(cli, "_fresh_enough", lambda _panel, _args, targets: ["AAPL", "NVDA"])
    monkeypatch.setattr(
        cli, "forecast_universe", lambda *_args, **_kwargs: [pick("AAPL", 0.70, auc=0.62)]
    )
    main(["shortlist", "AAPL", "MSFT", "NVDA", "--gainers", "3"])

    out = capsys.readouterr().out
    assert "1 of the 3 biggest gainers of session 2026-08-14, out of 3 candidates" in out


def test_the_gainers_line_claims_the_ranking_only_when_it_kept_every_mover():
    """The dropped mover can be the largest riser, so survivors are not the top K."""
    from gapmodel.cli import _mover_selection

    kept_all = _mover_selection(3, 3, 158, "2026-08-14")
    assert kept_all.startswith("the 3 biggest gainers of session 2026-08-14, out of 158")
    assert _mover_selection(2, 3, 158, "2026-08-14").startswith("2 of the 3 biggest gainers")
    assert _mover_selection(0, 3, 158, "2026-08-14").startswith("0 of the 3 biggest gainers")
    assert _mover_selection(1, 1, 158, "2026-08-14").startswith("the biggest gainer of session")


def _all_dropped_shortlist(monkeypatch, movers):
    """A ``--gainers`` run whose every chosen mover leaves before the forecast."""
    from gapmodel import cli

    bars = pd.DataFrame(
        {"Close": [100.0, 105.0]}, index=pd.to_datetime(["2026-08-13", "2026-08-14"])
    )
    monkeypatch.setattr(cli, "_panel", lambda _args: {})
    monkeypatch.setattr(cli, "load_panel", lambda **_kwargs: {name: bars for name in movers})
    monkeypatch.setattr(cli, "biggest_gainers", lambda *_args: list(movers))
    monkeypatch.setattr(cli, "_fresh_enough", lambda _panel, _args, targets: list(targets))

    def no_model(*_args, **_kwargs):
        raise RuntimeError("no stock could be modelled")

    monkeypatch.setattr(cli, "forecast_universe", no_model)


def test_losing_every_mover_names_them_instead_of_blaming_the_universe(monkeypatch):
    """ "No stock could be modelled" reads as a broken cache; the movers dropped."""
    _all_dropped_shortlist(monkeypatch, ["AAPL", "MSFT", "NVDA"])
    with pytest.raises(SystemExit) as excinfo:
        main(["shortlist", "AAPL", "MSFT", "NVDA", "--gainers", "3"])

    message = str(excinfo.value)
    assert "all 3 biggest gainers of session 2026-08-14 were dropped" in message
    assert "AAPL, MSFT, NVDA" in message
    # The reader is told how to get a report, not just that there isn't one.
    assert "--refresh" in message and "--gainers" in message
    # A stale mover is stopped by _fresh_enough before the fit, so offering
    # staleness here would send the reader after a cause this path cannot have.
    assert "stale" not in message


def test_losing_the_only_mover_is_singular(monkeypatch):
    """ "All 1 biggest gainers were dropped" would be a plural about one name."""
    _all_dropped_shortlist(monkeypatch, ["AAPL"])
    with pytest.raises(SystemExit) as excinfo:
        main(["shortlist", "AAPL", "--gainers", "1"])

    message = str(excinfo.value)
    assert "the biggest gainer of session 2026-08-14 was dropped" in message
    # The tail is singular too: "none of them" about one name is the same slip.
    assert "(AAPL), not fittable; the warning says why" in message


def test_a_run_without_gainers_keeps_the_universe_wide_error(monkeypatch):
    """Without a mover pass there are no chosen names to blame, so nothing is claimed."""
    _all_dropped_shortlist(monkeypatch, ["AAPL", "MSFT"])
    with pytest.raises(SystemExit) as excinfo:
        main(["shortlist", "AAPL", "MSFT"])

    assert str(excinfo.value) == "error: no stock could be modelled"


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


def test_screen_rejects_combining_the_universe_forms(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("AAPL\n", encoding="utf-8")
    with pytest.raises(SystemExit) as both:
        main(["screen", "AAPL", "--universe", str(path)])
    assert "not both" in str(both.value)
    with pytest.raises(SystemExit) as etfs:
        main(["screen", "--universe", str(path), "--etfs"])
    assert "default universe only" in str(etfs.value)


def test_screen_rejects_negative_thresholds(capsys):
    with pytest.raises(SystemExit):
        main(["screen", "--min-avg-volume", "-5"])
    assert "must not be negative" in capsys.readouterr().err


def test_screen_rejects_an_unreadable_universe_file(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["screen", "--universe", str(tmp_path / "missing.txt")])
    assert "error:" in str(exit_info.value)


def test_score_rejects_a_comparison_universe_without_relative(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("AAPL\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["score", "IVZ", "--universe", str(path)])
    assert "--universe applies to --relative" in str(exit_info.value)


def test_score_rejects_an_unreadable_comparison_universe(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["score", "IVZ", "--relative", "--universe", str(tmp_path / "missing.txt")])
    assert "error:" in str(exit_info.value)


def test_score_relative_compares_against_the_us_universe_by_default(monkeypatch, capsys):
    from gapmodel import cli
    from gapmodel.score import Reference, RelativeScore
    from gapmodel.universe import us_universe

    seen = {}

    def fake_relative(symbols, universe, **kwargs):
        seen["symbols"] = symbols
        seen["universe"] = universe
        scored = [RelativeScore("IVZ", 2.59, 1.4, 92.0, 32.55, pd.Timestamp("2026-08-14"), 200)]
        return scored, Reference(pd.Timestamp("2026-08-14"), 150, 1.2, 0.98, stale=("KHC",))

    monkeypatch.setattr(cli, "relative_scores", fake_relative)
    main(["score", "ivz", "--relative"])

    assert seen["symbols"] == ["IVZ"]
    assert seen["universe"] == us_universe()
    out = capsys.readouterr().out
    assert "IVZ" in out and "1.4" in out
    assert "universe: 150 names as of 2026-08-14" in out
    # The count is the headline of the stale line, and one laggard exercises the
    # short-list branch the long-list unit test never reaches.
    assert "stale (1 of 150, still in the mean at an earlier close): KHC" in out


def test_score_without_relative_prints_the_raw_table(monkeypatch, capsys):
    from gapmodel import cli
    from gapmodel.score import TrendScore

    def fake_relative(*_args, **_kwargs):
        raise AssertionError("relative_scores must not run without --relative")

    monkeypatch.setattr(cli, "relative_scores", fake_relative)
    monkeypatch.setattr(
        cli,
        "score_symbols",
        lambda symbols, **kwargs: [TrendScore("IVZ", 2.59, 32.55, pd.Timestamp("2026-08-14"), 200)],
    )
    main(["score", "IVZ"])

    out = capsys.readouterr().out
    assert "IVZ" in out
    assert "universe:" not in out
    assert "relative" not in out


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


def _stub_forecast(name: str, caveats: tuple[str, ...], session: str = "2026-09-04"):
    from gapmodel.predict import Forecast

    return Forecast(
        symbol="^GSPC",
        name=name,
        region="Americas",
        session=pd.Timestamp(session),
        probability_up=0.6,
        backtest={},
        contributions=pd.Series(dtype=float),
        caveats=caveats,
    )


def test_a_scheduled_release_is_reported_next_to_the_probability(capsys):
    from gapmodel import cli

    cli._print_caveats([_stub_forecast("S&P 500", ("US payrolls at 13:30 UTC, before this open",))])
    out = capsys.readouterr().out
    assert "cannot see" in out
    assert "S&P 500: US payrolls at 13:30 UTC" in out


def test_an_uneventful_run_prints_no_caveat_section(capsys):
    from gapmodel import cli

    cli._print_caveats([_stub_forecast("S&P 500", ())])
    assert capsys.readouterr().out == ""


def test_a_session_past_a_calendar_is_told_the_calendar_ran_out(capsys):
    """Only the FOMC table reaches 2027, so the rest must say they were unread."""
    from gapmodel import cli

    cli._print_caveats([_stub_forecast("S&P 500", (), session="2027-09-15")])
    out = capsys.readouterr().out
    assert "not checked for 2027-09-15" in out
    assert "US CPI: table ends 2026-12-31" in out
    assert "FOMC decision" not in out


def test_a_run_spanning_the_year_end_warns_about_the_session_that_needs_it(capsys):
    """Tokyo's next session can be past the tables while New York's is not."""
    from gapmodel import cli

    cli._print_caveats(
        [
            _stub_forecast("S&P 500", (), session="2026-12-31"),
            _stub_forecast("Nikkei 225", (), session="2027-01-04"),
        ]
    )
    out = capsys.readouterr().out
    assert "not checked for 2027-01-04" in out
    assert "2026-12-31 —" not in out
