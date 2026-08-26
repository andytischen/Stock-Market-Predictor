import math

import pytest

from gapmodel.cli import build_parser, main
from gapmodel.markets import INDICATORS, MARKETS_BY_SYMBOL
from gapmodel.scenarios import SCENARIOS, scenario


def test_every_scenario_moves_known_instruments():
    known = set(MARKETS_BY_SYMBOL) | {i.symbol for i in INDICATORS}
    for name, s in SCENARIOS.items():
        assert s.moves, f"{name} moves nothing"
        assert set(s.moves) <= known, f"{name} moves an unknown instrument"


def test_opec_supply_increase_sells_crude_off():
    moves = scenario("opec-supply-increase").moves
    assert moves["CL=F"] < 0 and moves["BZ=F"] < moves["CL=F"]
    shocks = scenario("opec-supply-increase").shocks()
    assert shocks["CL=F"] == pytest.approx(math.log1p(moves["CL=F"]))


def test_unknown_scenario_is_rejected_at_parse_time(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["predict", "--scenario", "nope"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_shock_overrides_a_leg_of_the_scenario(monkeypatch):
    seen: dict[str, float] = {}

    def capture(_panel, **kwargs):
        seen.update(kwargs["shocks"])
        raise RuntimeError("stop here")

    monkeypatch.setattr("gapmodel.cli.forecast_all", capture)
    monkeypatch.setattr("gapmodel.cli._panel", lambda _args: {})
    with pytest.raises(SystemExit):
        main(["predict", "--scenario", "opec-supply-increase", "--shock", "CL=F=-6%"])
    assert seen["CL=F"] == pytest.approx(math.log(0.94))
    assert seen["BZ=F"] == pytest.approx(math.log1p(-0.04))


def test_markets_command_lists_scenarios(capsys):
    main(["markets"])
    assert "opec-supply-increase" in capsys.readouterr().out


def test_predict_without_a_scenario_defaults_to_none():
    assert build_parser().parse_args(["predict"]).scenario is None
