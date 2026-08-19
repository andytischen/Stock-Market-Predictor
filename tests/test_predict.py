import pytest

from gapmodel.predict import forecast_all


def test_an_empty_symbol_list_forecasts_nothing_rather_than_everything(monkeypatch):
    """A caller whose filter dropped every candidate asked for no forecasts."""
    asked = []

    def fake_forecast_market(symbol, *_args, **_kwargs):
        asked.append(symbol)
        raise RuntimeError("not modelled in this test")

    monkeypatch.setattr("gapmodel.predict.forecast_market", fake_forecast_market)
    with pytest.raises(RuntimeError, match="no market could be modelled"):
        forecast_all({}, symbols=[])
    assert asked == []
