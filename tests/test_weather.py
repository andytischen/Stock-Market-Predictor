import json
from urllib.error import URLError

import pytest

from gapmodel import weather


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_search_locations_parses_open_meteo_geocoding_payload(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse(
            {
                "results": [
                    {
                        "name": "Seoul",
                        "country": "South Korea",
                        "admin1": "Seoul",
                        "latitude": 37.566,
                        "longitude": 126.9784,
                        "timezone": "Asia/Seoul",
                    }
                ]
            }
        )

    monkeypatch.setattr(weather, "_urlopen", fake_urlopen)
    found = weather.search_locations("Seoul")
    assert len(found) == 1
    assert found[0].name == "Seoul"
    assert found[0].timezone == "Asia/Seoul"
    assert "name=Seoul" in seen["url"]
    assert seen["timeout"] == 20


def test_search_locations_skips_the_network_when_query_is_blank(monkeypatch):
    monkeypatch.setattr(weather, "_urlopen", lambda *_args, **_kwargs: pytest.fail("no call"))
    assert weather.search_locations("   ") == []


def test_fetch_weather_parses_current_and_forecast(monkeypatch):
    payload = {
        "timezone": "America/New_York",
        "current": {
            "temperature_2m": 24.2,
            "apparent_temperature": 25.1,
            "precipitation": 0.2,
            "weather_code": 3,
            "wind_speed_10m": 18.4,
        },
        "daily": {
            "time": ["2026-09-26", "2026-09-27"],
            "weather_code": [3, 1],
            "temperature_2m_min": [18.1, 19.0],
            "temperature_2m_max": [26.8, 27.2],
            "precipitation_probability_max": [30, 10],
            "wind_speed_10m_max": [25.0, 20.0],
        },
    }

    monkeypatch.setattr(weather, "_urlopen", lambda *_args, **_kwargs: FakeResponse(payload))
    report = weather.fetch_weather(40.7, -74.0, name="New York", country="US")
    assert report.location.name == "New York"
    assert report.current.temperature_c == pytest.approx(24.2)
    assert report.current.feels_like_c == pytest.approx(25.1)
    assert report.current.condition == "Overcast"
    assert len(report.forecast) == 2
    assert report.forecast[0].date == "2026-09-26"
    assert report.forecast[0].condition == "Overcast"


def test_weather_api_errors_are_wrapped(monkeypatch):
    def fail(*_args, **_kwargs):
        raise URLError("timeout")

    monkeypatch.setattr(weather, "_urlopen", fail)
    with pytest.raises(weather.WeatherApiError, match="could not read weather API response"):
        weather.search_locations("London")
