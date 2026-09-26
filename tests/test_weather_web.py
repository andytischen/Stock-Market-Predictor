import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gapmodel import weather_web
from gapmodel.weather import CurrentWeather, ForecastDay, Location, WeatherApiError, WeatherReport


@pytest.fixture
def served(monkeypatch):
    sample_location = Location(
        name="London",
        country="United Kingdom",
        admin1="England",
        latitude=51.5072,
        longitude=-0.1276,
        timezone="Europe/London",
    )
    sample_report = WeatherReport(
        location=sample_location,
        current=CurrentWeather(
            temperature_c=19.5,
            feels_like_c=18.8,
            precipitation_mm=0.0,
            wind_speed_kmh=12.4,
            condition="Partly cloudy",
        ),
        forecast=(
            ForecastDay(
                date="2026-09-26",
                condition="Partly cloudy",
                temp_min_c=13.0,
                temp_max_c=21.5,
                precipitation_chance=20,
                wind_speed_kmh=17,
            ),
        ),
    )

    monkeypatch.setattr(weather_web, "search_locations", lambda _q: [sample_location])
    monkeypatch.setattr(weather_web, "fetch_weather", lambda *_args, **_kwargs: sample_report)

    server = ThreadingHTTPServer(("127.0.0.1", 0), weather_web._handler("London"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read().decode()


def test_index_page_contains_accessible_weather_controls(served):
    status, body = _get(f"{served}/")
    assert status == 200
    assert '<label for="location-search">' in body
    assert 'aria-label="Location results"' in body
    assert 'role="status"' in body
    assert 'role="alert"' in body


def test_search_endpoint_requires_a_query(served):
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{served}/api/weather/search")
    assert error.value.code == 400
    assert "search query is required" in error.value.read().decode()


def test_search_endpoint_returns_locations(served):
    status, body = _get(f"{served}/api/weather/search?q=London")
    payload = json.loads(body)
    assert status == 200
    assert payload["locations"][0]["name"] == "London"


def test_forecast_endpoint_returns_current_weather_and_forecast(served):
    status, body = _get(
        f"{served}/api/weather/forecast?latitude=51.5&longitude=-0.12&name=London&country=United+Kingdom"
    )
    payload = json.loads(body)
    assert status == 200
    assert payload["location"]["name"] == "London"
    assert payload["current"]["temperature_c"] == pytest.approx(19.5)
    assert payload["forecast"][0]["date"] == "2026-09-26"


def test_forecast_endpoint_rejects_invalid_coordinates(served):
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{served}/api/weather/forecast?latitude=bad&longitude=0")
    assert error.value.code == 400


def test_api_failures_are_returned_as_502(monkeypatch):
    def fail(_q):
        raise WeatherApiError("upstream down")

    monkeypatch.setattr(weather_web, "search_locations", fail)
    server = ThreadingHTTPServer(("127.0.0.1", 0), weather_web._handler(""))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(f"http://127.0.0.1:{server.server_port}/api/weather/search?q=Paris")
        assert error.value.code == 502
        assert "upstream down" in error.value.read().decode()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
