"""Open-Meteo weather client used by the local weather dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen as _urlopen

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherApiError(RuntimeError):
    """Raised when a weather API call fails or returns an unexpected payload."""


@dataclass(frozen=True)
class Location:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None = None


@dataclass(frozen=True)
class CurrentWeather:
    temperature_c: float
    feels_like_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    condition: str


@dataclass(frozen=True)
class ForecastDay:
    date: str
    condition: str
    temp_min_c: float
    temp_max_c: float
    precipitation_chance: float
    wind_speed_kmh: float


@dataclass(frozen=True)
class WeatherReport:
    location: Location
    current: CurrentWeather
    forecast: tuple[ForecastDay, ...]


def _json_from(url: str) -> dict:
    try:
        with _urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise WeatherApiError(f"could not read weather API response: {exc}") from exc


def describe_weather(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WEATHER_CODES.get(code, f"Weather code {code}")


def search_locations(query: str, *, count: int = 8) -> list[Location]:
    text = query.strip()
    if not text:
        return []
    url = f"{GEOCODING_API}?{urlencode({'name': text, 'count': count, 'format': 'json'})}"
    payload = _json_from(url)
    results = payload.get("results") or []
    return [
        Location(
            name=str(row["name"]),
            country=str(row.get("country") or ""),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            timezone=str(row.get("timezone") or "UTC"),
            admin1=str(row["admin1"]) if row.get("admin1") else None,
        )
        for row in results
        if all(key in row for key in ("name", "latitude", "longitude"))
    ]


def fetch_weather(
    latitude: float,
    longitude: float,
    *,
    name: str,
    country: str = "",
    timezone: str = "auto",
    days: int = 3,
) -> WeatherReport:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,wind_speed_10m_max",
        "timezone": timezone,
        "forecast_days": max(1, days),
    }
    payload = _json_from(f"{FORECAST_API}?{urlencode(params)}")

    current = payload.get("current")
    daily = payload.get("daily")
    if not isinstance(current, dict) or not isinstance(daily, dict):
        raise WeatherApiError("weather API response is missing current or daily data")

    weather = CurrentWeather(
        temperature_c=float(current["temperature_2m"]),
        feels_like_c=float(current["apparent_temperature"]),
        precipitation_mm=float(current["precipitation"]),
        wind_speed_kmh=float(current["wind_speed_10m"]),
        condition=describe_weather(int(current.get("weather_code", -1))),
    )

    dates = list(daily.get("time") or [])
    forecast = tuple(
        ForecastDay(
            date=str(dates[i]),
            condition=describe_weather(int((daily.get("weather_code") or [None])[i])),
            temp_min_c=float((daily.get("temperature_2m_min") or [float("nan")])[i]),
            temp_max_c=float((daily.get("temperature_2m_max") or [float("nan")])[i]),
            precipitation_chance=float(
                (daily.get("precipitation_probability_max") or [float("nan")])[i]
            ),
            wind_speed_kmh=float((daily.get("wind_speed_10m_max") or [float("nan")])[i]),
        )
        for i in range(len(dates))
    )

    return WeatherReport(
        location=Location(
            name=name,
            country=country,
            latitude=float(latitude),
            longitude=float(longitude),
            timezone=str(payload.get("timezone") or timezone),
        ),
        current=weather,
        forecast=forecast,
    )


def report_dict(report: WeatherReport) -> dict:
    return {
        "location": asdict(report.location),
        "current": asdict(report.current),
        "forecast": [asdict(day) for day in report.forecast],
    }
