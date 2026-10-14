"""Local browser weather dashboard backed by the public Open-Meteo APIs."""

from __future__ import annotations

import json
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .weather import WeatherApiError, fetch_weather, report_dict, search_locations
from .web import bind_family, browser_url, reachable_beyond_this_machine


def _index_html(default_query: str) -> str:
    seed = escape(default_query, quote=True)
    return f"""<!doctype html>
<html lang=\"en\">
<meta charset=\"utf-8\">
<title>Gapmodel weather dashboard</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 1rem; color: #222; }}
 form {{ display: flex; gap: .75rem; align-items: end; flex-wrap: wrap; }}
 label {{ display: grid; gap: .25rem; }}
 select {{ min-width: 20rem; }}
 button {{ cursor: pointer; }}
 [hidden] {{ display: none !important; }}
 table {{ border-collapse: collapse; margin-top: .75rem; width: min(60rem, 100%); }}
 th, td {{ padding: .35rem .6rem; border-bottom: 1px solid #ddd; text-align: left; }}
 th {{ font-weight: 600; }}
 .meta {{ color: #555; margin-top: .35rem; }}
 .error {{ color: #8b0000; }}
</style>
<h1>Gapmodel weather dashboard</h1>
<form id=\"search-form\" aria-label=\"Weather location search\">
  <label for=\"location-search\">Location search
    <input id=\"location-search\" name=\"q\" value=\"{seed}\" autocomplete=\"off\">
  </label>
  <button id=\"search-button\" type=\"submit\">Search</button>
  <label for=\"location-results\">Select location
    <select id=\"location-results\" aria-label=\"Location results\" size=\"6\" disabled></select>
  </label>
  <button id=\"show-weather\" type=\"button\" disabled>Show weather</button>
</form>
<p id=\"loading\" role=\"status\" aria-live=\"polite\" hidden>Loading weather…</p>
<p id=\"empty\" role=\"status\" aria-live=\"polite\">Search for a city to view weather.</p>
<p id=\"error\" role=\"alert\" class=\"error\" hidden></p>
<section id=\"weather-panel\" hidden>
  <h2 id=\"location-name\"></h2>
  <p id=\"location-meta\" class=\"meta\"></p>
  <table aria-label=\"Current weather\">
    <tr><th>Temperature</th><td id=\"current-temp\"></td></tr>
    <tr><th>Feels like</th><td id=\"current-feels\"></td></tr>
    <tr><th>Condition</th><td id=\"current-condition\"></td></tr>
    <tr><th>Precipitation</th><td id=\"current-precip\"></td></tr>
    <tr><th>Wind</th><td id=\"current-wind\"></td></tr>
  </table>
  <h3>Forecast</h3>
  <table aria-label=\"Weather forecast\">
    <thead>
      <tr>
        <th>Date</th><th>Condition</th><th>Min</th><th>Max</th>
        <th>Precipitation chance</th><th>Wind</th>
      </tr>
    </thead>
    <tbody id=\"forecast-body\"></tbody>
  </table>
</section>
<script>
const loading = document.getElementById('loading');
const empty = document.getElementById('empty');
const error = document.getElementById('error');
const panel = document.getElementById('weather-panel');
const searchForm = document.getElementById('search-form');
const input = document.getElementById('location-search');
const select = document.getElementById('location-results');
const showBtn = document.getElementById('show-weather');

const showState = ({{isLoading = false, emptyText = '', errorText = ''}} = {{}}) => {{
  loading.hidden = !isLoading;
  empty.hidden = !emptyText;
  empty.textContent = emptyText;
  error.hidden = !errorText;
  error.textContent = errorText;
}};

const optionLabel = (loc) => [loc.name, loc.admin1, loc.country].filter(Boolean).join(', ');

const render = (payload) => {{
  const current = payload.current;
  const forecast = payload.forecast || [];
  document.getElementById('location-name').textContent = optionLabel(payload.location);
  const locationMeta = document.getElementById('location-meta');
  locationMeta.textContent = `${{payload.location.latitude.toFixed(3)}}, ` +
    `${{payload.location.longitude.toFixed(3)}} (` + payload.location.timezone + ')';
  document.getElementById('current-temp').textContent = `${{current.temperature_c.toFixed(1)}} °C`;
  document.getElementById('current-feels').textContent = `${{current.feels_like_c.toFixed(1)}} °C`;
  document.getElementById('current-condition').textContent = current.condition;
  document.getElementById('current-precip').textContent =
    `${{current.precipitation_mm.toFixed(1)}} mm`;
  document.getElementById('current-wind').textContent =
    `${{current.wind_speed_kmh.toFixed(1)}} km/h`;
  const rows = forecast.map((day) =>
    `<tr><td>${{day.date}}</td><td>${{day.condition}}</td>` +
    `<td>${{day.temp_min_c.toFixed(1)}} °C</td>` +
    `<td>${{day.temp_max_c.toFixed(1)}} °C</td><td>${{day.precipitation_chance.toFixed(0)}}%</td>` +
    `<td>${{day.wind_speed_kmh.toFixed(1)}} km/h</td></tr>`
  ).join('');
  document.getElementById('forecast-body').innerHTML =
    rows || '<tr><td colspan="6">No forecast data</td></tr>';
  panel.hidden = false;
}};

searchForm.addEventListener('submit', async (event) => {{
  event.preventDefault();
  const q = input.value.trim();
  panel.hidden = true;
  select.innerHTML = '';
  showBtn.disabled = true;
  select.disabled = true;
  if (!q) {{
    showState({{emptyText: 'Type a location to search.'}});
    return;
  }}
  showState({{isLoading: true}});
  try {{
    const response = await fetch(`/api/weather/search?${{new URLSearchParams({{q}})}}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Search failed');
    if (!body.locations.length) {{
      showState({{emptyText: 'No locations found.'}});
      return;
    }}
    body.locations.forEach((loc, index) => {{
      const option = document.createElement('option');
      option.value = JSON.stringify(loc);
      option.textContent = optionLabel(loc);
      option.selected = index === 0;
      select.appendChild(option);
    }});
    select.disabled = false;
    showBtn.disabled = false;
    select.focus();
    showState();
  }} catch (err) {{
    showState({{errorText: `Could not search locations: ${{err.message}}`}});
  }}
}});

showBtn.addEventListener('click', async () => {{
  if (!select.value) {{
    showState({{emptyText: 'Select a location first.'}});
    return;
  }}
  panel.hidden = true;
  showState({{isLoading: true}});
  const loc = JSON.parse(select.value);
  const query = new URLSearchParams({{
    latitude: String(loc.latitude),
    longitude: String(loc.longitude),
    name: loc.name,
    country: loc.country || '',
    timezone: loc.timezone || 'auto',
  }});
  try {{
    const response = await fetch(`/api/weather/forecast?${{query}}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Forecast failed');
    render(body);
    showState();
  }} catch (err) {{
    showState({{errorText: `Could not load weather: ${{err.message}}`}});
  }}
}});

if (input.value.trim()) {{
  searchForm.dispatchEvent(new Event('submit'));
}}
</script>
</html>
"""


def _json(status: int, payload: dict) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode("utf-8")


def _handler(default_query: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/":
                self._reply(
                    200,
                    _index_html(default_query).encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if parsed.path == "/api/weather/search":
                q = (query.get("q") or [""])[0].strip()
                if not q:
                    self._reply_json(*_json(400, {"error": "search query is required"}))
                    return
                try:
                    locations = [
                        {
                            "name": loc.name,
                            "country": loc.country,
                            "admin1": loc.admin1,
                            "latitude": loc.latitude,
                            "longitude": loc.longitude,
                            "timezone": loc.timezone,
                        }
                        for loc in search_locations(q)
                    ]
                except WeatherApiError as exc:
                    self._reply_json(*_json(502, {"error": str(exc)}))
                    return
                self._reply_json(*_json(200, {"locations": locations}))
                return

            if parsed.path == "/api/weather/forecast":
                try:
                    latitude = float((query.get("latitude") or [""])[0])
                    longitude = float((query.get("longitude") or [""])[0])
                except ValueError:
                    self._reply_json(
                        *_json(400, {"error": "latitude and longitude must be numbers"})
                    )
                    return
                name = (query.get("name") or ["Selected location"])[0]
                country = (query.get("country") or [""])[0]
                timezone = (query.get("timezone") or ["auto"])[0]
                try:
                    report = fetch_weather(
                        latitude,
                        longitude,
                        name=name,
                        country=country,
                        timezone=timezone,
                    )
                except WeatherApiError as exc:
                    self._reply_json(*_json(502, {"error": str(exc)}))
                    return
                self._reply_json(*_json(200, report_dict(report)))
                return

            self._reply(404, b'{"error": "not found"}', "application/json; charset=utf-8")

        def _reply_json(self, status: int, body: bytes) -> None:
            self._reply(status, body, "application/json; charset=utf-8")

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def serve_weather(
    *,
    host: str,
    port: int,
    launch_browser: bool,
    default_query: str,
) -> None:
    host = host.strip("[]")

    class Server(ThreadingHTTPServer):
        address_family = bind_family(host)

    server = Server((host, port), _handler(default_query))
    address = browser_url(host, server.server_port)
    if reachable_beyond_this_machine(host):
        print(
            f"warning: bound to {host}, so anyone who can reach this machine can "
            "use this weather dashboard; there is no authentication"
        )
    print(f"serving weather dashboard at {address} (Ctrl+C to stop)")
    if launch_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()
