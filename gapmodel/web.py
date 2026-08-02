"""Lightweight browser interface for the regional dashboard."""

from __future__ import annotations

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import pandas as pd

from .dashboard import build_dashboard, render_html
from .markets import MARKETS, REGIONS
from .model import MIN_TRAIN
from .predict import forecast_all

# Keep warm-up aligned with the CLI's intraday mode.
INTRADAY_MIN_TRAIN = 200


def parse_utc_time(value: str | None) -> float | None:
    """Return HH:MM UTC as hours from midnight."""
    if value is None or value == "":
        return None
    try:
        moment = pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a time of day (use HH:MM)") from exc
    return moment.hour + moment.minute / 60


def _as_of(hours: float | None) -> pd.Timestamp | None:
    if hours is None:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    return now.normalize() + pd.Timedelta(hours=hours)


def dashboard_document(
    panel: dict[str, pd.DataFrame],
    hourly: dict[str, pd.Series] | None,
    region: str,
    hour: float | None,
    regularisation: float,
) -> str:
    symbols = [m.symbol for m in MARKETS if m.region == region]
    forecasts = forecast_all(
        panel,
        symbols=symbols,
        c=regularisation,
        hourly=hourly,
        min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
    )
    board = build_dashboard(panel, forecasts, as_of=_as_of(hour), region=region)
    return render_html(board)


def _index_html(default_region: str, default_at: float | None) -> str:
    time_value = ""
    if default_at is not None:
        h = int(default_at)
        m = int(round((default_at - h) * 60))
        time_value = f"{h:02d}:{m:02d}"
    options = "".join(
        f'<option value="{r}"{" selected" if r == default_region else ""}>{r}</option>'
        for r in REGIONS
    )
    query = urlencode({"region": default_region, **({"at": time_value} if time_value else {})})
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Gapmodel dashboard</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 1rem; color: #222; }}
 .controls {{ display: flex; gap: .75rem; align-items: end; flex-wrap: wrap; margin-bottom: .75rem; }}
 label {{ display: grid; gap: .25rem; }}
 iframe {{ width: 100%; height: 85vh; border: 1px solid #ddd; }}
</style>
<h1>Gapmodel browser dashboard</h1>
<form class="controls" action="/dashboard" method="get" target="board">
  <label>Region
    <select name="region">{options}</select>
  </label>
  <label>UTC time (optional)
    <input type="time" name="at" value="{time_value}">
  </label>
  <button type="submit">Render</button>
</form>
<iframe name="board" src="/dashboard?{query}" title="dashboard"></iframe>
</html>
"""


def _handler(
    panel: dict[str, pd.DataFrame],
    hourly: dict[str, pd.Series] | None,
    default_region: str,
    default_at: float | None,
    regularisation: float,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/":
                self._reply(200, _index_html(default_region, default_at))
                return
            if parsed.path != "/dashboard":
                self._reply(404, "<h1>Not found</h1>")
                return

            region = query.get("region", [default_region])[0]
            if region not in REGIONS:
                self._reply(400, f"<h1>Unknown region: {region}</h1>")
                return

            at_raw = query.get("at", [None])[0]
            try:
                at = parse_utc_time(at_raw)
            except ValueError as exc:
                self._reply(400, f"<h1>{exc}</h1>")
                return
            at = default_at if at is None else at

            try:
                html = dashboard_document(panel, hourly, region, at, regularisation)
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                self._reply(500, f"<h1>error: {exc}</h1>")
                return
            self._reply(200, html)

        def _reply(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def serve_dashboard(
    panel: dict[str, pd.DataFrame],
    hourly: dict[str, pd.Series] | None,
    *,
    host: str,
    port: int,
    region: str,
    at: float | None,
    regularisation: float,
    launch_browser: bool,
) -> None:
    server = ThreadingHTTPServer((host, port), _handler(panel, hourly, region, at, regularisation))
    address = f"http://{host}:{server.server_port}/"
    print(f"serving dashboard at {address} (Ctrl+C to stop)")
    if launch_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped at {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
    finally:
        server.server_close()
