"""Lightweight browser interface for the regional dashboard."""

from __future__ import annotations

import logging
import webbrowser
from collections.abc import Mapping, Sequence
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd

from .dashboard import build_dashboard, render_html
from .markets import REGIONS
from .model import INTRADAY_MIN_TRAIN, MIN_TRAIN
from .predict import forecast_all
from .utctime import as_of, format_utc_time, parse_utc_time

log = logging.getLogger(__name__)

# Bind addresses that mean "every interface": not reachable as a URL host.
_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]"}

# Addresses only this machine can reach. Anything else exposes an interface that
# refits models on request, with no authentication in front of it.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def browser_url(host: str, port: int) -> str:
    """The address a local browser can actually open for this bind address."""
    if host in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/"


def dashboard_document(
    panel: dict[str, pd.DataFrame],
    hourly: dict[str, pd.Series] | None,
    symbols: Sequence[str],
    region: str,
    hour: float | None,
    regularisation: float,
) -> str:
    forecasts = forecast_all(
        panel,
        symbols=symbols,
        c=regularisation,
        hourly=hourly,
        min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
    )
    board = build_dashboard(panel, forecasts, as_of=as_of(hour), region=region)
    return render_html(board)


def _index_html(default_region: str, default_at: float | None) -> str:
    time_value = "" if default_at is None else format_utc_time(default_at)
    options = "".join(
        f'<option value="{escape(r, quote=True)}"'
        f"{' selected' if r == default_region else ''}>{escape(r)}</option>"
        for r in REGIONS
    )
    query = urlencode({"region": default_region, **({"at": time_value} if time_value else {})})
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Gapmodel dashboard</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 1rem; color: #222; }}
 .controls {{
   display: flex; gap: .75rem; align-items: end; flex-wrap: wrap; margin-bottom: .75rem;
 }}
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
    symbols: Mapping[str, Sequence[str]],
    default_region: str,
    default_at: float | None,
    regularisation: float,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            # Blank values are kept: an emptied time field means "now", which is
            # not the same as never having asked for a time at all.
            query = parse_qs(parsed.query, keep_blank_values=True)

            if parsed.path == "/":
                self._reply(200, _index_html(default_region, default_at))
                return
            if parsed.path != "/dashboard":
                self._reply(404, "<h1>Not found</h1>")
                return

            region = query.get("region", [""])[0] or default_region
            if region not in symbols:
                self._reply(400, f"<h1>Unknown region: {escape(region)}</h1>")
                return

            # An "at" that was submitted empty means "now"; only an absent one
            # falls back to the time the server was started with.
            if "at" not in query:
                at = default_at
            else:
                at_raw = query["at"][0]
                if at_raw.strip() == "":
                    at = None
                else:
                    try:
                        at = parse_utc_time(at_raw)
                    except ValueError as exc:
                        self._reply(400, f"<h1>{escape(str(exc))}</h1>")
                        return

            try:
                html = dashboard_document(
                    panel, hourly, symbols[region], region, at, regularisation
                )
            # Every failure below the render is answered rather than raised: an
            # exception out of `do_GET` closes the socket mid-response, so the
            # browser reports a network error and the reason is lost. Catching
            # `Exception` and not the types seen so far because the render walks
            # third-party frames whose next failure will have a new type; the
            # traceback goes to the log, where the request lines are suppressed
            # but a broken server should still be diagnosable.
            except Exception as exc:
                log.exception("rendering %s failed", self.path)
                self._reply(500, f"<h1>error: {escape(str(exc))}</h1>")
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
    symbols: Mapping[str, Sequence[str]],
    host: str,
    port: int,
    region: str,
    at: float | None,
    regularisation: float,
    launch_browser: bool,
) -> None:
    handler = _handler(panel, hourly, symbols, region, at, regularisation)
    server = ThreadingHTTPServer((host, port), handler)
    address = browser_url(host, server.server_port)
    if host not in _LOOPBACK_HOSTS:
        # Said once, at the point the choice is made: there is no login on this
        # server, and each request it answers fits models, so a reachable one is
        # both readable and expensive to anyone who can route to it.
        print(
            f"warning: bound to {host}, so anyone who can reach this machine can "
            "load the dashboard and make it refit models; there is no authentication"
        )
    print(f"serving dashboard at {address} (Ctrl+C to stop)")
    if launch_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stopped = pd.Timestamp.now("UTC").tz_localize(None)
        print(f"\nstopped at {stopped:%Y-%m-%d %H:%M:%S} UTC")
    finally:
        server.server_close()
