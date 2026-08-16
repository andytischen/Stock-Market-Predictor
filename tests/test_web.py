import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gapmodel import web


@pytest.fixture
def served(monkeypatch):
    """A dashboard server whose rendering is replaced by the arguments it got."""
    seen = []

    def fake_document(_panel, _hourly, region, hour, _regularisation):
        seen.append((region, hour))
        return "<h1>board</h1>"

    monkeypatch.setattr(web, "dashboard_document", fake_document)
    handler = web._handler({}, None, "Asia", 5.0, 0.1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read().decode()


def test_clearing_the_time_field_returns_to_the_live_view(served):
    base, seen = served
    _get(f"{base}/dashboard?region=Asia&at=")
    assert seen == [("Asia", None)]


def test_omitting_the_time_keeps_the_startup_time(served):
    base, seen = served
    _get(f"{base}/dashboard?region=Asia")
    assert seen == [("Asia", 5.0)]


def test_unknown_region_is_reported_without_reflecting_markup(served):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{base}/dashboard?region=%3Cscript%3Ealert(1)%3C/script%3E")
    body = error.value.read().decode()
    assert error.value.code == 400
    assert "<script>" not in body
    assert "&lt;script&gt;" in body


def test_a_time_that_is_not_hh_mm_is_refused(served):
    base, seen = served
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{base}/dashboard?region=Asia&at=2024-01-01")
    assert error.value.code == 400
    assert seen == []


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8000/"),
        ("0.0.0.0", "http://127.0.0.1:8000/"),
        ("::", "http://127.0.0.1:8000/"),
        ("::1", "http://[::1]:8000/"),
    ],
)
def test_browser_url_is_always_reachable(host, expected):
    assert web.browser_url(host, 8000) == expected
