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

    def fake_document(_panel, _hourly, symbols, region, hour, _regularisation):
        seen.append((region, hour, list(symbols)))
        return "<h1>board</h1>"

    monkeypatch.setattr(web, "dashboard_document", fake_document)
    handler = web._handler({}, None, {"Asia": ["^N225"], "Europe": []}, "Asia", 5.0, 0.1)
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
    assert seen == [("Asia", None, ["^N225"])]


def test_omitting_the_time_keeps_the_startup_time(served):
    base, seen = served
    _get(f"{base}/dashboard?region=Asia")
    assert seen == [("Asia", 5.0, ["^N225"])]


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


def test_a_region_with_nothing_fresh_is_told_so_rather_than_rendered(served):
    base, seen = served
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{base}/dashboard?region=Europe")
    assert error.value.code == 503
    assert "recent enough" in error.value.read().decode()
    assert seen == []


def test_an_unexpected_rendering_failure_is_answered_as_a_500(served, monkeypatch):
    base, _ = served

    def explode(*_args, **_kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(web, "dashboard_document", explode)
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(f"{base}/dashboard?region=Asia")
    assert error.value.code == 500
    assert "list index out of range" in error.value.read().decode()


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


@pytest.fixture
def bound(monkeypatch):
    """``serve_dashboard`` with the socket and the serving loop taken out."""

    class FakeServer:
        server_port = 8000

        def __init__(self, address, _handler):
            self.address = address

        def serve_forever(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr(web, "ThreadingHTTPServer", FakeServer)

    def serve(host):
        web.serve_dashboard(
            {},
            None,
            symbols={"Asia": []},
            host=host,
            port=0,
            region="Asia",
            at=None,
            regularisation=0.1,
            launch_browser=False,
        )

    return serve


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.internal"])
def test_binding_beyond_loopback_warns_that_the_dashboard_is_unprotected(bound, capsys, host):
    bound(host)
    assert "no authentication" in capsys.readouterr().out


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2", "::1", "0:0:0:0:0:0:0:1"])
def test_binding_an_address_only_this_machine_can_reach_serves_without_a_warning(
    bound, capsys, host
):
    bound(host)
    assert "warning" not in capsys.readouterr().out
