import pytest

from pm.board import Board
from pm.cli import main
from pm.deck import render_deck


@pytest.fixture
def board():
    b = Board(name="Predictor")
    b.add("Ship model")
    b.add("Wire dashboard", owner="ana")
    b.add("Buy auction data", resources=["market data budget"])
    b.add("Late thing", due="2000-01-01")
    b.set_status(1, "done")
    b.set_status(2, "doing")
    b.set_status(3, "blocked")
    b.add_note(3, "vendor quote pending")
    return b


def test_deck_has_one_slide_per_populated_section(board):
    html = render_deck(board)
    assert html.count("<section>") == 6  # title + shipped/doing/blocked/next + resources
    for heading in ("Shipped", "In progress", "Blocked", "Up next", "Resources needed"):
        assert f"<h2>{heading}</h2>" in html
    assert "1 of 4 tasks complete · 25%" in html


def test_empty_sections_do_not_get_a_slide():
    board = Board(name="Fresh")
    board.add("only task")
    html = render_deck(board)
    assert "<h2>Up next</h2>" in html
    assert "<h2>Blocked</h2>" not in html
    assert "Nothing outstanding" in html


def test_slides_carry_owners_notes_and_overdue_marks(board):
    html = render_deck(board)
    assert "ana" in html
    assert "vendor quote pending" in html
    assert 'class="overdue"' in html and "due 2000-01-01 — overdue" in html
    assert "<strong>market data budget</strong>" in html


def test_titles_are_escaped():
    board = Board(name="<script>x</script>")
    board.add("<b>bold</b> & brash")
    html = render_deck(board)
    assert "<script>x</script>" not in html.split("<script src")[0]
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; brash" in html


def test_cli_deck_creates_missing_directories(tmp_path, capsys):
    board_file = tmp_path / "project.json"
    out = tmp_path / "docs" / "deck.html"
    main(["--file", str(board_file), "add", "plan the quarter"])
    main(["--file", str(board_file), "deck", "--out", str(out)])

    assert "plan the quarter" in out.read_text()
    assert "reveal.js" in out.read_text()
    assert f"wrote {out}" in capsys.readouterr().out


def test_cli_deck_open_calls_webbrowser(tmp_path, monkeypatch):
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    board_file = tmp_path / "project.json"
    out = tmp_path / "deck.html"
    main(["--file", str(board_file), "add", "plan the quarter"])
    main(["--file", str(board_file), "deck", "--out", str(out), "--open"])

    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("file://")
    assert "deck.html" in opened_urls[0]


def test_cli_deck_no_open_by_default(tmp_path, monkeypatch):
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    board_file = tmp_path / "project.json"
    out = tmp_path / "deck.html"
    main(["--file", str(board_file), "add", "plan the quarter"])
    main(["--file", str(board_file), "deck", "--out", str(out)])

    assert opened_urls == []
