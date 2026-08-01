import json

import pytest

from pm.board import Board
from pm.cli import main
from pm.report import render_report


@pytest.fixture
def board_file(tmp_path):
    return tmp_path / "project.json"


def run(board_file, *argv):
    main(["--file", str(board_file), *argv])


def test_tasks_round_trip_through_the_json_file(board_file):
    board = Board(name="demo")
    board.add("write docs", owner="ana", due="2030-01-01", resources=["tech writer"])
    board.save(board_file)

    reloaded = Board.load(board_file)
    assert reloaded.name == "demo"
    assert reloaded.get(1).resources == ["tech writer"]


def test_ids_increment_and_titles_are_required():
    board = Board()
    assert board.add("one").id == 1
    assert board.add("two").id == 2
    with pytest.raises(ValueError):
        board.add("   ")


def test_progress_and_resources_ignore_completed_work():
    board = Board()
    board.add("a", resources=["gpu"])
    board.add("b", resources=["gpu", "data licence"])
    board.set_status(1, "done")

    assert board.progress == 0.5
    assert list(board.resources_needed()) == ["gpu", "data licence"]
    assert [t.id for t in board.resources_needed()["gpu"]] == [2]


def test_overdue_only_applies_to_open_tasks():
    board = Board()
    board.add("late", due="2000-01-01")
    assert board.get(1).is_overdue
    board.set_status(1, "done")
    assert not board.get(1).is_overdue


def test_report_lists_sections_and_resource_requests():
    board = Board(name="predictor")
    board.add("ship model", owner="ana")
    board.add("buy data", resources=["market data budget"])
    board.set_status(1, "doing")
    board.set_status(2, "blocked")
    board.add_note(2, "vendor quote pending")

    report = render_report(board)
    assert "predictor — status update" in report
    assert "0/2 tasks complete (0%)" in report
    assert "## In progress" in report and "## Blocked" in report
    assert "vendor quote pending" in report
    assert "**market data budget** — needed for #2" in report


def test_report_without_requests_says_so():
    board = Board()
    board.add("solo")
    assert "None outstanding." in render_report(board)


def test_cli_add_status_and_report(board_file, capsys):
    run(board_file, "add", "ship model", "--owner", "ana", "--resource", "gpu")
    run(board_file, "status", "1", "doing")
    run(board_file, "need", "1", "extra reviewer")
    capsys.readouterr()

    run(board_file, "report")
    out = capsys.readouterr().out
    assert "## In progress" in out
    assert "**gpu**" in out and "**extra reviewer**" in out

    saved = json.loads(board_file.read_text())
    assert saved["tasks"][0]["status"] == "doing"


def test_cli_reports_unknown_task_as_an_error(board_file):
    with pytest.raises(SystemExit) as exit_info:
        run(board_file, "status", "9", "done")
    assert "no task with id 9" in str(exit_info.value)


def test_cli_rejects_a_bad_due_date(board_file):
    with pytest.raises(SystemExit) as exit_info:
        run(board_file, "add", "x", "--due", "tomorrow")
    assert "error:" in str(exit_info.value)


def test_cli_list_filters_by_status(board_file, capsys):
    run(board_file, "add", "alpha")
    run(board_file, "add", "beta")
    run(board_file, "status", "2", "done")
    capsys.readouterr()

    run(board_file, "list", "--status", "done")
    out = capsys.readouterr().out
    assert "beta" in out and "alpha" not in out
