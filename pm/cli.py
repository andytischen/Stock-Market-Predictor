"""Command line interface: ``python -m pm <command>``."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .board import DEFAULT_BOARD, STATUSES, Board
from .deck import render_deck
from .report import render_report


def _board(args: argparse.Namespace) -> Board:
    return Board.load(Path(args.file))


def _cmd_add(args: argparse.Namespace) -> None:
    board = _board(args)
    task = board.add(args.title, owner=args.owner, due=args.due, resources=args.resource)
    board.save(Path(args.file))
    print(f"added #{task.id} {task.title}")


def _cmd_list(args: argparse.Namespace) -> None:
    board = _board(args)
    tasks = board.by_status(args.status) if args.status else board.tasks
    if not tasks:
        print("no tasks")
        return
    for task in tasks:
        due = f" due {task.due}" if task.due else ""
        owner = f" @{task.owner}" if task.owner else ""
        flag = " !" if task.is_overdue else ""
        print(f"#{task.id:<3} {task.status:<8} {task.title}{owner}{due}{flag}")


def _cmd_status(args: argparse.Namespace) -> None:
    board = _board(args)
    task = board.set_status(args.id, args.new_status)
    board.save(Path(args.file))
    print(f"#{task.id} -> {task.status}")


def _cmd_note(args: argparse.Namespace) -> None:
    board = _board(args)
    board.add_note(args.id, args.text)
    board.save(Path(args.file))
    print(f"noted on #{args.id}")


def _cmd_need(args: argparse.Namespace) -> None:
    board = _board(args)
    board.add_resource(args.id, args.resource)
    board.save(Path(args.file))
    print(f"#{args.id} needs {args.resource}")


def _cmd_report(args: argparse.Namespace) -> None:
    report = render_report(_board(args))
    print(report, end="")
    if args.out:
        path = Path(args.out)
        path.write_text(report)
        print(f"wrote {args.out}")
        if args.open:
            webbrowser.open(path.resolve().as_uri())


def _cmd_deck(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_deck(_board(args)))
    print(f"wrote {args.out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm", description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_BOARD), help="board JSON file")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add a task")
    add.add_argument("title")
    add.add_argument("--owner", default="")
    add.add_argument("--due", default="", help="YYYY-MM-DD")
    add.add_argument("--resource", action="append", help="something the task needs")
    add.set_defaults(func=_cmd_add)

    listing = sub.add_parser("list", help="list tasks")
    listing.add_argument("--status", choices=STATUSES)
    listing.set_defaults(func=_cmd_list)

    status = sub.add_parser("status", help="move a task to a new status")
    status.add_argument("id", type=int)
    status.add_argument("new_status", choices=STATUSES)
    status.set_defaults(func=_cmd_status)

    note = sub.add_parser("note", help="append a dated note to a task")
    note.add_argument("id", type=int)
    note.add_argument("text")
    note.set_defaults(func=_cmd_note)

    need = sub.add_parser("need", help="record a resource a task needs")
    need.add_argument("id", type=int)
    need.add_argument("resource")
    need.set_defaults(func=_cmd_need)

    report = sub.add_parser("report", help="markdown status update")
    report.add_argument("--out", help="also write the report here")
    report.add_argument("--open", action="store_true", help="open the output file in the browser (requires --out)")
    report.set_defaults(func=_cmd_report)

    deck = sub.add_parser("deck", help="render the board as a reveal.js slide deck")
    deck.add_argument("--out", default="docs/status-deck.html", help="where to write the deck")
    deck.add_argument("--open", action="store_true", help="open the deck in the default browser after writing")
    deck.set_defaults(func=_cmd_deck)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
