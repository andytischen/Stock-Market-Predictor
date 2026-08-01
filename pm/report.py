"""Markdown status update rendered from a board."""

from __future__ import annotations

from datetime import datetime, timezone

from .board import Board, Task

_STATUS_SECTIONS = (
    ("done", "Completed"),
    ("doing", "In progress"),
    ("blocked", "Blocked"),
    ("todo", "Up next"),
)


def _line(task: Task) -> str:
    bits = [f"- **{task.title}** (#{task.id})"]
    if task.owner:
        bits.append(f"— {task.owner}")
    if task.due:
        bits.append(f"— due {task.due}{' **OVERDUE**' if task.is_overdue else ''}")
    line = " ".join(bits)
    if task.notes:
        line += f"\n  - {task.notes[-1]}"
    return line


def render_report(board: Board) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    done = len(board.by_status("done"))
    blocked = board.by_status("blocked")
    overdue = [t for t in board.tasks if t.is_overdue]

    out = [
        f"# {board.name} — status update {today}",
        "",
        f"Progress: {done}/{len(board.tasks)} tasks complete ({board.progress:.0%}).",
    ]
    if blocked:
        out.append(f"{len(blocked)} task(s) blocked.")
    if overdue:
        out.append(f"{len(overdue)} task(s) past their due date.")

    for status, heading in _STATUS_SECTIONS:
        tasks = board.by_status(status)
        if not tasks:
            continue
        out += ["", f"## {heading}", ""]
        out += [_line(t) for t in tasks]

    wanted = board.resources_needed()
    out += ["", "## Resources needed", ""]
    if wanted:
        for resource, tasks in sorted(wanted.items()):
            ids = ", ".join(f"#{t.id}" for t in tasks)
            out.append(f"- **{resource}** — needed for {ids}")
    else:
        out.append("- None outstanding.")

    return "\n".join(out) + "\n"
