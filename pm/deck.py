"""The board rendered as a reveal.js slide deck.

The deck is the same content as the markdown report, cut into slides for
showing to someone rather than reading: where the project is, what shipped,
what is moving, what is stuck, and what it needs to get unstuck.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from .board import Board, Task

REVEAL_VERSION = "4.6.1"
REVEAL_CDN = f"https://cdn.jsdelivr.net/npm/reveal.js@{REVEAL_VERSION}"


def _bullet(task: Task, show_note: bool = True) -> str:
    meta = []
    if task.owner:
        meta.append(escape(task.owner))
    if task.due:
        meta.append(f"due {task.due}" + (" — overdue" if task.is_overdue else ""))
    line = f"<strong>{escape(task.title)}</strong>"
    if meta:
        line += f' <span class="meta">{escape(" · ".join(meta))}</span>'
    if show_note and task.notes:
        line += f'<br><span class="note">{escape(task.notes[-1])}</span>'
    css = ' class="overdue"' if task.is_overdue else ""
    return f"<li{css}>{line}</li>"


def _task_slide(heading: str, tasks: list[Task], show_notes: bool = True) -> str:
    if not tasks:
        return ""
    items = "\n".join(_bullet(t, show_notes) for t in tasks)
    return f"<section><h2>{escape(heading)}</h2>\n<ul>\n{items}\n</ul></section>"


def _resource_slide(board: Board) -> str:
    wanted = board.resources_needed()
    if not wanted:
        body = "<p>Nothing outstanding — the roadmap is unblocked.</p>"
    else:
        items = "\n".join(
            f"<li><strong>{escape(resource)}</strong>"
            f' <span class="meta">{escape(", ".join(t.title for t in tasks))}</span></li>'
            for resource, tasks in sorted(wanted.items())
        )
        body = f"<ul>\n{items}\n</ul>"
    return f"<section><h2>Resources needed</h2>\n{body}</section>"


def render_deck(board: Board) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    done, total = len(board.by_status("done")), len(board.tasks)
    blocked = board.by_status("blocked")

    title = f"""<section>
<h1>{escape(board.name)}</h1>
<p class="lead">Status update — {today}</p>
<p class="headline">{done} of {total} tasks complete · {board.progress:.0%}</p>
<p class="meta">{len(blocked)} blocked · {len(board.by_status("doing"))} in progress</p>
</section>"""

    slides = [
        title,
        _task_slide("Shipped", board.by_status("done")),
        _task_slide("In progress", board.by_status("doing")),
        _task_slide("Blocked", blocked),
        _task_slide("Up next", board.by_status("todo"), show_notes=False),
        _resource_slide(board),
    ]
    body = "\n".join(s for s in slides if s)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(board.name)} — status deck {today}</title>
<link rel="stylesheet" href="{REVEAL_CDN}/dist/reveal.css">
<link rel="stylesheet" href="{REVEAL_CDN}/dist/theme/white.css">
<style>
 .reveal {{ font-family: system-ui, sans-serif; }}
 .reveal h1 {{ font-size: 2.1rem; }}
 .reveal h2 {{ font-size: 1.6rem; text-transform: none; }}
 .reveal ul {{ font-size: 1.05rem; }}
 .reveal li {{ margin-bottom: .6rem; }}
 .reveal .lead {{ font-size: 1.2rem; }}
 /* Not .progress: reveal.js owns that class for its own progress bar. */
 .reveal .headline {{ font-size: 1.6rem; font-weight: 600; }}
 .reveal .meta {{ color: #777; font-size: .85em; font-weight: 400; }}
 .reveal .note {{ color: #555; font-size: .8em; }}
 .reveal li.overdue strong {{ color: #a00; }}
</style>
</head>
<body>
<div class="reveal"><div class="slides">
{body}
</div></div>
<script src="{REVEAL_CDN}/dist/reveal.js"></script>
<script>Reveal.initialize({{hash: true, slideNumber: true}});</script>
</body>
</html>
"""
