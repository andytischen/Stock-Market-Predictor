"""Rendering of the Asian session dashboard as text or a standalone HTML page."""

from __future__ import annotations

from html import escape

import pandas as pd

from .asia import DATA_GAPS, AsiaDashboard, IndexSnapshot

CONSTITUENT_COLUMNS = (
    ("name", "Company"),
    ("sector", "Sector"),
    ("weight", "Weight %"),
    ("return_1d", "1d %"),
    ("return_5d", "5d %"),
    ("contribution_bp", "Index bp"),
    ("beta_to_index", "Beta"),
    ("volume_vs_average", "Vol x avg"),
    ("turnover_share", "Turnover %"),
)

DRIVER_COLUMNS = (
    ("driver", "Driver"),
    ("theme", "Theme"),
    ("lag_days", "Lag"),
    ("beta", "Beta"),
    ("t_stat", "t"),
    ("r2", "R2"),
    ("correlation", "Corr"),
    ("last_move", "Last %"),
    ("implied_bp", "Implied bp"),
)

STYLE = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1100px; padding: 24px; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 32px; border-bottom: 2px solid currentColor; padding-bottom: 4px; }
h3 { margin: 20px 0 6px; font-size: 15px; text-transform: uppercase; letter-spacing: .05em; }
.sub { opacity: .7; margin-top: 0; }
.cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
.card { border: 1px solid rgba(128,128,128,.4); border-radius: 6px;
        padding: 6px 10px; min-width: 108px; }
.card b { display: block; font-size: 17px; }
.card span { font-size: 11px; text-transform: uppercase; opacity: .7; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 14px; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid rgba(128,128,128,.3); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { font-size: 11px; text-transform: uppercase; opacity: .75; }
tr:hover td { background: rgba(128,128,128,.12); }
.up { color: #0a7d38; } .down { color: #c02626; }
.note { opacity: .75; font-style: italic; }
.gaps td { text-align: left; }
"""

SIGNED_FIELDS = frozenset(
    {"return_1d", "return_5d", "return_20d", "contribution_bp", "implied_bp", "last_move", "beta"}
)


def _format(value: object) -> str:
    if isinstance(value, float):
        if value != value:
            return "-"
        return f"{value:,.2f}" if abs(value) >= 1000 else f"{value:.2f}"
    return str(value)


def _headline(snapshot: IndexSnapshot) -> list[tuple[str, str]]:
    m = snapshot.metrics
    cards = [
        ("Close", _format(m["close"])),
        ("1d", f"{m['return_1d']:+.2f}%"),
        ("5d", f"{m['return_5d']:+.2f}%"),
        ("20d", f"{m['return_20d']:+.2f}%"),
        ("Vol 20d", f"{m['volatility_20d']:.1f}%"),
        ("Weight up", f"{m['weight_advancing']:.0f}%"),
        ("Top names", f"{m['weight_covered']:.0f}% of index"),
    ]
    if "opening_gap" in m and m["opening_gap"] == m["opening_gap"]:
        cards.insert(1, ("Open gap", f"{m['opening_gap']:+.2f}%"))
    if "volume_vs_average" in m:
        cards.append(("Volume x avg", f"{m['volume_vs_average']:.2f}"))
    return cards


def _text_table(frame: pd.DataFrame, columns: tuple[tuple[str, str], ...]) -> str:
    present = [(key, title) for key, title in columns if key in frame.columns]
    view = frame.loc[:, [key for key, _ in present]].copy()
    view.columns = [title for _, title in present]
    return view.to_string(index=False, float_format=lambda v: f"{v:.2f}")


def _source_note(snapshot: IndexSnapshot) -> str:
    if snapshot.source == snapshot.profile.symbol:
        return ""
    return (
        f"{snapshot.profile.symbol} has stopped printing regularly; "
        f"index level and drivers are read from {snapshot.source}."
    )


def render_asia_text(dashboard: AsiaDashboard) -> str:
    lines = [
        f"Asia session dashboard — {dashboard.generated:%Y-%m-%d %H:%M UTC}",
        "=" * 72,
    ]
    for snapshot in dashboard.snapshots:
        profile = snapshot.profile
        lines.append("")
        lines.append(f"{profile.name} ({profile.symbol}) — {profile.country}")
        lines.append(f"  session {snapshot.session:%Y-%m-%d}")
        if _source_note(snapshot):
            lines.append(f"  {_source_note(snapshot)}")
        lines.append("  " + "  ".join(f"{label} {value}" for label, value in _headline(snapshot)))
        if profile.note:
            lines.append(f"  {profile.note}")
        lines.append("")
        lines.append("  Dominant companies")
        lines.append(_text_table(snapshot.constituents, CONSTITUENT_COLUMNS))
        lines.append("")
        lines.append("  Outside drivers (known before this index opened)")
        lines.append(_text_table(snapshot.drivers, DRIVER_COLUMNS))
        lines.append("")
        lines.append("  Explained variance by theme")
        lines.append(snapshot.themes.to_string(index=False))
    lines.append("")
    lines.append("Theme R2 across Asia")
    lines.append(dashboard.theme_matrix().to_string(index=False))
    lines.append("")
    lines.append("Not in this dashboard (feed needed)")
    for topic, feed in DATA_GAPS:
        lines.append(f"  - {topic}: {feed}")
    return "\n".join(lines)


def _cell(key: str, value: object) -> str:
    text = escape(_format(value))
    if key in SIGNED_FIELDS and isinstance(value, float) and value == value:
        css = "up" if value > 0 else "down" if value < 0 else ""
        if css:
            sign = "+" if value > 0 else ""
            return f'<td class="{css}">{sign}{text}</td>'
    return f"<td>{text}</td>"


def _html_table(frame: pd.DataFrame, columns: tuple[tuple[str, str], ...]) -> str:
    present = [(key, title) for key, title in columns if key in frame.columns]
    head = "".join(f"<th>{escape(title)}</th>" for _, title in present)
    body = "".join(
        "<tr>" + "".join(_cell(key, row[key]) for key, _ in present) + "</tr>"
        for _, row in frame.iterrows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _plain_table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(v))}</td>" for v in row) + "</tr>"
        for row in frame.to_numpy()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _html_snapshot(snapshot: IndexSnapshot) -> str:
    profile = snapshot.profile
    cards = "".join(
        f"<div class='card'><span>{escape(label)}</span><b>{escape(value)}</b></div>"
        for label, value in _headline(snapshot)
    )
    notes = [text for text in (_source_note(snapshot), profile.note) if text]
    note = "".join(f"<p class='note'>{escape(text)}</p>" for text in notes)
    return f"""
<h2>{escape(profile.name)} <small>{escape(profile.symbol)} · {escape(profile.country)} ·
session {snapshot.session:%Y-%m-%d}</small></h2>
{note}
<div class="cards">{cards}</div>
<h3>Dominant companies</h3>
{_html_table(snapshot.constituents, CONSTITUENT_COLUMNS)}
<h3>Outside drivers known before the open</h3>
{_html_table(snapshot.drivers, DRIVER_COLUMNS)}
<h3>Explained variance by theme</h3>
{_plain_table(snapshot.themes)}
"""


def render_asia_html(dashboard: AsiaDashboard) -> str:
    gaps = "".join(
        f"<tr><td>{escape(topic)}</td><td>{escape(feed)}</td></tr>" for topic, feed in DATA_GAPS
    )
    sections = "".join(_html_snapshot(s) for s in dashboard.asia)
    europe = "".join(_html_snapshot(s) for s in dashboard.europe)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Asia session dashboard</title><style>{STYLE}</style></head>
<body>
<h1>Asia session dashboard</h1>
<p class="sub">Generated {dashboard.generated:%Y-%m-%d %H:%M} UTC. Every driver is lagged so it was
knowable before the index opened. Not investment advice.</p>
<h2>Theme R<sup>2</sup> across Asia</h2>
{_plain_table(dashboard.theme_matrix().round(4))}
{sections}
<h1>European indices behind the Asian afternoon</h1>
{europe}
<h1>Not in this dashboard</h1>
<table class="gaps"><thead><tr><th>Measure</th><th>Feed needed</th></tr></thead>
<tbody>{gaps}</tbody></table>
</body></html>
"""
