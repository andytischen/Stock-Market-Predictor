"""ASCII-text and HTML rendering for :class:`~gapmodel.social.signals.SocialSignal`.

Style mirrors :mod:`gapmodel.asia_report` (text) and :mod:`gapmodel.dashboard`
(HTML) so output is visually consistent with the rest of the toolkit.
"""

from __future__ import annotations

from html import escape

from .signals import SocialSignal

# ── text widths ──────────────────────────────────────────────────────────────
_COL_TICKER = 6
_COL_MENTIONS = 8
_COL_SENT = 8
_COL_BULL = 7
_COL_VEL = 7
_COL_SIG = 6

_HEADER = (
    f"{'Ticker':<{_COL_TICKER}}  "
    f"{'Mentions':>{_COL_MENTIONS}}  "
    f"{'SentMean':>{_COL_SENT}}  "
    f"{'Bull%':>{_COL_BULL}}  "
    f"{'Vel':>{_COL_VEL}}  "
    f"{'Signal':<{_COL_SIG}}"
)
_SEP = "-" * len(_HEADER)


# ── shared HTML style ────────────────────────────────────────────────────────
_HTML_STYLE = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 900px; padding: 24px; }
h1 { margin-bottom: 4px; }
.sub { opacity: .7; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 14px; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid rgba(128,128,128,.3); }
th:first-child, td:first-child { text-align: left; }
th { font-size: 11px; text-transform: uppercase; opacity: .75; }
tr:hover td { background: rgba(128,128,128,.12); }
.long { color: #0a7d38; font-weight: 600; }
.weak { color: #c02626; font-weight: 600; }
.hold { opacity: .85; }
.note { opacity: .75; font-style: italic; font-size: 12px; }
"""


def _signal_css(signal: str) -> str:
    return {"LONG": "long", "WEAK": "weak"}.get(signal, "hold")


def render_text(signals: list[SocialSignal]) -> str:
    """Render *signals* as an ASCII table.

    Matches the width and style used throughout :mod:`gapmodel.asia_report`.

    .. warning::
        This output is a model-derived sentiment summary, not investment advice.
    """
    if not signals:
        return "Social scan — no signals\n"

    lines = ["Social scan", "=" * len(_HEADER), _HEADER, _SEP]
    for sig in signals:
        lines.append(
            f"{sig.ticker:<{_COL_TICKER}}  "
            f"{sig.mention_count:>{_COL_MENTIONS}}  "
            f"{sig.sentiment_mean:>{_COL_SENT}.4f}  "
            f"{sig.bullish_ratio:>{_COL_BULL}.2%}  "
            f"{sig.velocity:>{_COL_VEL}.2f}  "
            f"{sig.signal:<{_COL_SIG}}"
        )
    lines.append(_SEP)
    lines.append(
        "Signals are derived from public social-media posts. "
        "Not investment advice."
    )
    return "\n".join(lines) + "\n"


def render_html(signals: list[SocialSignal]) -> str:
    """Render *signals* as a minimal standalone HTML page.

    Matches the stylesheet used in :mod:`gapmodel.dashboard`.

    .. warning::
        This output is a model-derived sentiment summary, not investment advice.
    """
    head = "".join(
        f"<th>{h}</th>"
        for h in ("Ticker", "Mentions", "Sent mean", "Bull %", "Velocity", "Signal")
    )

    def _row(sig: SocialSignal) -> str:
        css = _signal_css(sig.signal)
        return (
            "<tr>"
            f"<td>{escape(sig.ticker)}</td>"
            f"<td>{sig.mention_count}</td>"
            f"<td>{sig.sentiment_mean:.4f}</td>"
            f"<td>{sig.bullish_ratio:.2%}</td>"
            f"<td>{sig.velocity:.2f}</td>"
            f'<td class="{css}">{escape(sig.signal)}</td>'
            "</tr>"
        )

    rows = "".join(_row(s) for s in signals)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Social scan</title><style>{_HTML_STYLE}</style></head>\n"
        "<body>\n"
        "<h1>Social scan</h1>\n"
        '<p class="sub">Derived from public social-media posts. '
        "Not investment advice.</p>\n"
        f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>\n"
        "</body></html>\n"
    )
