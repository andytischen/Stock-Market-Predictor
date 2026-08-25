"""The customer-facing brief: the calls, the tape behind them, and the caveats.

Every other renderer here is written for whoever fitted the model: `predict`
prints standardised log-odds, `dashboard` prints an oil column in log-odds too.
This one is written for the person the call is sent to, which changes what has
to be on the page rather than how it is styled. Three things are separated so
they can never be read as one another:

* what the model produced — a probability for one opening auction, with the
  out-of-sample record that earns it;
* what the market did — closes and moves, each stamped with the session they
  come from, so a series that stopped printing is visible as a date and not
  inferred from a suspiciously round move;
* what a person wrote — the week-ahead commentary, which is read from a notes
  file and labelled as somebody's opinion, because nothing in this repository
  forecasts a commodity or a closing level and a paragraph that reads as though
  it does would be the most damaging thing on the page.

The blind spots and the "not forecast" list are part of the output rather than
small print: an opening-gap probability sent without them is routinely read as a
view on the day, which is a claim no model here makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .events import CALENDAR_END, Event, unmaintained_on, upcoming
from .features import log_return
from .markets import INDICATORS
from .predict import Forecast
from .staleness import STALE_DAYS

# Calendar days of releases the brief looks ahead over: a week, so a Friday
# brief covers the week it is read in rather than stopping at the weekend.
HORIZON_DAYS = 7

# The tape a reader wants beside the calls, grouped as they would be quoted.
# Deliberately a subset of the indicators the model reads: everything here is a
# number the reader can check against their own screen.
QUOTE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Commodities", ("BZ=F", "CL=F", "GC=F", "SI=F", "HG=F")),
    ("Rates, volatility and the dollar", ("^VIX", "^TNX", "^TYX", "DX-Y.NYB")),
)

# Sessions behind the last bar for each move shown, and its label. 21 sessions
# is a month of trading, which is the horizon the commentary tends to discuss.
RETURN_WINDOWS: tuple[tuple[int, str], ...] = ((1, "1d"), (5, "1w"), (21, "1m"))

# Series quoted as a rate, whose moves are shown in basis points. A percentage
# move in a yield is the one number on this page a reader would reliably
# misread: "+0.1%" on a 4.7% ten-year is five basis points, not ten.
RATES: frozenset[str] = frozenset({"^TNX", "^FVX", "^TYX", "^IRX"})

NAMES = {instrument.symbol: instrument.name for instrument in INDICATORS}

# What the number on the page is, in one line, and the things it is repeatedly
# taken for. Said before any probability appears rather than in a footer.
FORECAST = (
    "the probability that one equity index opens above its own previous close, "
    "in the opening auction of the session named beside it"
)

NOT_FORECAST: tuple[str, ...] = (
    "commodity prices. Crude, gold, silver and copper enter as inputs — level, "
    "one-day and five-day return, realised volatility, curve shape — and never "
    "as targets: there is no oil or gold forecast in this system to quote",
    "the rest of the session after that auction, or where the index closes",
    "close-to-close direction, or a move over several days",
    "a price target, and so nothing derived from one: no probability of "
    "finishing above a strike, and no option outcome",
)

# Categories of information with no feature behind them, and what each would
# need before it could have one. Phrased as the gap rather than as a hedge: a
# reader who knows which of these is live today knows exactly how much of the
# call is being made in the dark.
UNSEEN: tuple[tuple[str, str], ...] = (
    (
        "Headlines, news and sentiment",
        "every feature is a price, so a story reaches the model only once "
        "something has traded on it — and never before the auction it moves",
    ),
    (
        "Scheduled releases",
        "flagged per call from the published calendars, never anticipated: the "
        "features are built from a world in which the number is unpublished",
    ),
    (
        "Results, guidance and after-hours announcements",
        "an index-level model reads no company disclosure, so an overnight "
        "warning from a heavyweight reaches it only as tomorrow's price",
    ),
    (
        "Geopolitics, sanctions and policy decisions",
        "firm crude is read as firm crude; whether the level is demand or a "
        "closed shipping lane is not in the features, and the two do not "
        "transmit to equities the same way",
    ),
    (
        "Positioning, flows, breadth and borrow",
        "an exchange or prime-broker feed would be needed; a crowded trade and "
        "a quiet one look identical in a close",
    ),
    (
        "A stale overnight print",
        "a market that closed hours before its listed peers kept trading is "
        "read at its own last close, which the peers may already have repriced",
    ),
)


@dataclass(frozen=True)
class Quote:
    """One instrument's last close and recent moves, with its session date."""

    symbol: str
    name: str
    session: pd.Timestamp
    close: float
    returns: dict[str, float]
    lag_days: int
    # Whether ``returns`` holds proportional moves or changes in the rate.
    is_rate: bool = False

    @property
    def is_stale(self) -> bool:
        """Whether this series has stopped printing rather than stopped moving.

        A holiday and a broken feed are indistinguishable in the data and only
        differ in how long the silence has lasted, which is what the lag
        measures — the same threshold the forecasting guard uses.
        """
        return self.lag_days > STALE_DAYS


@dataclass(frozen=True)
class Commentary:
    """A titled block of somebody's written view, as read from a notes file."""

    title: str
    paragraphs: tuple[str, ...] = ()
    bullets: tuple[str, ...] = ()


@dataclass
class Brief:
    generated: pd.Timestamp
    calls: list[Forecast] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    releases: tuple[Event, ...] = ()
    unmaintained: tuple[str, ...] = ()
    commentary: tuple[Commentary, ...] = ()
    horizon_days: int = HORIZON_DAYS

    @property
    def caveats(self) -> list[tuple[str, str]]:
        """Every call's release caveats, as (market, note) pairs."""
        return [(call.name, note) for call in self.calls for note in call.caveats]


def read_commentary(path: Path) -> tuple[Commentary, ...]:
    """Written commentary from a small subset of Markdown.

    ``## Title`` opens a block, ``- `` lines are bullets and everything else is
    a paragraph. A subset and not a Markdown library because the file is the
    only part of the page a human writes and it goes out to customers: nothing
    here can emit markup, so a stray tag in the notes reaches the page as the
    text it was typed as, and a heading is the only structure available.
    """
    blocks: list[Commentary] = []
    title = ""
    paragraphs: list[str] = []
    bullets: list[str] = []

    def close() -> None:
        if title:
            blocks.append(Commentary(title, tuple(paragraphs), tuple(bullets)))

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            close()
            title, paragraphs, bullets = line.lstrip("#").strip(), [], []
        elif line.startswith(("- ", "* ")):
            bullets.append(line[2:].strip())
        elif line:
            paragraphs.append(line)
    close()
    if not blocks:
        raise ValueError(f"{path}: no commentary found; a block starts with a '## Heading' line")
    return tuple(blocks)


def quotes(panel: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> list[Quote]:
    """The tape for the grouped instruments, skipping any that did not arrive."""
    found: list[Quote] = []
    for _, symbols in QUOTE_GROUPS:
        for symbol in symbols:
            frame = panel.get(symbol)
            if frame is None or frame.empty:
                continue
            close = frame["Close"].dropna()
            if close.empty:
                continue
            session = close.index[-1]
            found.append(
                Quote(
                    symbol=symbol,
                    name=NAMES.get(symbol, symbol),
                    session=session,
                    close=float(close.iloc[-1]),
                    # A window longer than the history says nothing rather than
                    # being silently relabelled: "1m" has to mean a month.
                    returns={
                        label: _move_over(close, sessions, symbol in RATES)
                        for sessions, label in RETURN_WINDOWS
                    },
                    is_rate=symbol in RATES,
                    lag_days=int((as_of.normalize() - session.normalize()).days),
                )
            )
    return found


def build_brief(
    panel: dict[str, pd.DataFrame],
    forecasts: list[Forecast],
    as_of: pd.Timestamp | None = None,
    commentary: tuple[Commentary, ...] = (),
    horizon_days: int = HORIZON_DAYS,
) -> Brief:
    moment = as_of or pd.Timestamp.now(tz="UTC").tz_localize(None)
    calls = sorted(forecasts, key=lambda f: -f.probability_up)
    # Measured from the first session called rather than from today: the brief
    # is about the sessions in the table, and a run made the evening before
    # would otherwise drop the very release the earliest call is exposed to.
    first = min((call.session for call in calls), default=moment).normalize()
    return Brief(
        generated=moment,
        calls=calls,
        quotes=quotes(panel, moment),
        releases=upcoming(first, horizon_days),
        unmaintained=unmaintained_on(first + pd.Timedelta(days=horizon_days - 1)),
        commentary=commentary,
        horizon_days=horizon_days,
    )


def _move_over(close: pd.Series, sessions: int, is_rate: bool) -> float:
    """The move over ``sessions``: basis points for a rate, a log return otherwise."""
    if len(close) <= sessions:
        return float("nan")
    if is_rate:
        return float(close.iloc[-1] - close.iloc[-1 - sessions]) * 100.0
    return float(log_return(close, sessions).iloc[-1])


def _pct(value: float | None, is_rate: bool = False) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:+.1f}bp" if is_rate else f"{value:+.2%}"


def _auc(call: Forecast) -> str:
    value = call.backtest.get("auc", float("nan"))
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def _release_rows(brief: Brief) -> list[tuple[str, str, str]]:
    return [
        (
            f"{event.date:%a %d %b}",
            f"{int(event.time_utc):02d}:{round(event.time_utc % 1 * 60):02d} UTC",
            event.name,
        )
        for event in brief.releases
    ]


def render_text(brief: Brief) -> str:
    lines = [
        f"Week-ahead brief — {brief.generated:%Y-%m-%d %H:%M} UTC",
        "",
        f"What is forecast: {FORECAST}.",
        "What is not forecast:",
    ]
    lines += [f"  - {item}" for item in NOT_FORECAST]

    lines += ["", "Opening calls (model output):"]
    lines.append(
        f"  {'market':<20} {'region':<9} {'session':<11} {'p(open up)':>10} {'oos AUC':>8}"
    )
    for call in brief.calls:
        lines.append(
            f"  {call.name:<20} {call.region:<9} {call.session.date().isoformat():<11} "
            f"{call.probability_up:>9.1%} {_auc(call):>8}"
        )

    lines += ["", "Market data:"]
    for group, symbols in QUOTE_GROUPS:
        shown = [q for q in brief.quotes if q.symbol in symbols]
        if not shown:
            continue
        lines.append(f"  {group}")
        for quote in shown:
            moves = "  ".join(
                f"{label} {_pct(quote.returns[label], quote.is_rate)}"
                for _, label in RETURN_WINDOWS
            )
            flag = "  <- stopped printing" if quote.is_stale else ""
            lines.append(
                f"    {quote.name:<24} {quote.close:>12,.2f}  {moves}"
                f"  session {quote.session.date().isoformat()}{flag}"
            )

    lines += ["", f"Scheduled releases in the next {brief.horizon_days} days:"]
    rows = _release_rows(brief)
    lines += [f"  {day}  {clock}  {name}" for day, clock, name in rows] or [
        "  none published for this window"
    ]
    if brief.unmaintained:
        lines.append(
            f"  calendar not maintained past {CALENDAR_END.date().isoformat()} for: "
            + ", ".join(brief.unmaintained)
        )

    if brief.caveats:
        lines += ["", "Releases the calls in this table cannot price:"]
        lines += [f"  {market}: {note}" for market, note in brief.caveats]

    lines += ["", "Not in the model:"]
    lines += [f"  - {topic}: {why}" for topic, why in UNSEEN]

    for block in brief.commentary:
        lines += ["", f"{block.title} (written view, not model output):"]
        lines += [f"  {text}" for text in block.paragraphs]
        lines += [f"  - {text}" for text in block.bullets]
    return "\n".join(lines) + "\n"


STYLE = """
:root { color-scheme: light dark; --line: rgba(128,128,128,.35); }
body { font: 15px/1.55 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1000px; padding: 24px; }
header { border-bottom: 3px solid currentColor; padding-bottom: 10px; }
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { margin: 32px 0 8px; font-size: 18px; }
.sub { opacity: .7; margin: 0; }
.tag { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
       border: 1px solid var(--line); border-radius: 10px; padding: 1px 8px;
       vertical-align: middle; opacity: .8; font-weight: 400; }
.scope { border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 16px 0; }
.scope p { margin: 0 0 6px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 4px; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; min-width: 150px; }
.card b { display: block; font-size: 19px; }
.card .name { font-size: 11px; text-transform: uppercase; opacity: .7; }
.card .moves { font-size: 12px; }
.card .when { font-size: 11px; opacity: .65; }
.card.stale { border-color: #c02626; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; }
th, td { text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { font-size: 11px; text-transform: uppercase; opacity: .75; }
.up { color: #0a7d38; } .down { color: #c02626; }
dl { margin: 6px 0; } dt { font-weight: 600; margin-top: 8px; } dd { margin: 0 0 0 18px; }
footer { margin-top: 36px; border-top: 1px solid var(--line); padding-top: 10px;
         font-size: 13px; opacity: .8; }
"""

CALLS_COLUMNS = ("Market", "Region", "Session", "p(open up)", "Out-of-sample AUC")


def _tag(kind: str) -> str:
    return f'<span class="tag">{escape(kind)}</span>'


def _move(value: float, is_rate: bool = False) -> str:
    if value is None or np.isnan(value):
        return '<span class="moves">n/a</span>'
    css = "up" if value > 0 else "down" if value < 0 else ""
    return f'<span class="{css}">{_pct(value, is_rate)}</span>'


def _card(quote: Quote) -> str:
    moves = " · ".join(
        f"{_move(quote.returns[label], quote.is_rate)} <span class='when'>{escape(label)}</span>"
        for _, label in RETURN_WINDOWS
    )
    stale = " stale" if quote.is_stale else ""
    note = " · stopped printing" if quote.is_stale else ""
    return (
        f"<div class='card{stale}'><span class='name'>{escape(quote.name)}</span>"
        f"<b>{quote.close:,.2f}</b><div class='moves'>{moves}</div>"
        f"<div class='when'>session {quote.session.date().isoformat()}"
        f"{note}</div></div>"
    )


def _calls_table(brief: Brief) -> str:
    head = "".join(f"<th>{escape(column)}</th>" for column in CALLS_COLUMNS)
    body = "".join(
        "<tr>"
        f"<td>{escape(call.name)}</td><td>{escape(call.region)}</td>"
        f"<td>{call.session.date().isoformat()}</td>"
        f"<td>{call.probability_up:.1%}</td><td>{_auc(call)}</td>"
        "</tr>"
        for call in brief.calls
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _releases_table(brief: Brief) -> str:
    rows = _release_rows(brief)
    if not rows:
        return "<p>No release on the published calendars falls in this window.</p>"
    body = "".join(
        f"<tr><td>{escape(day)}</td><td>{escape(name)}</td><td>{escape(clock)}</td></tr>"
        for day, clock, name in rows
    )
    return (
        "<table><thead><tr><th>Day</th><th>Release</th><th>Published</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _commentary_html(block: Commentary) -> str:
    paragraphs = "".join(f"<p>{escape(text)}</p>" for text in block.paragraphs)
    bullets = (
        "<ul>" + "".join(f"<li>{escape(text)}</li>" for text in block.bullets) + "</ul>"
        if block.bullets
        else ""
    )
    return f"<h2>{escape(block.title)} {_tag('written view')}</h2>{paragraphs}{bullets}"


def render_html(brief: Brief) -> str:
    groups = "".join(
        f"<h3>{escape(group)}</h3><div class='cards'>"
        + "".join(_card(q) for q in brief.quotes if q.symbol in symbols)
        + "</div>"
        for group, symbols in QUOTE_GROUPS
        if any(q.symbol in symbols for q in brief.quotes)
    )
    caveats = (
        "<h2>Releases these calls cannot price</h2><ul>"
        + "".join(
            f"<li><b>{escape(market)}</b>: {escape(note)}</li>" for market, note in brief.caveats
        )
        + "</ul>"
        if brief.caveats
        else ""
    )
    unmaintained = (
        f"<p>Nothing is checked past {CALENDAR_END.date().isoformat()} for "
        f"{escape(', '.join(brief.unmaintained))}: no warning there means nothing "
        "was checked, not that nothing is scheduled.</p>"
        if brief.unmaintained
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Week-ahead brief — {brief.generated:%Y-%m-%d %H:%M} UTC</title>
<style>{STYLE}</style></head>
<body>
<header>
<h1>Week-ahead brief</h1>
<p class="sub">Generated {brief.generated:%Y-%m-%d %H:%M} UTC</p>
</header>

<div class="scope">
<p><b>What is forecast:</b> {escape(FORECAST)}.</p>
<p><b>What is not forecast:</b></p>
<ul>{"".join(f"<li>{escape(item)}</li>" for item in NOT_FORECAST)}</ul>
</div>

<h2>Opening calls {_tag("model output")}</h2>
{_calls_table(brief)}
<p class="sub">Each index has its own model, fitted and scored on its own history.
The AUC column is that model's out-of-sample ranking ability over a walk-forward
backtest — 0.5 is a coin flip — and it is the reason to believe a probability,
or not.</p>
{caveats}

<h2>The tape behind the calls {_tag("market data")}</h2>
{groups}
<p class="sub">Closes, with the session each one comes from. A card marked
stale has stopped printing rather than stopped moving, and every model reading
it carries its last value forward.</p>

<h2>Scheduled releases, next {brief.horizon_days} days {_tag("calendar")}</h2>
{_releases_table(brief)}
{unmaintained}

<h2>Not in the model {_tag("limitations")}</h2>
<dl>{"".join(f"<dt>{escape(topic)}</dt><dd>{escape(why)}</dd>" for topic, why in UNSEEN)}</dl>

{"".join(_commentary_html(block) for block in brief.commentary)}

<footer>Probabilities describe one opening auction and nothing after it.
Written views are opinion, not model output. Market data from Yahoo Finance,
stamped with the session it was read from. This is not investment advice.</footer>
</body></html>
"""
