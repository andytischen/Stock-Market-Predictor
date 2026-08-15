"""Command line interface: ``python -m gapmodel <command>``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .asia import ACTIVITY_WINDOW, REGRESSION_WINDOW, build_asia_dashboard
from .asia_report import render_asia_html, render_asia_text
from .dashboard import build_dashboard, oil_readings, render_html, render_text
from .data import DEFAULT_CACHE, load_panel
from .events import SCHEDULES, unmaintained_on
from .export import build_snapshot, dumps
from .features import build_features, feature_symbols
from .intraday import load_hourly_panel
from .journal import (
    DEFAULT_LOG,
    MIN_SETTLED,
    decayed,
    read_log,
    record,
    settle,
    skills,
    write_log,
)
from .journal import DEFAULT_WINDOW as JOURNAL_WINDOW
from .journal import (
    render_text as render_journal_text,
)
from .markets import (
    BILL_YIELD,
    CURVE_FRONT,
    CURVE_STRIP,
    FUNDS_FUTURE,
    INDICATORS,
    MARKETS,
    MARKETS_BY_SYMBOL,
    REGIONS,
)
from .model import MIN_TRAIN, walk_forward
from .predict import Forecast, forecast_all, parse_shock, to_frame
from .regions import dashboard_symbols
from .scenarios import SCENARIOS, scenario
from .score import DEFAULT_WINDOW, relative_scores, render_reference, score_symbols
from .score import to_frame as score_to_frame
from .score import to_relative_frame as score_to_relative_frame
from .scorecard import RECENT_WINDOW, build_scorecard, calls_frame
from .scorecard import append_log as append_scorecard_log
from .scorecard import render_text as render_scorecard_text
from .screener import (
    ATR_WINDOW,
    AVG_WINDOW,
    MIN_ATR,
    MIN_AVG_VOLUME,
    MIN_CHANGE,
    MIN_PRICE,
    MIN_REL_VOLUME,
    MIN_VOLUME,
    Criteria,
    screen,
)
from .screener import DEFAULT_START as SCREEN_START
from .screener import render_text as render_screen_text
from .screener import to_frame as screen_to_frame
from .sectors import build_sector_board
from .sectors import render_text as render_sector_text
from .shortlist import biggest_gainers, forecast_universe
from .shortlist import discarded as discarded_shortlist
from .shortlist import rank as rank_shortlist
from .shortlist import render_text as render_shortlist_text
from .shortlist import to_frame as shortlist_to_frame
from .social_arb import CORRELATION_WINDOW, build_social_arb
from .social_arb import to_frame as social_arb_to_frame
from .staleness import STALE_DAYS, fresh_targets, guard, today
from .stocks import (
    BLIND_SPOTS,
    SHORTLISTED,
    STOCKS,
    STOCKS_BY_SYMBOL,
    is_stock,
    peers_of,
    stock_symbols,
)
from .universe import modelled_universe, read_universe, us_universe

log = logging.getLogger(__name__)

# The hourly window is short, so the intraday variant needs a smaller warm-up.
INTRADAY_MIN_TRAIN = 200


def _market_symbol(value: str) -> str:
    if value not in MARKETS_BY_SYMBOL:
        known = ", ".join(MARKETS_BY_SYMBOL)
        raise argparse.ArgumentTypeError(f"unknown market {value!r}; choose from {known}")
    return value


def _stock_symbol(value: str) -> str:
    symbol = value.upper()
    if symbol not in STOCKS_BY_SYMBOL:
        known = ", ".join(STOCKS_BY_SYMBOL)
        raise argparse.ArgumentTypeError(f"unknown stock {value!r}; choose from {known}")
    return symbol


def _shortlisted_symbol(value: str) -> str:
    """A name the shortlist models.

    Refused up front rather than warned about per name: a ticker outside the
    universe has no session clock here, and a typo that merely downloaded
    whatever Yahoo returned would be ranked beside the rest as if it belonged.
    """
    symbol = value.upper()
    if symbol not in SHORTLISTED:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not in the modelled US universe; add it to LARGE_CAP or "
            "MID_CAP in gapmodel/universe.py to forecast it"
        )
    return symbol


def _target_symbol(value: str) -> str:
    """An index or a modelled single stock: anything with a model behind it."""
    symbol = value if value in MARKETS_BY_SYMBOL else value.upper()
    if symbol in MARKETS_BY_SYMBOL or symbol in STOCKS_BY_SYMBOL:
        return symbol
    known = ", ".join(list(MARKETS_BY_SYMBOL) + list(STOCKS_BY_SYMBOL))
    raise argparse.ArgumentTypeError(f"unknown market {value!r}; choose from {known}")


def _parse_shock(value: str, known: set[str]) -> tuple[str, float]:
    try:
        symbol, move = parse_shock(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if symbol not in known:
        raise argparse.ArgumentTypeError(f"unknown instrument {symbol!r}")
    return symbol, move


def _index_instruments() -> set[str]:
    return set(MARKETS_BY_SYMBOL) | {i.symbol for i in INDICATORS} | {CURVE_FRONT, CURVE_STRIP}


def _shock(value: str) -> tuple[str, float]:
    """A move in something an index model actually reads.

    The single-name peers are deliberately not accepted here: no index feature
    is derived from them, so the shock would be applied to nothing and print an
    unchanged probability, which reads as "no effect" rather than "not modelled".
    """
    return _parse_shock(value, _index_instruments())


def _stock_shock(value: str) -> tuple[str, float]:
    """A move in anything a single-name model reads, peers included."""
    return _parse_shock(value, _index_instruments() | set(stock_symbols()))


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _utc_time(value: str) -> float:
    """``HH:MM`` (UTC) as hours from midnight."""
    try:
        moment = pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a time of day (use HH:MM)") from exc
    return moment.hour + moment.minute / 60


def _panel(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return load_panel(start=args.start, cache_dir=Path(args.cache), refresh=args.refresh)


def _model_inputs(
    panel: dict[str, pd.DataFrame], targets: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """The panel restricted to the series the requested models actually read.

    One download serves every command, so a loaded panel is wider than any run
    of it: the European sector trackers are skipped for a target outside Europe,
    and an opening-price stand-in like ``ISF.L`` is read only as the gap source
    of its own index. Judged on the whole panel, a quiet European sector ETF
    refuses a US forecast that reads nothing it publishes.
    """
    if not targets:
        return dict(panel)
    # Which series arrived decides the paired blocks: a curve or policy leg whose
    # partner failed to download builds no feature, so its silence is nobody's.
    carried = {symbol for symbol, bars in panel.items() if not bars.empty}
    read = set().union(*(feature_symbols(symbol, carried) for symbol in targets))
    return {symbol: bars for symbol, bars in panel.items() if symbol in read}


def _shared_inputs(
    panel: dict[str, pd.DataFrame], targets: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """The series this run reads for someone other than themselves.

    A stock panel is loaded whole — every curated name and every peer — whatever
    was asked for, and a shortlist panel carries the sixty-odd listings it ranks.
    Those series are read by one model each, so holding the whole run to their
    freshness would let a single halted listing cancel sixty-five sound
    forecasts. A name that is a *peer* of something requested stays here: it is
    then a feature, read by a model other than its own, and its silence is
    everyone's problem.
    """
    peers = {peer.symbol for symbol in targets for peer in peers_of(symbol)}
    inputs = _model_inputs(panel, targets)
    target_only = {s for s in inputs if s in SHORTLISTED or s in STOCKS_BY_SYMBOL} - peers
    return {symbol: bars for symbol, bars in inputs.items() if symbol not in target_only}


def _forecast_inputs(
    panel: dict[str, pd.DataFrame], targets: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Exactly the series the run read: the shared inputs and the names kept.

    What the report's stale-input footer should describe. Handed the whole loaded
    panel it would count a name the run skipped, and say its last value was
    carried forward when the reason it is not in the table is that it was not.
    """
    shared = _shared_inputs(panel, targets)
    return shared | {symbol: panel[symbol] for symbol in targets if symbol in panel}


def _fresh_enough(
    panel: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    targets: Sequence[str] = (),
) -> list[str]:
    """Stop a forecasting command before it fits anything on dead inputs.

    Checked here rather than inside the model because it is a question about the
    run and not about the arithmetic: the fit is correct either way, and the
    backtest metrics beside it are still honestly earned. What is wrong is the
    conclusion a reader draws from a probability built by forward-filling a feed
    that stopped a week ago.

    Returns the targets still worth forecasting: the shared inputs either pass
    for everyone or fail the run, while a target with no recent bar of its own
    is dropped by name.
    """
    guard(
        _shared_inputs(panel, targets),
        today(),
        max_days=args.max_stale_days,
        allow=args.allow_stale,
    )
    return fresh_targets(
        panel,
        targets,
        today(),
        max_days=args.max_stale_days,
        allow=args.allow_stale,
    )


def _stock_panel(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """The index panel plus the single names and their peers.

    Only the equity legs are asked for ``Adj Close``: they are the ones whose
    dividends would otherwise be read as opening gaps. Requiring it of the whole
    panel would re-download every index for a column that means nothing to them.
    """
    panel = _panel(args)
    panel.update(
        load_panel(
            symbols=stock_symbols(),
            start=args.start,
            cache_dir=Path(args.cache),
            refresh=args.refresh,
            require=("Adj Close",),
        )
    )
    return panel


def _hourly(args: argparse.Namespace) -> dict[str, pd.Series] | None:
    """The hourly futures panel, or None when it was not asked for or is missing.

    A refused or rate-limited hourly endpoint is treated the same way as bars
    too stale to use: the daily model still has something to say.
    """
    if not getattr(args, "intraday", False):
        return None
    try:
        return load_hourly_panel(cache_dir=Path(args.cache), refresh=args.refresh)
    except RuntimeError as exc:
        log.warning("no hourly futures data (%s): falling back to the daily model", exc)
        return None


def _cmd_markets(_: argparse.Namespace) -> None:
    print("Markets (open / close, UTC hours):")
    for m in MARKETS:
        print(f"  {m.symbol:<12} {m.name:<20} {m.region:<9} {m.open_utc:>6} {m.close_utc:>6}")
    print("\nIndicators:")
    for i in INDICATORS:
        print(f"  {i.symbol:<12} {i.name}")
    curve = f"{CURVE_FRONT}/{CURVE_STRIP}"
    print(f"  {curve:<12} crude curve: front month against the 12-month strip")
    policy = f"{FUNDS_FUTURE}/{BILL_YIELD}"
    print(f"  {policy:<12} priced policy rate and the tightening priced 3 months out")
    print("\nSingle stocks (gapmodel stock):")
    for s in STOCKS:
        peers = ", ".join(p.symbol for p in s.peers if p.symbol != s.symbol)
        print(f"  {s.symbol:<12} {s.name:<24} {s.theme}\n  {'':<12} peers: {peers}")
    print("\nScenarios (predict --scenario):")
    for s in SCENARIOS.values():
        legs = ", ".join(f"{sym} {move:+.1%}" for sym, move in s.moves.items())
        print(f"  {s.name:<22} {s.description}\n  {'':<22} {legs}")


def _forecast(
    panel: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    hourly: dict[str, pd.Series] | None,
    shocks: dict[str, float] | None = None,
    symbols: list[str] | None = None,
) -> list[Forecast]:
    """Forecast every requested market, dropping the pre-open features if need be.

    The intraday variant depends on futures bars running into the bell, which
    a stale or halted feed may not provide. Rather than return nothing, the
    run is repeated on the daily features alone and the loss of sharpness is
    reported.
    """
    wanted = symbols if symbols is not None else args.market
    try:
        return forecast_all(
            panel,
            symbols=wanted,
            c=args.regularisation,
            hourly=hourly,
            min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
            shocks=shocks,
        )
    except RuntimeError:
        if hourly is None:
            raise
        log.warning("no pre-open futures bars: falling back to the daily model")
        return forecast_all(
            panel,
            symbols=wanted,
            c=args.regularisation,
            min_train=MIN_TRAIN,
            shocks=shocks,
        )


def _cmd_predict(args: argparse.Namespace) -> None:
    hourly = _hourly(args)
    shocks = dict(scenario(args.scenario).shocks()) if args.scenario else {}
    # An explicit --shock on the same instrument replaces the scenario's leg.
    shocks.update(args.shock or [])
    panel = _panel(args)
    symbols = _fresh_enough(panel, args, args.market or [m.symbol for m in MARKETS])
    forecasts = _forecast(panel, args, hourly, shocks, symbols=symbols)
    if args.scenario:
        print(f"scenario: {args.scenario} — {scenario(args.scenario).description}")
    if shocks:
        described = ", ".join(f"{s} {np.expm1(m):+.2%}" for s, m in shocks.items())
        print(f"hypothetical: {described}\n")
    frame = to_frame(forecasts).sort_values("p_open_up", ascending=False)
    print(frame.to_string(index=False))
    _print_caveats(forecasts)
    if args.explain:
        for f in forecasts:
            print(f"\n{f.name} — top drivers (log-odds contribution):")
            for feature, value in f.drivers.items():
                print(f"  {feature:<28} {value:+.3f}")
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _print_caveats(forecasts: list[Forecast]) -> None:
    """Warn where a scheduled release makes a probability less than it looks."""
    flagged = [f for f in forecasts if f.caveats]
    if flagged:
        print("\nscheduled releases this model cannot see:")
        for forecast in flagged:
            for note in forecast.caveats:
                print(f"  {forecast.name}: {note}")
    _print_unmaintained(forecasts)


def _print_unmaintained(forecasts: list[Forecast]) -> None:
    """Say when a quiet session is merely an unread calendar.

    The release tables are copied from the agencies' pages and run out at
    different dates. Once a session is past one of them, the lack of a warning
    for that series carries no information, and saying so is the only way the
    silence stays honest.

    One run can forecast more than one date — Tokyo's next session is the
    following day while New York is still on today — so each date is checked on
    its own rather than through the earliest of them.
    """
    for session in sorted({f.session for f in forecasts}):
        stale = unmaintained_on(session)
        if not stale:
            continue
        print(f"\nnot checked for {session.date()} — these calendars end earlier:")
        for schedule in SCHEDULES:
            if schedule.name in stale:
                print(f"  {schedule.name}: table ends {schedule.covers_until} ({schedule.source})")


def _cmd_stock(args: argparse.Namespace) -> None:
    """Next-open probabilities for the modelled single stocks.

    The peer series live outside the index panel, so the download is widened
    rather than reusing whatever ``predict`` happens to need.
    """
    symbols = args.symbols or [s.symbol for s in STOCKS]
    panel = _stock_panel(args)
    symbols = _fresh_enough(panel, args, symbols)
    forecasts = _forecast(panel, args, _hourly(args), dict(args.shock or []), symbols=symbols)
    frame = to_frame(forecasts).sort_values("p_open_up", ascending=False)
    print(frame.to_string(index=False))
    _print_caveats(forecasts)
    print("\nnot in the model, and larger than the overnight tape for one company:")
    for note in BLIND_SPOTS:
        print(f"  {note}")
    if args.explain:
        for f in forecasts:
            print(f"\n{f.name} — top drivers (log-odds contribution):")
            for feature, value in f.drivers.items():
                print(f"  {feature:<28} {value:+.3f}")
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _cmd_export(args: argparse.Namespace) -> None:
    panel = _panel(args)
    symbols = _fresh_enough(panel, args, args.market or [m.symbol for m in MARKETS])
    hourly = _hourly(args)
    forecasts = _forecast(panel, args, hourly, symbols=symbols)
    snapshot = build_snapshot(forecasts, oil_readings(panel))
    text = dumps(snapshot)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


def _last_monday_5am() -> pd.Timestamp:
    """Most recent Monday at 05:00 UTC — the start of last week's trading window."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    # weekday(): Mon=0 … Sun=6.  Roll back to the most recent Monday.
    days_back = now.weekday()  # 0 on Monday, 6 on Sunday
    if days_back == 0:
        days_back = 7  # today IS Monday — use the previous Monday
    return now.normalize() - pd.Timedelta(days=days_back) + pd.Timedelta(hours=5)


def _since_timestamp(args: argparse.Namespace) -> pd.Timestamp | None:
    """Return the ``since`` cutoff implied by ``--last-week`` or ``--since``."""
    if getattr(args, "last_week", False):
        return _last_monday_5am()
    value = getattr(args, "since", None)
    if value is not None:
        try:
            return pd.Timestamp(value)
        except ValueError as exc:
            raise SystemExit(f"error: --since {value!r} is not a valid date: {exc}") from exc
    return None


def _cmd_backtest(args: argparse.Namespace) -> None:
    wanted = args.market or [m.symbol for m in MARKETS]
    # A single name needs its peers loaded, and nothing else does.
    panel = _stock_panel(args) if any(is_stock(s) for s in wanted) else _panel(args)
    hourly = _hourly(args)
    since = _since_timestamp(args)
    rows: list[dict] = []
    window_rows: list[dict] = []
    for symbol in wanted:
        try:
            features, labels = build_features(symbol, panel, hourly=hourly)
            result = walk_forward(
                features,
                labels,
                min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
                c=args.regularisation,
            )
        except Exception as exc:
            print(f"skipping {symbol}: {exc}")
            continue
        rows.append({"symbol": symbol, **result.metrics})
        if since is not None:
            try:
                window_rows.append({"symbol": symbol, **result.window_metrics(since=since)})
            except ValueError:
                window_rows.append({"symbol": symbol, "n": 0})
        if args.reliability:
            print(f"\n{symbol} reliability:")
            print(result.reliability().to_string())
    if not rows:
        raise SystemExit("nothing to back-test")
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    if since is not None:
        label = f"Window: {since.date()} 05:00 UTC → present"
        print(f"\n{label}")
        print(pd.DataFrame(window_rows).round(4).to_string(index=False))


def _cmd_scorecard(args: argparse.Namespace) -> None:
    wanted = args.market or [m.symbol for m in MARKETS]
    panel = _stock_panel(args) if any(is_stock(s) for s in wanted) else _panel(args)
    hourly = _hourly(args)
    records = build_scorecard(
        panel,
        symbols=wanted,
        window=args.window,
        c=args.regularisation,
        min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
        hourly=hourly,
    )
    print(render_scorecard_text(records, window=args.window), end="")
    if args.csv:
        calls_frame(records).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    if args.log:
        merged = append_scorecard_log(records, args.log)
        print(f"\n{args.log}: {len(merged)} scored sessions logged")


def _cmd_asia(args: argparse.Namespace) -> None:
    # Volume drives the turnover and participation columns, so a cache written
    # before it was collected is re-downloaded rather than shown as blank.
    panel = load_panel(
        symbols=dashboard_symbols(),
        start=args.start,
        cache_dir=Path(args.cache),
        refresh=args.refresh,
        require=("Volume",),
    )
    board = build_asia_dashboard(
        panel,
        window=args.window,
        regression_window=args.regression_window,
    )
    if args.out:
        Path(args.out).write_text(render_asia_html(board), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.out is None or args.text:
        print(render_asia_text(board))


def _cmd_social_arb(args: argparse.Namespace) -> None:
    panel = _panel(args)
    symbols = _fresh_enough(panel, args, [m.symbol for m in MARKETS])
    forecasts = forecast_all(panel, symbols=symbols, c=args.regularisation, min_train=MIN_TRAIN)
    signals = build_social_arb(panel, forecasts, window=args.window)
    frame = social_arb_to_frame(signals)
    print(frame.to_string(index=False))
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    panel = _panel(args)
    hourly = _hourly(args)
    symbols = [m.symbol for m in MARKETS if m.region == args.region]
    # A board is read for the probabilities on it, so it is held to the same
    # freshness as `predict`: one command refusing a dead cache while another
    # prints a number from it tells the reader the cache is fine.
    symbols = _fresh_enough(panel, args, symbols)
    forecasts = forecast_all(
        panel,
        symbols=symbols,
        c=args.regularisation,
        hourly=hourly,
        min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
    )
    board = build_dashboard(panel, forecasts, as_of=_as_of(args.at), region=args.region)
    print(render_text(board), end="")
    if args.html:
        Path(args.html).write_text(render_html(board))
        print(f"\nwrote {args.html}")


def _as_of(hours: float | None) -> pd.Timestamp | None:
    """Today's date at the given UTC hour, or now when no hour is given."""
    if hours is None:
        return None
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return now.normalize() + pd.Timedelta(hours=hours)


def _score_universe(args: argparse.Namespace) -> list[str]:
    """The comparison list for ``--relative``: a file if given, else the US list."""
    if args.universe:
        try:
            return read_universe(Path(args.universe))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: --universe {args.universe}: {exc}") from exc
    return us_universe()


def _cmd_score(args: argparse.Namespace) -> None:
    symbols = [s.upper() for s in args.symbols]
    asof: pd.Timestamp | None = None
    if args.asof:
        try:
            asof = pd.Timestamp(args.asof).tz_localize(None)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"error: --asof {args.asof!r} is not a valid date: {exc}") from exc
    # Refused rather than quietly ignored: a caller who names a comparison list
    # plainly wants the comparison.
    if args.universe and not args.relative:
        raise SystemExit("error: --universe applies to --relative; pass --relative too")
    footer = ""
    if args.relative:
        scores, reference = relative_scores(
            symbols,
            _score_universe(args),
            window=args.window,
            asof=asof,
            start=args.start,
            cache_dir=Path(args.cache),
            refresh=args.refresh,
        )
        frame = score_to_relative_frame(scores)
        footer = render_reference(reference)
    else:
        frame = score_to_frame(
            score_symbols(
                symbols,
                window=args.window,
                asof=asof,
                start=args.start,
                cache_dir=Path(args.cache),
                refresh=args.refresh,
            )
        )
    print(frame.to_string(index=False))
    if footer:
        print(f"\n{footer}")
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _screen_universe(args: argparse.Namespace) -> list[str]:
    """Whatever was asked for: explicit symbols, a file, or the default US list.

    The three are alternatives, so asking for more than one is an error rather
    than a silent precedence rule.
    """
    if args.symbols and args.universe:
        raise SystemExit("error: pass symbols or --universe, not both")
    if args.etfs and (args.symbols or args.universe):
        raise SystemExit("error: --etfs applies to the default universe only")
    if args.symbols:
        return [s.upper() for s in args.symbols]
    if args.universe:
        try:
            return read_universe(Path(args.universe))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: --universe {args.universe}: {exc}") from exc
    return us_universe(include_etfs=args.etfs)


def _cmd_screen(args: argparse.Namespace) -> None:
    asof: pd.Timestamp | None = None
    if args.asof:
        try:
            asof = pd.Timestamp(args.asof).tz_localize(None)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"error: --asof {args.asof!r} is not a valid date: {exc}") from exc
    criteria = Criteria(
        min_price=args.min_price,
        min_volume=args.min_volume * 1e6,
        min_avg_volume=args.min_avg_volume * 1e6,
        min_rel_volume=args.min_rel_volume,
        min_change=args.min_change / 100,
        min_atr=args.min_atr / 100,
        avg_window=args.avg_window,
        atr_window=args.atr_window,
    )
    result = screen(
        _screen_universe(args),
        criteria=criteria,
        asof=asof,
        start=args.screen_start,
        cache_dir=Path(args.cache),
        refresh=args.refresh,
    )
    print(render_screen_text(result), end="")
    if args.csv:
        screen_to_frame(result.readings).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _cmd_sectors(args: argparse.Namespace) -> None:
    panel = _panel(args)
    symbols = _fresh_enough(panel, args, [args.market])
    forecasts = forecast_all(panel, symbols=symbols, c=args.regularisation)
    print(render_sector_text(build_sector_board(panel, forecasts[0])), end="")


def _shortlist_equities(symbols: list[str]) -> list[str]:
    """The names to download for a shortlist run: the targets and any peers.

    A curated name keeps its peers here so that a name common to both commands is
    read from the same features either way. Without them ``shortlist MU`` would
    quietly print a different probability from ``stock MU``, having silently
    dropped the columns that make it the better forecast.
    """
    peers = [peer.symbol for symbol in symbols for peer in peers_of(symbol)]
    return list(dict.fromkeys(symbols + peers))


def _cmd_shortlist(args: argparse.Namespace) -> None:
    """Rank the universe by how much edge each name's own record supports."""
    candidates = args.symbols or modelled_universe()
    # The names are not part of the default panel — they are targets, never
    # features — so they are loaded on top of it, and only the equity legs are
    # asked for ``Adj Close``: a company's dividend would otherwise be read as an
    # opening gap, which means nothing to an index.
    panel = _panel(args)
    panel.update(
        load_panel(
            symbols=_shortlist_equities(candidates),
            start=args.start,
            cache_dir=Path(args.cache),
            refresh=args.refresh,
            require=("Adj Close",),
        )
    )
    # Every candidate is downloaded and only the chosen ones are fitted: the
    # bars are cheap and each walk-forward is not, so the mover pass narrows
    # after the panel exists rather than guessing which names moved beforehand.
    symbols = candidates
    selection: str | None = None
    if args.gainers:
        symbols = biggest_gainers(panel, candidates, args.gainers)
        if not symbols:
            raise SystemExit("error: no candidate had two closes to compare")
        # The session is named, and so is the ranking rule: sorting descending
        # and slicing gives the smallest fallers on a session where everything
        # fell, and calling those gainers would assert a rise the data denies.
        moved = max(panel[symbol].index.max() for symbol in symbols).date().isoformat()
        selection = (
            f"the {len(symbols)} biggest gainers of session {moved}, out of "
            f"{len(candidates)} candidates, ranked on their move in that session"
        )
    # After the mover pass, so that a stale listing is judged only when it is one
    # of the names about to be fitted.
    symbols = _fresh_enough(panel, args, symbols)
    picks = forecast_universe(panel, symbols=symbols, c=args.regularisation)
    print(
        render_shortlist_text(
            picks,
            top=args.top,
            panel=_forecast_inputs(panel, symbols),
            max_stale_days=args.max_stale_days,
            selection=selection,
            as_of=today(),
        ),
        end="",
    )
    if args.csv:
        # Written in the report's order, with the verdict as a column, so that
        # sorting the file on the raw probability is not the obvious next step.
        frame = shortlist_to_frame(rank_shortlist(picks) + discarded_shortlist(picks))
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


def _cmd_journal(args: argparse.Namespace) -> None:
    """Journal today's forecasts, settle the ones that have printed, and score them.

    Recording and settling are one command on purpose: the forecast has to be
    written down before the auction it describes, and the only run that is
    certain to happen every morning is the one that makes the forecast.
    """
    path = Path(args.log)
    journal = read_log(path)
    panel = _panel(args)
    if not args.settle_only:
        forecasts = _forecast(panel, args, _hourly(args))
        journal, added = record(journal, forecasts, panel)
        print(f"recorded {len(added)} of {len(forecasts)} forecasts")
    journal, filled, retired = settle(journal, panel)
    print(f"settled {filled} session(s) against realised opens, retired {retired} unscorable")
    write_log(journal, path)
    print(f"wrote {path}\n")
    measured = skills(journal, window=args.window, min_settled=args.min_settled)
    print(render_journal_text(journal, measured, args.window, args.min_settled))
    if args.csv:
        journal.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    if args.fail_on_decay and decayed(measured):
        raise SystemExit(1)


def _cmd_fetch(args: argparse.Namespace) -> None:
    panel = _panel(args)
    for symbol, frame in panel.items():
        span = f"{frame.index.min().date()} -> {frame.index.max().date()}"
        print(f"{symbol:<12} {len(frame):>6} rows  {span}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gapmodel", description=__doc__)
    parser.add_argument("--start", default="2005-01-01", help="first date to download")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="cache directory")
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    parser.add_argument(
        "--max-stale-days",
        type=_positive_int,
        default=STALE_DAYS,
        help=f"refuse to forecast from inputs older than this many days (default: {STALE_DAYS})",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="forecast from stale inputs anyway, warning instead of failing",
    )
    parser.add_argument("--regularisation", type=_positive_float, default=0.1, help="logistic C")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    markets = sub.add_parser("markets", help="list modelled markets and indicators")
    markets.set_defaults(func=_cmd_markets)

    fetch = sub.add_parser("fetch", help="download and cache the price panel")
    fetch.set_defaults(func=_cmd_fetch)

    score = sub.add_parser(
        "score",
        help="trend score (price z-score) for arbitrary tickers, strongest first",
    )
    score.add_argument("symbols", nargs="+", help="tickers to score, e.g. IVZ JPM DDOG")
    score.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"trailing sessions for the z-score (default {DEFAULT_WINDOW})",
    )
    score.add_argument(
        "--asof", metavar="DATE", help="score as of this date (ISO) instead of latest"
    )
    score.add_argument(
        "--relative",
        action="store_true",
        help="normalise each score across a comparison universe, so 0 is an average stock today",
    )
    score.add_argument(
        "--universe",
        metavar="FILE",
        help="comparison universe for --relative, one ticker per line (default: the US list)",
    )
    score.add_argument("--csv", help="also write the table to this path")
    score.set_defaults(func=_cmd_score)

    screener = sub.add_parser(
        "screen",
        help="narrow a US universe to liquid, unusually active, moving stocks",
    )
    screener.add_argument("symbols", nargs="*", help="screen these tickers instead of the universe")
    screener.add_argument(
        "--universe", metavar="FILE", help="read tickers from a file, one per line"
    )
    screener.add_argument(
        "--etfs", action="store_true", help="include the heavily traded ETFs in the universe"
    )
    screener.add_argument(
        "--min-price",
        type=_non_negative_float,
        default=MIN_PRICE,
        help=f"price floor (default {MIN_PRICE:g})",
    )
    screener.add_argument(
        "--min-volume",
        type=_non_negative_float,
        default=MIN_VOLUME / 1e6,
        metavar="MILLIONS",
        help=f"today's volume floor in millions of shares (default {MIN_VOLUME / 1e6:g})",
    )
    screener.add_argument(
        "--min-avg-volume",
        type=_non_negative_float,
        default=MIN_AVG_VOLUME / 1e6,
        metavar="MILLIONS",
        help=f"average volume floor in millions of shares (default {MIN_AVG_VOLUME / 1e6:g})",
    )
    screener.add_argument(
        "--min-rel-volume",
        type=_non_negative_float,
        default=MIN_REL_VOLUME,
        metavar="TIMES",
        help=f"relative volume floor (default {MIN_REL_VOLUME:g})",
    )
    screener.add_argument(
        "--min-change",
        type=float,
        default=MIN_CHANGE * 100,
        metavar="PERCENT",
        help=f"daily move floor in percent (default {MIN_CHANGE * 100:g})",
    )
    screener.add_argument(
        "--min-atr",
        type=_non_negative_float,
        default=MIN_ATR * 100,
        metavar="PERCENT",
        help=f"average true range floor, percent of price (default {MIN_ATR * 100:g})",
    )
    screener.add_argument(
        "--avg-window",
        type=int,
        default=AVG_WINDOW,
        help=f"sessions behind today for the volume baseline (default {AVG_WINDOW})",
    )
    screener.add_argument(
        "--atr-window",
        type=int,
        default=ATR_WINDOW,
        help=f"sessions for the ATR (default {ATR_WINDOW})",
    )
    screener.add_argument(
        "--asof", metavar="DATE", help="screen this session (ISO) instead of the latest"
    )
    screener.add_argument(
        "--screen-start",
        default=SCREEN_START,
        metavar="DATE",
        help=f"first date to download for the screen (default {SCREEN_START})",
    )
    screener.add_argument("--csv", help="also write the surviving names to this path")
    screener.set_defaults(func=_cmd_screen)

    predict = sub.add_parser("predict", help="probability that the next open is up")
    predict.add_argument(
        "--market", action="append", type=_market_symbol, help="restrict to a symbol"
    )
    predict.add_argument("--explain", action="store_true", help="show top drivers")
    predict.add_argument("--csv", help="also write the table to this path")
    predict.add_argument(
        "--shock",
        action="append",
        type=_shock,
        metavar="SYMBOL=MOVE",
        help="re-run under a hypothetical move, e.g. --shock '^KS11=+2%%'",
    )
    predict.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        help="re-run under a named bundle of moves; --shock overrides a leg of it",
    )
    predict.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    predict.set_defaults(func=_cmd_predict)

    stock = sub.add_parser(
        "stock",
        help="probability that one company's next open is up",
    )
    stock.add_argument(
        "symbols",
        nargs="*",
        type=_stock_symbol,
        help=f"modelled stocks (default all): {', '.join(STOCKS_BY_SYMBOL)}",
    )
    stock.add_argument("--explain", action="store_true", help="show top drivers")
    stock.add_argument("--csv", help="also write the table to this path")
    stock.add_argument(
        "--shock",
        action="append",
        type=_stock_shock,
        metavar="SYMBOL=MOVE",
        help="re-run under a hypothetical move, e.g. --shock '000660.KS=+3%%'",
    )
    stock.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    stock.set_defaults(func=_cmd_stock)

    backtest = sub.add_parser("backtest", help="walk-forward out-of-sample metrics")
    backtest.add_argument(
        "--market",
        action="append",
        type=_target_symbol,
        help="restrict to a symbol; a modelled single stock is accepted too",
    )
    backtest.add_argument("--reliability", action="store_true", help="calibration table")
    backtest.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    backtest.add_argument(
        "--since",
        metavar="DATE",
        help="show a second metrics table restricted to sessions on or after DATE (ISO format)",
    )
    backtest.add_argument(
        "--last-week",
        action="store_true",
        help="shorthand for --since last-Monday-05:00-UTC (the opening of last week)",
    )
    backtest.set_defaults(func=_cmd_backtest)

    scorecard = sub.add_parser(
        "scorecard", help="recent out-of-sample record: what was called, what opened"
    )
    scorecard.add_argument(
        "--market",
        action="append",
        type=_target_symbol,
        help="restrict to a symbol; a modelled single stock is accepted too",
    )
    scorecard.add_argument(
        "--window",
        type=_positive_int,
        default=RECENT_WINDOW,
        help=f"scored sessions in the recent window (default {RECENT_WINDOW})",
    )
    scorecard.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    scorecard.add_argument("--csv", help="write the scored sessions to this file")
    scorecard.add_argument(
        "--log",
        metavar="PATH",
        help="merge the scored sessions into this CSV log, one row per session",
    )
    scorecard.set_defaults(func=_cmd_scorecard)

    asia = sub.add_parser(
        "asia", help="evaluate the Asian session: heavyweights and outside drivers"
    )
    asia.add_argument("--out", help="write a standalone HTML page to this path")
    asia.add_argument("--text", action="store_true", help="also print the text version")
    asia.add_argument(
        "--window", type=int, default=ACTIVITY_WINDOW, help="sessions for betas and volume averages"
    )
    asia.add_argument(
        "--regression-window",
        type=int,
        default=REGRESSION_WINDOW,
        help="sessions used for the driver regressions",
    )
    asia.set_defaults(func=_cmd_asia)

    dashboard = sub.add_parser(
        "dashboard", help="crude readings next to one region's session state and open calls"
    )
    dashboard.add_argument("--region", choices=REGIONS, default="Asia")
    dashboard.add_argument(
        "--at", type=_utc_time, help="UTC time of day to render for, e.g. 05:00 (default: now)"
    )
    dashboard.add_argument("--html", help="also write an HTML dashboard here")
    dashboard.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    dashboard.set_defaults(func=_cmd_dashboard)

    export = sub.add_parser(
        "export", help="write the forecast run as a JSON snapshot for the mobile app"
    )
    export.add_argument(
        "--market", action="append", type=_market_symbol, help="restrict to a symbol"
    )
    export.add_argument("--out", help="write the JSON here instead of stdout")
    export.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    export.set_defaults(func=_cmd_export)

    shortlist = sub.add_parser(
        "shortlist",
        help="next-open probability across the US universe, ranked by demonstrated edge",
    )
    shortlist.add_argument(
        "symbols",
        nargs="*",
        type=_shortlisted_symbol,
        help="forecast these tickers instead of the whole US universe",
    )
    shortlist.add_argument(
        "--gainers",
        type=_positive_int,
        help="forecast only the N biggest movers of the latest session",
    )
    shortlist.add_argument(
        "--top",
        type=_positive_int,
        help="show only the strongest N ranked names (default: all)",
    )
    shortlist.add_argument("--csv", help="also write every name, ranked or not, to this path")
    shortlist.set_defaults(func=_cmd_shortlist)

    sectors = sub.add_parser(
        "sectors", help="split one European index's open call by STOXX 600 sector"
    )
    sectors.add_argument("--market", type=_market_symbol, default="^STOXX50E")
    sectors.set_defaults(func=_cmd_sectors)

    journal = sub.add_parser(
        "journal",
        help="journal today's forecasts and score the ones whose opens have printed",
    )
    journal.add_argument(
        "--market", action="append", type=_market_symbol, help="restrict to a symbol"
    )
    journal.add_argument(
        "--log", default=str(DEFAULT_LOG), help=f"journal CSV (default {DEFAULT_LOG})"
    )
    journal.add_argument(
        "--window",
        type=_positive_int,
        default=JOURNAL_WINDOW,
        help=f"settled sessions per market to score (default {JOURNAL_WINDOW})",
    )
    journal.add_argument(
        "--min-settled",
        type=_positive_int,
        default=MIN_SETTLED,
        help=f"settled sessions before a market's record is reported (default {MIN_SETTLED})",
    )
    journal.add_argument(
        "--settle-only",
        action="store_true",
        help="score what is already journalled without forecasting today",
    )
    journal.add_argument(
        "--fail-on-decay",
        action="store_true",
        help="exit non-zero when a market's live record is below its own drift",
    )
    journal.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    journal.add_argument("--csv", help="also write a copy of the journal to this path")
    journal.set_defaults(func=_cmd_journal)

    social_arb = sub.add_parser(
        "social-arb",
        help="markets where the model probability diverges from what correlated peers imply",
    )
    social_arb.add_argument(
        "--window",
        type=_positive_int,
        default=CORRELATION_WINDOW,
        help="sessions used for the peer correlation matrix (default: %(default)s)",
    )
    social_arb.add_argument("--csv", help="also write the table to this path")
    social_arb.set_defaults(func=_cmd_social_arb)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        args.func(args)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        # A failed run is an expected outcome (no data, too little history, an
        # unwritable cache): report it as an error, not as a crash.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
