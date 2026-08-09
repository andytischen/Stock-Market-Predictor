"""Command line interface: ``python -m gapmodel <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .asia import ACTIVITY_WINDOW, REGRESSION_WINDOW, build_asia_dashboard
from .asia_report import render_asia_html, render_asia_text
from .dashboard import build_dashboard, oil_readings, render_html, render_text
from .data import DEFAULT_CACHE, load_panel
from .export import build_snapshot, dumps
from .features import build_features
from .intraday import load_hourly_panel
from .markets import (
    CURVE_FRONT,
    CURVE_STRIP,
    INDICATORS,
    MARKETS,
    MARKETS_BY_SYMBOL,
    REGIONS,
)
from .model import MIN_TRAIN, walk_forward
from .predict import Forecast, forecast_all, parse_shock, to_frame
from .regions import dashboard_symbols
from .scenarios import SCENARIOS, scenario
from .score import DEFAULT_WINDOW, score_symbols
from .score import to_frame as score_to_frame
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
from .universe import read_universe, us_universe

log = logging.getLogger(__name__)

# The hourly window is short, so the intraday variant needs a smaller warm-up.
INTRADAY_MIN_TRAIN = 200


def _market_symbol(value: str) -> str:
    if value not in MARKETS_BY_SYMBOL:
        known = ", ".join(MARKETS_BY_SYMBOL)
        raise argparse.ArgumentTypeError(f"unknown market {value!r}; choose from {known}")
    return value


def _shock(value: str) -> tuple[str, float]:
    known = set(MARKETS_BY_SYMBOL) | {i.symbol for i in INDICATORS} | {CURVE_FRONT, CURVE_STRIP}
    try:
        symbol, move = parse_shock(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if symbol not in known:
        raise argparse.ArgumentTypeError(f"unknown instrument {symbol!r}")
    return symbol, move


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
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
    print("\nScenarios (predict --scenario):")
    for s in SCENARIOS.values():
        legs = ", ".join(f"{sym} {move:+.1%}" for sym, move in s.moves.items())
        print(f"  {s.name:<22} {s.description}\n  {'':<22} {legs}")


def _forecast(
    panel: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    hourly: dict[str, pd.Series] | None,
    shocks: dict[str, float] | None = None,
) -> list[Forecast]:
    """Forecast every requested market, dropping the pre-open features if need be.

    The intraday variant depends on futures bars running into the bell, which
    a stale or halted feed may not provide. Rather than return nothing, the
    run is repeated on the daily features alone and the loss of sharpness is
    reported.
    """
    try:
        return forecast_all(
            panel,
            symbols=args.market,
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
            symbols=args.market,
            c=args.regularisation,
            min_train=MIN_TRAIN,
            shocks=shocks,
        )


def _cmd_predict(args: argparse.Namespace) -> None:
    hourly = _hourly(args)
    shocks = dict(scenario(args.scenario).shocks()) if args.scenario else {}
    # An explicit --shock on the same instrument replaces the scenario's leg.
    shocks.update(args.shock or [])
    forecasts = _forecast(_panel(args), args, hourly, shocks)
    if args.scenario:
        print(f"scenario: {args.scenario} — {scenario(args.scenario).description}")
    if shocks:
        described = ", ".join(f"{s} {np.expm1(m):+.2%}" for s, m in shocks.items())
        print(f"hypothetical: {described}\n")
    frame = to_frame(forecasts).sort_values("p_open_up", ascending=False)
    print(frame.to_string(index=False))
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
    hourly = _hourly(args)
    forecasts = _forecast(panel, args, hourly)
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
    panel = _panel(args)
    hourly = _hourly(args)
    since = _since_timestamp(args)
    rows: list[dict] = []
    window_rows: list[dict] = []
    for symbol in args.market or [m.symbol for m in MARKETS]:
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


def _cmd_dashboard(args: argparse.Namespace) -> None:
    panel = _panel(args)
    hourly = _hourly(args)
    symbols = [m.symbol for m in MARKETS if m.region == args.region]
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


def _cmd_score(args: argparse.Namespace) -> None:
    symbols = [s.upper() for s in args.symbols]
    asof: pd.Timestamp | None = None
    if args.asof:
        try:
            asof = pd.Timestamp(args.asof).tz_localize(None)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"error: --asof {args.asof!r} is not a valid date: {exc}") from exc
    scores = score_symbols(
        symbols,
        window=args.window,
        asof=asof,
        start=args.start,
        cache_dir=Path(args.cache),
        refresh=args.refresh,
    )
    frame = score_to_frame(scores)
    print(frame.to_string(index=False))
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
    forecasts = forecast_all(panel, symbols=[args.market], c=args.regularisation)
    print(render_sector_text(build_sector_board(panel, forecasts[0])), end="")


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

    backtest = sub.add_parser("backtest", help="walk-forward out-of-sample metrics")
    backtest.add_argument(
        "--market", action="append", type=_market_symbol, help="restrict to a symbol"
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

    sectors = sub.add_parser(
        "sectors", help="split one European index's open call by STOXX 600 sector"
    )
    sectors.add_argument("--market", type=_market_symbol, default="^STOXX50E")
    sectors.set_defaults(func=_cmd_sectors)

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
