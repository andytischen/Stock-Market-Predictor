"""Command line interface: ``python -m gapmodel <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .dashboard import build_dashboard, render_html, render_text
from .data import DEFAULT_CACHE, load_panel
from .features import build_features
from .intraday import load_hourly_panel
from .markets import INDICATORS, MARKETS, MARKETS_BY_SYMBOL, REGIONS
from .model import MIN_TRAIN, walk_forward
from .predict import forecast_all, to_frame
from .web import serve_dashboard

# The hourly window is short, so the intraday variant needs a smaller warm-up.
INTRADAY_MIN_TRAIN = 200


def _market_symbol(value: str) -> str:
    if value not in MARKETS_BY_SYMBOL:
        known = ", ".join(MARKETS_BY_SYMBOL)
        raise argparse.ArgumentTypeError(f"unknown market {value!r}; choose from {known}")
    return value


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
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
    if not getattr(args, "intraday", False):
        return None
    return load_hourly_panel(cache_dir=Path(args.cache), refresh=args.refresh)


def _cmd_markets(_: argparse.Namespace) -> None:
    print("Markets (open / close, UTC hours):")
    for m in MARKETS:
        print(f"  {m.symbol:<12} {m.name:<20} {m.region:<9} {m.open_utc:>6} {m.close_utc:>6}")
    print("\nIndicators:")
    for i in INDICATORS:
        print(f"  {i.symbol:<12} {i.name}")


def _cmd_predict(args: argparse.Namespace) -> None:
    hourly = _hourly(args)
    forecasts = forecast_all(
        _panel(args),
        symbols=args.market,
        c=args.regularisation,
        hourly=hourly,
        min_train=INTRADAY_MIN_TRAIN if hourly else MIN_TRAIN,
    )
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


def _cmd_backtest(args: argparse.Namespace) -> None:
    panel = _panel(args)
    hourly = _hourly(args)
    rows = []
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
        if args.reliability:
            print(f"\n{symbol} reliability:")
            print(result.reliability().to_string())
    if not rows:
        raise SystemExit("nothing to back-test")
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))


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


def _cmd_web(args: argparse.Namespace) -> None:
    serve_dashboard(
        _panel(args),
        _hourly(args),
        host=args.host,
        port=args.port,
        region=args.region,
        at=args.at,
        regularisation=args.regularisation,
        launch_browser=not args.no_browser,
    )


def _as_of(hours: float | None) -> pd.Timestamp | None:
    """Today's date at the given UTC hour, or now when no hour is given."""
    if hours is None:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    return now.normalize() + pd.Timedelta(hours=hours)


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

    predict = sub.add_parser("predict", help="probability that the next open is up")
    predict.add_argument(
        "--market", action="append", type=_market_symbol, help="restrict to a symbol"
    )
    predict.add_argument("--explain", action="store_true", help="show top drivers")
    predict.add_argument("--csv", help="also write the table to this path")
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
    backtest.set_defaults(func=_cmd_backtest)

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

    web = sub.add_parser("web", help="serve the dashboard in a local browser interface")
    web.add_argument("--region", choices=REGIONS, default="Asia")
    web.add_argument(
        "--at", type=_utc_time, help="UTC time of day to render for, e.g. 05:00 (default: now)"
    )
    web.add_argument("--host", default="127.0.0.1", help="address to bind")
    web.add_argument("--port", type=int, default=8000, help="port to bind")
    web.add_argument("--no-browser", action="store_true", help="do not auto-open a browser tab")
    web.add_argument(
        "--intraday",
        action="store_true",
        help="add pre-open futures moves (recent ~2 years only)",
    )
    web.set_defaults(func=_cmd_web)

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
