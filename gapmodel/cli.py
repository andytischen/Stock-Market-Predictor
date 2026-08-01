"""Command line interface: ``python -m gapmodel <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CACHE, load_panel
from .features import build_features
from .markets import INDICATORS, MARKETS
from .model import walk_forward
from .predict import forecast_all, to_frame


def _panel(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return load_panel(start=args.start, cache_dir=Path(args.cache), refresh=args.refresh)


def _cmd_markets(_: argparse.Namespace) -> None:
    print("Markets (open / close, UTC hours):")
    for m in MARKETS:
        print(f"  {m.symbol:<12} {m.name:<20} {m.region:<9} {m.open_utc:>6} {m.close_utc:>6}")
    print("\nIndicators:")
    for i in INDICATORS:
        print(f"  {i.symbol:<12} {i.name}")


def _cmd_predict(args: argparse.Namespace) -> None:
    forecasts = forecast_all(_panel(args), symbols=args.market, c=args.regularisation)
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
    rows = []
    for symbol in args.market or [m.symbol for m in MARKETS]:
        try:
            features, labels = build_features(symbol, panel)
            result = walk_forward(features, labels, c=args.regularisation)
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
    parser.add_argument("--regularisation", type=float, default=0.1, help="logistic C")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    markets = sub.add_parser("markets", help="list modelled markets and indicators")
    markets.set_defaults(func=_cmd_markets)

    fetch = sub.add_parser("fetch", help="download and cache the price panel")
    fetch.set_defaults(func=_cmd_fetch)

    predict = sub.add_parser("predict", help="probability that the next open is up")
    predict.add_argument("--market", action="append", help="restrict to a symbol")
    predict.add_argument("--explain", action="store_true", help="show top drivers")
    predict.add_argument("--csv", help="also write the table to this path")
    predict.set_defaults(func=_cmd_predict)

    backtest = sub.add_parser("backtest", help="walk-forward out-of-sample metrics")
    backtest.add_argument("--market", action="append", help="restrict to a symbol")
    backtest.add_argument("--reliability", action="store_true", help="calibration table")
    backtest.set_defaults(func=_cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
