"""Command line entry point: `radar <command>`."""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from radar.config import CACHE_DIR, ConfigError, ensure_dirs
from radar.data import cache, returns, universe
from radar.data.tiingo import DEFAULT_START


def _parse_date(value: str | None) -> date | None:
    return pd.Timestamp(value).date() if value else None


def cmd_universes(_: argparse.Namespace) -> int:
    for key, uni in universe.UNIVERSES.items():
        print(f"\n{key}  ({len(uni)} tickers)  -- {uni.title}")
        print(f"  {uni.description}")
        counts = uni.to_frame()["group"].value_counts().sort_index()
        for group, n in counts.items():
            print(f"    {group:<26} {n}")
        if uni.caveats:
            print(f"  caveat: {uni.caveats}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    ensure_dirs()
    if args.universe == "all":
        tickers = list(universe.all_tickers())
    else:
        tickers = list(universe.get_universe(args.universe).tickers)

    print(f"Fetching {len(tickers)} tickers into {CACHE_DIR}")
    report = returns_safe_fetch(tickers, args)
    if report is None:
        return 1

    failures = report[report["status"].str.startswith("error")]
    print(f"\n{len(report) - len(failures)}/{len(report)} tickers up to date.")
    if not failures.empty:
        print("\nFailures (re-run to resume; the cache is incremental):")
        for ticker, row in failures.iterrows():
            print(f"  {ticker:<6} {row['status']}")
        return 1
    return 0


def returns_safe_fetch(tickers: list[str], args: argparse.Namespace) -> pd.DataFrame | None:
    from radar.data import tiingo

    try:
        return tiingo.fetch_universe(
            tickers,
            start=_parse_date(args.start) or DEFAULT_START,
            end=_parse_date(args.end),
            force=args.force,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def cmd_status(args: argparse.Namespace) -> int:
    tickers = (
        list(universe.get_universe(args.universe).tickers)
        if args.universe != "all"
        else None
    )
    summary = cache.cache_summary(tickers)
    if summary.empty:
        print(f"Cache is empty ({CACHE_DIR}). Run `radar fetch`.")
        return 0
    missing = summary[summary["rows"] == 0]
    print(summary.to_string())
    print(f"\n{len(summary) - len(missing)} cached, {len(missing)} missing.")
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    rets, report = returns.returns_panel(
        universe=args.universe,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    print(report.summary())
    if rets.empty:
        print("\nNo usable panel. Run `radar fetch` first.")
        return 1
    for window in (90, 252, 504):
        q = report.q(window)
        flag = "  <-- singular, unusable" if q >= 1 else ("  <-- noisy" if q > 0.25 else "")
        print(f"  q = N/T at {window:>3}d window: {q:.3f}{flag}")
    print(f"\nAnnualised vol of the equal-weight basket: "
          f"{rets.mean(axis=1).std() * (252 ** 0.5):.1%}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    print(returns.coverage_frontier(universe=args.universe).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="radar", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("universes", help="list the fixed universes and their sectors")
    p.set_defaults(func=cmd_universes)

    p = sub.add_parser("fetch", help="populate the price cache from Tiingo")
    p.add_argument("--universe", default="all", help="universe key, or 'all' (default)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default 1998-01-01)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default today)")
    p.add_argument("--force", action="store_true", help="re-fetch full history")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("status", help="show what is cached")
    p.add_argument("--universe", default="all")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("panel", help="build the return panel and report alignment")
    p.add_argument("--universe", default=universe.DEFAULT_UNIVERSE)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.set_defaults(func=cmd_panel)

    p = sub.add_parser("coverage", help="assets retained vs. start date trade-off")
    p.add_argument("--universe", default=universe.DEFAULT_UNIVERSE)
    p.set_defaults(func=cmd_coverage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
