#!/usr/bin/env python3
"""
scan_70d_highs.py
Finds stocks (US common stock) with market cap >= CAP_MIN that made a 70-day high
during the current week. Outputs a CSV with the Top-N by market cap and a short digest.

Requirements: requests
Environment: POLYGON_API_KEY must be set (or pass via --api-key)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

# ----------------------------
# HTTP helpers (simple retries)
# ----------------------------

def _sleep_backoff(k: int) -> None:
    time.sleep(min(0.5 * (2**k), 8.0))  # cap at 8s


def http_get_json(url: str, params: Dict[str, str], max_tries: int = 6) -> dict:
    last_exc: Optional[Exception] = None
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            # Polygon returns 200 with {"status":"ERROR",...} sometimes
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and j.get("status") in ("ERROR", "NOT_FOUND"):
                    # Treat as failure to retry if appropriate
                    last_exc = RuntimeError(f"Polygon error body: {j}")
                    if "API Key was not provided" in str(j):
                        # No point retrying
                        break
                else:
                    return j
            elif r.status_code == 429:
                # rate limited, backoff and retry
                last_exc = RuntimeError("429 Too Many Requests")
            else:
                last_exc = RuntimeError(f"{r.status_code} {r.text[:600]}")
        except Exception as e:
            last_exc = e

        _sleep_backoff(k)

    raise RuntimeError(f"GET {url} failed after retries: {last_exc}")


# -----------------------------------------
# Reference listing (with date → market_cap)
# -----------------------------------------

def list_common_stocks(
    api_key: str,
    pages: int = 5,
    max_syms: int = 5000,
    asof_date: Optional[str] = None,
) -> List[dict]:
    """
    Returns a list of reference ticker dicts from Polygon with fields including:
      - ticker
      - market_cap (populated when date= is supplied)
    We query market=stocks, type=CS, active=true.
    """
    url = "https://api.polygon.io/v3/reference/tickers"
    out: List[dict] = []
    next_url: Optional[str] = None

    for p in range(pages):
        params = {
            "market": "stocks",
            "type": "CS",
            "active": "true",
            "sort": "ticker",
            "order": "asc",
            "limit": "1000",
            "apiKey": api_key,
        }
        if asof_date:
            params["date"] = asof_date

        j = http_get_json(next_url or url, params if next_url is None else {})
        results = j.get("results") or []
        if not isinstance(results, list):
            break

        out.extend(results)
        if len(out) >= max_syms:
            break

        # navigate pagination
        next_url = (j.get("next_url") or j.get("next_url")).strip() if j.get("next_url") else None
        if not next_url:
            break

    # Trim to max_syms
    if len(out) > max_syms:
        out = out[:max_syms]

    return out


# ----------------------------------
# Aggregates fetch (daily bar window)
# ----------------------------------

def fetch_daily_bars(
    api_key: str, ticker: str, start: str, end: str
) -> List[dict]:
    """
    Fetch daily aggregates for [start, end], adjusted=true, ascending.
    Returns a list of bar dicts with 't' (ms), 'h' (high).
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    j = http_get_json(url, params)
    results = j.get("results") or []
    if not isinstance(results, list):
        return []
    return results


# -------------------------------------
# 70-day high detection within a window
# -------------------------------------

def made_70d_high_this_week(
    bars: List[dict],
    week_start: dt.date,
    week_end: dt.date,
    lookback_days: int = 70,
) -> Optional[Tuple[dt.date, float]]:
    """
    Given daily bars (ascending), returns the (date, high) for the first day
    in [week_start..week_end] that hit a 70-day (lookback_days) rolling high.
    If none, returns None.

    bars: items like {"t": epoch_ms, "h": high}
    """
    if not bars:
        return None

    # Convert to simple (date, high)
    seq: List[Tuple[dt.date, float]] = []
    for b in bars:
        try:
            d = dt.datetime.utcfromtimestamp(b["t"] / 1000).date()
            h = float(b["h"])
            seq.append((d, h))
        except Exception:
            continue

    # Rolling max of the prior lookback_days highs (exclude same day)
    highs = [h for (_, h) in seq]
    dates = [d for (d, _) in seq]

    for i in range(len(seq)):
        d = dates[i]
        if d < week_start or d > week_end:
            continue

        # prior window indices
        start_idx = max(0, i - lookback_days)
        prev_highs = highs[start_idx:i]  # exclude today
        if not prev_highs:
            continue

        today_h = highs[i]
        # hit 70D high if today's high >= max of prior highs
        if today_h >= max(prev_highs):
            return d, today_h

    return None


# ------------
# CLI plumbing
# ------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan for 70-day highs this week among US common stocks with market cap filter."
    )
    parser.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY") or os.getenv("POLY_KEY"), help="Polygon API key (or set POLYGON_API_KEY)")
    parser.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Market cap floor (USD), default 1e9")
    parser.add_argument("--top", type=int, default=10, help="Top N results to output (by market cap)")
    parser.add_argument("--pages", type=int, default=5, help="Reference pages (x1000 rows) to scan")
    parser.add_argument("--max-syms", type=int, default=5000, help="Max symbols to consider")
    parser.add_argument("--window", type=int, default=70, help="Lookback days for 70D high")
    parser.add_argument("--since-days", type=int, default=140, help="Calendar days of bars to fetch (must be > window)")
    parser.add_argument("--friday", type=str, default="", help="Friday YYYY-MM-DD to define the week (optional)")
    parser.add_argument("--out-dir", type=str, default="backtests", help="Output folder")
    return parser.parse_args(argv)


def find_week(friday_str: str = "") -> Tuple[dt.date, dt.date, dt.date]:
    """
    Returns (week_monday, week_friday, friday_date).
    If friday_str empty → use 'last Friday' in Europe/Oslo timezone notion.
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Oslo")
    except Exception:
        tz = dt.timezone.utc

    if friday_str:
        friday = dt.date.fromisoformat(friday_str)
    else:
        now = dt.datetime.now(tz).date()
        # walk back to Friday
        d = now
        while d.weekday() != 4:  # Fri
            d -= dt.timedelta(days=1)
        friday = d

    monday = friday - dt.timedelta(days=4)  # Mon..Fri
    return monday, friday, friday


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("FATAL: Missing Polygon API key (set POLYGON_API_KEY or use --api-key).", file=sys.stderr)
        return 2

    if args.since_days <= args.window:
        args.since_days = max(args.window + 30, 120)

    week_mon, week_fri, friday = find_week(args.friday)

    print(f"Week window: {week_mon} .. {week_fri} | bars from: {(week_fri - dt.timedelta(days=args.since_days))} .. {week_fri}")

    # 1) Reference universe snapshot at Friday (gets market_cap)
    ref = list_common_stocks(
        api_key=args.api_key,
        pages=args.pages,
        max_syms=args.max_syms,
        asof_date=friday.isoformat(),
    )
    print(f"Fetched {len(ref)} active common stocks from Polygon (pages={args.pages}).")

    # Count how many have non-null market_cap
    with_mcap = sum(1 for r in ref if (r.get("market_cap") or 0) > 0)
    print(f"Reference rows with non-null market_cap: {with_mcap}/{len(ref)}")

    # 2) Cap filter
    keep = []
    for r in ref:
        cap = r.get("market_cap")
        try:
            cap_f = float(cap) if cap is not None else 0.0
        except Exception:
            cap_f = 0.0
        if cap_f >= float(args.cap_min):
            keep.append((r.get("ticker"), cap_f))

    # Just in case
    if not keep:
        print("No symbols after cap filter; aborting.")
        ensure_dir(args.out_dir)
        write_csv(
            os.path.join(args.out_dir, "hi70_thisweek.csv"),
            [],
            ["symbol", "market_cap", "hit_date", "hit_price", "week_start"],
        )
        open(os.path.join(args.out_dir, "hi70_digest.txt"), "w", encoding="utf-8").write(
            f"week {week_mon}..{week_fri}: 0 total hits. Top-0 by mcap:\n"
        )
        return 0

    print(f"After cap filter (>= {int(args.cap_min):,}): {len(keep)} symbols.")

    # 3) For each, fetch bars and test 70D high this week
    start_date = (week_fri - dt.timedelta(days=args.since_days)).isoformat()
    end_date = week_fri.isoformat()

    hits: List[Tuple[str, float, dt.date, float]] = []
    scanned = 0
    for sym, mcap in keep:
        scanned += 1
        try:
            bars = fetch_daily_bars(args.api_key, sym, start_date, end_date)
            hit = made_70d_high_this_week(bars, week_mon, week_fri, lookback_days=args.window)
            if hit:
                d_hit, px = hit
                hits.append((sym, mcap, d_hit, px))
        except Exception as e:
            # don't crash—just continue
            print(f"WARN: {sym} fetch/scan error: {e}", file=sys.stderr)
        # light pacing
        if scanned % 200 == 0:
            time.sleep(0.2)

    # 4) Sort by mcap desc and keep Top-N
    hits.sort(key=lambda x: x[1], reverse=True)
    top_hits = hits[: max(1, args.top)]

    # 5) Write outputs
    ensure_dir(args.out_dir)

    csv_rows = [
        {
            "symbol": s,
            "market_cap": f"{mcap:.0f}",
            "hit_date": d.isoformat(),
            "hit_price": f"{px:.4f}",
            "week_start": week_mon.isoformat(),
        }
        for (s, mcap, d, px) in top_hits
    ]
    csv_path = os.path.join(args.out_dir, "hi70_thisweek.csv")
    write_csv(csv_path, csv_rows, ["symbol", "market_cap", "hit_date", "hit_price", "week_start"])

    digest_path = os.path.join(args.out_dir, "hi70_digest.txt")
    lines = [f"week {week_mon}..{week_fri}: {len(hits)} total hits. Top-{len(top_hits)} by mcap:\n"]
    for (s, mcap, d, px) in top_hits:
        lines.append(f"  {s:<6}  mcap=${int(mcap):,}  hit={d}  px={px:.2f}")
    open(digest_path, "w", encoding="utf-8").write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Wrote: {csv_path}  and  {digest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
