#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan for 70-day highs made during a target week.

- Pulls active common stocks (CS) from Polygon (paged, sorted by market cap).
- For each symbol, downloads daily bars covering:
    [week_start - since_days,  week_end]
- Computes prior 70D high using bars strictly before week_start.
- Flags symbols that make a new 70D high during [week_start .. week_end].
- Optional filters: min price, min avg dollar volume (pre-week), market cap min.
- Writes:
    backtests/hi70_thisweek.csv
    backtests/hi70_digest.txt
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
from datetime import date, datetime, timedelta, timezone
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests


# --------------------------
# Utilities / configuration
# --------------------------

@dc.dataclass
class Config:
    api_key: str
    pages: int = 3
    since_days: int = 150
    top: int = 10
    out_dir: str = "backtests"
    min_price: float = 0.0
    min_dollar_vol: float = 0.0
    cap_min: float = 0.0
    friday: Optional[date] = None


def parse_args():
    import os, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", dest="api_key", default=os.getenv("POLYGON_API_KEY"))
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--since-days", type=int, default=130)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out-dir", default="backtests")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=3_000_000)
    # ✱ NEW: make friday optional
    p.add_argument("--friday", dest="friday", default=None, help="YYYY-MM-DD; default = last Friday (NY)")
    return p.parse_args()

    fri: Optional[date] = None
    if a.friday:
        fri = datetime.fromisoformat(a.friday).date()

    cfg = Config(
        api_key=a.api_key.strip(),
        pages=int(a.pages),
        since_days=int(a.since_days),
        top=int(a.top),
        out_dir=str(a.out_dir),
        min_price=float(a.min_price),
        min_dollar_vol=float(a.min_dollar_vol),
        cap_min=float(a.cap_min),
        friday=fri,
    )

    if not cfg.api_key:
        print("FATAL: Missing Polygon API key. Use --api-key or set POLYGON_API_KEY.", file=sys.stderr)
        sys.exit(1)

    return cfg


def last_friday_ny(today_utc: Optional[datetime] = None) -> date:
    """Return last Friday (America/New_York)."""
    # We do not import zoneinfo to keep dependencies minimal; Polygon uses UTC, but "week" is US trading week.
    # We take "today" in UTC and treat weekday() on UTC; that’s perfectly fine for determining last business Friday.
    if today_utc is None:
        today_utc = datetime.now(timezone.utc)
    d = today_utc.date()
    while d.weekday() != 4:  # 0=Mon .. 4=Fri
        d -= timedelta(days=1)
    return d


def monday_of_week(friday: date) -> date:
    return friday - timedelta(days=4)


def fmt_usd(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:,.0f}"


def sleep_backoff(k: int):
    time.sleep(min(1.0 * (2 ** k), 10.0))


# --------------------------
# HTTP helpers (Polygon)
# --------------------------

AUTH: Dict[str, str] = {}  # set in main()


def http_get_json(url: str, params: Optional[dict] = None, timeout: int = 30, retries: int = 6) -> dict:
    for k in range(retries):
        r = requests.get(url, headers=AUTH, params=params, timeout=timeout)
        if r.status_code not in (429, 502, 503, 504):
            r.raise_for_status()
            try:
                return r.json() or {}
            except Exception:
                return {}
        sleep_backoff(k)
    # final try (raise if bad)
    r = requests.get(url, headers=AUTH, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json() or {}


# --------------------------
# Reference universe
# --------------------------

@dc.dataclass
class RefRow:
    symbol: str
    name: str
    mcap: Optional[float]


def list_common_stocks(api_key: str, pages: int = 3) -> list[str]:
    import requests, time

    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": 1000,
        # REMOVE the invalid sort:
        # "sort": "market_cap",
        # "order": "desc",
    }

    tickers = []
    next_url = url
    for _ in range(max(1, int(pages))):
        r = requests.get(next_url, params={**params, "apiKey": api_key}, timeout=30)
        # Fallback if any future invalid param slips in:
        if r.status_code == 400:
            # retry once without any client-side sort/order
            params.pop("sort", None)
            params.pop("order", None)
            r = requests.get(url, params={**params, "apiKey": api_key}, timeout=30)
        r.raise_for_status()

        j = r.json() or {}
        for row in j.get("results", []) or []:
            t = (row.get("ticker") or "").strip()
            if t:
                tickers.append(t)

        next_url = j.get("next_url")
        if not next_url:
            break
        time.sleep(0.05)

    return tickers


# --------------------------
# Bars + detection
# --------------------------

@dc.dataclass
class Bar:
    d: date
    o: float
    h: float
    l: float
    c: float
    v: float


def fetch_daily_bars(sym: str, start: date, end: date) -> List[Bar]:
    """
    Fetch daily bars (adjusted) for [start, end], ascending.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "limit": 50000, "sort": "asc"}
    j = http_get_json(url, params=params, timeout=30)
    res = j.get("results") or []
    out: List[Bar] = []
    for r in res:
        try:
            tms = int(r["t"])  # ms since epoch (UTC)
            d = datetime.utcfromtimestamp(tms / 1000.0).date()
            out.append(Bar(
                d=d,
                o=float(r["o"]),
                h=float(r["h"]),
                l=float(r["l"]),
                c=float(r["c"]),
                v=float(r["v"]),
            ))
        except Exception:
            continue
    return out


def avg_dollar_vol_preweek(bars: List[Bar], week_start: date, n: int = 20) -> float:
    pre = [b for b in bars if b.d < week_start]
    pre = pre[-n:]
    if not pre:
        return 0.0
    return sum(b.c * b.v for b in pre) / len(pre)


@dc.dataclass
class Hit:
    symbol: str
    name: str
    mcap: Optional[float]
    gap: float           # (best_high / prior70H - 1)
    bar_date: date       # date of best bar this week
    best_high: float
    best_close: float
    prior70H: float


def find_week_hi70(
    sym: str, name: str, mcap: Optional[float],
    bars: List[Bar], week_start: date, week_end: date,
    min_price: float, min_dollar_vol: float
) -> Optional[Hit]:
    """
    Return Hit if symbol makes a new 70D high during [week_start .. week_end]
    relative to the prior 70D high (computed strictly on bars < week_start).
    Applies price/volume filters based on pre-week data.
    """
    if len(bars) < 80:
        return None

    # Pre-week filters
    pre = [b for b in bars if b.d < week_start]
    if len(pre) < 70:
        return None

    # price filter: last close before week
    last_pre_close = pre[-1].c
    if last_pre_close < min_price:
        return None

    if min_dollar_vol > 0.0:
        adv = avg_dollar_vol_preweek(bars, week_start, n=20)
        if adv < min_dollar_vol:
            return None

    # prior 70D high (strictly before week_start)
    prior70 = max(b.h for b in pre[-70:])

    weekbars = [b for b in bars if week_start <= b.d <= week_end]
    if not weekbars:
        return None

    # Check if any bar exceeds prior70
    best: Optional[Tuple[Bar, float]] = None  # (bar, gap)
    for b in weekbars:
        if b.h > prior70:
            g = b.h / prior70 - 1.0
            if best is None or g > best[1]:
                best = (b, g)

    if best is None:
        return None

    b, gap = best
    return Hit(
        symbol=sym, name=name, mcap=mcap, gap=gap,
        bar_date=b.d, best_high=b.h, best_close=b.c, prior70H=prior70
    )


# --------------------------
# Main
# --------------------------

def main():
    cfg = parse_args()
    global AUTH
    AUTH = {"Authorization": f"Bearer {cfg.api_key}"}

    # Determine friday + week window
    fri_str = cfg.friday or last_friday_ny().isoformat()
    import datetime as dt
    fri = dt.date.fromisoformat(fri_str)
    mon = monday_of_week(fri)
    bars_from = mon - timedelta(days=cfg.since_days)
    print(f"Week window: {mon} .. {fri} | bars from: {bars_from} .. {fri}")

    # Universe
    ref = list_common_stocks(api_key=cfg.api_key, pages=cfg.pages)
    print(f"Fetched {len(ref)} active common stocks from Polygon (pages={cfg.pages}).")

    # Apply market cap min only if provided
    if cfg.cap_min > 0:
        before = len(ref)
        ref = [r for r in ref if r.mcap is not None and r.mcap >= cfg.cap_min]
        print(f"After cap filter (>= {cfg.cap_min:,.0f}): {len(ref)} kept (was {before}).")

    # Ensure output dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    total = len(ref)
    hits: List[Hit] = []
    okbars = 0

    for i, row in enumerate(ref, start=1):
        if i % 200 == 0 or i == 1:
            print(f"— scanned {i}/{total}; candidates so far = {len(hits)}")

        try:
            bars = fetch_daily_bars(row.symbol, bars_from, fri)
        except Exception as e:
            # soft-fail per symbol
            continue

        if len(bars) >= 70:
            okbars += 1

        h = find_week_hi70(
            row.symbol, row.name, row.mcap, bars, mon, fri,
            min_price=cfg.min_price, min_dollar_vol=cfg.min_dollar_vol
        )
        if h:
            hits.append(h)

    # Summary
    any_break_high = sum(1 for _ in hits)
    print(f"[hi70][dbg] SUMMARY: scanned={total} | okbars={okbars} | any_breakout(high)={any_break_high}")

    # Sort by market cap desc (None goes last), then by gap desc
    def _sort_key(h: Hit):
        mc = h.mcap if h.mcap is not None else -1.0
        return (mc, h.gap)

    hits_sorted = sorted(hits, key=_sort_key, reverse=True)

    # Write CSV
    csv_path = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "name", "market_cap", "bar_date",
            "prior70H", "best_high", "best_close", "gap_pct",
            "week_start", "week_end"
        ])
        for h in hits_sorted:
            w.writerow([
                h.symbol,
                h.name,
                f"{h.mcap:.0f}" if h.mcap is not None else "",
                h.bar_date.isoformat(),
                f"{h.prior70H:.2f}",
                f"{h.best_high:.2f}",
                f"{h.best_close:.2f}",
                f"{100.0 * h.gap:.2f}",
                mon.isoformat(),
                fri.isoformat(),
            ])

    # Write digest (text) + print top-N
    digest_path = os.path.join(cfg.out_dir, "hi70_digest.txt")
    topN = hits_sorted[: cfg.top]
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(f"Candidates that made a 70D high this week: {len(hits_sorted)}\n\n")
        f.write("Top-{n} by market cap:\n".format(n=len(topN)))
        f.write("symbol  mcap    prior70H  high      gap%   date\n")
        for h in topN:
            f.write(
                f"{h.symbol:<6} {fmt_usd(h.mcap):>8}  {h.prior70H:>8.2f}  {h.best_high:>8.2f}  "
                f"{100.0 * h.gap:>6.2f}  {h.bar_date.isoformat()}\n"
            )

    # Console preview (same topN)
    if topN:
        print("\n[hi70][dbg] Top-{N} nearest to prior-70D HIGH (by mcap):".format(N=len(topN)))
        print(f"{'SYM':<6} {'mcap':>8}   {'prior70H':>8}  {'high':>8}  {'gap%':>6}  {'date':>10}")
        for h in topN:
            print(
                f"{h.symbol:<6} {fmt_usd(h.mcap):>8}   {h.prior70H:>8.2f}  {h.best_high:>8.2f}  "
                f"{100.0 * h.gap:>6.2f}  {h.bar_date.isoformat():>10}"
            )
    else:
        print("No candidates hit a new 70D high this week.")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {digest_path}")


if __name__ == "__main__":
    main()
