#!/usr/bin/env python3
"""
Scan for 70-day highs in the week *after* the last Friday.

Outputs:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt

Defaults:
  - API key from env POLYGON_API_KEY (or --api-key)
  - last Friday inferred from America/New_York timezone (or --friday)
  - US common stocks (type=CS), active=true
  - up to --pages * 1000 symbols from Polygon reference/tickers
  - optional filters: --cap-min, --min-price, --min-dollar-vol
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore


# --------------------------
# Utilities / configuration
# --------------------------

NY = ZoneInfo("America/New_York")

def last_friday_ny(today_utc: Optional[datetime] = None) -> date:
    """Return last New York Friday (date)."""
    if today_utc is None:
        today_utc = datetime.utcnow().replace(tzinfo=None)
    now_ny = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(NY)
    d = now_ny.date()
    # Go back to the most recent Friday (weekday=4)
    while d.weekday() != 4:
        d -= timedelta(days=1)
    return d

def ymd(d: date) -> str:
    return d.isoformat()

def backoff_sleep(k: int) -> None:
    time.sleep(min(0.5 * (2 ** k), 6.0))

def must_env(key: str, fallback: Optional[str] = None) -> str:
    v = os.getenv(key) or (fallback or "")
    if not v:
        sys.exit(f"FATAL: environment variable {key} is empty")
    return v


# --------------------------
# HTTP helpers
# --------------------------

def http_get_json(url: str, params: Dict[str, str], max_tries: int = 6) -> Dict:
    last = None
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                backoff_sleep(k)
                continue
            r.raise_for_status()
            return r.json() or {}
        except requests.HTTPError as e:
            last = e
            if r.status_code >= 500:
                backoff_sleep(k)
                continue
            break
        except Exception as e:
            last = e
            backoff_sleep(k)
    raise RuntimeError(f"GET failed {url} ({last})")


# --------------------------
# Polygon adapters
# --------------------------

def list_common_stocks(api_key: str, pages: int, on_date: Optional[date]) -> List[Dict]:
    """
    Return up to pages*1000 tickers for US common stocks (active).
    We *do not* sort by market cap here (Polygon rejects 'sort=market_cap' for this endpoint).
    If on_date is provided, Polygon will attach 'market_cap' for that date; if not,
    we still get the list, and market_cap may be missing.
    """
    base = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": "1000",
        "apiKey": api_key,
    }
    if on_date:
        params["date"] = ymd(on_date)

    results: List[Dict] = []
    page_n = 0
    cursor: Optional[str] = None

    while True:
        p = params.copy()
        if cursor:
            p["cursor"] = cursor
        j = http_get_json(base, p)
        batch = j.get("results", []) or []
        results.extend(batch)
        page_n += 1

        nxt = j.get("next_url") or ""
        if nxt:
            q = parse_qs(urlparse(nxt).query)
            cur = q.get("cursor", [None])[0]
            cursor = cur
        else:
            cursor = None

        if not cursor or page_n >= pages:
            break

    return results


def fetch_daily_bars(api_key: str, symbol: str, start: date, end: date) -> List[Dict]:
    """
    Fetch daily bars (adjusted) for symbol within [start, end].
    """
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{ymd(start)}/{ymd(end)}"
    p = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    j = http_get_json(base, p)
    return j.get("results", []) or []


# --------------------------
# Core scan logic
# --------------------------

@dataclass
class Candidate:
    symbol: str
    name: str
    market_cap: float
    prior70h: float
    week_high: float
    week_close: float
    gap_high_pct: float
    gap_close_pct: float
    first_break_day: str  # YYYY-MM-DD or ""
    bars: int

def detect_week_breakout(bars: List[Dict], mon: date, week_end: date) -> Tuple[float, float, float, float, str]:
    """
    Given Polygon 'bars' (list of dicts) sorted asc by time,
    compute:
      - prior70h (max high of the *previous* 70 trading bars before Monday),
      - week_high, week_close (max high / last close in Mon..Fri),
      - gaps to prior70h, and first breakout day (if any).
    """
    if not bars:
        return (0.0, 0.0, 0.0, 0.0, "")

    # Convert milliseconds to date
    def as_date(ms: int) -> date:
        return datetime.utcfromtimestamp(ms / 1000).date()

    # Split bars into before-mon and mon..fri
    before: List[Dict] = []
    week: List[Dict] = []
    for b in bars:
        d = as_date(int(b.get("t", 0)))
        if d < mon:
            before.append(b)
        elif mon <= d <= week_end:
            week.append(b)

    if not before or not week:
        return (0.0, 0.0, 0.0, 0.0, "")

    # prior 70 trading highs (use last 70 bars from 'before')
    prev = before[-70:] if len(before) >= 70 else before
    if not prev:
        return (0.0, 0.0, 0.0, 0.0, "")

    prior70h = max(float(x.get("h", 0.0) or 0.0) for x in prev)
    week_high = max(float(x.get("h", 0.0) or 0.0) for x in week)
    week_close = float(week[-1].get("c", 0.0) or 0.0)

    # Find first day where high or close >= prior70h
    first_day = ""
    for x in week:
        h = float(x.get("h", 0.0) or 0.0)
        c = float(x.get("c", 0.0) or 0.0)
        if h >= prior70h or c >= prior70h:
            d = datetime.utcfromtimestamp(int(x.get("t", 0)) / 1000).date()
            first_day = ymd(d)
            break

    gap_high_pct = 0.0 if prior70h <= 0 else (week_high / prior70h - 1.0) * 100.0
    gap_close_pct = 0.0 if prior70h <= 0 else (week_close / prior70h - 1.0) * 100.0

    return (prior70h, week_high, week_close, max(gap_high_pct, gap_close_pct), first_day)


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan 70D highs for the week after last Friday.")
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY", ""), help="Polygon API key (or env POLYGON_API_KEY)")
    p.add_argument("--friday", default=os.getenv("FRIDAY", ""), help="Friday anchor date (YYYY-MM-DD). If empty, auto NY Friday.")
    p.add_argument("--since-days", type=int, default=130, help="How many calendar days of bars to request (safe >= 120).")
    p.add_argument("--pages", type=int, default=3, help="How many 1000-ticker pages to fetch (1..N).")
    p.add_argument("--top", type=int, default=10, help="Top N names to keep in the CSV (sorted by market cap desc).")
    p.add_argument("--out-dir", default="backtests", help="Output folder.")
    p.add_argument("--min-price", type=float, default=5.0, help="Minimum close price (last close in week).")
    p.add_argument("--min-dollar-vol", type=float, default=3_000_000.0, help="Min (week last close × latest volume) rough proxy.")
    p.add_argument("--cap-min", type=float, default=0.0, help="Minimum market cap in USD (0 disables).")
    return p.parse_args()


# --------------------------
# Main
# --------------------------

def main() -> None:
    cfg = parse_args()

    api_key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        sys.exit("FATAL: missing Polygon API key (set env POLYGON_API_KEY or use --api-key)")

    # Determine the Friday anchor
    if cfg.friday:
        try:
            fri = datetime.strptime(cfg.friday, "%Y-%m-%d").date()
        except Exception:
            sys.exit(f"FATAL: --friday must be YYYY-MM-DD, got {cfg.friday!r}")
    else:
        fri = last_friday_ny()

    fri_anchor = last_friday_ny()             # Monday after last Friday
    mon = fri_anchor + timedelta(days=3)
    week_end = mon + timedelta(days=4)          # the following Friday (Mon..Fri)
    today_ny = datetime.now(NY).date()
    if week_end > today_ny:
        week_end = today_ny                     # don’t ask for future bars
    start = mon - timedelta(days=cfg.since_days)
    end = week_end

    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"Week window: {ymd(mon)} .. {ymd(fri)} | bars from: {ymd(start)} .. {ymd(fri)}")
    print(f"Fetching tickers (pages={cfg.pages})…")

    tickers = list_common_stocks(api_key, pages=cfg.pages, on_date=fri)
    print(f"Fetched {len(tickers)} active common stocks from Polygon (pages={cfg.pages}).")

    # Build a compact list: symbol, name, market_cap
    refs: List[Tuple[str, str, float]] = []
    for t in tickers:
        sym = str(t.get("ticker") or "").upper().strip()
        name = str(t.get("name") or "").strip()
        mc = float(t.get("market_cap") or 0.0)
        if not sym:
            continue
        refs.append((sym, name, mc))

    # Scan
    cands: List[Candidate] = []
    total = len(refs)
    done = 0
    kept = 0

    for sym, name, mc in refs:
        done += 1
        if done % 200 == 0:
            print(f"… scanned {done}/{total}; candidates so far = {kept}")

        # Fetch bars
        try:
            bars = fetch_daily_bars(api_key, sym, start, end)
        except Exception as e:
            # Skip if API error
            continue

        if not bars:
            continue

        # Quick rolling filters: last close & rough liquidity
        last_close = float(bars[-1].get("c", 0.0) or 0.0)
        latest_vol = float(bars[-1].get("v", 0.0) or 0.0)
        dollar_vol = last_close * latest_vol

        if last_close < float(cfg.min_price or 0.0):
            continue
        if dollar_vol < float(cfg.min_dollar_vol or 0.0):
            continue
        if float(cfg.cap_min or 0.0) > 0 and float(mc or 0.0) < float(cfg.cap_min):
            continue

        prior70h, week_high, week_close, gap_pct, first_day = detect_week_breakout(bars, mon, week_end)
        if prior70h <= 0:
            continue

        broke = (week_high >= prior70h) or (week_close >= prior70h)
        if not broke:
            continue

        gap_high_pct = 0.0 if prior70h <= 0 else (week_high / prior70h - 1.0) * 100.0
        gap_close_pct = 0.0 if prior70h <= 0 else (week_close / prior70h - 1.0) * 100.0

        cands.append(
            Candidate(
                symbol=sym,
                name=name,
                market_cap=mc,
                prior70h=prior70h,
                week_high=week_high,
                week_close=week_close,
                gap_high_pct=gap_high_pct,
                gap_close_pct=gap_close_pct,
                first_break_day=first_day,
                bars=len(bars),
            )
        )
        kept += 1

    # Sort & keep top N by market cap (desc)
    cands.sort(key=lambda x: (x.market_cap, x.gap_high_pct, x.gap_close_pct), reverse=True)
    topN = cands[: int(cfg.top or 10)]

    # Write CSV
    csv_path = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "symbol",
                "name",
                "market_cap",
                "prior70h",
                "week_high",
                "week_close",
                "gap_high_pct",
                "gap_close_pct",
                "first_break_day",
                "bars",
            ]
        )
        for c in topN:
            w.writerow(
                [
                    c.symbol,
                    c.name,
                    f"{c.market_cap:.0f}",
                    f"{c.prior70h:.2f}",
                    f"{c.week_high:.2f}",
                    f"{c.week_close:.2f}",
                    f"{c.gap_high_pct:.2f}",
                    f"{c.gap_close_pct:.2f}",
                    c.first_break_day,
                    c.bars,
                ]
            )

    # Digest TXT
    digest_path = os.path.join(cfg.out_dir, "hi70_digest.txt")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(f"hi70 scan — week {ymd(mon)} .. {ymd(fri)}\n")
        f.write(f"symbols scanned: {total}\n")
        f.write(f"candidates found: {len(cands)}\n")
        f.write(f"top {min(len(topN), int(cfg.top or 10))} by market cap:\n\n")
        for i, c in enumerate(topN, 1):
            f.write(
                f"{i:>2}. {c.symbol:<6} cap={c.market_cap:,.0f} "
                f"prior70H={c.prior70h:.2f} high={c.week_high:.2f} "
                f"close={c.week_close:.2f} gap(max)={max(c.gap_high_pct, c.gap_close_pct):.2f}% "
                f"first={c.first_break_day or '-'}\n"
            )

    print(f"Candidates that made a 70D high this week: {len(cands)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {digest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
