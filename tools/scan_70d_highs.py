#!/usr/bin/env python3
"""
Scan for 70-day highs in the week AFTER the last Friday, with a controllable universe:
- Pre-limit to top-N by market cap (fast) OR (optional) keep all, but still apply cap/dollar-vol floors.
- Filter to primary exchanges (default: XNYS, XNAS, ARCX).

Outputs:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt

Notes:
- API key from env POLYGON_API_KEY (or --api-key)
- last Friday inferred from America/New_York (or --friday YYYY-MM-DD)
- tickers from /v3/reference/tickers (active, type=CS)
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

NY = ZoneInfo("America/New_York")


# -------------------------- Utilities --------------------------

def last_friday_ny() -> date:
    now_ny = datetime.now(NY).date()
    d = now_ny
    while d.weekday() != 4:  # 4 = Friday
        d -= timedelta(days=1)
    return d

def ymd(d: date) -> str:
    return d.isoformat()

def backoff_sleep(k: int) -> None:
    time.sleep(min(0.5 * (2 ** k), 6.0))

def http_get_json(url: str, params: Dict[str, str], max_tries: int = 6) -> Dict:
    last = None
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                backoff_sleep(k); continue
            r.raise_for_status()
            return r.json() or {}
        except requests.HTTPError as e:
            last = e
            if r.status_code >= 500:
                backoff_sleep(k); continue
            break
        except Exception as e:
            last = e
            backoff_sleep(k)
    raise RuntimeError(f"GET failed {url} ({last})")


# -------------------------- Polygon adapters --------------------------

def list_common_stocks(api_key: str, pages: int, on_date: Optional[date]) -> List[Dict]:
    """
    Return up to pages*1000 US common stocks (active).
    If on_date is provided, Polygon attaches 'market_cap' for that date.
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
    cursor: Optional[str] = None
    fetched_pages = 0

    while True:
        p = params.copy()
        if cursor:
            p["cursor"] = cursor
        j = http_get_json(base, p)
        results.extend(j.get("results", []) or [])
        fetched_pages += 1
        nxt = j.get("next_url") or ""
        if nxt:
            q = parse_qs(urlparse(nxt).query)
            cursor = q.get("cursor", [None])[0]
        else:
            cursor = None
        if not cursor or fetched_pages >= pages:
            break
    return results


def fetch_daily_bars(api_key: str, symbol: str, start: date, end: date) -> List[Dict]:
    """Fetch daily bars (adjusted) for symbol within [start, end]."""
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{ymd(start)}/{ymd(end)}"
    p = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": api_key}
    j = http_get_json(base, p)
    return j.get("results", []) or []


# -------------------------- Core logic --------------------------

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
    first_break_day: str
    bars: int

def detect_week_breakout(bars: List[Dict], mon: date, week_end: date) -> Tuple[float, float, float, float, str]:
    """
    From ascending daily bars, compute:
      prior70h = max high of the previous 70 trading bars before Monday
      week_high, week_close in Mon..Fri
      gap% vs prior70h, and first day that broke.
    """
    if not bars:
        return (0.0, 0.0, 0.0, 0.0, "")

    def as_date(ms: int) -> date:
        return datetime.utcfromtimestamp(ms / 1000).date()

    before: List[Dict] = []
    this_week: List[Dict] = []

    for b in bars:
        d = as_date(int(b.get("t", 0)))
        if d < mon:
            before.append(b)
        elif mon <= d <= week_end:
            this_week.append(b)

    if not before or not this_week:
        return (0.0, 0.0, 0.0, 0.0, "")

    prev = before[-70:] if len(before) >= 70 else before
    if not prev:
        return (0.0, 0.0, 0.0, 0.0, "")

    prior70h = max(float(x.get("h", 0.0) or 0.0) for x in prev)
    week_high = max(float(x.get("h", 0.0) or 0.0) for x in this_week)
    week_close = float(this_week[-1].get("c", 0.0) or 0.0)

    first_day = ""
    for x in this_week:
        h = float(x.get("h", 0.0) or 0.0)
        c = float(x.get("c", 0.0) or 0.0)
        if h >= prior70h or c >= prior70h:
            d = datetime.utcfromtimestamp(int(x.get("t", 0)) / 1000).date()
            first_day = ymd(d)
            break

    gap_high_pct = 0.0 if prior70h <= 0 else (week_high / prior70h - 1.0) * 100.0
    gap_close_pct = 0.0 if prior70h <= 0 else (week_close / prior70h - 1.0) * 100.0

    return (prior70h, week_high, week_close, max(gap_high_pct, gap_close_pct), first_day)


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan 70D highs for the week after last Friday.")
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY", ""), help="Polygon API key")
    p.add_argument("--friday", default=os.getenv("FRIDAY", ""), help="Anchor Friday YYYY-MM-DD (default=auto NY Friday)")
    p.add_argument("--pages", type=int, default=5, help="How many 1000-ticker pages to fetch from Polygon")
    p.add_argument("--universe-method", choices=["marketcap","all"], default="marketcap",
                   help="Pre-limit universe by: marketcap (fast) or all (no pre-limit)")
    p.add_argument("--universe-limit", type=int, default=1000, help="Keep at most this many symbols (pre-limit)")
    p.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Min market cap USD (0 disables)")
    p.add_argument("--min-price", type=float, default=5.0, help="Min last close price")
    p.add_argument("--min-dollar-vol", type=float, default=10_000_000.0,
                   help="Min (last close × last volume) rough proxy")
    p.add_argument("--exchanges", default="XNYS,XNAS,ARCX",
                   help="Comma list of primary_exchange codes to keep (e.g. XNYS,XNAS,ARCX). Empty=keep all")
    p.add_argument("--since-days", type=int, default=130, help="Calendar days of bars to request")
    p.add_argument("--top", type=int, default=10, help="Top N names (by market cap) to keep in CSV")
    p.add_argument("--out-dir", default="backtests", help="Output folder")
    return p.parse_args()


# -------------------------- Main --------------------------

def main() -> None:
    cfg = parse_args()

    api_key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        sys.exit("FATAL: missing Polygon API key (POLYGON_API_KEY)")

    # Friday anchor → week window (Mon..Fri)
    if cfg.friday:
        try:
            fri = datetime.strptime(cfg.friday, "%Y-%m-%d").date()
        except Exception:
            sys.exit(f"FATAL: --friday must be YYYY-MM-DD, got {cfg.friday!r}")
    else:
        fri = last_friday_ny()

    mon = fri + timedelta(days=3)
    week_end = mon + timedelta(days=4)
    today_ny = datetime.now(NY).date()
    if week_end > today_ny:
        week_end = today_ny

    start = mon - timedelta(days=cfg.since_days)
    end = week_end

    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"Week window: {ymd(mon)} .. {ymd(week_end)} (anchor Friday={ymd(fri)})")
    print(f"Fetching tickers (pages={cfg.pages})…")

    tickers = list_common_stocks(api_key, pages=cfg.pages, on_date=fri)
    print(f"Reference universe fetched: {len(tickers)}")

    # Build refs: (sym, name, market_cap, exchange)
    exch_allowed = {e.strip().upper() for e in str(cfg.exchanges or "").split(",") if e.strip()}
    refs: List[Tuple[str, str, float, str]] = []
    for t in tickers:
        sym = str(t.get("ticker") or "").upper().strip()
        if not sym:
            continue
        name = str(t.get("name") or "").strip()
        mc = float(t.get("market_cap") or 0.0)
        ex = str(t.get("primary_exchange") or "").upper().strip()
        if exch_allowed and ex not in exch_allowed:
            continue
        refs.append((sym, name, mc, ex))

    print(f"After exchange filter ({','.join(sorted(exch_allowed)) or 'ALL'}): {len(refs)}")

    # Pre-limit: by market cap (fast)
    if cfg.universe_method == "marketcap":
        refs.sort(key=lambda r: r[2], reverse=True)  # by market cap desc
        if cfg.universe_limit > 0 and len(refs) > cfg.universe_limit:
            refs = refs[:cfg.universe_limit]
        print(f"Pre-limited to top {len(refs)} by market cap.")

    total_scan = len(refs)
    print(f"Scanning {total_scan} symbols for 70D breakouts…")

    cands: List[Candidate] = []
    kept = 0

    for i, (sym, name, mc, ex) in enumerate(refs, 1):
        if i % 200 == 0:
            print(f"… {i}/{total_scan} scanned; candidates so far = {kept}")

        try:
            bars = fetch_daily_bars(api_key, sym, start, end)
        except Exception:
            continue
        if not bars:
            continue

        # quick liquidity floors (last day proxy)
        last_close = float(bars[-1].get("c", 0.0) or 0.0)
        latest_vol = float(bars[-1].get("v", 0.0) or 0.0)
        dollar_vol = last_close * latest_vol

        if last_close < float(cfg.min_price or 0.0):
            continue
        if dollar_vol < float(cfg.min_dollar_vol or 0.0):
            continue
        if float(cfg.cap_min or 0.0) > 0 and float(mc or 0.0) < float(cfg.cap_min):
            continue

        prior70h, week_high, week_close, _gap_max, first_day = detect_week_breakout(bars, mon, week_end)
        if prior70h <= 0:
            continue

        if (week_high >= prior70h) or (week_close >= prior70h):
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

    # Sort candidates by market cap desc, then gap
    cands.sort(key=lambda x: (x.market_cap, x.gap_high_pct, x.gap_close_pct), reverse=True)
    topN = cands[: int(cfg.top or 10)]

    # Write CSV
    csv_path = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","market_cap","prior70h","week_high","week_close",
                    "gap_high_pct","gap_close_pct","first_break_day","bars"])
        for c in topN:
            w.writerow([c.symbol, c.name, f"{c.market_cap:.0f}",
                        f"{c.prior70h:.2f}", f"{c.week_high:.2f}", f"{c.week_close:.2f}",
                        f"{c.gap_high_pct:.2f}", f"{c.gap_close_pct:.2f}",
                        c.first_break_day, c.bars])

    # Digest TXT
    digest_path = os.path.join(cfg.out_dir, "hi70_digest.txt")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(f"hi70 scan — week {ymd(mon)} .. {ymd(week_end)} (anchor Friday={ymd(fri)})\n")
        f.write(f"symbols scanned: {total_scan}\n")
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
