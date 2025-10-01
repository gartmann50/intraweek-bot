#!/usr/bin/env python3
"""
Scan 70-day highs within the most-recent trading week and write:
  <out-dir>/hi70_thisweek.csv  (always created, even if empty)
  <out-dir>/hi70_digest.txt    (one-line summary)

Selection:
  - US stocks, type=CS, active, market_cap >= cap_min (default $1B)
  - A ticker is a "hit" if the most-recent 70-day high date
    (up to and including Friday) falls within [Monday..Friday].

Environment / CLI:
  POLYGON_API_KEY (or POLY_KEY)
  --friday YYYY-MM-DD      optional; overrides env FRIDAY
  --out-dir PATH           output dir (default: backtests)
"""

from __future__ import annotations

import os
import time
import csv
import sys
import math
import argparse
import requests
from dataclasses import dataclass
from pathlib import Path
from datetime import date, datetime, timedelta

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

API_TICKERS = "https://api.polygon.io/v3/reference/tickers"
API_AGGS    = "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"

@dataclass
class SymbolInfo:
    ticker: str
    name: str
    market_cap: float

def log(*a): print(*a, flush=True)

def get_env_api_key() -> str:
    key = os.getenv("POLYGON_API_KEY") or os.getenv("POLY_KEY") or os.getenv("POLYGON_KEY")
    if not key:
        sys.exit("ERROR: Missing POLYGON_API_KEY / POLY_KEY in env.")
    return key

def last_friday_europe_oslo() -> date:
    if ZoneInfo:
        tz = ZoneInfo("Europe/Oslo")
        now = datetime.now(tz).date()
    else:
        now = datetime.utcnow().date()
    d = now
    while d.weekday() != 4:  # 0=Mon ... 4=Fri
        d -= timedelta(days=1)
    return d

def week_bounds(friday: date) -> tuple[date, date]:
    monday = friday - timedelta(days=4)
    return monday, friday

def http_get_json(url: str, params: dict, max_tries: int = 6, backoff: float = 0.6):
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (2 ** k)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if k == max_tries - 1: raise
            time.sleep(backoff * (2 ** k))
    raise RuntimeError("http_get_json: exhausted retries")

def get_all_symbols_by_cap(api_key: str, cap_min: float, pages: int) -> list[SymbolInfo]:
    out: list[SymbolInfo] = []
    url = API_TICKERS
    params = {
        "market":"stocks", "type":"CS", "active":"true",
        "sort":"market_cap", "order":"desc", "limit":1000,
        "apiKey": api_key,
    }
    for _ in range(pages):
        j = http_get_json(url, params)
        results = (j or {}).get("results") or []
        for rec in results:
            mc = float(rec.get("market_cap") or 0.0)
            if mc >= cap_min:
                out.append(SymbolInfo(
                    ticker=str(rec.get("ticker") or "").upper(),
                    name=str(rec.get("name") or ""),
                    market_cap=mc,
                ))
        next_url = j.get("next_url")
        if not next_url: break
        url, params = next_url, {}  # next_url contains query (incl apiKey)
    log(f"Symbols meeting cap_min={int(cap_min):,}: {len(out)}")
    return out

def get_daily_bars(api_key: str, sym: str, start: date, end: date) -> list[dict]:
    url = API_AGGS.format(sym=sym, start=start.isoformat(), end=end.isoformat())
    params = {"adjusted":"true","limit":50000,"sort":"asc","apiKey":api_key}
    j = http_get_json(url, params)
    results = (j or {}).get("results") or []
    out = []
    for r in results:
        d = datetime.utcfromtimestamp((r.get("t") or 0)/1000.0).date()
        out.append({"d": d, "h": float(r.get("h") or 0.0)})
    return out

def most_recent_70d_high_in_week(bars: list[dict], week_mon: date, week_fri: date, window: int) -> date|None:
    if not bars: return None
    up_to_fri = [b for b in bars if b["d"] <= week_fri]
    if len(up_to_fri) < 10: return None
    lastN = up_to_fri[-window:] if len(up_to_fri) >= window else up_to_fri
    if not lastN: return None
    max_h = max(b["h"] for b in lastN)
    eps = max(1e-8, 1e-6 * max_h)
    candidates = [b["d"] for b in lastN if (max_h - b["h"]) <= eps]
    if not candidates: return None
    hit_date = max(candidates)
    return hit_date if week_mon <= hit_date <= week_fri else None

def parse_friday(cli_friday: str|None) -> date:
    if cli_friday:
        try:
            return date.fromisoformat(cli_friday)
        except Exception:
            log(f"WARNING: --friday '{cli_friday}' not ISO date; falling back.")
    env_fri = (os.getenv("FRIDAY") or "").strip()
    if env_fri:
        try:
            return date.fromisoformat(env_fri)
        except Exception:
            log(f"WARNING: FRIDAY env '{env_fri}' not ISO date; falling back.")
    return last_friday_europe_oslo()

def main():
    ap = argparse.ArgumentParser(description="Scan 70-day highs in the most recent week.")
    ap.add_argument("--cap-min", type=float, default=1e9, help="Market cap threshold (USD). Default 1e9.")
    ap.add_argument("--top", type=int, default=10, help="Max number of results to keep (sorted by market cap).")
    ap.add_argument("--pages", type=int, default=3, help="How many Polygon tickers pages to scan (1000 per page).")
    ap.add_argument("--max-syms", type=int, default=800, help="Stop after this many symbols (perf guard).")
    ap.add_argument("--window", type=int, default=70, help="Lookback window (trading days).")
    ap.add_argument("--since-days", type=int, default=140, help="Calendar days of bars to pull.")
    ap.add_argument("--friday", type=str, default=None, help="Override Friday YYYY-MM-DD (CLI wins over env).")
    ap.add_argument("--out-dir", type=str, default="backtests", help="Output directory (default: backtests).")
    args = ap.parse_args()

    api_key = get_env_api_key()

    fri = parse_friday(args.friday)
    week_mon, week_fri = week_bounds(fri)
    log(f"Week window: {week_mon} .. {week_fri}")

    syms = get_all_symbols_by_cap(api_key, args.cap_min, args.pages)
    if args.max_syms and len(syms) > args.max_syms:
        syms = syms[:args.max_syms]
        log(f"Truncated to top {args.max_syms} symbols by market cap for performance.")

    since = week_mon - timedelta(days=args.since_days)
    hits = []

    total = len(syms)
    for i, info in enumerate(syms, 1):
        if i % 50 == 0 or i == total:
            log(f"{i}/{total} {info.ticker} …")
        try:
            bars = get_daily_bars(api_key, info.ticker, since, week_fri)
        except Exception:
            continue
        hitd = most_recent_70d_high_in_week(bars, week_mon, week_fri, args.window)
        if hitd:
            hits.append({
                "ticker": info.ticker,
                "name": info.name,
                "market_cap": info.market_cap,
                "hit_date": hitd,
            })

    hits.sort(key=lambda r: (r["market_cap"], r["ticker"]), reverse=True)
    top = hits[: max(1, args.top)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hi70_thisweek.csv"
    digest_path = out_dir / "hi70_digest.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["week_start", "ticker", "name", "market_cap"])
        for r in top:
            w.writerow([week_mon.isoformat(), r["ticker"], r["name"], int(r["market_cap"] or 0)])

    summary = (
        f"Week {week_mon.isoformat()}..{week_fri.isoformat()}: "
        f"{len(hits)} total hits. Top-{len(top)} by mcap: "
        f"{', '.join([r['ticker'] for r in top])}\n"
    )
    digest_path.write_text(summary, encoding="utf-8")

    log("Wrote:", csv_path, "and", digest_path)
    log(summary.strip())

if __name__ == "__main__":
    main()
