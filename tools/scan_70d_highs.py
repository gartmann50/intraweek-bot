#!/usr/bin/env python3
"""
Scan for 70-day highs in the WEEK AFTER the last Friday (Mon..Fri).

Writes:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt

Key robustness:
- Restrict universe by market-cap (default >= $1B) and/or take only the Top N
  by market-cap (default top 1000) to avoid thin names.
- Correct week window: Monday after last Friday → following Friday, capped at today (NY).
- Defensive against API hiccups + rate limits.

CLI:
  --cap-min 1000000000   (default $1B)
  --topcap 1000          (keep only top 1000 by market cap)
  --pages 5              (how many 1000-ticker pages to fetch at most)
  --since-days 130       (history to compute prior 70 bars)
  --top 10               (top N by market cap in the final CSV)
"""

from __future__ import annotations
import argparse, csv, os, sys, time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # py<3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")

# ---------- helpers ----------
def ny_today() -> date:
    return datetime.now(NY).date()

def last_friday_ny() -> date:
    d = ny_today()
    while d.weekday() != 4:  # Friday=4
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

# ---------- Polygon ----------
def list_common_stocks(api_key: str, pages: int, on_date: Optional[date]) -> List[Dict]:
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
    page_n = 0
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        j = http_get_json(base, p)
        results.extend(j.get("results", []) or [])
        page_n += 1
        nxt = j.get("next_url") or ""
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0] if nxt else None
        if not cursor or page_n >= pages:
            break
    return results

def fetch_daily_bars(api_key: str, symbol: str, start: date, end: date) -> List[Dict]:
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{ymd(start)}/{ymd(end)}"
    p = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": api_key}
    j = http_get_json(base, p)
    return j.get("results", []) or []

# ---------- core ----------
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
    if not bars:
        return (0.0, 0.0, 0.0, 0.0, "")

    def as_date(ms: int) -> date:
        return datetime.utcfromtimestamp(ms / 1000).date()

    before, during = [], []
    for b in bars:
        d = as_date(int(b.get("t", 0)))
        if d < mon:
            before.append(b)
        elif mon <= d <= week_end:
            during.append(b)

    if not before or not during:
        return (0.0, 0.0, 0.0, 0.0, "")

    prev = before[-70:] if len(before) >= 70 else before
    if not prev:
        return (0.0, 0.0, 0.0, 0.0, "")

    prior70h = max(float(x.get("h", 0.0) or 0.0) for x in prev)
    week_high = max(float(x.get("h", 0.0) or 0.0) for x in during)
    week_close = float(during[-1].get("c", 0.0) or 0.0)

    first_day = ""
    for x in during:
        h = float(x.get("h", 0.0) or 0.0)
        c = float(x.get("c", 0.0) or 0.0)
        if h >= prior70h or c >= prior70h:
            first_day = ymd(datetime.utcfromtimestamp(int(x.get("t", 0)) / 1000).date())
            break

    gap_high_pct = 0.0 if prior70h <= 0 else (week_high / prior70h - 1.0) * 100.0
    gap_close_pct = 0.0 if prior70h <= 0 else (week_close / prior70h - 1.0) * 100.0
    return (prior70h, week_high, week_close, max(gap_high_pct, gap_close_pct), first_day)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan 70D highs for the week after last Friday (Mon..Fri).")
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY", ""))
    p.add_argument("--since-days", type=int, default=130)
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--topcap", type=int, default=1000, help="Keep only top N by market cap before scanning (0=disable).")
    p.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Min market cap in USD (0=disable).")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=3_000_000.0)
    p.add_argument("--out-dir", default="backtests")
    return p.parse_args()

def main() -> None:
    cfg = parse_args()
    key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        sys.exit("FATAL: missing Polygon API key (POLYGON_API_KEY).")

    fri = last_friday_ny()
    mon = fri + timedelta(days=3)
    today = ny_today()
    # if we're already in or before that week, push back one week
    if today < mon:
        fri -= timedelta(days=7)
        mon = fri + timedelta(days=3)
    week_end = min(mon + timedelta(days=4), today)


    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Week window: {ymd(mon)} .. {ymd(week_end)} | bars from {ymd(start)} .. {ymd(week_end)}")

    tickers = list_common_stocks(key, pages=cfg.pages, on_date=fri)
    print(f"Fetched {len(tickers)} active common stocks (pages={cfg.pages}).")

    # compact refs + basic filters
    refs: List[Tuple[str, str, float]] = []
    for t in tickers:
        sym = str(t.get("ticker") or "").upper().strip()
        if not sym:
            continue
        # Weed out OTC explicitly if present
        ex = (t.get("primary_exchange") or "").upper()
        if ex == "OTC":
            continue
        mc = float(t.get("market_cap") or 0.0)
        if cfg.cap_min and mc < cfg.cap_min:
            continue
        name = str(t.get("name") or "").strip()
        refs.append((sym, name, mc))

    # keep only largest by market-cap, if requested
    if cfg.topcap and len(refs) > cfg.topcap:
        refs.sort(key=lambda x: x[2], reverse=True)
        refs = refs[: cfg.topcap]
    print(f"Universe after filters: {len(refs)} symbols.")

    # scan
    cands: List[Candidate] = []
    total = len(refs)
    for i, (sym, name, mc) in enumerate(refs, 1):
        if i % 200 == 0 or i == total:
            print(f"… scanned {i}/{total}; candidates={len(cands)}")
        try:
            bars = fetch_daily_bars(key, sym, start, week_end)
        except Exception:
            continue
        if not bars:
            continue

        last_close = float(bars[-1].get("c", 0.0) or 0.0)
        last_vol   = float(bars[-1].get("v", 0.0) or 0.0)
        if last_close < cfg.min_price or (last_close * last_vol) < cfg.min_dollar_vol:
            continue

        prior70h, week_high, week_close, _, first_day = detect_week_breakout(bars, mon, week_end)
        if prior70h <= 0:
            continue
        if not (week_high >= prior70h or week_close >= prior70h):
            continue

        gap_high_pct = (week_high / prior70h - 1.0) * 100.0 if prior70h > 0 else 0.0
        gap_close_pct = (week_close / prior70h - 1.0) * 100.0 if prior70h > 0 else 0.0

        cands.append(Candidate(
            symbol=sym, name=name, market_cap=mc,
            prior70h=prior70h, week_high=week_high, week_close=week_close,
            gap_high_pct=gap_high_pct, gap_close_pct=gap_close_pct,
            first_break_day=first_day, bars=len(bars)
        ))

    cands.sort(key=lambda x: (x.market_cap, x.gap_high_pct, x.gap_close_pct), reverse=True)
    topN = cands[: int(cfg.top or 10)]

    # CSV
    csv_path = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","market_cap","prior70h","week_high","week_close",
                    "gap_high_pct","gap_close_pct","first_break_day","bars"])
        for c in topN:
            w.writerow([c.symbol, c.name, f"{c.market_cap:.0f}", f"{c.prior70h:.2f}",
                        f"{c.week_high:.2f}", f"{c.week_close:.2f}", f"{c.gap_high_pct:.2f}",
                        f"{c.gap_close_pct:.2f}", c.first_break_day, c.bars])

    # TXT digest
    digest_path = os.path.join(cfg.out_dir, "hi70_digest.txt")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(f"hi70 scan — week {ymd(mon)} .. {ymd(week_end)}\n")
        f.write(f"universe scanned: {total}\n")
        f.write(f"candidates found: {len(cands)}\n")
        f.write(f"top {min(len(topN), int(cfg.top or 10))} by market cap:\n\n")
        for i, c in enumerate(topN, 1):
            f.write(
                f"{i:>2}. {c.symbol:<6} cap={c.market_cap:,.0f} "
                f"prior70H={c.prior70h:.2f} high={c.week_high:.2f} "
                f"close={c.week_close:.2f} gap(max)={max(c.gap_high_pct, c.gap_close_pct):.2f}% "
                f"first={c.first_break_day or '-'}\n"
            )

    print(f"Candidates: {len(cands)} | wrote {csv_path} and {digest_path}")

if __name__ == "__main__":
    main()
