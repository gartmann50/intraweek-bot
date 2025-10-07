#!/usr/bin/env python3
"""
70-day Highs scanner.

Outputs:
  backtests/hi70_thisweek.csv
  backtests/hi70_digest.txt

Week window (default):
  Monday..Friday of the week AFTER the most recent Friday, but never in the future.

Debug overrides:
  --use-current-week           Scan Monday..today of *this* week (capped at Fri).
  --anchor-friday YYYY-MM-DD   Use a specific anchor Friday (scan the following week).

Universe strategy (fast + unbiased):
  1) Pull ALL active common stocks (no alpha bias).
  2) Pull grouped-daily for 'week_end' to get close & volume.
  3) Keep Top-N by dollar volume (close*volume), default 1000 (--topvol).
  4) Apply min price / min dollar volume / optional market-cap lower bound (only if cap present).
  5) Scan those names for 70D breakouts over the chosen week window.

Requires POLYGON_API_KEY (env or --api-key).
"""

from __future__ import annotations
import argparse, csv, os, sys, time, requests
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from collections import Counter

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
POLY_TICKERS = "https://api.polygon.io/v3/reference/tickers"
POLY_AGGS    = "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"
POLY_GROUPED = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d}"

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
    time.sleep(min(0.5 * (2**k), 6.0))

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
def list_common_stocks_all(api_key: str, on_date: Optional[date]) -> List[Dict]:
    """Return ALL pages of active common stocks (no alpha bias)."""
    params = {"market":"stocks","active":"true","type":"CS","limit":"1000","apiKey":api_key}
    if on_date: params["date"] = ymd(on_date)
    results: List[Dict] = []
    cursor: Optional[str] = None
    while True:
        p = dict(params)
        if cursor: p["cursor"] = cursor
        j = http_get_json(POLY_TICKERS, p)
        results.extend(j.get("results", []) or [])
        nxt = j.get("next_url") or ""
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0] if nxt else None
        if not cursor: break
    return results

def fetch_grouped_daily(api_key: str, d: date) -> Dict[str, Tuple[float, float, float]]:
    """Return {symbol: (close, volume, dollar_vol)} for a single day."""
    url = POLY_GROUPED.format(d=ymd(d))
    j = http_get_json(url, {"adjusted":"true","apiKey":api_key})
    data: Dict[str, Tuple[float, float, float]] = {}
    for r in j.get("results", []) or []:
        sym = str(r.get("T") or "").upper()
        c   = float(r.get("c", 0.0) or 0.0)
        v   = float(r.get("v", 0.0) or 0.0)
        if sym:
            data[sym] = (c, v, c * v)
    return data

def fetch_daily_bars(api_key: str, symbol: str, start: date, end: date) -> List[Dict]:
    url = POLY_AGGS.format(sym=symbol, start=ymd(start), end=ymd(end))
    j = http_get_json(url, {"adjusted":"true","sort":"asc","limit":"50000","apiKey":api_key})
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
        if d < mon: before.append(b)
        elif mon <= d <= week_end: during.append(b)
    if not before or not during: return (0.0, 0.0, 0.0, 0.0, "")
    prev = before[-70:] if len(before) >= 70 else before
    if not prev: return (0.0, 0.0, 0.0, 0.0, "")
    prior70h  = max(float(x.get("h", 0) or 0) for x in prev)
    week_high = max(float(x.get("h", 0) or 0) for x in during)
    week_close= float(during[-1].get("c", 0) or 0)
    first_day = ""
    for x in during:
        h = float(x.get("h", 0) or 0); c = float(x.get("c", 0) or 0)
        if h >= prior70h or c >= prior70h:
            first_day = ymd(datetime.utcfromtimestamp(int(x.get("t",0))/1000).date()); break
    gap_high_pct  = 0.0 if prior70h<=0 else (week_high/prior70h - 1.0)*100.0
    gap_close_pct = 0.0 if prior70h<=0 else (week_close/prior70h - 1.0)*100.0
    return (prior70h, week_high, week_close, max(gap_high_pct, gap_close_pct), first_day)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan 70D highs for a chosen week window.")
    p.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY",""))
    p.add_argument("--since-days", type=int, default=130)
    p.add_argument("--pages", type=int, default=9999)   # kept for compat (unused)
    p.add_argument("--top", type=int, default=10)
    # universe / filters
    p.add_argument("--topvol", type=int, default=1000, help="Keep top N by dollar volume at week_end (0=all).")
    p.add_argument("--cap-min", type=float, default=1_000_000_000.0, help="Min market cap USD (only enforced if cap present).")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=10_000_000.0)
    p.add_argument("--universe-method", default="marketcap")  # compat no-op
    # overrides
    p.add_argument("--anchor-friday", type=str, default="")
    p.add_argument("--use-current-week", action="store_true")
    # output
    p.add_argument("--out-dir", default="backtests")
    return p.parse_args()

# ---------- main ----------
def main() -> None:
    cfg = parse_args()
    key = (cfg.api_key or os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        sys.exit("FATAL: missing Polygon API key (POLYGON_API_KEY).")

    today = ny_today()
    if cfg.anchor_friday:
        fri = date.fromisoformat(cfg.anchor_friday)
        mon = fri + timedelta(days=3)
        week_end = min(mon + timedelta(days=4), today)
    elif cfg.use_current_week:
        mon = today - timedelta(days=today.weekday())
        fri = mon + timedelta(days=4)
        week_end = min(today, fri)
    else:
        fri = last_friday_ny()
        mon = fri + timedelta(days=3)
        if today < mon:
            fri -= timedelta(days=7); mon = fri + timedelta(days=3)
        week_end = min(mon + timedelta(days=4), today)

    start = mon - timedelta(days=cfg.since_days)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Week window: {ymd(mon)} .. {ymd(week_end)} | bars from {ymd(start)} .. {ymd(week_end)}")

    # 1) Full active CS list (latest reference snapshot for better market_cap coverage)
    tickers = list_common_stocks_all(key, on_date=None)
    print(f"Fetched {len(tickers)} active common stocks (all pages).")

    # 2) Liquidity map for week_end
    gd = fetch_grouped_daily(key, week_end)
    print(f"Grouped-daily rows for {ymd(week_end)}: {len(gd)}")

    # 3) Build refs with filters, keep Top-N by dollar volume
    refs: List[Tuple[str, str, float, float]] = []  # (sym, name, mc, dv)
    for t in tickers:
        sym = str(t.get("ticker") or "").upper().strip()
        if not sym: continue
        if (t.get("primary_exchange") or "").upper() == "OTC": continue
        g = gd.get(sym)
        if not g: continue
        close, vol, dv = g
        if close <= 0 or vol <= 0: continue
        if close < cfg.min_price or dv < cfg.min_dollar_vol: continue
        mc = float(t.get("market_cap") or 0.0)
        if cfg.cap_min and mc and mc < cfg.cap_min: continue  # only enforce when cap is present
        name = str(t.get("name") or "").strip()
        refs.append((sym, name, mc, dv))

    if cfg.topvol and len(refs) > cfg.topvol:
        refs.sort(key=lambda x: x[3], reverse=True)  # by $ volume
        refs = refs[: cfg.topvol]
    print(f"Universe after $-vol filters: {len(refs)} symbols (topvol={cfg.topvol}).")

    # 4) Scan
    cands: List[Candidate] = []
    total = len(refs)
    for i, (sym, name, mc, _dv) in enumerate(refs, 1):
        if i % 200 == 0 or i == total:
            print(f"… scanned {i}/{total}; candidates={len(cands)}")
        try:
            bars = fetch_daily_bars(key, sym, start, week_end)
        except Exception:
            continue
        if not bars: continue
        last_close = float(bars[-1].get("c", 0) or 0)
        last_vol   = float(bars[-1].get("v", 0) or 0)
        if last_close < cfg.min_price or (last_close * last_vol) < cfg.min_dollar_vol:
            continue
        prior70h, week_high, week_close, _, first_day = detect_week_breakout(bars, mon, week_end)
        if prior70h <= 0: continue
        if not (week_high >= prior70h or week_close >= prior70h): continue
        gap_high_pct  = (week_high/prior70h - 1.0)*100.0 if prior70h > 0 else 0.0
        gap_close_pct = (week_close/prior70h - 1.0)*100.0 if prior70h > 0 else 0.0
        cands.append(Candidate(sym, name, mc, prior70h, week_high, week_close,
                               gap_high_pct, gap_close_pct, first_day, len(bars)))

    # 5) Sort & outputs
    cands.sort(key=lambda x: (x.market_cap, x.gap_high_pct, x.gap_close_pct), reverse=True)
    topN = cands[: int(cfg.top or 10)]

    letters = Counter([c.symbol[:1] for c in topN if c.symbol])
    if letters:
        print("[diag] first-letter distribution:", dict(sorted(letters.items())))

    csv_path = os.path.join(cfg.out_dir, "hi70_thisweek.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","name","market_cap","prior70h","week_high","week_close",
                    "gap_high_pct","gap_close_pct","first_break_day","bars"])
        for c in topN:
            w.writerow([c.symbol, c.name, f"{c.market_cap:.0f}", f"{c.prior70h:.2f}",
                        f"{c.week_high:.2f}", f"{c.week_close:.2f}", f"{c.gap_high_pct:.2f}",
                        f"{c.gap_close_pct:.2f}", c.first_break_day, c.bars])

    digest_path = os.path.join(cfg.out_dir, "hi70_digest.txt")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(f"hi70 scan — week {ymd(mon)} .. {ymd(week_end)} (anchor Friday={ymd(fri)})\n")
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

    if not topN:
        print(f"[hi70] No breakouts found for week {ymd(mon)} .. {ymd(week_end)} (universe={total})")
    print(f"Candidates: {len(cands)} | wrote {csv_path} and {digest_path}")

if __name__ == "__main__":
    main()
