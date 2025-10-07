#!/usr/bin/env python3
from __future__ import annotations
import os, sys, csv, time, argparse, requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
BASE = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{}"

def ymd(d: date) -> str: return d.isoformat()
def parse_date(s: str) -> date: return date.fromisoformat(s)

def http_json(url: str, params: Dict[str,str], tries: int = 6) -> Dict:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(min(0.5*(2**k), 6.0)); continue
            r.raise_for_status()
            return r.json() or {}
        except Exception as e:
            last = e
            time.sleep(min(0.5*(2**k), 6.0))
    raise RuntimeError(f"GET {url} failed: {last}")

def grouped_close_map(apikey: str, d: date) -> Dict[str, float]:
    """Return {SYMBOL: adjusted close} for date d (empty if market closed)."""
    j = http_json(BASE.format(ymd(d)), {"adjusted":"true","apiKey":apikey})
    out: Dict[str, float] = {}
    for r in j.get("results", []) or []:
        t = str(r.get("T") or "").upper()
        c = float(r.get("c", 0.0) or 0.0)
        if t:
            out[t] = c
    return out

def weekly_close(apikey: str, fri: date, symbol: str) -> Optional[float]:
    """Last available close in that Mon..Fri week (Fri→Thu→…→Mon)."""
    s = symbol.strip().upper()
    for dd in range(0,5):
        mp = grouped_close_map(apikey, fri - timedelta(days=dd))
        if s in mp:
            return mp[s]
    return None

def detect_symbol_col(header: List[str]) -> int:
    cand = ["symbol", "ticker", "sym", "ticker_symbol"]
    lower = [h.strip().lower() for h in header]
    for k in cand:
        if k in lower: return lower.index(k)
    return 0

def main():
    p = argparse.ArgumentParser(description="Light trend veto: disallow ≥2 consecutive down weeks.")
    p.add_argument("--picklist", required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--anchor-friday", required=True, help="YYYY-MM-DD of the anchor Friday")
    p.add_argument("--weeks", type=int, default=3, help="How many recent Fridays to consider (>=3)")
    p.add_argument("--max-consecutive-down", type=int, default=1, help="Allow at most this many down weeks in a row")
    p.add_argument("--down-eps", type=float, default=0.002, help="Noise band: drop counted only if c_t < c_{t+1}*(1-eps)")
    args = p.parse_args()

    key = (os.getenv("POLYGON_API_KEY") or "").strip()
    if not key:
        sys.exit("FATAL: POLYGON_API_KEY missing")

    if args.weeks < 3:
        sys.exit("--weeks must be >= 3")

    fri0 = parse_date(args.anchor_friday)
    fridays = [fri0 - timedelta(days=7*i) for i in range(args.weeks)]
    print(f"[trend] fridays considered (most recent first): {', '.join(ymd(x) for x in fridays)}")

    # Load picklist
    with open(args.picklist, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("[trend] empty picklist")
        open(args.out,"w",encoding="utf-8").write("")
        return

    header, body = rows[0], rows[1:]
    sym_col = detect_symbol_col(header)
    kept, dropped = [], 0

    for r in body:
        if not r or sym_col >= len(r): continue
        sym = r[sym_col].strip().upper()

        closes: List[Optional[float]] = []
        for fri in fridays:
            closes.append(weekly_close(key, fri, sym))

        # if insufficient data, fail-open (keep)
        if any(c is None for c in closes[:3]):  # need at least 3 to test 2 consecutive drops
            kept.append(r); continue

        # count consecutive "down" weeks with epsilon
        eps = args.down_eps
        consec, max_consec = 0, 0
        for i in range(len(closes)-1):
            c_now, c_prev = closes[i], closes[i+1]
            if c_now is None or c_prev is None:
                break
            is_down = (c_now < c_prev * (1.0 - eps))
            consec = consec + 1 if is_down else 0
            max_consec = max(max_consec, consec)

        if max_consec > args.max_consecutive_down:
            dropped += 1
        else:
            kept.append(r)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(kept)

    print(f"[trend] kept {len(kept)} / {len(body)} | dropped(two+ down): {dropped} | eps={args.down_eps} weeks={args.weeks} maxConsec={args.max_consecutive_down}")

if __name__ == "__main__":
    main()
