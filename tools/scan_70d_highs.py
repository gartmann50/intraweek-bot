#!/usr/bin/env python3
# tools/scan_70d_highs.py
import os, sys, time, math, argparse, datetime as dt, requests, pandas as pd
from typing import List, Tuple

API = "https://api.polygon.io"

def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)

def week_bounds(friday: dt.date) -> Tuple[dt.date, dt.date]:
    """Return (mon, fri) for the ISO week that ends on the given Friday."""
    # friday is provided by the workflow; still compute Monday defensively
    mon = friday - dt.timedelta(days=4)
    return mon, friday

# ---------- HTTP helpers ----------

def http_get(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json() or {}

def list_common_stocks(api_key: str,
                       pages: int = 5,
                       max_syms: int = 5000) -> List[dict]:
    """
    Robust pager for /v3/reference/tickers.
    We *always* include the apiKey, including when following next_url.
    """
    url = f"{API}/v3/reference/tickers"
    params = {
        "active": "true",
        "asset_class": "stocks",     # NOT 'market=stocks'
        "type": "CS",                # common stock
        "sort": "ticker",
        "order": "asc",
        "limit": 1000,
        "apiKey": api_key
    }
    out: List[dict] = []
    page = 0
    while True:
        js = http_get(url, params=params)
        res = js.get("results") or []
        out.extend(res)
        page += 1
        next_url = js.get("next_url")
        if not next_url or page >= pages or len(out) >= max_syms:
            break
        # When following next_url, pass apiKey as a query parameter explicitly.
        url = f"{next_url}&apiKey={api_key}"
        params = None
    return out

def agg_bars(session: requests.Session,
             api_key: str,
             symbol: str,
             start: dt.date,
             end: dt.date) -> List[dict]:
    url = f"{API}/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "limit": 50000, "sort": "asc", "apiKey": api_key}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("results") or []

# ---------- Core logic ----------

def find_70d_high_this_week(bars: List[dict], week_mon: dt.date, week_fri: dt.date) -> bool:
    if not bars:
        return False
    df = pd.DataFrame(bars)
    # Require the typical Polygon fields
    if not {"t", "h"}.issubset(df.columns):
        return False
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df["high"] = pd.to_numeric(df["h"], errors="coerce")
    df.sort_values("date", inplace=True)

    # Rolling 70-day high on 'high'
    df["roll_max_70"] = df["high"].rolling(70, min_periods=70).max()
    wk = df[(df["date"] >= week_mon) & (df["date"] <= week_fri)].copy()
    if wk.empty:
        return False
    eps = 1e-8
    return bool((wk["high"] >= wk["roll_max_70"] - eps).any())

def main():
    ap = argparse.ArgumentParser(description="Scan for 70-day highs occurring within the current week.")
    ap.add_argument("--cap-min", type=float, default=1e9, help="Minimum market cap (USD). Default 1e9")
    ap.add_argument("--top", type=int, default=10, help="Return up to this many tickers (default 10)")
    ap.add_argument("--pages", type=int, default=5, help="How many /v3/reference pages (1000 each) to fetch")
    ap.add_argument("--max-syms", type=int, default=5000, help="Upper bound on universe to consider")
    ap.add_argument("--since-days", type=int, default=120, help="Download this many calendar days of bars to compute 70D high (default 120)")
    ap.add_argument("--friday", type=str, required=True, help="Week ending Friday (YYYY-MM-DD)")
    ap.add_argument("--out-dir", type=str, default="backtests", help="Where to write hi70_thisweek.csv and hi70_digest.txt")
    ap.add_argument("--reqs-per-min", type=int, default=5, help="Polygon rate limit to respect (default 5 for free tier)")
    args = ap.parse_args()

    api_key = os.getenv("POLYGON_API_KEY") or os.getenv("POLY_KEY") or ""
    if not api_key:
        print("FATAL: POLYGON_API_KEY not set.")
        sys.exit(1)

    friday = d(args.friday)
    mon, fri = week_bounds(friday)
    # We’ll pull enough history back to compute a 70-day rolling max safely.
    start = mon - dt.timedelta(days=args.since_days)
    print(f"Week window: {mon} .. {fri} | bars from: {start} .. {fri}")

    # 1) Universe
    all_meta = list_common_stocks(api_key, pages=args.pages, max_syms=args.max_syms)
    print(f"Fetched {len(all_meta)} active common stocks from Polygon (pages={args.pages}).")

    # client-side market cap filter
    keep: List[Tuple[str, float]] = []
    for m in all_meta:
        mc = m.get("market_cap")
        sym = (m.get("ticker") or "").upper()
        if not sym or mc is None:
            continue
        try:
            mc = float(mc)
        except Exception:
            continue
        if mc >= args.cap_min:
            keep.append((sym, mc))

    # sort by market cap desc to scan the biggest first
    keep.sort(key=lambda t: -t[1])
    print(f"After cap filter (>= {args.cap_min:,.0f}): {len(keep)} symbols.")
    if not keep:
        print("No symbols after cap filter; aborting.")
        # still write empty artifacts for the email step
        os.makedirs(args.out_dir, exist_ok=True)
        pd.DataFrame(columns=["symbol","week_start","market_cap"]).to_csv(
            os.path.join(args.out_dir, "hi70_thisweek.csv"), index=False
        )
        open(os.path.join(args.out_dir, "hi70_digest.txt"), "w").write("0 hits\n")
        sys.exit(0)

    # 2) Scan bars with polite rate limiting
    session = requests.Session()
    rpm = max(1, args.reqs_per_min)
    sleep_per = 60.0 / rpm
    print(f"Will scan up to {min(len(keep), args.max_syms)} symbols. "
          f"Rate limit ~{rpm}/min ⇒ ~{sleep_per:.1f}s between calls. "
          f"ETA (worst case) ≈ {math.ceil(min(len(keep), args.max_syms) * sleep_per / 60)} min.")

    hits: List[Tuple[str, float]] = []
    calls = 0
    last_call = 0.0

    for sym, mc in keep:
        # rate-limit
        now = time.monotonic()
        wait = sleep_per - (now - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()

        try:
            bars = agg_bars(session, api_key, sym, start, fri)
        except Exception as e:
            print(f"WARN: {sym} bars error: {e}")
            continue
        calls += 1

        if find_70d_high_this_week(bars, mon, fri):
            hits.append((sym, mc))
            print(f"  HIT {len(hits)}: {sym} (mcap {mc:,.0f})")
            if len(hits) >= args.top:
                break

    # 3) Write artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "hi70_thisweek.csv")
    out_txt = os.path.join(args.out_dir, "hi70_digest.txt")

    df = pd.DataFrame(hits, columns=["symbol","market_cap"])
    df.insert(1, "week_start", mon.isoformat())
    df.to_csv(out_csv, index=False)

    with open(out_txt, "w") as f:
        f.write(f"{len(hits)} hits (out of {len(keep)} candidates, {calls} bar calls)\n")
        f.write("\n".join([f"{s}  mcap={mc:,.0f}" for s,mc in hits]))

    print(f"\nDONE. hits={len(hits)}  candidates={len(keep)}  bar_calls={calls}")
    print(f"Wrote {out_csv} and {out_txt}")

if __name__ == "__main__":
    main()
