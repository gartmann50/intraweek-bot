#!/usr/bin/env python3
"""
Scan US common stocks (market cap >= $1B) and find symbols that printed a 70-day
high during THIS WEEK (Mon..Fri) of the given Friday.

Outputs:
  backtests/hi70_thisweek.csv   (full table)
  backtests/hi70_digest.txt     (human-friendly text snippet)

Requirements: requests, pandas
Secrets: POLYGON_API_KEY in env
"""

import os, sys, time, math, argparse, datetime as dt
from typing import List, Dict
import requests
import pandas as pd

POLY = "https://api.polygon.io"

def last_monday_of_week(friday: str) -> dt.date:
    f = dt.date.fromisoformat(friday)
    return f - dt.timedelta(days=4)  # Mon of that week

def iso(s: dt.date) -> str:
    return s.isoformat()

def get_all_symbols_by_cap(
    api_key: str,
    cap_min: float = 1_000_000_000,
    max_pages: int = 3,                # bump if you want more coverage
    limit_per_page: int = 1000
) -> List[Dict]:
    """Return list of dicts: {ticker, market_cap, name} for U.S. common stocks."""
    # polygon v3/reference/tickers supports filter & paging
    url = f"{POLY}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "locale": "us",
        "type": "CS",
        "sort": "market_cap",
        "order": "desc",
        "limit": limit_per_page,
        "apiKey": api_key,
    }
    page = 0
    out: List[Dict] = []
    while url and page < max_pages:
        page += 1
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json() or {}
        results = j.get("results") or []
        for row in results:
            mc = float(row.get("market_cap") or 0.0)
            if mc >= cap_min:
                out.append({
                    "ticker": str(row.get("ticker","")).upper(),
                    "market_cap": mc,
                    "name": row.get("name") or "",
                })
        next_url = j.get("next_url")
        if next_url:
            # next_url does not include apiKey
            url, params = next_url, {"apiKey": api_key}
        else:
            break
        # polite throttle
        time.sleep(0.25)
    return out

def get_daily_bars(api_key: str, sym: str, start: str, end: str) -> pd.DataFrame:
    u = f"{POLY}/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    for k in range(5):
        r = requests.get(u, params=params, timeout=30)
        if r.status_code != 429:
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            res = (r.json() or {}).get("results") or []
            if not res:
                return pd.DataFrame()
            df = pd.DataFrame(res)
            df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
            df.rename(columns={"h":"high","c":"close","v":"volume"}, inplace=True)
            return df[["date","high","close","volume"]]
        time.sleep(0.6 * (2**k))
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friday", required=True, help="Friday YYYY-MM-DD to define the week")
    ap.add_argument("--cap-min", type=float, default=1_000_000_000, help="Market cap floor (USD)")
    ap.add_argument("--lookback", type=int, default=70, help="Rolling high window (days)")
    ap.add_argument("--pages", type=int, default=3, help="How many 1000-ticker pages to scan")
    ap.add_argument("--since-buffer-days", type=int, default=140, help="Aggs lookback (calendar days)")
    ap.add_argument("--out-dir", default="backtests")
    args = ap.parse_args()

    key = os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY_ID") or os.getenv("POLYGON_KEY")
    if not key:
        sys.exit("ERROR: set POLYGON_API_KEY in Secrets.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "hi70_thisweek.csv")
    digest_path = os.path.join(out_dir, "hi70_digest.txt")

    fri = dt.date.fromisoformat(args.friday)
    mon = last_monday_of_week(args.friday)

    # Build symbol universe by cap
    syms = get_all_symbols_by_cap(key, cap_min=args.cap_min, max_pages=args.pages)
    if not syms:
        sys.exit("No symbols from reference API.")

    # Download a few months of daily bars per symbol
    start = (fri - dt.timedelta(days=args.since_buffer_days)).isoformat()
    end   = iso(fri)
    week_mask = pd.Series(pd.date_range(mon, fri, freq="D")).dt.date.tolist()

    rows = []
    scanned = 0
    for row in syms:
        sym = row["ticker"]
        df = get_daily_bars(key, sym, start, end)
        if df.empty or len(df) < args.lookback:
            continue

        df = df.sort_values("date").reset_index(drop=True)
        df["roll_high"] = df["high"].rolling(args.lookback, min_periods=args.lookback).max()

        # Did a 70D high occur during THIS week?
        wk = df[df["date"].isin(week_mask)].copy()
        if wk.empty:
            continue
        wk["is_hi"] = wk["high"] >= wk["roll_high"]
        hits = wk[wk["is_hi"]]
        if hits.empty:
            continue

        first_hit = hits.iloc[0]
        last_row  = df.iloc[-1]
        rows.append({
            "symbol": sym,
            "first_hit": first_hit["date"],
            "last_close": float(last_row["close"]),
            "market_cap": float(row["market_cap"]),
        })

        scanned += 1
        if scanned % 200 == 0:
            print(f"scanned {scanned}/{len(syms)} ...", flush=True)
        # gentle throttle (stay well below free-plan limits)
        time.sleep(0.06)

    if not rows:
        open(digest_path, "w").write(f"No 70-day highs this week ({iso(mon)}..{iso(fri)}).\n")
        open(csv_path, "w").write("symbol,first_hit,last_close,market_cap\n")
        print("No results.")
        return

    out = pd.DataFrame(rows).sort_values(["market_cap","symbol"], ascending=[False, True])
    out.to_csv(csv_path, index=False)

    # Short digest (top 20 by market cap)
    top = out.head(20).copy()
    top["market_cap_b"] = (top["market_cap"] / 1_000_000_000).round(1)
    lines = [f"70-day Highs this week (≥$1B mc)  —  {iso(mon)} .. {iso(fri)}  (top {len(top)}/{len(out)})",
             "symbol  first_hit   last_close   mcap($B)"]
    for _, r in top.iterrows():
        lines.append(f"{r['symbol']:5s}  {r['first_hit']}   {r['last_close']:>10.2f}   {r['market_cap_b']:>7.1f}")
    open(digest_path, "w").write("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}  (rows={len(out)})")
    print(f"Wrote {digest_path}")

if __name__ == "__main__":
    main()
